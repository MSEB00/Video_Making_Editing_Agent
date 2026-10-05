"""AI planning, rights-aware music selection, render critique, and one revision."""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Callable, Optional

from app.ai.creative_editor import CreativeAIConfigurationError, ShortFormEditingModel
from app.analysis.media_context import analyze_sources
from app.audio.jamendo_provider import JamendoMusicProvider, MusicRequirements
from app.audio.sfx_library import SfxLibrary
from app.config.config_loader import load_config
from app.editing.edit_plan import EditPlan
from app.editing.editor import EditOptions, render_edited_video
from app.editing.style_learner import EditingStyleLearner
from app.utilities.logger import get_logger

log = get_logger(__name__)


class ShortFormCreativeEditor:
    def __init__(
        self,
        model: Any = None,
        music_provider: Any = None,
        style_learner: Any = None,
        sfx_library: Any = None,
        renderer: Callable[..., pathlib.Path] = render_edited_video,
    ) -> None:
        self.settings = load_config().get("creative_editing", {})
        self.model = model
        self.model_configuration_error: Optional[CreativeAIConfigurationError] = None
        if self.model is None:
            try:
                self.model = ShortFormEditingModel()
            except CreativeAIConfigurationError as exc:
                self.model_configuration_error = exc
        self.music = music_provider or JamendoMusicProvider()
        examples_path = pathlib.Path(self.settings.get("learned_examples_path", "data/editing_examples.jsonl"))
        if not examples_path.is_absolute():
            examples_path = pathlib.Path(__file__).resolve().parents[2] / examples_path
        self.style_learner = style_learner or EditingStyleLearner(examples_path, promoted_only=True)
        self.sfx_library = sfx_library or SfxLibrary()
        self.renderer = renderer

    def create_edit(
        self,
        input_files: list[pathlib.Path],
        output_path: pathlib.Path,
        options: EditOptions,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> dict[str, Any]:
        def notify(message: str) -> None:
            if progress_callback:
                progress_callback("🧠", message)

        platform = options.platform
        platform_limit = 90 if platform == "instagram_reels" else 180
        if platform == "youtube_shorts":
            platform_limit = min(platform_limit, int(self.settings.get("max_short_duration_seconds", 60)))
        target_duration = max(1, min(options.target_duration, platform_limit))
        notify("Analyzing footage and sampling low-resolution frames...")
        media_context, frames = analyze_sources(
            input_files,
            max_sources=max(1, int(self.settings.get("max_source_videos", 12))),
        )
        events_by_source = self._detect_gameplay_events(input_files, media_context, options, notify)
        style_profile = self.style_learner.learn(platform)
        sfx_assets = self.sfx_library.list_assets()
        preferences = {
            "game": options.game or self.settings.get("default_game", "valorant"),
            "requested_style": options.creative_request,
            "music_preference": options.bgm_track,
            "sfx_preference": options.sfx_preference,
            "user_selected_aspect_ratio": options.aspect_ratio,
        }

        local_model = None
        self._active_local_model = None
        hosted_error = None
        try:
            from app.research.metadata_priors import youtube_duration_priors
            metadata_priors = youtube_duration_priors()
        except Exception:
            metadata_priors = {}
        try:
            if self.model is None:
                raise self.model_configuration_error or CreativeAIConfigurationError(
                    "No creative model provider is configured."
                )
            raw_plan = self.model.create_plan(
                media_context=media_context,
                frames=frames,
                style_profile=style_profile,
                platform=platform,
                target_duration=target_duration,
                user_request=options.creative_request,
                user_preferences=preferences,
                metadata_priors=metadata_priors,
                available_sfx=[
                    {
                        key: item[key]
                        for key in ("filename", "format", "duration", "description", "origin",
                                    "type", "mood", "intensity", "tags")
                        if item.get(key) is not None
                    }
                    for item in sfx_assets
                ],
            )
        except Exception as exc:
            from app.ai.local_editing import LocalShortFormEditingModel

            # Keep the (truncated) message, not just the type — remote diagnosis
            # of hosted-provider failures is otherwise impossible.
            hosted_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            log.warning("Hosted creative model unavailable (%s); using measured local editing fallback.", hosted_error)
            local_model = LocalShortFormEditingModel()
            self._active_local_model = local_model
            raw_plan = local_model.create_plan(
                input_files, media_context, platform, target_duration,
                seed=options.variation_seed,
            )
        max_shots = max(1, int(self.settings.get("max_planned_shots", 24)))
        plan = EditPlan.from_dict(raw_plan, len(media_context["sources"]), max_shots)
        self._validate_source_ranges(plan, media_context["sources"], target_duration)
        alignment = self._align_plan_to_events(plan, events_by_source, media_context["sources"])
        warnings: list[str] = []
        if hosted_error:
            warnings.append(
                f"Hosted creative model unavailable ({hosted_error}); selected strategy uses measured local features."
            )

        # Only an EXPLICIT "none/off/false" disables music. bgm_track=None is
        # the default meaning "no user preference" — the planner's own music
        # requirements (hosted or measured) must survive. (Previously None was
        # treated as a disable, so every default run silently dropped BGM.)
        bgm_pref = str(options.bgm_track or "").strip().lower()
        if bgm_pref in {"none", "off", "false", "no"}:
            plan.music_requirements = {}
        candidate_tracks = self._search_music(plan, warnings)
        if options.sfx_preference is True and not sfx_assets:
            warnings.append("SFX were requested, but assets/sfx has no usable sound files.")
        selected_track, license_record = self._select_music(
            media_context, plan, candidate_tracks, warnings
        )
        self._align_plan_to_music(plan, selected_track, options.transition_duration, media_context["sources"])
        render_options = self._renderer_options(
            options, plan, input_files, selected_track, sfx_assets, media_context["sources"]
        )
        notify(f"Rendering strategy '{plan.strategy}' ({len(plan.shots)} planned shots)...")
        self.renderer(
            [input_files[shot.source_index] for shot in plan.shots],
            output_path,
            render_options,
            progress_callback,
        )

        review_enabled = os.getenv(
            "CREATIVE_REVIEW_RENDER", str(self.settings.get("review_render", True))
        ).lower() not in {"0", "false", "no"}
        max_revisions = _revision_pass_limit(self.settings)
        revision_count = 0
        if review_enabled:
            reviewer = local_model or self.model
            review_context, review_frames = analyze_sources([output_path], max_sources=1)
            try:
                critique = reviewer.review_render(plan.to_dict(), review_context, review_frames)
            except Exception as exc:
                critique = None
                warnings.append(
                    f"Render review failed ({type(exc).__name__}: {str(exc)[:140]}); "
                    "shipping the render without critique."
                )
            if isinstance(critique, dict):
                plan.review = critique
            while (
                isinstance(critique, dict)
                and revision_count < max_revisions
                and critique.get("needs_revision")
                and critique.get("revision_request")
            ):
                notify(
                    f"Review found a material issue; generating revision "
                    f"{revision_count + 1} of up to {max_revisions}..."
                )
                previous_critique = str(critique.get("critique", ""))[:500]
                try:
                    revised_raw = reviewer.revise_plan(
                        plan=plan.to_dict(),
                        critique=critique,
                        media_context=media_context,
                        platform=platform,
                        target_duration=target_duration,
                    )
                    revised = EditPlan.from_dict(revised_raw, len(media_context["sources"]), max_shots)
                    self._validate_source_ranges(revised, media_context["sources"], target_duration)
                except Exception as exc:
                    warnings.append(
                        f"Revision failed ({type(exc).__name__}: {str(exc)[:140]}); "
                        "keeping the current render."
                    )
                    break
                alignment = self._align_plan_to_events(revised, events_by_source, media_context["sources"]) or alignment
                if revised.music_requirements != plan.music_requirements:
                    candidate_tracks = self._search_music(revised, warnings)
                    selected_track, license_record = self._select_music(
                        media_context, revised, candidate_tracks, warnings
                    )
                revised.selected_track = plan.selected_track
                if selected_track:
                    revised.selected_track = self._public_track(selected_track)
                self._align_plan_to_music(
                    revised, selected_track, options.transition_duration, media_context["sources"]
                )
                revised.review = critique
                revised.version = plan.version + 1
                plan = revised
                render_options = self._renderer_options(
                    options, plan, input_files, selected_track, sfx_assets, media_context["sources"]
                )
                notify(f"Rendering revision {revision_count + 1}...")
                self.renderer(
                    [input_files[shot.source_index] for shot in plan.shots],
                    output_path,
                    render_options,
                    progress_callback,
                )
                revision_count += 1
                review_context, review_frames = analyze_sources([output_path], max_sources=1)
                try:
                    critique = reviewer.review_render(plan.to_dict(), review_context, review_frames)
                except Exception as exc:
                    plan.review = {
                        "needs_revision": False,
                        "critique": f"Re-review unavailable ({type(exc).__name__}).",
                        "revision_request": "",
                        "revision_applied": True,
                        "revision_pass": revision_count,
                        "previous_critique": previous_critique,
                    }
                    warnings.append("Re-review after revision failed; keeping the revised render.")
                    break
                plan.review = {
                    **critique,
                    "revision_applied": True,
                    "revision_pass": revision_count,
                    "previous_critique": previous_critique,
                }

        artifact = plan.to_dict()
        artifact["revision_count"] = revision_count
        artifact["planning_mode"] = "local_feature_fallback" if local_model else "hosted_creative_model"
        artifact["model_version"] = (
            "local_feature_fallback_untrained"
            if local_model else getattr(self.model, "model", "hosted_model")
        )
        artifact["dataset_version"] = "rights_cleared_references_v001"
        artifact["music_license"] = license_record
        artifact["warnings"] = warnings
        if metadata_priors:
            artifact["youtube_metadata_priors"] = metadata_priors
        artifact["gameplay_events"] = {
            str(source.get("source_index", index)): source.get("gameplay_events") or []
            for index, source in enumerate(media_context.get("sources") or [])
        }
        artifact["event_alignment"] = {str(key): value for key, value in (alignment or {}).items()}
        width, height = (1080, 1920) if options.aspect_ratio == "9:16" else (1920, 1080)
        timeline = []
        cut_timestamps = []
        output_time = 0.0
        previous_duration = 0.0
        for shot_index, shot in enumerate(plan.shots):
            shot_duration = shot.end - shot.start
            if shot_index:
                if shot.transition != "cut":
                    output_time -= min(options.transition_duration, min(previous_duration, shot_duration) * 0.45)
                cut_timestamps.append(round(max(0.0, output_time), 3))
            timeline.append({
                "shot_index": shot_index,
                "source_index": shot.source_index,
                "source_path": str(input_files[shot.source_index].resolve()),
                "source_start": shot.start,
                "source_end": shot.end,
                "duration": round(shot_duration, 3),
                "output_start": round(max(0.0, output_time), 3),
                "transition": shot.transition,
                "caption": shot.caption,
                "visual_emphasis": shot.visual_emphasis,
                "playback_speed": 1.0,
                "source_audio_gain_db": self._source_audio_gain_db(
                    media_context["sources"][shot.source_index]
                ),
                "aligned_event": (alignment or {}).get(shot_index),
            })
            output_time += shot_duration
            previous_duration = shot_duration
        selection = (selected_track or {}).get("selection", {})
        track_duration = _as_float((selected_track or {}).get("duration"), 0.0)
        section_start = _as_float(selection.get("section_start"), 0.0)
        artifact["source_files"] = [
            {"source_index": index, "path": str(path.resolve())}
            for index, path in enumerate(input_files)
        ]
        artifact["timeline"] = timeline
        artifact["clip_order"] = [item["source_index"] for item in timeline]
        artifact["cut_timestamps"] = cut_timestamps
        artifact["captions"] = [
            {"time": item["output_start"], "text": item["caption"]}
            for item in timeline if item["caption"]
        ]
        artifact["sfx"] = plan.sound_design
        artifact["speed_changes"] = []
        artifact["render_settings"] = {
            "aspect_ratio": options.aspect_ratio,
            "resolution": f"{width}x{height}",
            "vertical_crop_mode": "center_fill" if options.aspect_ratio == "9:16" else None,
            "fps": 60,
            "video_codec": "h264",
            "audio_codec": "aac",
            "audio_normalization_target_lufs": options.audio_normalization_target_lufs,
            "source_audio_normalization_target_db": -27.0,
            "source_audio_max_gain_db": 24.0,
        }
        artifact["music_mix"].update({
            "enabled": selected_track is not None,
            "track_start_seconds": section_start if selected_track else None,
            "track_end_seconds": min(track_duration, section_start + plan.target_duration) if selected_track else None,
            "looping": bool(selected_track and section_start + plan.target_duration > track_duration),
            "volume": _as_float(selection.get("volume"), 0.0),
            "duck_under_original_audio": bool(selection.get("duck_under_original_audio", False)),
            "fade_in_seconds": 0.8 if selected_track else 0.0,
            "fade_out_seconds": 1.5 if selected_track else 0.0,
        })
        artifact_path = pathlib.Path(output_path).with_suffix(".edit-plan.json")
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact

    def _search_music(self, plan: EditPlan, warnings: list[str]) -> list[dict[str, Any]]:
        request = plan.music_requirements
        if not request:
            return []
        if not self.music.client_id:
            warnings.append("Jamendo music search skipped; configure JAMENDO_CLIENT_ID to enable it.")
            return []
        requirements = MusicRequirements(
            search=str(request.get("search", "")),
            tags=_as_tuple(request.get("tags")),
            fuzzytags=_as_tuple(request.get("fuzzytags")),
            speed=_as_tuple(request.get("speed")),
            instrumental=request.get("instrumental") if isinstance(request.get("instrumental"), bool) else None,
            duration_min=_as_int(request.get("duration_min")),
            duration_max=_as_int(request.get("duration_max")),
            content_id_free=True,
        )
        try:
            candidate_limit = int(self.settings.get("jamendo_candidate_limit", 20))
            candidates = self.music.search(requirements, limit=candidate_limit)
            if not candidates:
                broader_requirements = MusicRequirements(
                    search=requirements.search,
                    speed=requirements.speed,
                    instrumental=requirements.instrumental,
                    duration_min=requirements.duration_min,
                    duration_max=max(requirements.duration_max or 0, 600),
                    content_id_free=True,
                )
                candidates = self.music.search(broader_requirements, limit=candidate_limit)
                if candidates:
                    warnings.append("Jamendo returned no tracks for the narrow tags/duration; a broader licensed search was used.")
            for candidate in candidates[:5]:
                candidate["audio_features"] = self.music.analyze_audio(candidate.get("audio_url", ""))
            return candidates
        except Exception as exc:
            warnings.append(f"Jamendo search unavailable: {exc}")
            return []

    def _select_music(
        self,
        media_context: dict[str, Any],
        plan: EditPlan,
        candidates: list[dict[str, Any]],
        warnings: list[str],
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        if not candidates:
            return None, None
        public_candidates = [self._music_candidate_summary(track) for track in candidates[:12]]
        ranker = getattr(self, "_active_local_model", None) or self.model
        try:
            selection = ranker.rank_music(media_context, plan.to_dict(), public_candidates)
        except Exception as exc:
            # Music is an optional enhancement — a ranker failure must never
            # kill an otherwise renderable job.
            detail = f"{type(exc).__name__}: {str(exc)[:140]}"
            warnings.append(f"Music ranking failed ({detail}); keeping original gameplay audio.")
            log.warning("Music ranking failed; continuing without BGM: %s", detail)
            return None, None
        selected_id = str(selection.get("track_id") or "")
        track = next((item for item in candidates if item["id"] == selected_id), None)
        if track is None:
            warnings.append("The creative model declined the available Jamendo tracks; original gameplay audio will be retained.")
            return None, None
        try:
            destination = pathlib.Path(__file__).resolve().parents[2] / "assets" / "bgm"
            path, license_record = self.music.download_track(track, destination)
        except Exception as exc:
            warnings.append(f"Selected Jamendo track could not be downloaded: {exc}")
            return None, None
        license_record["commercial_use"] = "verify_with_provider_for_intended_use"
        if license_record.get("warning"):
            warnings.append(str(license_record["warning"]))
        track["local_path"] = str(path)
        track["selection"] = {
            "rationale": str(selection.get("rationale", ""))[:500],
            "section_start": max(
                0.0,
                min(
                    max(0.0, (track.get("duration") or 0.0) - 0.5),
                    _as_float(selection.get("section_start"), 0.0),
                ),
            ),
            "volume": max(0.0, min(0.6, _as_float(selection.get("volume"), 0.2))),
            "duck_under_original_audio": bool(selection.get("duck_under_original_audio", False)),
            "beat_sync_strength": max(0.0, min(1.0, _as_float(selection.get("beat_sync_strength"), 0.0))),
        }
        plan.selected_track = self._public_track(track)
        return track, license_record

    def _renderer_options(
        self,
        options: EditOptions,
        plan: EditPlan,
        input_files: list[pathlib.Path],
        selected_track: Optional[dict[str, Any]],
        sfx_assets: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> EditOptions:
        options.creative_mode = True
        options.randomize_clips = False
        options.transition_type = "fade"
        options.transition_sequence = [shot.transition for shot in plan.shots[1:]]
        options.planned_segments = [(shot.start, shot.end - shot.start) for shot in plan.shots]
        options.planned_captions = [shot.caption for shot in plan.shots]
        options.planned_emphasis = [shot.visual_emphasis for shot in plan.shots]
        options.bgm_track = None
        options.bgm_audio_path = pathlib.Path(selected_track["local_path"]) if selected_track else None
        options.bgm_start_seconds = selected_track.get("selection", {}).get("section_start", 0.0) if selected_track else 0.0
        options.bgm_volume = selected_track.get("selection", {}).get("volume", 0.0) if selected_track else 0.0
        options.duck_bgm_to_original_audio = selected_track.get("selection", {}).get("duck_under_original_audio", False) if selected_track else False
        options.sfx_clips = self._resolve_sfx(plan, sfx_assets, options.sfx_preference)
        options.planned_audio_gains_db = [
            self._source_audio_gain_db(sources[shot.source_index]) for shot in plan.shots
        ]
        return options

    @staticmethod
    def _source_audio_gain_db(source: dict[str, Any]) -> float:
        if not source.get("has_audio"):
            return 0.0
        source_level = _as_float(source.get("audio_mean_db"), -27.0)
        return round(max(0.0, min(24.0, -27.0 - source_level)), 2)

    def _resolve_sfx(
        self,
        plan: EditPlan,
        assets: list[dict[str, Any]],
        preference: Optional[bool],
    ) -> list[tuple[pathlib.Path, float, float]]:
        if preference is False:
            return []
        available = {item["filename"]: pathlib.Path(item["path"]) for item in assets}
        planned = []
        for item in plan.sound_design:
            filename = pathlib.Path(str(item.get("asset_filename", ""))).name
            path = available.get(filename)
            if path is None:
                continue
            when = max(0.0, min(plan.target_duration, _as_float(item.get("time"), 0.0)))
            volume = max(0.0, min(1.0, _as_float(item.get("level"), 0.25)))
            planned.append((path, when, volume))
        return planned

    @staticmethod
    def _align_plan_to_music(
        plan: EditPlan,
        track: Optional[dict[str, Any]],
        transition_duration: float,
        sources: list[dict[str, Any]],
    ) -> None:
        if not track:
            return
        features = track.get("audio_features", {})
        selection = track.get("selection", {})
        bpm = _as_float(features.get("bpm_estimate"), 0.0)
        confidence = _as_float(features.get("bpm_confidence"), 0.0)
        strength = max(0.0, min(1.0, _as_float(selection.get("beat_sync_strength"), 0.0)))
        if bpm <= 0 or confidence < 0.12 or strength <= 0:
            return

        durations = [shot.end - shot.start for shot in plan.shots]
        overlap = min(transition_duration, min(durations) * 0.45)
        overlap = max(0.0, overlap)
        beat_interval = 60.0 / bpm
        section_start = max(0.0, _as_float(selection.get("section_start"), 0.0))
        total_duration = sum(durations)
        aligned_boundaries = []

        for boundary in range(1, len(plan.shots)):
            incoming = plan.shots[boundary]
            if incoming.transition == "cut":
                continue
            cumulative = sum(durations[:boundary])
            midpoint = cumulative - (boundary - 0.5) * overlap
            beat_index = round((midpoint + section_start) / beat_interval)
            desired_midpoint = beat_index * beat_interval - section_start
            desired_cumulative = desired_midpoint + (boundary - 0.5) * overlap
            previous_duration = durations[boundary - 1]
            duration_before_previous = sum(durations[:boundary - 1])
            desired_duration = desired_cumulative - duration_before_previous
            adjusted_duration = previous_duration + (desired_duration - previous_duration) * strength
            source_duration = float(sources[plan.shots[boundary - 1].source_index]["duration"])
            maximum_duration = source_duration - plan.shots[boundary - 1].start
            adjusted_duration = max(0.35, min(adjusted_duration, maximum_duration))
            if adjusted_duration > previous_duration and total_duration + adjusted_duration - previous_duration > plan.target_duration:
                adjusted_duration = previous_duration
            if abs(adjusted_duration - previous_duration) < 0.02:
                continue
            durations[boundary - 1] = adjusted_duration
            plan.shots[boundary - 1].end = plan.shots[boundary - 1].start + adjusted_duration
            total_duration += adjusted_duration - previous_duration
            aligned_boundaries.append(boundary)

        plan.music_mix["beat_alignment"] = {
            "bpm_estimate": bpm,
            "confidence": confidence,
            "strength": strength,
            "boundaries": aligned_boundaries,
        }

    @staticmethod
    def _validate_source_ranges(
        plan: EditPlan,
        sources: list[dict[str, Any]],
        target_duration: float,
    ) -> None:
        plan.target_duration = min(plan.target_duration, target_duration)
        remaining_duration = plan.target_duration
        valid_shots = []
        for shot in plan.shots:
            if remaining_duration < 0.35:
                break
            duration = float(sources[shot.source_index]["duration"])
            if shot.start >= duration:
                continue
            shot.end = min(shot.end, duration)
            shot.end = min(shot.end, shot.start + remaining_duration)
            if shot.end - shot.start >= 0.35:
                valid_shots.append(shot)
                remaining_duration -= shot.end - shot.start
        if not valid_shots:
            raise ValueError("AI edit plan contains no usable footage ranges.")
        plan.shots = valid_shots

    def _detect_gameplay_events(
        self,
        input_files: list[pathlib.Path],
        media_context: dict[str, Any],
        options: EditOptions,
        notify: Callable[[str], None],
    ) -> dict[int, list[dict[str, Any]]]:
        """Detect timestamped gameplay events (e.g. kills) per analyzed source.

        Events are attached to media_context sources so hosted planners can
        reason about them, and returned per source_index for deterministic
        shot alignment. Absence of events is a valid outcome — planning then
        proceeds from measured motion/audio as before.
        """
        if not self.settings.get("align_to_gameplay_events", True):
            return {}
        allowed = {
            str(game).strip().lower()
            for game in (self.settings.get("event_detection_games") or ["valorant"])
        }
        game = str(options.game or self.settings.get("default_game", "")).strip().lower()
        if game not in allowed:
            return {}
        from app.analysis.games import get_event_detector

        detector = get_event_detector(game)
        if detector is None:
            return {}
        sources = media_context.get("sources") or []
        events_by_source: dict[int, list[dict[str, Any]]] = {}
        notify(f"Scanning {len(sources)} source(s) for {game} gameplay events...")
        for index, source in enumerate(sources):
            if index >= len(input_files):
                break
            try:
                events = detector.detect(input_files[index])
            except Exception as exc:
                log.warning("Event detection failed for %s: %s", input_files[index].name, exc)
                events = []
            source["gameplay_events"] = [
                {
                    "kind": event.get("kind"),
                    "start": event.get("start"),
                    "end": event.get("end", event.get("start")),
                    "confidence": event.get("confidence"),
                }
                for event in events
            ]
            if events:
                events_by_source[int(source.get("source_index", index))] = events
        total = sum(len(value) for value in events_by_source.values())
        notify(f"Detected {total} gameplay event(s) across {len(events_by_source)} source(s).")
        log.info(
            "Gameplay event detection complete",
            extra={"game": game, "events": total, "sources_with_events": len(events_by_source)},
        )
        return events_by_source

    def _align_plan_to_events(
        self,
        plan: EditPlan,
        events_by_source: dict[int, list[dict[str, Any]]],
        sources: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """Deterministically snap shots so detected events become the payoff.

        For each shot with events in its source: place the event end
        ``event_hold_seconds`` before the shot end (the elimination is fully
        visible plus a brief hold), extend the planned duration backwards as
        buildup, and never cut inside the event. Planner intent is respected:
        windows only move up to ``event_max_snap_shift_seconds``. This runs
        for BOTH hosted and local plans, so alignment never depends on the
        planner guessing timestamps.
        """
        if not events_by_source or not self.settings.get("align_to_gameplay_events", True):
            return {}
        hold = max(0.2, float(self.settings.get("event_hold_seconds", 1.0)))
        max_shift = max(0.0, float(self.settings.get("event_max_snap_shift_seconds", 6.0)))
        alignment: dict[int, dict[str, Any]] = {}
        for shot_index, shot in enumerate(plan.shots):
            events = events_by_source.get(shot.source_index)
            if not events or shot.source_index >= len(sources):
                continue
            try:
                source_duration = float(sources[shot.source_index].get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            if source_duration <= 1.0:
                continue
            length = shot.end - shot.start
            best: tuple[float, dict[str, Any], float, float] | None = None
            for event in events:
                event_start = float(event.get("start", 0.0))
                event_end = float(event.get("end") or event_start)
                new_end = min(source_duration - 0.05, event_end + hold)
                new_start = max(0.0, new_end - length)
                if event_start < new_start:
                    # Never cut mid-event: pull the window back to include it.
                    new_start = max(0.0, event_start - 0.3)
                    new_end = min(source_duration - 0.05, max(new_end, new_start + min(length, 1.5)))
                shift = max(abs(new_start - shot.start), abs(new_end - shot.end))
                if shift > max_shift:
                    continue
                contained = event_start >= shot.start and event_end <= shot.end
                cost = shift - (2.0 if contained else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, event, new_start, new_end)
            if best is None:
                continue
            _, event, new_start, new_end = best
            if abs(new_start - shot.start) < 0.05 and abs(new_end - shot.end) < 0.05:
                continue  # already aligned
            shot.start = round(new_start, 3)
            shot.end = round(new_end, 3)
            alignment[shot_index] = {
                "event_kind": event.get("kind"),
                "event_start": event.get("start"),
                "event_end": event.get("end", event.get("start")),
                "event_confidence": event.get("confidence"),
                "shot_start": shot.start,
                "shot_end": shot.end,
            }
        if alignment:
            log.info(
                "Aligned %d shot(s) to detected gameplay events", len(alignment),
                extra={"aligned_shots": len(alignment)},
            )
        return alignment

    @staticmethod
    def _public_track(track: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in track.items()
            if key not in {"audio_url", "download_url", "local_path"}
        }

    @staticmethod
    def _music_candidate_summary(track: dict[str, Any]) -> dict[str, Any]:
        summary = ShortFormCreativeEditor._public_track(track)
        musicinfo = track.get("musicinfo") or {}
        if isinstance(musicinfo, dict):
            summary["musicinfo"] = {
                key: musicinfo[key]
                for key in ("tags", "vocalinstrumental", "speed", "acousticelectric")
                if key in musicinfo
            }
        return summary


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(str(item) for item in value if item)
    return ()


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _revision_pass_limit(settings: dict[str, Any]) -> int:
    """Review→revise budget: env MAX_REVISION_PASSES overrides config, clamped 0..3."""
    raw = os.getenv("MAX_REVISION_PASSES", settings.get("max_revision_passes", 1))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(0, min(3, value))
