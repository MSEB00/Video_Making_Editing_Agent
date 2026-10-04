"""Rights-aware offline reference analysis, feedback capture, and model versioning."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
from typing import Any

from app.editing.style_learner import EditingStyleLearner, MIN_REFERENCE_EXAMPLES
from app.training.dataset_quality import split_by_creator
from app.utilities.ffmpeg_utils import get_audio_stream, get_ffmpeg_path, get_video_stream, probe


ALLOWED_RIGHTS_BASES = {"user_owned", "licensed", "public_domain", "explicitly_permitted"}
PLATFORMS = ("youtube_shorts", "instagram_reels", "gaming_shorts", "all")
DATASET_CATEGORIES = {"valorant", "fps", "gaming", "esports", "montage", "highlights", "funny"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Human validation vocabulary shared by dashboard and CLI feedback paths.
FEEDBACK_TAGS = frozenset({
    "too_many_effects", "too_slow", "too_fast", "bgm_mismatch",
    "captions_good", "transitions_bad", "hook_good",
    "kills_well_aligned", "kills_misaligned",
})


class ReferenceTrainingPipeline:
    """Analyze only media the caller affirmatively identifies as rights-cleared."""

    def __init__(self, training_root: pathlib.Path | None = None, examples_path: pathlib.Path | None = None) -> None:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        self.root = pathlib.Path(training_root or project_root / "training")
        self.references = self.root / "raw"
        self.metadata = self.root / "metadata"
        self.analysis = self.root / "analysis"
        self.features = self.root / "features"
        self.timelines = self.root / "timelines"
        self.datasets = self.root / "datasets"
        self.patterns = self.root / "models"
        self.manifest_path = self.root / "dataset.json"
        self.examples_path = pathlib.Path(examples_path or self.features / "examples.jsonl")
        for directory in (
            self.references, self.metadata, self.analysis, self.features,
            self.timelines, self.datasets, self.patterns,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def import_local_video(
        self,
        video_path: pathlib.Path,
        rights_basis: str,
        platform: str = "all",
        style_tags: list[str] | None = None,
        creator_group: str | None = None,
        annotations: dict[str, Any] | None = None,
        category: str = "gaming",
        source_url: str | None = None,
        license_url: str | None = None,
        attribution_required: bool = True,
        view_count: int | None = None,
        license_name: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = pathlib.Path(video_path)
        source_metadata = source_metadata or {}
        source_url = source_url or source_metadata.get("url")
        license_url = license_url or source_metadata.get("license_url")
        license_name = license_name or source_metadata.get("license")
        creator_group = creator_group or source_metadata.get("channel_id") or source_metadata.get("channel")
        view_count = view_count if view_count is not None else _optional_int(source_metadata.get("view_count"))
        attribution_required = bool(source_metadata.get("attribution_required", attribution_required))
        if rights_basis not in ALLOWED_RIGHTS_BASES:
            raise ValueError(f"rights_basis must be one of: {', '.join(sorted(ALLOWED_RIGHTS_BASES))}")
        if platform not in PLATFORMS:
            raise ValueError(f"platform must be one of: {', '.join(PLATFORMS)}")
        if category not in DATASET_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(DATASET_CATEGORIES))}")
        if source.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video type: {source.suffix}")
        if not source.is_file():
            raise FileNotFoundError(source)

        reference_id = self._sha256(source)
        existing = self._manifest_item(reference_id)
        if existing:
            existing_features = self.features / f"{reference_id}.json"
            if existing_features.is_file():
                return json.loads(existing_features.read_text(encoding="utf-8"))
        stored = self.references / category / f"{reference_id[:20]}{source.suffix.lower()}"
        if not stored.exists():
            metadata = probe(source)
            video = get_video_stream(metadata)
            if not video:
                raise ValueError("Reference media does not contain a video stream.")
            duration = _positive_float(metadata.get("format", {}).get("duration"))
            if duration is None:
                raise ValueError("Reference media has no valid duration.")
            audio = get_audio_stream(metadata)
            cuts = self._scene_cut_times(source)
            audio_levels = self._audio_levels(source) if audio else (None, None)
            silence_ratio = self._silence_ratio(source, duration) if audio else None
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, stored)
        else:
            metadata = probe(stored)
            video = get_video_stream(metadata) or {}
            duration = _positive_float(metadata.get("format", {}).get("duration"))
            if duration is None:
                raise ValueError("Reference media has no valid duration.")
            audio = get_audio_stream(metadata)
            cuts = self._scene_cut_times(stored)
            audio_levels = self._audio_levels(stored) if audio else (None, None)
            silence_ratio = self._silence_ratio(stored, duration) if audio else None
        audio_db, audio_peak_db = audio_levels
        shot_std, pacing_irregularity = _shot_pacing_statistics(cuts, duration)
        fps = _parse_rate(video.get("avg_frame_rate"))
        width, height = video.get("width"), video.get("height")
        video_format = None
        if width is not None and height is not None:
            video_format = "square" if width == height else "landscape" if width > height else "vertical"
        file_size = stored.stat().st_size
        features: dict[str, Any] = {
            "duration": round(duration, 3),
            "width": width,
            "height": height,
            "fps": fps,
            "format": video_format,
            "video_codec": video.get("codec_name"),
            "has_audio": audio is not None,
            "scene_cut_count": len(cuts),
            "cut_density": round(len(cuts) / duration, 4) if duration > 0 else 0.0,
            "average_shot_duration": round(duration / (len(cuts) + 1), 3) if duration > 0 else None,
            "shot_duration_std": shot_std,
            "pacing_irregularity": pacing_irregularity,
            "silence_ratio": silence_ratio,
            "audio_mean_db": audio_db,
            "audio_peak_db": audio_peak_db,
            "audio_dynamic_range_db": (
                round(audio_peak_db - audio_db, 3)
                if audio_db is not None and audio_peak_db is not None else None
            ),
            "audio_intensity": round(max(0.0, min(1.0, (audio_db + 60) / 60)), 3) if audio_db is not None else None,
            "bgm_presence": None,
            "caption_density": None,
            "effect_density": None,
            "sfx_density": None,
        }
        annotations = annotations or {}
        features.update(annotations.get("features", {}))
        example = {
            "reference_id": reference_id,
            "platform": platform,
            "style_tags": style_tags or ["general"],
            "creator_group": creator_group,
            "category": category,
            "rights_basis": rights_basis,
            "source_url": source_url,
            "license_url": license_url,
            "license_name": license_name,
            "attribution_required": attribution_required,
            "view_count": view_count,
            "sha256": reference_id,
            "size_bytes": file_size,
            "features": features,
            "moments": annotations.get("moments", []),
            "source_path": str(stored.resolve()),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self._write_json(self.metadata / f"{reference_id}.json", {
            "reference_id": reference_id,
            "title": str(source_metadata.get("title") or source.stem),
            "filename": stored.name,
            "platform": platform,
            "rights_basis": rights_basis,
            "creator_group": creator_group,
            "category": category,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "format": video_format,
            "video_codec": video.get("codec_name"),
            "has_audio": audio is not None,
            "sha256": reference_id,
            "size_bytes": file_size,
            "source_url": source_url,
            "license": license_name,
            "rights_basis": rights_basis,
            "license_url": license_url,
            "attribution_required": attribution_required,
            "view_count": view_count,
            "like_count": _optional_int(source_metadata.get("like_count")),
            "comment_count": _optional_int(source_metadata.get("comment_count")),
            "published_at": source_metadata.get("published_at"),
            "description": str(source_metadata.get("description", ""))[:5000],
            "channel_id": source_metadata.get("channel_id"),
            "query": source_metadata.get("query"),
        })
        self._write_json(self.analysis / f"{reference_id}.json", {
            "reference_id": reference_id,
            "scene_cut_times": cuts,
            "annotations": annotations,
        })
        self._write_json(self.features / f"{reference_id}.json", example)
        self._write_json(self.timelines / f"{reference_id}.json", {
            "reference_id": reference_id,
            "segments": annotations.get("segments", []),
            "scene_cut_times": cuts,
        })
        self._append_jsonl(self.examples_path, example)
        self._append_manifest({
            "sample_id": f"{category}_{reference_id[:12]}",
            "source": {
                "platform": source_metadata.get("platform") or ("youtube" if _youtube_video_id(source_url) else "local"),
                "video_id": source_metadata.get("video_id") or _youtube_video_id(source_url),
                "title": str(source_metadata.get("title") or source.stem),
                "creator": source_metadata.get("channel") or creator_group,
                "channel_id": source_metadata.get("channel_id"),
                "source_url": source_url,
                "license": license_name,
                "license_name": license_name,
                "rights_basis": rights_basis,
                "license_url": license_url,
                "attribution_required": attribution_required,
                "category": category,
                "view_count": view_count,
                "like_count": _optional_int(source_metadata.get("like_count")),
                "comment_count": _optional_int(source_metadata.get("comment_count")),
                "published_at": source_metadata.get("published_at"),
                "query": source_metadata.get("query"),
            },
            "file": str(stored.relative_to(self.root)).replace("\\", "/"),
            "sha256": reference_id,
            "size_bytes": file_size,
            "metadata_path": str((self.metadata / f"{reference_id}.json").relative_to(self.root)).replace("\\", "/"),
            "analysis_path": str((self.analysis / f"{reference_id}.json").relative_to(self.root)).replace("\\", "/"),
            "features_path": str((self.features / f"{reference_id}.json").relative_to(self.root)).replace("\\", "/"),
            "timeline_path": str((self.timelines / f"{reference_id}.json").relative_to(self.root)).replace("\\", "/"),
        })
        return example

    def _manifest_item(self, reference_id: str) -> dict[str, Any] | None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return next((item for item in manifest.get("items", []) if item.get("sha256") == reference_id), None)

    def _append_manifest(self, item: dict[str, Any]) -> None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            manifest = {
                "dataset_name": "short_form_gaming_cc",
                "version": "v001",
                "created_at": now,
                "items": [],
            }
        if not any(existing.get("sha256") == item["sha256"] for existing in manifest["items"]):
            manifest["items"].append(item)
        manifest["total_items"] = len(manifest["items"])
        manifest["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._write_json(self.manifest_path, manifest)

    def record_feedback(
        self,
        edit_id: str,
        rating: int,
        tags: list[str] | None = None,
        notes: str = "",
        context: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        if rating not in (-1, 1):
            raise ValueError("rating must be 1 (good) or -1 (bad).")
        record = {
            "edit_id": str(edit_id),
            "rating": rating,
            "tags": [str(tag)[:80] for tag in (tags or [])[:20]],
            "notes": str(notes)[:1000],
            "context": context or {},
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        path = self.root / "feedback" / "edits.jsonl"
        self._append_jsonl(path, record)
        return path

    def train_candidate(self, platforms: tuple[str, ...] = PLATFORMS[:-1]) -> dict[str, Any]:
        examples = self._read_examples()
        unique = {str(item.get("reference_id")): item for item in examples if item.get("reference_id")}
        if len(unique) < MIN_REFERENCE_EXAMPLES:
            return {"status": "insufficient_references", "example_count": len(unique), "minimum": MIN_REFERENCE_EXAMPLES}

        partitions = split_by_creator(list(unique.values()))
        training = partitions["train"]
        validation = partitions["validation"]
        testing = partitions["test"]
        if len(training) < MIN_REFERENCE_EXAMPLES:
            return {
                "status": "insufficient_training_references",
                "example_count": len(unique),
                "training_count": len(training),
                "minimum_training": MIN_REFERENCE_EXAMPLES,
            }
        if not validation or not testing:
            creator_count = len({item.get("creator_group") or item.get("reference_id") for item in unique.values()})
            return {"status": "insufficient_creator_diversity", "creator_count": creator_count}

        learner = EditingStyleLearner(self.examples_path)
        profiles = {
            platform: learner.build_profile(
                platform,
                [item for item in training if item.get("platform") in (platform, "all")],
            )
            for platform in platforms
        }
        if not any(profile.get("training_status") == "learned" for profile in profiles.values()):
            return {"status": "insufficient_platform_references", "example_count": len(unique)}
        score = self._validation_mae(profiles, validation, platforms)
        baseline_score = self._baseline_mae(training, validation, platforms)
        test_score = self._validation_mae(profiles, testing, platforms)
        if not math.isfinite(score):
            return {"status": "insufficient_validation_features", "example_count": len(unique)}

        versions = sorted(self.patterns.glob("style_model_v*.json"))
        version = f"style_model_v{len(versions) + 1:03d}"
        creator_count = len({item.get("creator_group") or item.get("reference_id") for item in unique.values()})
        candidate = {
            "version": version,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset": {
                "reference_count": len(unique),
                "creator_count": creator_count,
                "training_count": len(training),
                "validation_count": len(validation),
                "test_count": len(testing),
                "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
            "validation": {
                "mean_absolute_error": score,
                "baseline_mean_absolute_error": baseline_score,
                "test_mean_absolute_error": test_score,
            },
            "feedback_signals": self._feedback_signals(),
            "profiles": profiles,
        }
        candidate_path = self.patterns / f"{version}.json"
        self._write_json(candidate_path, candidate)
        self._write_json(self.datasets / f"splits_{version}.json", {
            "version": version,
            "train_reference_ids": [item.get("reference_id") for item in training],
            "validation_reference_ids": [item.get("reference_id") for item in validation],
            "test_reference_ids": [item.get("reference_id") for item in testing],
        })

        active_path = self.patterns / "active.json"
        active = None
        if active_path.is_file():
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                active = None
        active_score = None
        if active is not None:
            active_score = self._validation_mae(active.get("profiles", {}), validation, platforms)
        beats_baseline = math.isfinite(baseline_score) and score < baseline_score
        promoted = beats_baseline and (
            active is None or (active_score is not None and math.isfinite(active_score) and score < active_score)
        )
        candidate["validation"]["active_model_mae"] = active_score
        self._write_json(candidate_path, candidate)
        self._write_patterns(version, profiles)
        if promoted:
            self._write_json(active_path, candidate)
        return {
            "status": "promoted" if promoted else "rejected",
            "version": version,
            "training_count": len(training),
            "validation_count": len(validation),
            "test_count": len(testing),
            "example_count": len(unique),
            "creator_count": creator_count,
            "validation_mae": score,
            "test_mae": test_score,
            "baseline_mae": baseline_score,
            "active_model_mae": active_score,
            "path": str(candidate_path),
        }

    def _write_patterns(self, version: str, profiles: dict[str, dict[str, Any]]) -> pathlib.Path:
        """Persist an inspectable JSONL digest of what the model version learned."""
        path = self.patterns / f"{version}_patterns.jsonl"
        records: list[dict[str, Any]] = []
        for platform, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            for relationship in profile.get("relationships", []) or []:
                records.append({"type": "relationship", "platform": platform, "model_version": version, **relationship})
            clusters = profile.get("style_clusters") or {}
            for cluster in clusters.get("clusters", []) or []:
                records.append({
                    "type": "style_cluster",
                    "platform": platform,
                    "model_version": version,
                    "cluster_id": cluster.get("cluster_id"),
                    "label": cluster.get("label"),
                    "size": cluster.get("size"),
                    "dominant_tags": cluster.get("dominant_tags"),
                    "distinctive_features": cluster.get("distinctive_features"),
                })
            for context, decisions in (profile.get("conditional_edit_probabilities") or {}).items():
                for decision, stats in decisions.items():
                    records.append({
                        "type": "conditional_edit_probability",
                        "platform": platform,
                        "model_version": version,
                        "context": context,
                        "decision": decision,
                        **stats,
                    })
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        return path

    def _feedback_signals(self) -> dict[str, dict[str, int]]:
        path = self.root / "feedback" / "edits.jsonl"
        counts: dict[str, dict[str, int]] = {}
        if not path.is_file():
            return counts
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rating = record.get("rating")
                if rating not in (-1, 1):
                    continue
                for tag in record.get("tags", []):
                    bucket = counts.setdefault(str(tag), {"positive": 0, "negative": 0})
                    bucket["positive" if rating > 0 else "negative"] += 1
        return counts

    @staticmethod
    def _audio_levels(path: pathlib.Path) -> tuple[float | None, float | None]:
        """Return (mean_volume_db, max_volume_db) via a single volumedetect pass."""
        command = [
            get_ffmpeg_path(), "-hide_banner", "-i", str(path),
            "-map", "a:0", "-af", "volumedetect", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        import re

        def _db(pattern: str) -> float | None:
            match = re.search(pattern + r":\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB", result.stderr)
            return float(match.group(1)) if match else None

        return _db(r"mean_volume"), _db(r"max_volume")

    @staticmethod
    def _silence_ratio(path: pathlib.Path, duration: float) -> float | None:
        """Fraction of the timeline that is silent (-35 dB, >= 0.4 s windows)."""
        if not duration or duration <= 0:
            return None
        command = [
            get_ffmpeg_path(), "-hide_banner", "-i", str(path),
            "-map", "a:0", "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            return None
        import re

        silent_seconds = sum(
            float(value)
            for value in re.findall(r"silence_duration:\s*([0-9]+(?:\.[0-9]+)?)", result.stderr)
        )
        return round(min(1.0, silent_seconds / duration), 4)

    @staticmethod
    def _scene_cut_times(path: pathlib.Path) -> list[float]:
        command = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info", "-i", str(path),
            "-vf", "select='gt(scene,0.3)',showinfo", "-an", "-f", "null", "-",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg scene analysis failed: {result.stderr[-500:]}")
        import re

        return [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]

    def _read_examples(self) -> list[dict[str, Any]]:
        records = []
        if self.examples_path.is_file():
            with self.examples_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            records.append(item)
                    except json.JSONDecodeError:
                        continue
        records.extend(self._read_approved_research_examples())
        return records

    def _read_approved_research_examples(self) -> list[dict[str, Any]]:
        research_root = self.root / "research"
        videos_path = research_root / "videos.jsonl"
        observations_path = research_root / "observations.jsonl"
        if not videos_path.is_file() or not observations_path.is_file():
            return []

        videos = {}
        with videos_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("video_id"):
                    videos[str(item["video_id"])] = item

        grouped: dict[str, list[dict[str, Any]]] = {}
        with observations_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    observation = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(observation, dict) or observation.get("training_eligible") is not True:
                    continue
                source = observation.get("source") or {}
                video_id = str(source.get("video_id") or "")
                video = videos.get(video_id)
                rights = observation.get("rights_confirmation") or {}
                if (
                    not video_id
                    or video is None
                    or video.get("license") != "creativeCommon"
                    or rights.get("creator_and_audio_rights_confirmed_by_operator") is not True
                    or rights.get("youtube_derived_data_approval_configured") is not True
                ):
                    continue
                grouped.setdefault(video_id, []).append(observation)

        examples = []
        for video_id, items in grouped.items():
            video = videos[video_id]
            feature_values: dict[str, list[float]] = {}
            moments = []
            for item in items:
                context = item.get("content_context") or {}
                editing = item.get("editing_decisions") or {}
                observed_edit = {}
                for field, value in editing.items():
                    if value == "observed":
                        observed_edit[field] = 1.0
                    elif value == "not_observed":
                        observed_edit[field] = 0.0
                moment_features = {}
                for field in ("visual_intensity", "audio_intensity"):
                    value = _finite_number_or_none(context.get(field))
                    if value is not None:
                        moment_features[field] = value
                        feature_values.setdefault(field, []).append(value)
                speech = context.get("speech_present")
                if isinstance(speech, bool):
                    moment_features["speech_present"] = 1.0 if speech else 0.0
                    feature_values.setdefault("speech_present", []).append(moment_features["speech_present"])
                for field, value in observed_edit.items():
                    feature_values.setdefault(field, []).append(value)
                if moment_features or observed_edit:
                    moments.append({
                        "moment": moment_features,
                        "observed_edit": observed_edit,
                        "start_seconds": item.get("start_seconds"),
                        "end_seconds": item.get("end_seconds"),
                    })
            if not moments:
                continue
            features = {
                name: round(sum(values) / len(values), 4)
                for name, values in feature_values.items() if values
            }
            duration = _positive_float(video.get("duration_seconds"))
            if duration is not None:
                features["duration"] = round(duration, 3)
            creator_group = str(video.get("channel_id") or video.get("channel") or video_id)
            examples.append({
                "reference_id": f"youtube_{video_id}",
                "platform": "youtube_shorts",
                "style_tags": [str(video.get("category") or "gaming")],
                "creator_group": creator_group,
                "category": str(video.get("category") or "gaming"),
                "rights_basis": "explicitly_permitted",
                "features": features,
                "moments": moments,
                "provenance": {
                    "platform": "youtube",
                    "video_id": video_id,
                    "url": video.get("url"),
                    "license": video.get("license"),
                    "media_saved": False,
                    "observation_ids": [item.get("observation_id") for item in items],
                },
            })
        return examples

    @staticmethod
    def _validation_mae(profiles: dict[str, dict[str, Any]], validation: list[dict[str, Any]], platforms: tuple[str, ...]) -> float:
        errors = []
        for example in validation:
            platform = example.get("platform")
            if platform not in platforms:
                continue
            profile = profiles.get(platform, {})
            predicted = profile.get("learned_averages", {})
            style_profiles = profile.get("style_profiles", {})
            tags = example.get("style_tags") or []
            if isinstance(tags, str):
                tags = [tags]
            matching = [style_profiles[tag] for tag in tags if tag in style_profiles]
            if matching:
                feature_values: dict[str, list[float]] = {}
                for style_profile in matching:
                    for name, value in style_profile.items():
                        feature_values.setdefault(name, []).append(float(value))
                predicted = {name: sum(values) / len(values) for name, values in feature_values.items()}
            observed = example.get("features", {})
            for name, actual in observed.items():
                try:
                    errors.append(abs(float(actual) - float(predicted[name])))
                except (KeyError, TypeError, ValueError):
                    continue
        return round(sum(errors) / len(errors), 6) if errors else float("inf")

    @staticmethod
    def _baseline_mae(training: list[dict[str, Any]], validation: list[dict[str, Any]], platforms: tuple[str, ...]) -> float:
        values: dict[tuple[str, str], list[float]] = {}
        for example in training:
            platform = example.get("platform")
            if platform not in platforms:
                continue
            for name, value in (example.get("features") or {}).items():
                try:
                    values.setdefault((platform, name), []).append(float(value))
                except (TypeError, ValueError):
                    continue
        baselines = {key: sorted(items)[len(items) // 2] for key, items in values.items() if items}
        errors = []
        for example in validation:
            platform = example.get("platform")
            if platform not in platforms:
                continue
            for name, value in (example.get("features") or {}).items():
                try:
                    errors.append(abs(float(value) - baselines[(platform, name)]))
                except (KeyError, TypeError, ValueError):
                    continue
        return round(sum(errors) / len(errors), 6) if errors else float("inf")

    @staticmethod
    def _sha256(path: pathlib.Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _append_jsonl(path: pathlib.Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=True) + "\n")

    @staticmethod
    def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def _finite_number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        return round(float(numerator) / float(denominator), 3)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def _youtube_video_id(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/") or None
    if parsed.hostname and parsed.hostname.endswith("youtube.com"):
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

def _shot_pacing_statistics(cuts: list[float], duration: float) -> tuple[float | None, float | None]:
    """Return (shot_duration_std, pacing_irregularity) from scene-cut times.

    pacing_irregularity is the coefficient of variation of shot durations —
    low values indicate steady rhythm, high values indicate uneven pacing.
    """
    if not cuts or not duration or duration <= 0:
        return None, None
    import statistics as _statistics

    boundaries = [0.0] + [float(cut) for cut in cuts if 0 < cut < duration] + [duration]
    shots = [right - left for left, right in zip(boundaries, boundaries[1:]) if right > left]
    if len(shots) < 2:
        return None, None
    std = _statistics.stdev(shots)
    mean = _statistics.mean(shots)
    irregularity = round(std / mean, 4) if mean > 0 else None
    return round(std, 4), irregularity
