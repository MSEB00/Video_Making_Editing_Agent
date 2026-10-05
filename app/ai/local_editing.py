"""Feature-based editing fallback for environments without a hosted model."""
from __future__ import annotations

import math
import pathlib
import random
import struct
import subprocess
from typing import Any

from PIL import Image, ImageChops, ImageStat

from app.utilities.ffmpeg_utils import get_ffmpeg_path


class LocalShortFormEditingModel:
    def __init__(self) -> None:
        self.features: list[dict[str, Any]] = []
        self._rng: random.Random | None = None

    def create_plan(
        self,
        source_paths: list[pathlib.Path],
        media_context: dict[str, Any],
        platform: str,
        target_duration: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # Seeded jitter (±4%) breaks exact ties between near-equal windows so
        # repeated fallback runs on the same folder are not byte-identical,
        # while remaining fully reproducible for a given seed (job id).
        self._rng = random.Random(seed) if seed is not None else None
        self.features = [self._analyze_source(path) for path in source_paths]
        all_activity = [score for item in self.features for score in item["activity"]]
        all_motion = [score for item in self.features for score in item["motion"]]
        all_audio = [score for item in self.features for score in item["audio"]]
        mean_activity = _mean(all_activity)
        mean_motion = _mean(all_motion)
        mean_audio = _mean(all_audio)

        strategies = [
            ("fast_aggressive", 2.5, "cut"),
            ("cinematic_build", 4.0, "fade"),
            ("minimal_high_impact", 5.5, "cut"),
        ]
        candidates = []
        continuous = self._best_contiguous_window(target_duration)
        if continuous:
            candidates.append({
                "name": "continuous_sequence",
                "transition": "cut",
                "windows": [continuous],
                "score": round(0.7 * continuous["score"] + 0.3 * continuous["peak"], 4),
                "coverage": continuous["score"],
                "diversity": 1.0,
            })
        for name, shot_duration, transition in strategies:
            windows = self._rank_windows(shot_duration)
            selected = self._diverse_windows(windows, target_duration, shot_duration)
            if not selected:
                continue
            selected = sorted(selected, key=lambda item: (item["source_index"], item["start"]))
            coverage = _mean([window["score"] for window in selected])
            diversity = len({window["source_index"] for window in selected}) / len(selected)
            peak = max(window["score"] for window in selected)
            source_switches = sum(
                left["source_index"] != right["source_index"]
                for left, right in zip(selected, selected[1:])
            )
            cut_count = len(selected) - 1
            quality = (
                0.7 * coverage
                + 0.3 * peak
                + 0.03 * diversity
                - 0.02 * cut_count
                - 0.08 * source_switches
            )
            candidates.append({
                "name": name,
                "transition": transition,
                "windows": selected,
                "score": round(quality, 4),
                "coverage": coverage,
                "diversity": diversity,
                "cut_count": cut_count,
                "source_switches": source_switches,
            })
        if not candidates:
            raise RuntimeError("Local analysis found no usable source segments.")
        chosen = max(candidates, key=lambda item: (item["score"], item["diversity"], item["name"]))
        ordered = self._story_order(chosen["windows"])
        shots = []
        remaining = float(target_duration)
        for index, window in enumerate(ordered):
            duration = min(window["end"] - window["start"], remaining)
            if duration < 0.5:
                break
            shots.append({
                "source_index": window["source_index"],
                "start": round(window["start"], 3),
                "end": round(window["start"] + duration, 3),
                "role": "hook" if index == 0 else "payoff" if index == len(ordered) - 1 else "action",
                "transition": "cut" if index == 0 else chosen["transition"],
                "caption": None,
                "visual_emphasis": [],
            })
            remaining -= duration

        if mean_activity >= 0.58:
            music_search = "energetic electronic instrumental with driving rhythm"
            music_tags, speed = ["energetic", "electronic", "instrumental"], ["high", "veryhigh"]
        elif mean_activity >= 0.35:
            music_search = "driving cinematic instrumental with a clear pulse"
            music_tags, speed = ["cinematic", "instrumental", "electronic"], ["medium", "high"]
        else:
            music_search = "restrained atmospheric instrumental with gradual build"
            music_tags, speed = ["cinematic", "ambient", "instrumental"], ["low", "medium"]
        return {
            "platform": platform,
            "strategy": chosen["name"],
            "rationale": (
                "Compared a continuous passage with progressively faster alternatives using measured "
                "motion, game-audio activity, continuity, and cut/source-switch costs. No reference-trained "
                "model was available; effects were omitted because no context-aware evidence justified them."
            ),
            "target_duration": round(sum(shot["end"] - shot["start"] for shot in shots), 3),
            "alternatives": [
                {
                    "strategy": item["name"],
                    "score": item["score"],
                    "activity_coverage": round(item["coverage"], 3),
                    "cuts": item.get("cut_count", 0),
                    "source_switches": item.get("source_switches", 0),
                }
                for item in sorted(candidates, key=lambda item: item["score"], reverse=True)
            ],
            "shots": shots,
            "music_requirements": {
                "search": music_search,
                "tags": music_tags,
                "speed": speed,
                "instrumental": True,
                "duration_min": max(20, int(target_duration * 0.7)),
                "duration_max": max(600, target_duration * 3),
            },
            "music_mix": {
                "volume": 0.12 if mean_audio >= 0.35 else 0.2,
                "duck_under_original_audio": mean_audio >= 0.25,
                "fade_in_seconds": 0.8,
                "fade_out_seconds": 1.5,
                "beat_alignment": "only when track analysis has sufficient confidence",
            },
            "sound_design": [],
            "ending": "Resolve on the strongest remaining distinct gameplay segment.",
            "local_analysis": {
                "mean_activity": round(mean_activity, 4),
                "mean_visual_motion": round(mean_motion, 4),
                "mean_audio_energy": round(mean_audio, 4),
                "window_seconds": 0.5,
                "training_status": "insufficient_references",
            },
        }

    def rank_music(
        self,
        media_context: dict[str, Any],
        plan: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        requirements = plan.get("music_requirements", {})
        wanted = set(" ".join([str(requirements.get("search", "")), *requirements.get("tags", [])]).lower().split())
        target_duration = float(plan.get("target_duration", 0))
        ranked = []
        for track in candidates:
            info = track.get("musicinfo") or {}
            searchable = " ".join(str(value) for value in (track.get("title"), track.get("artist"), info)).lower()
            words = set(searchable.replace("{", " ").replace("}", " ").replace("'", " ").replace(",", " ").split())
            overlap = len(wanted & words) / max(1, len(wanted))
            duration = float(track.get("duration") or 0)
            duration_fit = 1.0 if duration >= target_duration else duration / max(1.0, target_duration)
            license_fit = 1.0 if track.get("download_allowed") and track.get("license_url") else 0.0
            speed = str(info.get("speed", "")).lower()
            speed_fit = 1.0 if speed in requirements.get("speed", []) else 0.4
            quality = 0.35 * overlap + 0.25 * duration_fit + 0.25 * license_fit + 0.15 * speed_fit
            ranked.append((quality, track))
        if not ranked:
            return {"track_id": None, "rationale": "No downloadable, suitable music candidate was available."}
        _, selected = max(ranked, key=lambda item: (item[0], item[1].get("id", "")))
        audio_features = selected.get("audio_features", {})
        beat_confidence = float(audio_features.get("bpm_confidence") or 0.0)
        if beat_confidence < 0.32:
            return {
                "track_id": None,
                "rationale": "No candidate had enough beat evidence to synchronize with this cut; preserve the original gameplay audio instead.",
            }
        mean_audio = _mean([score for source in self.features for score in source["audio"]])
        return {
            "track_id": selected.get("id"),
            "rationale": "Highest match for the footage-derived mood, speed, duration, and available license metadata.",
            "section_start": 0.0,
            "volume": 0.11 if mean_audio >= 0.35 else 0.18,
            "duck_under_original_audio": mean_audio >= 0.25,
            "beat_sync_strength": min(1.0, max(0.55, beat_confidence)),
        }

    @staticmethod
    def review_render(
        plan: dict[str, Any],
        review_context: dict[str, Any],
        frames: list[dict[str, Any]],
    ) -> dict[str, Any]:
        source = (review_context.get("sources") or [{}])[0]
        issues = []
        if source.get("width") != 1080 or source.get("height") != 1920:
            issues.append("Rendered dimensions are not 1080x1920.")
        if not source.get("has_audio"):
            issues.append("Rendered output has no audio stream.")
        if float(source.get("duration") or 0) <= 0:
            issues.append("Rendered output has no valid duration.")
        if not frames:
            issues.append("No review frame could be decoded.")
        return {
            "needs_revision": bool(issues),
            "critique": " ".join(issues) if issues else "Render metadata and sampled frames passed the local output checks.",
            "revision_request": "",
        }

    def revise_plan(self, plan: dict[str, Any], **_: Any) -> dict[str, Any]:
        return plan

    def _analyze_source(self, path: pathlib.Path) -> dict[str, Any]:
        frame_width, frame_height = 160, 90
        frame_size = frame_width * frame_height * 3
        frames_result = subprocess.run(
            [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-vf", "fps=2,scale=160:90", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            capture_output=True,
            timeout=120,
        )
        motion = []
        previous = None
        for offset in range(0, len(frames_result.stdout) - frame_size + 1, frame_size):
            current = Image.frombytes("RGB", (frame_width, frame_height), frames_result.stdout[offset:offset + frame_size])
            if previous is not None:
                difference = ImageChops.difference(previous, current)
                pixel_delta = sum(ImageStat.Stat(difference).mean) / 3
                motion.append(min(1.0, pixel_delta / 18.0))
            previous = current

        audio_result = subprocess.run(
            [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1",
            ],
            capture_output=True,
            timeout=120,
        )
        sample_bytes = audio_result.stdout[:len(audio_result.stdout) // 2 * 2]
        samples = [sample[0] for sample in struct.iter_unpack("<h", sample_bytes)]
        audio = []
        window_size = 4000
        for offset in range(0, len(samples), window_size):
            window = samples[offset:offset + window_size]
            if not window:
                continue
            rms = math.sqrt(sum(sample * sample for sample in window) / len(window)) / 32768
            decibels = 20 * math.log10(max(rms, 1e-6))
            audio.append(max(0.0, min(1.0, (decibels + 48) / 40)))
        count = min(len(motion), len(audio))
        motion, audio = motion[:count], audio[:count]
        activity = [0.7 * visual + 0.3 * sound for visual, sound in zip(motion, audio)]
        return {"path": path, "motion": motion, "audio": audio, "activity": activity}

    def _jitter(self, score: float) -> float:
        if self._rng is None or score == 0:
            return score
        return score * (1.0 + (self._rng.random() - 0.5) * 0.04)

    def _rank_windows(self, shot_duration: float) -> list[dict[str, Any]]:
        windows = []
        window_samples = max(1, round(shot_duration * 2))
        for source_index, source in enumerate(self.features):
            activity = source["activity"]
            for offset in range(0, max(1, len(activity) - window_samples + 1), 2):
                values = activity[offset:offset + window_samples]
                if len(values) < max(2, window_samples // 2):
                    continue
                average = _mean(values)
                peak = max(values)
                active_fraction = sum(value >= 0.25 for value in values) / len(values)
                windows.append({
                    "source_index": source_index,
                    "start": offset / 2,
                    "end": (offset + len(values)) / 2,
                    "score": self._jitter(0.6 * average + 0.25 * peak + 0.15 * active_fraction),
                })
        return sorted(windows, key=lambda item: item["score"], reverse=True)

    def _best_contiguous_window(self, target_duration: int) -> dict[str, Any] | None:
        sample_count = max(2, math.ceil(target_duration * 2))
        best = None
        for source_index, source in enumerate(self.features):
            activity = source["activity"]
            if len(activity) < sample_count:
                continue
            step = max(1, sample_count // 24)
            for offset in range(0, len(activity) - sample_count + 1, step):
                values = activity[offset:offset + sample_count]
                average = _mean(values)
                active_fraction = sum(value >= 0.25 for value in values) / len(values)
                peak = max(values)
                score = self._jitter(0.65 * average + 0.20 * peak + 0.15 * active_fraction)
                candidate = {
                    "source_index": source_index,
                    "start": offset / 2,
                    "end": (offset + sample_count) / 2,
                    "score": score,
                    "peak": peak,
                }
                if best is None or (candidate["score"], candidate["peak"]) > (best["score"], best["peak"]):
                    best = candidate
        return best

    @staticmethod
    def _diverse_windows(
        windows: list[dict[str, Any]],
        target_duration: int,
        shot_duration: float,
    ) -> list[dict[str, Any]]:
        count = max(1, min(24, math.ceil(target_duration / shot_duration)))
        max_per_source = max(1, math.ceil(target_duration / shot_duration))
        selected = []
        source_uses: dict[int, int] = {}
        for window in windows:
            source_index = window["source_index"]
            if any(
                existing["source_index"] == source_index
                and window["start"] < existing["end"]
                and window["end"] > existing["start"]
                for existing in selected
            ):
                continue
            if source_uses.get(source_index, 0) >= max_per_source:
                continue
            selected.append(window)
            source_uses[source_index] = source_uses.get(source_index, 0) + 1
            if len(selected) >= count:
                break
        return selected

    @staticmethod
    def _story_order(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(windows, key=lambda item: (item["source_index"], item["start"]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0