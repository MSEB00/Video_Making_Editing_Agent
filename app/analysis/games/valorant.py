"""VALORANT kill-feed event detection.

Detects kill-feed activity bursts in the top-right HUD region (configured in
``config/valorant.yaml``) so shots can be aligned to actual eliminations.

Method (no new dependencies — FFmpeg sampling + Pillow statistics):
  1. Crop the kill-feed region and sample it at a few frames per second as
     small grayscale frames.
  2. Measure per-sample "structure" (grayscale standard deviation). Kill-feed
     entries add a sustained high-structure plateau (~4 s) above the rolling
     baseline of ordinary gameplay motion.
  3. Threshold robustly (lower-quartile baseline + noise scale), require
     sustained activity (kills
     transient flickers), merge near bursts (multi-kills), and emit events
     with honest confidence scores.

This module only OBSERVES timestamps; it makes no creative decisions and
maps no events to effects.
"""
from __future__ import annotations

import pathlib
import statistics
import subprocess
from typing import Any

from PIL import Image, ImageStat

from app.utilities.ffmpeg_utils import get_ffmpeg_path, get_video_stream, probe
from app.utilities.logger import get_logger

log = get_logger(__name__)

DEFAULT_REGION = {"x": 0.72, "y": 0.02, "width": 0.28, "height": 0.25}


class ValorantEventDetector:
    """Kill-feed burst detector for VALORANT recordings."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        region = (config.get("hud") or {}).get("kill_feed") or {}
        self.region = {
            "x": float(region.get("x", DEFAULT_REGION["x"])),
            "y": float(region.get("y", DEFAULT_REGION["y"])),
            "width": float(region.get("width", DEFAULT_REGION["width"])),
            "height": float(region.get("height", DEFAULT_REGION["height"])),
        }
        det = config.get("event_detection") or {}
        self.sample_fps = max(1.0, float(det.get("sample_fps", 4.0)))
        self.analysis_width = max(32, int(det.get("analysis_width", 224)))
        self.min_structure_delta = float(det.get("min_structure_delta", 6.0))
        self.mad_multiplier = float(det.get("mad_multiplier", 2.0))
        self.min_consecutive_samples = max(1, int(det.get("min_consecutive_samples", 2)))
        self.min_event_gap = float(det.get("min_event_gap_seconds", 1.2))
        self.max_events = max(1, int(det.get("max_events_per_source", 24)))
        self.timeout = max(10, int(det.get("analysis_timeout_seconds", 300)))

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, video_path: pathlib.Path | str) -> list[dict[str, Any]]:
        """Return chronologically sorted kill events for *video_path*.

        Each event: {"kind": "kill", "start": s, "end": s, "confidence": 0..1,
        "peak_structure": float}. Self-calibrating: when the configured
        thresholds find nothing, one more permissive pass runs automatically
        (flagged "relaxed_pass", confidence discounted). Empty list remains a
        valid outcome (callers must treat "no events" gracefully).
        """
        path = pathlib.Path(video_path)
        try:
            metadata = probe(path)
        except Exception:
            log.warning("Event detection: probe failed for %s", path.name)
            return []
        video = get_video_stream(metadata) or {}
        width, height = video.get("width"), video.get("height")
        if not width or not height:
            return []
        frames = self._sample_region(path, int(width), int(height))
        if len(frames) < 4:
            return []
        levels = [ImageStat.Stat(frame).stddev[0] for frame in frames]
        events = self._events_from_levels(levels)
        if not events:
            relaxed = self._events_from_levels(
                levels,
                min_delta=self.min_structure_delta * 0.5,
                mad_multiplier=self.mad_multiplier * 1.5,
            )
            for event in relaxed:
                event["confidence"] = round(event["confidence"] * 0.8, 2)
                event["relaxed_pass"] = True
            if relaxed:
                log.info(
                    "Kill-feed detection: %s yielded events only on the permissive pass "
                    "(consider tuning config/valorant.yaml)", path.name,
                )
            events = relaxed
        return events

    # ── internals ───────────────────────────────────────────────────────────

    def _sample_region(self, path: pathlib.Path, width: int, height: int) -> list[Image.Image]:
        crop_w = self._even(max(16, int(round(width * self.region["width"]))))
        crop_h = self._even(max(16, int(round(height * self.region["height"]))))
        crop_x = min(self._even(max(0, int(round(width * self.region["x"])))), max(0, width - crop_w))
        crop_y = min(self._even(max(0, int(round(height * self.region["y"])))), max(0, height - crop_h))
        analysis_h = self._even(max(16, int(round(self.analysis_width * crop_h / crop_w))))
        vf = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={self.analysis_width}:{analysis_h},"
            f"fps={self.sample_fps},format=gray"
        )
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired):
            log.warning("Event detection: FFmpeg sampling failed for %s", path.name)
            return []
        if result.returncode != 0:
            log.warning(
                "Event detection: FFmpeg exited %s for %s: %s",
                result.returncode, path.name, result.stderr[-200:].decode("utf-8", "replace"),
            )
            return []
        frame_bytes = self.analysis_width * analysis_h
        buffer = result.stdout
        frames = []
        for offset in range(0, len(buffer) - frame_bytes + 1, frame_bytes):
            frames.append(Image.frombytes("L", (self.analysis_width, analysis_h), buffer[offset:offset + frame_bytes]))
        return frames

    def _events_from_levels(
        self,
        levels: list[float],
        min_delta: float | None = None,
        mad_multiplier: float | None = None,
    ) -> list[dict[str, Any]]:
        # Robust baseline: the quiet 25th percentile of structure levels (the
        # feed is empty most of the time) plus a noise scale from the lower
        # quartile of deviations. Median-based baselines fail when kill-feed
        # activity occupies a large share of the clip (extended teamfights).
        min_delta = self.min_structure_delta if min_delta is None else min_delta
        mad_multiplier = self.mad_multiplier if mad_multiplier is None else mad_multiplier
        quartiles = statistics.quantiles(levels, n=4, method="inclusive")
        baseline = quartiles[0]
        deviations = sorted(abs(value - baseline) for value in levels)
        noise = statistics.quantiles(deviations, n=4, method="inclusive")[0] if len(deviations) >= 2 else 0.0
        threshold = baseline + max(min_delta, mad_multiplier * noise)

        active = [level > threshold for level in levels]
        runs = self._active_runs(active)
        # Merge runs separated by less than min_event_gap (multi-kill bursts).
        gap_samples = max(1, int(round(self.min_event_gap * self.sample_fps)))
        merged: list[list[int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] <= gap_samples:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        events = []
        for start, end in merged:
            length = end - start + 1
            if length < self.min_consecutive_samples:
                continue  # transient flicker, not a feed entry
            segment = levels[start:end + 1]
            peak = max(segment)
            excess = peak - threshold
            confidence = round(min(0.95, 0.4 + excess / 40.0), 2)
            events.append({
                "kind": "kill",
                "start": round(start / self.sample_fps, 2),
                "end": round((end + 1) / self.sample_fps, 2),
                "confidence": confidence,
                "peak_structure": round(peak, 2),
                "detection": "kill_feed_structure_burst",
            })
        if not events:
            return []
        # Keep the strongest events, then restore chronological order.
        events.sort(key=lambda item: item["confidence"], reverse=True)
        events = events[: self.max_events]
        events.sort(key=lambda item: item["start"])
        return events

    @staticmethod
    def _active_runs(active: list[bool]) -> list[list[int]]:
        """Return [start, end] index runs of True values."""
        runs: list[list[int]] = []
        start = None
        for index, flag in enumerate(active):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                runs.append([start, index - 1])
                start = None
        if start is not None:
            runs.append([start, len(active) - 1])
        return runs

    @staticmethod
    def _even(value: int) -> int:
        return value - (value % 2)
