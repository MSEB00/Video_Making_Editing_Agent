"""Automatic deep analysis of legally obtained reference videos.

Extracts an editing-grammar sequence from a finished video WITHOUT requiring
paired raw/edited examples (spec §3): scene cuts, motion envelope (low-res
frame differencing), audio energy envelope + silence (mono RMS windows), then
segments the timeline at cuts and labels it via grammar.label_segments.

Only tools already in the project are used (FFmpeg + Pillow + stdlib).
Channels that cannot be measured reliably (captions, SFX, BGM presence) are
left unknown — never fabricated.
"""
from __future__ import annotations

import array
import datetime as dt
import pathlib
import statistics
import subprocess
from typing import Any, Optional

from PIL import Image, ImageChops, ImageStat

from app.learning.grammar import label_segments, sequence_stats
from app.utilities.ffmpeg_utils import get_audio_stream, get_duration, get_ffmpeg_path, probe
from app.utilities.logger import get_logger

log = get_logger(__name__)

ANALYZER_VERSION = "reference_analyzer_v1"
SAMPLE_FPS = 4.0            # motion/audio envelope rate
FRAME_W, FRAME_H = 160, 90  # motion analysis resolution
AUDIO_RATE = 8000           # mono PCM rate for RMS envelope
MOTION_FULL_SCALE = 32.0    # mean-abs-diff value treated as motion 1.0
AUDIO_FULL_SCALE = 0.25     # RMS value treated as audio energy 1.0
MIN_DURATION = 1.5


class AutomaticReferenceAnalyzer:
    """Video file → editing-grammar sequence record (or None when unusable)."""

    def __init__(self, timeout: int = 600) -> None:
        self.timeout = timeout

    # ── public API ──────────────────────────────────────────────────────────

    def analyze_file(self, path: pathlib.Path | str, meta: dict[str, Any] | None = None) -> Optional[dict[str, Any]]:
        path = pathlib.Path(path)
        meta = dict(meta or {})
        try:
            metadata = probe(path)
        except Exception:
            log.warning("[GRAMMAR] probe failed for %s", path.name)
            return None
        if not metadata.get("streams"):
            return None
        duration = _positive(get_duration(path))
        if duration is None or duration < MIN_DURATION:
            return None
        has_audio = get_audio_stream(metadata) is not None

        cuts = self._scene_cuts(path)
        motion = self._motion_envelope(path, duration)
        audio, silence_flags = self._audio_envelope(path, duration) if has_audio else (None, None)

        segments = self._build_segments(duration, cuts, motion, audio, silence_flags)
        if not segments:
            return None
        labeled = label_segments(segments)
        stats = sequence_stats(labeled, duration)
        record = {
            "reference_id": str(meta.get("reference_id") or _stable_id(path, duration)),
            "platform": str(meta.get("platform") or "all"),
            "style_tags": list(meta.get("style_tags") or ["general"]),
            "creator_group": str(meta.get("creator_group") or "unknown"),
            "category": str(meta.get("category") or "gaming"),
            "rights_basis": str(meta.get("rights_basis") or "unspecified"),
            "source": {
                "path": str(path),
                "source_url": meta.get("source_url"),
                "license_name": meta.get("license_name"),
                "provider": (meta.get("source_metadata") or {}).get("platform") if isinstance(meta.get("source_metadata"), dict) else None,
            },
            "duration": round(duration, 3),
            "has_audio": has_audio,
            "sequence": labeled,
            "stats": stats,
            "analyzer_version": ANALYZER_VERSION,
            "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        log.info(
            "[GRAMMAR] Extracted %d segments from %s (%.1fs)",
            len(labeled), path.name, duration,
        )
        return record

    # ── signal extraction ───────────────────────────────────────────────────

    def _scene_cuts(self, path: pathlib.Path) -> list[float]:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info", "-i", str(path),
            "-vf", "select='gt(scene,0.3)',showinfo", "-an", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired):
            return []
        import re
        return [float(v) for v in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]

    def _motion_envelope(self, path: pathlib.Path, duration: float) -> list[float]:
        """Per-0.25s mean absolute frame-difference, normalized to [0, 1]."""
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vf", f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired):
            return []
        frame_bytes = FRAME_W * FRAME_H
        buffer = result.stdout
        frames = [
            Image.frombytes("L", (FRAME_W, FRAME_H), buffer[offset:offset + frame_bytes])
            for offset in range(0, len(buffer) - frame_bytes + 1, frame_bytes)
        ]
        envelope: list[float] = []
        previous: Optional[Image.Image] = None
        for frame in frames:
            if previous is None:
                previous = frame
                envelope.append(0.0)
                continue
            diff = ImageChops.difference(previous, frame)
            mean = ImageStat.Stat(diff).mean[0]
            envelope.append(min(1.0, mean / MOTION_FULL_SCALE))
            previous = frame
        return envelope

    def _audio_envelope(self, path: pathlib.Path, duration: float) -> tuple[list[float], list[bool]]:
        """Per-0.25s RMS energy [0,1] + silence flags (relative threshold)."""
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired):
            return [], []
        samples = array.array("h")
        samples.frombytes(result.stdout[: len(result.stdout) - (len(result.stdout) % 2)])
        if not samples:
            return [], []
        window = int(AUDIO_RATE / SAMPLE_FPS)  # 2000 samples = 0.25 s
        rms_values: list[float] = []
        for start in range(0, max(1, len(samples) - window + 1), window):
            chunk = samples[start:start + window]
            if not chunk:
                break
            rms = (sum(int(v) * int(v) for v in chunk) / len(chunk)) ** 0.5 / 32768.0
            rms_values.append(rms)
        if not rms_values:
            return [], []
        positive = sorted(v for v in rms_values if v > 0)
        median_rms = statistics.median(positive) if positive else 0.0
        silence_threshold = max(0.008, 0.15 * median_rms)
        envelope = [min(1.0, v / AUDIO_FULL_SCALE) for v in rms_values]
        silence_flags = [v < silence_threshold for v in rms_values]
        return envelope, silence_flags

    # ── segmentation ────────────────────────────────────────────────────────

    def _build_segments(
        self,
        duration: float,
        cuts: list[float],
        motion: list[float],
        audio: Optional[list[float]],
        silence_flags: Optional[list[bool]],
    ) -> list[dict[str, Any]]:
        boundaries = [0.0] + [c for c in cuts if 0.05 < c < duration - 0.05] + [duration]
        segments: list[dict[str, Any]] = []
        for left, right in zip(boundaries, boundaries[1:]):
            seg_duration = right - left
            if seg_duration < 0.12:
                continue
            w0, w1 = int(left * SAMPLE_FPS), max(int(left * SAMPLE_FPS) + 1, int(right * SAMPLE_FPS))
            motion_window = motion[w0:min(w1, len(motion))] if motion else []
            audio_window = audio[w0:min(w1, len(audio))] if audio else None
            silence_window = silence_flags[w0:min(w1, len(silence_flags))] if silence_flags else None
            segments.append({
                "t": round(left, 3),
                "duration": round(seg_duration, 3),
                "motion": round(statistics.mean(motion_window), 4) if motion_window else 0.0,
                "audio": (round(statistics.mean(audio_window), 4) if audio_window else None),
                "silence_ratio": (
                    round(sum(1 for flag in silence_window if flag) / len(silence_window), 3)
                    if silence_window else None
                ),
            })
        return segments


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _stable_id(path: pathlib.Path, duration: float) -> str:
    import hashlib
    return hashlib.sha256(f"{path.name}:{duration:.3f}".encode("utf-8")).hexdigest()[:20]
