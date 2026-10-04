"""Low-cost local video/audio facts and low-resolution frames for AI planning."""
from __future__ import annotations

import base64
import pathlib
import re
import subprocess
from typing import Any

from app.utilities.ffmpeg_utils import get_audio_stream, get_duration, get_ffmpeg_path, get_video_stream, probe


def analyze_sources(paths: list[pathlib.Path], max_sources: int = 12) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = []
    sampled_frames = []
    for source_index, path in enumerate(paths[:max_sources]):
        metadata = probe(path)
        video = get_video_stream(metadata) or {}
        audio = get_audio_stream(metadata)
        duration = get_duration(path)
        source = {
            "source_index": source_index,
            "filename": path.name,
            "duration": round(duration, 3),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": _parse_rate(video.get("avg_frame_rate")),
            "has_audio": audio is not None,
            "audio_mean_db": _audio_mean_db(path, duration) if audio else None,
        }
        sources.append(source)
        if duration > 0:
            for fraction in (0.18, 0.62):
                timestamp = min(duration - 0.05, max(0.0, duration * fraction))
                frame_data = _sample_frame(path, timestamp)
                if frame_data:
                    sampled_frames.append({
                        "source_index": source_index,
                        "time": round(timestamp, 2),
                        "data_url": "data:image/jpeg;base64," + base64.b64encode(frame_data).decode("ascii"),
                    })
    return {"sources": sources}, sampled_frames


def _sample_frame(path: pathlib.Path, timestamp: float) -> bytes:
    result = subprocess.run(
        [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
            "-i", str(path), "-frames:v", "1", "-vf", "scale=512:288:force_original_aspect_ratio=decrease",
            "-q:v", "7", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ],
        capture_output=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else b""


def _audio_mean_db(path: pathlib.Path, duration: float) -> float | None:
    result = subprocess.run(
        [
            get_ffmpeg_path(), "-hide_banner", "-i", str(path), "-t", str(min(duration, 12)),
            "-vn", "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    return float(match.group(1)) if match else None


def _parse_rate(value: Any) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        return round(float(numerator) / float(denominator), 3)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None
