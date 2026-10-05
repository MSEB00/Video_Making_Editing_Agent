"""
app/utilities/ffmpeg_utils.py
-----------------------------
Thin wrapper around FFmpeg / FFprobe.

Provides:
  - get_ffmpeg_path() / get_ffprobe_path() — binary resolution (env override)
  - probe()            — runs ffprobe and returns parsed JSON metadata
  - get_duration()     — returns duration in seconds
  - get_video_stream() / get_audio_stream() — stream selectors
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from app.utilities.logger import get_logger

log = get_logger(__name__)

# ── Binary resolution ─────────────────────────────────────────────────────────

def _resolve_bin(env_key: str, default: str) -> str:
    """Return the binary path from the environment or system PATH."""
    path = os.environ.get(env_key, "").strip()
    return path if path else default


def get_ffmpeg_path() -> str:
    return _resolve_bin("FFMPEG_PATH", "ffmpeg")


def get_ffprobe_path() -> str:
    return _resolve_bin("FFPROBE_PATH", "ffprobe")


# ── Probe ─────────────────────────────────────────────────────────────────────

def probe(path: str | Path) -> dict[str, Any]:
    """
    Run ffprobe on *path* and return parsed JSON metadata.

    Returns a dict with keys ``streams`` and ``format``.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path!r}: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def get_duration(path: str | Path) -> float:
    """Return duration of media file in seconds."""
    data = probe(path)
    duration = float(data.get("format", {}).get("duration", 0))
    return duration


def get_video_stream(probe_data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first video stream dict from ffprobe output."""
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


def get_audio_stream(probe_data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first audio stream dict from ffprobe output."""
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return None


# ── FFmpeg runner ─────────────────────────────────────────────────────────────
