"""
app/utilities/ffmpeg_utils.py
-----------------------------
Thin wrapper around FFmpeg / FFprobe.

Provides:
  - check_ffmpeg()     — validates binaries exist and returns versions
  - probe()            — runs ffprobe and returns parsed JSON metadata
  - run_ffmpeg()       — runs an ffmpeg command with structured logging
  - get_duration()     — returns duration in seconds
  - extract_frames()   — extracts frames at given timestamps
"""
from __future__ import annotations

import json
import os
import shutil
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


# ── Validation ────────────────────────────────────────────────────────────────

class FFmpegNotFoundError(RuntimeError):
    pass


def check_ffmpeg() -> dict[str, str]:
    """
    Verify FFmpeg and FFprobe are available.

    Returns a dict with ``ffmpeg`` and ``ffprobe`` version strings.
    Raises ``FFmpegNotFoundError`` if either binary is missing.
    """
    versions: dict[str, str] = {}
    for name, path in [("ffmpeg", get_ffmpeg_path()), ("ffprobe", get_ffprobe_path())]:
        if not shutil.which(path):
            raise FFmpegNotFoundError(
                f"{name!r} binary not found at {path!r}. "
                "Install FFmpeg and ensure it is on your PATH, "
                "or set FFMPEG_PATH / FFPROBE_PATH in your .env."
            )
        result = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        versions[name] = first_line
        log.debug(f"{name} found", extra={"version": first_line})
    return versions


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

def run_ffmpeg(
    args: list[str],
    job_id: str = "",
    timeout: int = 3600,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute an FFmpeg command.

    Parameters
    ----------
    args:     List of arguments (do NOT include the ``ffmpeg`` binary itself).
    job_id:   Passed to the logger for structured output.
    timeout:  Max seconds to wait (default 1 hour).
    check:    If True, raise CalledProcessError on non-zero exit.
    """
    cmd = [get_ffmpeg_path(), "-hide_banner", "-loglevel", "warning"] + args
    log.debug("Running FFmpeg", extra={"job_id": job_id, "cmd": " ".join(cmd)})
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        log.error(
            "FFmpeg failed",
            extra={"job_id": job_id, "stderr": result.stderr[-2000:]},
        )
        raise RuntimeError(
            f"FFmpeg exited {result.returncode}: {result.stderr[-500:]}"
        )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_clip(
    src: str | Path,
    dst: str | Path,
    start: float,
    end: float,
    job_id: str = "",
    encoder: str = "libx264",
) -> Path:
    """
    Extract a sub-clip from *src* between *start* and *end* seconds.
    Uses stream-copy for speed if ``encoder`` is empty, otherwise re-encodes.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    if encoder:
        codec_args = ["-c:v", encoder, "-crf", "18", "-preset", "fast",
                      "-c:a", "aac", "-b:a", "192k"]
    else:
        codec_args = ["-c", "copy"]

    run_ffmpeg(
        ["-ss", str(start), "-i", str(src),
         "-t", str(duration)] + codec_args + ["-y", str(dst)],
        job_id=job_id,
    )
    log.info(
        "Clip extracted",
        extra={"job_id": job_id, "src": str(src), "dst": str(dst),
               "start": start, "end": end},
    )
    return dst


def extract_frame(
    src: str | Path,
    dst: str | Path,
    timestamp: float,
    job_id: str = "",
) -> Path:
    """Extract a single frame at *timestamp* seconds as a JPEG."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-ss", str(timestamp), "-i", str(src),
         "-vframes", "1", "-q:v", "2", "-y", str(dst)],
        job_id=job_id,
    )
    return dst
# ---------------------------------------------------------------------
# Concatenation helper used by the orchestrator
# ---------------------------------------------------------------------
from pathlib import Path
import subprocess, tempfile
from app.utilities.logger import get_logger
log = get_logger(__name__)

def concat_videos(input_files: list[Path], output_path: Path) -> Path:
    """Concatenate *input_files* into *output_path* using FFmpeg.

    Creates a temporary ``list.txt`` with lines ``file '<abs>'`` and runs:
    ``ffmpeg -y -f concat -safe 0 -i list.txt -c copy <output>``.
    Falls back to re‑encode with libx264 if stream‑copy fails.
    Returns the Path to the final output file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as list_file:
        for p in input_files:
            list_file.write(f"file '{p.resolve()}'\n")
        list_path = Path(list_file.name)
    try:
        cmd = [
            get_ffmpeg_path(), '-y', '-f', 'concat', '-safe', '0', '-i', str(list_path),
            '-c', 'copy', str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.warning('concat copy failed, falling back to re‑encode', extra={'stderr': result.stderr})
            cmd = [
                get_ffmpeg_path(), '-y', '-f', 'concat', '-safe', '0', '-i', str(list_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'aac', '-b:a', '192k', str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f'FFmpeg concat failed: {result.stderr}')
        return output_path
    finally:
        try:
            list_path.unlink()
        except Exception:
            pass
