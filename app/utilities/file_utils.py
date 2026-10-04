"""
app/utilities/file_utils.py
---------------------------
File system helpers: path sanitisation, hashing, safe temp-file management.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm"}
)


def sanitise_filename(name: str) -> str:
    """
    Strip characters that are unsafe in file names on Windows and Unix.
    Replaces runs of whitespace / unsafe chars with underscores.
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name or "untitled"


def is_supported_video(path: str | Path) -> bool:
    """Return True if *path* has a supported video file extension."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def scan_for_videos(directory: str | Path) -> list[Path]:
    """
    Recursively scan *directory* for supported video files.
    Returns an alphabetically sorted list of absolute Path objects.
    """
    directory = Path(directory)
    videos: list[Path] = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        videos.extend(directory.rglob(f"*{ext}"))
        videos.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(videos))


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a file in streaming fashion (handles large files).
    Returns the hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist. Return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_copy(src: str | Path, dst_dir: str | Path) -> Path:
    """
    Copy *src* to *dst_dir*, creating the destination directory if needed.
    Returns the destination path.
    """
    dst = Path(dst_dir) / Path(src).name
    ensure_dir(dst_dir)
    shutil.copy2(src, dst)
    return dst


def human_readable_size(size_bytes: int) -> str:
    """Convert byte count to a human-readable string (e.g. '1.4 GB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} PB"
