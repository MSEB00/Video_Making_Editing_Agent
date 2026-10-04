"""Discover user-supplied sound effects without binding effects to game events."""
from __future__ import annotations

import pathlib
import json
import math
import random
import struct
import wave
from typing import Any

from app.utilities.ffmpeg_utils import get_duration


DEFAULT_SFX_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "sfx"
SAMPLE_RATE = 22050


class SfxLibrary:
    def __init__(self, directory: pathlib.Path = DEFAULT_SFX_DIR) -> None:
        self.directory = pathlib.Path(directory)

    def list_assets(self) -> list[dict[str, Any]]:
        self._ensure_original_assets()
        assets = []
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
                continue
            try:
                duration = get_duration(path)
            except Exception:
                duration = None
            metadata_path = path.with_suffix(".json")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            assets.append({
                "filename": path.name,
                "format": path.suffix.lower().lstrip("."),
                "duration": duration,
                "description": metadata.get("description", "User-provided sound effect"),
                "origin": metadata.get("origin", "user-provided"),
                "path": str(path.resolve()),
            })
        return assets

    def find(self, filename: str) -> pathlib.Path | None:
        for asset in self.list_assets():
            if asset["filename"] == pathlib.Path(filename).name:
                return pathlib.Path(asset["path"])
        return None

    def _ensure_original_assets(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        generators = {
            "original_sweep.wav": ("A short soft noise sweep for a subtle transition or reveal.", _make_sweep),
            "original_impact.wav": ("A short low tonal impact with a quick decay.", _make_impact),
            "original_glitch.wav": ("A brief gated digital texture with several small pulses.", _make_glitch),
        }
        for filename, (description, generator) in generators.items():
            audio_path = self.directory / filename
            if not audio_path.is_file():
                generator(audio_path)
                audio_path.with_suffix(".json").write_text(json.dumps({
                    "description": description,
                    "origin": "generated locally by Gaming Video Agent",
                    "license": "original synthesized audio",
                }, indent=2), encoding="utf-8")


def _write_mono_wave(path: pathlib.Path, samples: list[float]) -> None:
    pcm = [max(-32768, min(32767, int(sample * 32767))) for sample in samples]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def _make_sweep(path: pathlib.Path) -> None:
    duration = 0.52
    rng = random.Random(173)
    samples = []
    smooth_noise = 0.0
    for index in range(int(SAMPLE_RATE * duration)):
        time = index / SAMPLE_RATE
        smooth_noise = smooth_noise * 0.82 + rng.uniform(-1, 1) * 0.18
        envelope = math.sin(math.pi * time / duration) ** 1.5
        samples.append(smooth_noise * envelope * 0.65)
    _write_mono_wave(path, samples)


def _make_impact(path: pathlib.Path) -> None:
    duration = 0.42
    samples = []
    for index in range(int(SAMPLE_RATE * duration)):
        time = index / SAMPLE_RATE
        frequency = 115 - 70 * time / duration
        envelope = math.exp(-11 * time)
        samples.append(math.sin(2 * math.pi * frequency * time) * envelope * 0.8)
    _write_mono_wave(path, samples)


def _make_glitch(path: pathlib.Path) -> None:
    duration = 0.36
    samples = []
    for index in range(int(SAMPLE_RATE * duration)):
        time = index / SAMPLE_RATE
        pulse = int(time / 0.045) % 2 == 0
        frequency = 740 + (index % 97) * 13
        envelope = max(0.0, 1 - time / duration) * (0.55 if pulse else 0.08)
        samples.append(math.sin(2 * math.pi * frequency * time) * envelope * 0.45)
    _write_mono_wave(path, samples)
