"""Discover user-supplied sound effects without binding effects to game events."""
from __future__ import annotations

import pathlib
import json
import math
import random
import re
import struct
import wave
from typing import Any

from app.utilities.ffmpeg_utils import get_duration


DEFAULT_SFX_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "sfx"
SAMPLE_RATE = 22050

# Descriptive semantics for the locally generated starter assets. These
# describe the SOUND itself (type, mood, envelope, tags) — never a mapping
# from game events to specific effects.
GENERATED_SEMANTICS: dict[str, dict[str, Any]] = {
    "original_sweep.wav": {
        "type": "sweep",
        "mood": ["subtle", "airy", "forward"],
        "intensity": 0.3,
        "attack_seconds": 0.22,
        "decay_seconds": 0.06,
        "tags": ["transition", "reveal", "rise", "soft", "airy", "movement"],
    },
    "original_impact.wav": {
        "type": "impact",
        "mood": ["weighty", "dark", "punctual"],
        "intensity": 0.75,
        "attack_seconds": 0.005,
        "decay_seconds": 0.3,
        "tags": ["hit", "impact", "low", "thud", "emphasis", "punch"],
    },
    "original_glitch.wav": {
        "type": "texture",
        "mood": ["digital", "tense", "erratic"],
        "intensity": 0.5,
        "attack_seconds": 0.01,
        "decay_seconds": 0.12,
        "tags": ["glitch", "digital", "stutter", "pulse", "tech", "texture"],
    },
}


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
            semantic = metadata.get("semantic") or {}
            assets.append({
                "filename": path.name,
                "format": path.suffix.lower().lstrip("."),
                "duration": duration,
                "description": metadata.get("description", "User-provided sound effect"),
                "origin": metadata.get("origin", "user-provided"),
                "license": metadata.get("license"),
                "type": semantic.get("type"),
                "mood": semantic.get("mood") or [],
                "intensity": semantic.get("intensity"),
                "tags": semantic.get("tags") or [],
                "semantic": semantic,
                "path": str(path.resolve()),
            })
        return assets

    def rank_candidates(
        self,
        description: str = "",
        limit: int = 5,
        intensity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic (lexical) ranking of assets against a free-text need.

        Matches descriptive words/moods/tags/envelope only — it never maps
        game events to sounds. An empty result is a valid outcome meaning
        "nothing in the library fits"; callers may legitimately choose no SFX.
        """
        tokens = {token for token in re.findall(r"[a-z]+", (description or "").lower()) if len(token) > 2}
        scored = []
        for asset in self.list_assets():
            haystack = " ".join([
                str(asset.get("description") or ""),
                str(asset.get("type") or ""),
                " ".join(asset.get("mood") or []),
                " ".join(asset.get("tags") or []),
                asset["filename"].replace("_", " ").rsplit(".", 1)[0],
            ]).lower()
            hay_tokens = {token for token in re.findall(r"[a-z]+", haystack) if len(token) > 2}
            overlap = len(tokens & hay_tokens)
            score = overlap / len(tokens) if tokens else 0.0
            if intensity is not None and asset.get("intensity") is not None:
                score += max(0.0, 0.3 - abs(float(asset["intensity"]) - float(intensity)))
            if score > 0:
                scored.append({**asset, "match_score": round(score, 3)})
        scored.sort(key=lambda item: (-item["match_score"], item["filename"]))
        return scored[: max(1, int(limit))]

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
            metadata_path = audio_path.with_suffix(".json")
            if not audio_path.is_file():
                generator(audio_path)
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            generated_origin = metadata.get("origin") == "generated locally by Gaming Video Agent"
            if not metadata_path.is_file() or (generated_origin and "semantic" not in metadata):
                # (Re)write sidecars for generated assets only; user sidecars are never touched.
                metadata = {
                    "description": metadata.get("description", description),
                    "origin": metadata.get("origin", "generated locally by Gaming Video Agent"),
                    "license": metadata.get("license", "original synthesized audio"),
                    "semantic": GENERATED_SEMANTICS.get(filename, {}),
                }
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
