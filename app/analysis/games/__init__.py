"""Game-specific gameplay event detection.

Detectors extract timestamped gameplay events (kills, multi-kills) from a
game's on-screen HUD so the creative planner can align cuts to what actually
happened in the footage instead of guessing time windows.

Registry-based: each game provides a detector class and a config file at
``config/<game>.yaml``. Detectors are pure observation — they never hardcode
event-to-effect creative mappings.
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional

import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def load_game_config(game: str) -> dict[str, Any]:
    """Load ``config/<game>.yaml`` (empty dict when absent)."""
    path = PROJECT_ROOT / "config" / f"{str(game).lower()}.yaml"
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return data if isinstance(data, dict) else {}
    except OSError:
        return {}


def get_event_detector(game: str) -> Optional[Any]:
    """Return an event detector instance for *game*, or None if unsupported."""
    key = str(game or "").strip().lower()
    if key == "valorant":
        from app.analysis.games.valorant import ValorantEventDetector

        return ValorantEventDetector(load_game_config("valorant"))
    return None


SUPPORTED_GAMES = ("valorant",)
