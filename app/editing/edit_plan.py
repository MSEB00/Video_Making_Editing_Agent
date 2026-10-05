"""Validated creative plans shared by the planner and FFmpeg renderer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


ALLOWED_TRANSITIONS = {
    "cut", "fade", "wipeleft", "wiperight", "slideleft", "slideright",
    "circlecrop", "dissolve", "fadeblack", "fadewhite", "smoothleft", "smoothright",
}


@dataclass
class PlannedShot:
    source_index: int
    start: float
    end: float
    role: str
    transition: str = "cut"
    caption: Optional[str] = None
    visual_emphasis: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any], source_count: int) -> "PlannedShot":
        source_index = int(value["source_index"])
        start = max(0.0, float(value["start"]))
        end = float(value["end"])
        if not 0 <= source_index < source_count or end <= start:
            raise ValueError("AI edit plan contains an invalid source or time range.")
        transition = str(value.get("transition", "cut")).lower()
        if transition not in ALLOWED_TRANSITIONS:
            transition = "cut"
        caption = value.get("caption")
        emphasis = value.get("visual_emphasis") or []
        if isinstance(emphasis, str):
            # Hosted models sometimes return "punch_zoom" instead of a list;
            # iterating the raw string would split it into single characters.
            emphasis = [emphasis]
        return cls(
            source_index=source_index,
            start=start,
            end=end,
            role=str(value.get("role", "action"))[:80],
            transition=transition,
            caption=str(caption)[:100] if caption else None,
            visual_emphasis=[str(item).strip()[:40] for item in emphasis[:4] if str(item).strip()],
        )


@dataclass
class EditPlan:
    platform: str
    strategy: str
    rationale: str
    target_duration: float
    shots: list[PlannedShot]
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    music_requirements: dict[str, Any] = field(default_factory=dict)
    music_mix: dict[str, Any] = field(default_factory=dict)
    sound_design: list[dict[str, Any]] = field(default_factory=list)
    ending: str = ""
    selected_track: Optional[dict[str, Any]] = None
    review: Optional[dict[str, Any]] = None
    version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, Any], source_count: int, max_shots: int = 24) -> "EditPlan":
        if not isinstance(value, dict):
            raise ValueError("AI edit plan must be a JSON object.")
        # Shots are the critical payload: skip individually malformed entries
        # (LLMs vary shapes between runs/models) but require >=1 valid shot.
        raw_shots = value.get("shots") if isinstance(value.get("shots"), list) else []
        shots: list[PlannedShot] = []
        for item in raw_shots:
            if not isinstance(item, dict):
                continue
            try:
                shots.append(PlannedShot.from_dict(item, source_count))
            except (KeyError, TypeError, ValueError):
                continue
        if not shots:
            raise ValueError("AI returned an edit plan with no usable shots.")
        if len(shots) > max_shots:
            raise ValueError(f"AI edit plan exceeds the {max_shots}-shot safety limit.")
        return cls(
            platform=str(value.get("platform", "youtube_shorts"))[:40],
            strategy=str(value.get("strategy", "contextual highlight"))[:120],
            rationale=str(value.get("rationale", ""))[:1200],
            target_duration=_safe_positive_float(
                value.get("target_duration"),
                default=sum(s.end - s.start for s in shots),
            ),
            shots=shots,
            alternatives=_coerce_alternatives(value.get("alternatives")),
            music_requirements=_as_plain_dict(value.get("music_requirements")),
            music_mix=_as_plain_dict(value.get("music_mix")),
            sound_design=_coerce_sound_design(value.get("sound_design")),
            ending=str(value.get("ending") or "")[:300],
            selected_track=_as_optional_dict(value.get("selected_track")),
            review=_as_optional_dict(value.get("review")),
            version=_safe_int(value.get("version"), default=1, minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_shot_duration(self) -> float:
        return sum(max(0.0, shot.end - shot.start) for shot in self.shots)


def _as_plain_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_optional_dict(value: Any) -> Optional[dict[str, Any]]:
    return dict(value) if isinstance(value, dict) else None


def _coerce_alternatives(value: Any, limit: int = 2) -> list[dict[str, Any]]:
    """Normalize alternatives: dicts pass through, bare strings become names.

    Hosted models vary between {"name","description"} dicts and plain strings
    across runs/models; a decorative field must never crash a whole plan.
    """
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item:
            out.append({str(key)[:60]: item[key] for key in list(item)[:8]})
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip()[:200]})
        if len(out) >= max(1, limit):
            break
    return out


def _coerce_sound_design(value: Any, limit: int = 24) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item:
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"description": item.strip()[:200]})
        if len(out) >= max(1, limit):
            break
    return out


def _safe_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return max(1.0, float(default))
    return number if number > 0 else max(1.0, float(default))


def _safe_int(value: Any, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default
