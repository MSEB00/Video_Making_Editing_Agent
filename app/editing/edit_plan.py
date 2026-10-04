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
        shots = [PlannedShot.from_dict(item, source_count) for item in value.get("shots", [])]
        if not shots:
            raise ValueError("AI returned an edit plan with no shots.")
        if len(shots) > max_shots:
            raise ValueError(f"AI edit plan exceeds the {max_shots}-shot safety limit.")
        return cls(
            platform=str(value.get("platform", "youtube_shorts")),
            strategy=str(value.get("strategy", "contextual highlight"))[:120],
            rationale=str(value.get("rationale", ""))[:1200],
            target_duration=max(1.0, float(value.get("target_duration", sum(s.end - s.start for s in shots)))),
            shots=shots,
            alternatives=[dict(item) for item in value.get("alternatives", [])[:2]],
            music_requirements=dict(value.get("music_requirements") or {}),
            music_mix=dict(value.get("music_mix") or {}),
            sound_design=[dict(item) for item in value.get("sound_design", [])[:24]],
            ending=str(value.get("ending", ""))[:300],
            selected_track=dict(value["selected_track"]) if value.get("selected_track") else None,
            review=dict(value["review"]) if value.get("review") else None,
            version=max(1, int(value.get("version", 1))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_shot_duration(self) -> float:
        return sum(max(0.0, shot.end - shot.start) for shot in self.shots)
