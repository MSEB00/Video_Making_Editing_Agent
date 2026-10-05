"""Creative-model prompts and tolerant JSON parsing.

Shared by every creative-model transport (the web-chat paste workflow
today; any future key-free backend tomorrow). Prompts are plain data —
no provider SDKs live here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


PLAN_SYSTEM = (
    "You are a context-aware short-form gaming editor. Create an original edit plan for this footage; "
    "do not imitate or identify any individual creator, and never reproduce a reference video's "
    "scene sequence, exact timing, captions, or audio. Use only broad, recurring editing principles. "
    "Reason from the supplied frames, durations, "
    "audio levels, platform and learned aggregate timeline profile. Sources may include "
    "gameplay_events: timestamped highlight moments (e.g. eliminations) detected from the "
    "game HUD, each with start, end and confidence. When present, build shots around them: "
    "enter a few seconds before a strong event as buildup, place the event near the shot's "
    "end as the payoff with roughly a second of hold after it, and never cut in the middle "
    "of an event. Weigh event confidence and prefer multi-event sequences for the hook and "
    "climax. When a source has no events, fall back to motion and audio evidence. "
    "When user_preferences.chronological_assembly is true, shots will be reassembled in "
    "recording order (source capture order, then position within each source): select and "
    "caption moments for a chronological match narrative instead of cold-open reordering, "
    "and make the FIRST chronologically-selected moment strong enough to serve as the hook. "
    "When user_preferences.full_session_coverage is true, EVERY analyzed source will appear "
    "in the final cut (sources you omit get automatic event-anchored shots): spend your "
    "creativity on per-source moment selection, captions, music and pacing rather than on "
    "excluding sources. "
    "The context may include youtube_metadata_priors: metadata-only duration distributions "
    "from public short-form research (no media was accessed); use them as sanity checks for "
    "platform-typical pacing and length. "
    "Treat relationships as statistical "
    "evidence, not rules, and account for sample counts and user feedback. If references are insufficient, "
    "do not claim a learned style. Do not use fixed event-to-effect "
    "mappings. Select hook, shot timing, transitions, captions, effects, ending and music requirements "
    "only when they fit this footage. Compare at least two distinct candidate structures, select the one "
    "best supported by the footage, and summarize the rejected alternatives without copying creator identity. "
    "Return JSON only with keys: platform, strategy, rationale, target_duration, alternatives, shots, "
    "music_requirements, music_mix, sound_design, ending. Each shot has source_index, start, end, role, "
    "transition, caption, visual_emphasis. visual_emphasis may include punch_zoom only when warranted, "
    "or be empty. Transitions are cut, fade, wipeleft, wiperight, slideleft, "
    "slideright, circlecrop, dissolve, fadeblack, fadewhite, smoothleft, or smoothright. "
    "Music requirements should provide a dynamic search phrase, tags or fuzzytags, speed, instrumental, "
    "duration_min and duration_max. Sound design is a list of selected moments with time, description, "
    "asset_filename selected only from available_sfx_assets, and level; it may be empty when no asset fits. "
    "Keep captions optional and concise."
)

MUSIC_SYSTEM = (
    "Select the Jamendo candidate that best serves this edit, or return {\"track_id\": null} if none fit. "
    "Compare the music requirements with each candidate's title, artist, license, music information, "
    "duration and available audio features. Do not infer unknown licensing rights. Use the supplied audio "
    "facts to decide whether BGM should duck beneath the original mixed gameplay audio. Do not claim speech "
    "detection unless a transcript is supplied. Return JSON only with track_id, rationale, section_start, "
    "volume, duck_under_original_audio, beat_sync_strength, and warnings."
)

REVIEW_SYSTEM = (
    "Review this rendered short-form edit against its plan using the supplied sampled frames and media facts. "
    "Assess hook, pacing, dead time, crop, gameplay visibility, transition quality, captions, sound design, "
    "music fit and ending. Do not claim to hear audio from images; use audio metadata if supplied. "
    "Return JSON only with needs_revision (boolean), critique (string), and revision_request (string)."
)

REVISE_SYSTEM = (
    "Revise the existing edit plan only where the render critique identifies a material problem. "
    "Keep decisions grounded in the supplied footage and do not add fixed event-to-effect rules or imitate creators. "
    "Return the full revised JSON plan with the same schema as the original."
)


def build_plan_context(
    media_context: dict[str, Any],
    style_profile: dict[str, Any],
    platform: str,
    target_duration: int,
    user_request: str = "",
    user_preferences: Optional[dict[str, Any]] = None,
    available_sfx: Optional[list[dict[str, Any]]] = None,
    metadata_priors: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "platform": platform,
        "target_duration_seconds": target_duration,
        "user_request": user_request[:500],
        "media": media_context,
        "learned_style_profile": style_profile,
        "user_preferences": user_preferences or {},
        "available_sfx_assets": available_sfx or [],
    }
    if metadata_priors:
        # Metadata-only statistics from YouTube Data API research (duration
        # distributions per category); no media was accessed to compute them.
        context["youtube_metadata_priors"] = metadata_priors
    return context


def select_prompt_frames(frames: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    """Pick a bounded, source-balanced set of sample frames for the model."""
    if limit <= 0:
        return []
    sources: dict[Any, list[dict[str, Any]]] = {}
    for frame_index, frame in enumerate(frames):
        source_index = frame.get("source_index", frame_index)
        sources.setdefault(source_index, []).append(frame)
    source_frames = list(sources.values())
    if len(source_frames) > limit:
        source_positions = [
            round(index * (len(source_frames) - 1) / (limit - 1))
            for index in range(limit)
        ] if limit > 1 else [len(source_frames) // 2]
        source_frames = [source_frames[index] for index in source_positions]
    selected = [items[len(items) // 2] for items in source_frames]
    if len(selected) >= limit:
        return selected[:limit]
    selected_ids = {id(frame) for frame in selected}
    for sample_index in range(max((len(items) for items in source_frames), default=0)):
        for items in source_frames:
            if sample_index >= len(items) or id(items[sample_index]) in selected_ids:
                continue
            selected.append(items[sample_index])
            selected_ids.add(id(items[sample_index]))
            if len(selected) >= limit:
                return selected
    return selected[:limit]


def parse_json_object(text: Any) -> Optional[dict[str, Any]]:
    """Parse a JSON object from an LLM response.

    Tolerates the shapes chat models actually emit: bare JSON, markdown code
    fences, leading/trailing prose, and objects embedded in text. Returns
    None only when no JSON object can be recovered.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start:index + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
