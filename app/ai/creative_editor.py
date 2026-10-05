"""OpenAI-compatible planning, music ranking, and render critique."""
from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any, Optional


def _parse_json_object(text: Any) -> Optional[dict[str, Any]]:
    """Parse a JSON object from an LLM response.

    Tolerates the shapes hosted models actually emit: bare JSON, markdown
    code fences, leading/trailing prose, and objects embedded in text.
    Returns None only when no JSON object can be recovered.
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


def _select_prompt_frames(frames: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
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


class CreativeAIConfigurationError(RuntimeError):
    pass


class ShortFormEditingModel:
    def __init__(self, client: Any = None, model: Optional[str] = None) -> None:
        from dotenv import load_dotenv
        load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
        self.provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
        if self.provider not in {"gemini", "openai"}:
            raise CreativeAIConfigurationError("AI_PROVIDER must be 'gemini' or 'openai'.")
        if self.provider == "gemini":
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            key_name = "GEMINI_API_KEY"
        else:
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
            key_name = "OPENAI_API_KEY"
        self.client = client
        if self.client is None:
            if not api_key:
                raise CreativeAIConfigurationError(
                    f"AI creative editing with {self.provider} requires {key_name}. Add it to the project .env file."
                )
            from openai import OpenAI
            client_options = {"api_key": api_key, "timeout": 90.0, "max_retries": 2}
            if base_url:
                client_options["base_url"] = base_url
            self.client = OpenAI(**client_options)

    def create_plan(
        self,
        media_context: dict[str, Any],
        frames: list[dict[str, Any]],
        style_profile: dict[str, Any],
        platform: str,
        target_duration: int,
        user_request: str = "",
        user_preferences: Optional[dict[str, Any]] = None,
        available_sfx: Optional[list[dict[str, Any]]] = None,
        metadata_priors: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        context = {
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
        system = (
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
        return self._json_call(system, context, frames, max_tokens=3500)

    def rank_music(
        self,
        media_context: dict[str, Any],
        plan: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "Select the Jamendo candidate that best serves this edit, or return {\"track_id\": null} if none fit. "
            "Compare the music requirements with each candidate's title, artist, license, music information, "
            "duration and available audio features. Do not infer unknown licensing rights. Use the supplied audio "
            "facts to decide whether BGM should duck beneath the original mixed gameplay audio. Do not claim speech "
            "detection unless a transcript is supplied. Return JSON only with track_id, rationale, section_start, "
            "volume, duck_under_original_audio, beat_sync_strength, and warnings."
        )
        payload = {"media": media_context, "edit_plan": plan, "candidates": candidates}
        return self._json_call(system, payload, [], max_tokens=650)

    def review_render(
        self,
        plan: dict[str, Any],
        review_context: dict[str, Any],
        frames: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "Review this rendered short-form edit against its plan using the supplied sampled frames and media facts. "
            "Assess hook, pacing, dead time, crop, gameplay visibility, transition quality, captions, sound design, "
            "music fit and ending. Do not claim to hear audio from images; use audio metadata if supplied. "
            "Return JSON only with needs_revision (boolean), critique (string), and revision_request (string)."
        )
        return self._json_call(system, {"plan": plan, "render": review_context}, frames, max_tokens=500)

    def revise_plan(
        self,
        plan: dict[str, Any],
        critique: dict[str, Any],
        media_context: dict[str, Any],
        platform: str,
        target_duration: int,
    ) -> dict[str, Any]:
        system = (
            "Revise the existing edit plan only where the render critique identifies a material problem. "
            "Keep decisions grounded in the supplied footage and do not add fixed event-to-effect rules or imitate creators. "
            "Return the full revised JSON plan with the same schema as the original."
        )
        payload = {
            "existing_plan": plan,
            "critique": critique,
            "media": media_context,
            "platform": platform,
            "target_duration_seconds": target_duration,
        }
        return self._json_call(system, payload, [], max_tokens=3500)
    def _json_call(
        self,
        system: str,
        payload: dict[str, Any],
        frames: list[dict[str, Any]],
        max_tokens: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        }]
        for frame in _select_prompt_frames(frames):
            data_url = frame.get("data_url")
            if data_url:
                content.append({
                    "type": "text",
                    "text": f"Sample frame: source_index={frame.get('source_index')}, time={frame.get('time')} seconds.",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                })
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        message = response.choices[0].message.content
        if not message:
            raise RuntimeError("Creative AI returned an empty response.")
        value = _parse_json_object(message)
        if value is None:
            snippet = " ".join(str(message).split())[:160]
            raise RuntimeError(f"Creative AI returned unparseable JSON; response started: {snippet!r}")
        return value


OpenAICreativeEditor = ShortFormEditingModel
