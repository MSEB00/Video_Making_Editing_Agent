"""Transport-agnostic remote creative model.

Implements the four-call creative interface (create_plan / rank_music /
review_render / revise_plan) on top of an injected ``call`` transport:

    call(system: str, payload: dict, frames: list, max_tokens: int) -> str

The transport returns the model's raw text; parsing lives here (tolerant of
code fences and prose). The shipped transport is the web-chat paste workflow
(ManualPlanModel via `edit --plan-file`); runs without any model use the
measured local feature planner. No provider SDKs, no API keys.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from app.ai.prompts import (
    MUSIC_SYSTEM,
    PLAN_SYSTEM,
    REVIEW_SYSTEM,
    REVISE_SYSTEM,
    build_plan_context,
    parse_json_object,
    select_prompt_frames,
)


class RemoteJSONModelError(RuntimeError):
    pass


class RemoteJSONModel:
    """Creative model delegating JSON calls to an injected transport."""

    def __init__(
        self,
        call: Callable[..., str],
        model_name: str = "remote_model",
        max_frames: int = 8,
    ) -> None:
        if not callable(call):
            raise RemoteJSONModelError("RemoteJSONModel requires a callable transport.")
        self._call = call
        self.model = model_name
        self.max_frames = max(0, int(max_frames))

    # ── creative interface ──────────────────────────────────────────────────

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
        context = build_plan_context(
            media_context=media_context,
            style_profile=style_profile,
            platform=platform,
            target_duration=target_duration,
            user_request=user_request,
            user_preferences=user_preferences,
            available_sfx=available_sfx,
            metadata_priors=metadata_priors,
        )
        return self._json_call(PLAN_SYSTEM, context, frames, max_tokens=3500)

    def rank_music(
        self,
        media_context: dict[str, Any],
        plan: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {"media": media_context, "edit_plan": plan, "candidates": candidates}
        return self._json_call(MUSIC_SYSTEM, payload, [], max_tokens=1400)

    def review_render(
        self,
        plan: dict[str, Any],
        review_context: dict[str, Any],
        frames: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._json_call(
            REVIEW_SYSTEM, {"plan": plan, "render": review_context}, frames, max_tokens=1200
        )

    def revise_plan(
        self,
        plan: dict[str, Any],
        critique: dict[str, Any],
        media_context: dict[str, Any],
        platform: str,
        target_duration: int,
    ) -> dict[str, Any]:
        payload = {
            "existing_plan": plan,
            "critique": critique,
            "media": media_context,
            "platform": platform,
            "target_duration_seconds": target_duration,
        }
        return self._json_call(REVISE_SYSTEM, payload, [], max_tokens=3500)

    # ── internals ───────────────────────────────────────────────────────────

    def _json_call(
        self,
        system: str,
        payload: dict[str, Any],
        frames: list[dict[str, Any]],
        max_tokens: int,
    ) -> dict[str, Any]:
        bounded_frames = [
            {
                "source_index": frame.get("source_index"),
                "time": frame.get("time"),
                "data_url": frame.get("data_url"),
            }
            for frame in select_prompt_frames(frames or [], limit=self.max_frames)
            if frame.get("data_url")
        ]
        raw = self._call(
            system=system,
            payload=json.loads(json.dumps(payload, ensure_ascii=True, default=str)),
            frames=bounded_frames,
            max_tokens=max_tokens,
        )
        value = parse_json_object(raw)
        if value is None:
            snippet = " ".join(str(raw).split())[:160]
            raise RemoteJSONModelError(
                f"Remote creative model returned unparseable JSON; response started: {snippet!r}"
            )
        return value
