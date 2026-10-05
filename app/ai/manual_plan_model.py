"""Manual plan model — use any web chat AI (e.g. ChatGPT) as the planner.

Workflow (no APIs, no keys, no automation of anyone's website — a human
copies text between this tool and their chat of choice):

    python main.py plan-request input\\session ...   → writes a prompt file
    (paste it into ChatGPT web, attach frame images if desired,
     save the reply text to a file)
    python main.py edit input\\session --plan-file response.txt ...

create_plan returns the human-delivered plan (tolerantly parsed); music
ranking and render review delegate to the measured local model so the rest
of the pipeline behaves exactly like a hosted run. revise_plan is a no-op
(no second round-trip) — the local critic still gates the render.
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional

from app.ai.prompts import parse_json_object
from app.ai.remote_json_model import RemoteJSONModelError


class ManualPlanModel:
    """RemoteJSONModel-compatible planner fed by a pasted chat response."""

    def __init__(self, plan_text: str, source_paths: Optional[list[pathlib.Path]] = None) -> None:
        self.model = "chat-web-manual"
        self._plan_text = str(plan_text or "")
        self._source_paths = [pathlib.Path(p) for p in (source_paths or [])]
        self._local: Any = None

    def _ensure_local(self) -> Any:
        if self._local is None:
            from app.ai.local_editing import LocalShortFormEditingModel

            local = LocalShortFormEditingModel()
            local.features = [local._analyze_source(path) for path in self._source_paths]
            self._local = local
        return self._local

    def create_plan(self, **_: Any) -> dict[str, Any]:
        value = parse_json_object(self._plan_text)
        if value is None:
            snippet = " ".join(self._plan_text.split())[:160]
            raise RemoteJSONModelError(
                f"Plan file contains no JSON object; response started: {snippet!r}. "
                "Ask the chat AI to 'return JSON only' and save its full reply."
            )
        return value

    def rank_music(self, media_context: dict[str, Any], plan: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return self._ensure_local().rank_music(media_context, plan, candidates)

    def review_render(self, plan: dict[str, Any], review_context: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
        return self._ensure_local().review_render(plan, review_context, frames)

    def revise_plan(self, plan: dict[str, Any], **_: Any) -> dict[str, Any]:
        # No second paste round-trip in v1: keep the delivered plan; the
        # critic's verdict is still recorded in the artifact.
        return plan
