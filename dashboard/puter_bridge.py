"""Puter.js browser bridge — keyless AI for the dashboard's creative runs.

The dashboard's connected browser loads js.puter.com and executes creative
model calls through the user's own Puter session (free, "user-pays", no API
keys anywhere). The Python side sends a request over SocketIO and blocks for
the browser's ack:

    server                          browser
      │  socketio.call("puter_ai_request", {...})   │
      ├────────────────────────────────────────────►│ puter.ai.chat(messages,
      │                                             │   files, false, {model})
      │            ack {ok, text | error}           │
      │◄────────────────────────────────────────────┤

Any failure (no client, sign-in declined, timeout, transport error) raises
PuterBrowserError — ShortFormCreativeEditor then falls back to the measured
local planner, so a creative run never hard-fails because of the AI channel.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from app.utilities.logger import get_logger

log = get_logger(__name__)

DEFAULT_PUTER_MODEL = "qwen/qwen3.7-plus"
DEFAULT_TIMEOUT_SECONDS = 240.0


class PuterBrowserError(RuntimeError):
    pass


class PuterBrowserTransport:
    """Transport callable for app.ai.remote_json_model.RemoteJSONModel."""

    def __init__(
        self,
        socketio: Any,
        sid: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.socketio = socketio
        self.sid = sid
        self.model_name = (
            model
            or os.getenv("PUTER_MODEL", "").strip()
            or DEFAULT_PUTER_MODEL
        )
        self.timeout = float(timeout or DEFAULT_TIMEOUT_SECONDS)

    def call(
        self,
        system: str,
        payload: dict[str, Any],
        frames: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        request = {
            "kind": "creative_json",
            "system": system,
            "payload": payload,
            "frames": frames,
            "max_tokens": int(max_tokens),
            "model": self.model_name,
        }
        try:
            response = self.socketio.call(
                "puter_ai_request", request, to=self.sid, timeout=self.timeout
            )
        except PuterBrowserError:
            raise
        except Exception as exc:  # includes socketio ack timeouts / disconnects
            raise PuterBrowserError(
                f"browser Puter call failed ({type(exc).__name__}: {str(exc)[:160]}); "
                "is the dashboard page open and signed in to Puter?"
            ) from exc
        if not isinstance(response, dict):
            raise PuterBrowserError(f"malformed browser response: {type(response).__name__}")
        if not response.get("ok"):
            raise PuterBrowserError(str(response.get("error") or "unknown browser error")[:240])
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            raise PuterBrowserError("browser returned an empty Puter response")
        return text
