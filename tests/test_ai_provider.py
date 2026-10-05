"""Tests for the keyless creative-AI layer: RemoteJSONModel + Puter browser bridge."""
import json

import pytest

from app.ai.prompts import PLAN_SYSTEM, REVIEW_SYSTEM, REVISE_SYSTEM, MUSIC_SYSTEM
from app.ai.remote_json_model import RemoteJSONModel, RemoteJSONModelError
from app.agent.short_form_editor import ShortFormCreativeEditor
from dashboard.puter_bridge import (
    DEFAULT_PUTER_MODEL,
    PuterBrowserError,
    PuterBrowserTransport,
)


class FakeSocketIO:
    """Records requests and replies with a scripted ack."""

    def __init__(self, response=None, raises=None):
        self.requests = []
        self.response = response
        self.raises = raises

    def call(self, event, payload, to=None, timeout=None):
        self.requests.append({"event": event, "payload": payload, "to": to, "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        return self.response


# ── RemoteJSONModel ───────────────────────────────────────────────────────────

def test_remote_model_sends_prompts_payload_frames_and_parses():
    captured = {}

    def call(**kwargs):
        captured.update(kwargs)
        return '```json\n{"platform": "youtube_shorts", "shots": []}\n```'

    model = RemoteJSONModel(call=call, model_name="puter:qwen/qwen3.7-plus")
    frames = [{"source_index": 0, "time": 0.5, "data_url": "data:image/jpeg;base64,AA", "extra": "dropped"}]
    plan = model.create_plan(
        media_context={"sources": []}, frames=frames, style_profile={"training_status": "learned"},
        platform="youtube_shorts", target_duration=30, user_request="x" * 900,
        metadata_priors={"overall": {"median_seconds": 32.5}},
    )
    assert plan["platform"] == "youtube_shorts"
    assert captured["system"] == PLAN_SYSTEM
    assert captured["payload"]["user_request"] == "x" * 500          # request bounded
    assert captured["payload"]["youtube_metadata_priors"]["overall"]["median_seconds"] == 32.5
    assert captured["max_tokens"] == 3500
    # frames pass through whitelisted fields only
    assert captured["frames"] == [{"source_index": 0, "time": 0.5, "data_url": "data:image/jpeg;base64,AA"}]
    assert model.model == "puter:qwen/qwen3.7-plus"


def test_remote_model_uses_distinct_system_prompts_per_call():
    systems = []

    def call(**kwargs):
        systems.append(kwargs["system"])
        return "{}"

    model = RemoteJSONModel(call=call)
    model.rank_music({}, {"music_requirements": {}}, [])
    model.review_render({}, {}, [])
    model.revise_plan({}, {}, {}, "youtube_shorts", 10)
    assert systems == [MUSIC_SYSTEM, REVIEW_SYSTEM, REVISE_SYSTEM]


def test_remote_model_requires_callable_transport():
    with pytest.raises(RemoteJSONModelError):
        RemoteJSONModel(call="not callable")


# ── PuterBrowserTransport (SocketIO bridge) ──────────────────────────────────

def test_bridge_roundtrip_returns_browser_text():
    fake = FakeSocketIO(response={"ok": True, "text": '{"needs_revision": false}'})
    transport = PuterBrowserTransport(fake, sid="abc123", timeout=5)
    text = transport.call(system="sys", payload={"a": 1}, frames=[], max_tokens=999)
    assert text == '{"needs_revision": false}'
    request = fake.requests[0]
    assert request["event"] == "puter_ai_request"
    assert request["to"] == "abc123"
    assert request["timeout"] == 5
    assert request["payload"]["system"] == "sys"
    assert request["payload"]["payload"] == {"a": 1}
    assert request["payload"]["max_tokens"] == 999
    assert request["payload"]["model"] == DEFAULT_PUTER_MODEL


def test_bridge_model_from_env(monkeypatch):
    monkeypatch.setenv("PUTER_MODEL", "qwen/qwen3.8-max")
    transport = PuterBrowserTransport(FakeSocketIO(response={"ok": True, "text": "{}"}), sid="s")
    assert transport.model_name == "qwen/qwen3.8-max"


def test_bridge_surfaces_browser_errors():
    fake = FakeSocketIO(response={"ok": False, "error": "user declined Puter sign-in"})
    transport = PuterBrowserTransport(fake, sid="s")
    with pytest.raises(PuterBrowserError, match="declined Puter sign-in"):
        transport.call(system="s", payload={}, frames=[], max_tokens=10)


def test_bridge_handles_timeouts_and_malformed_acks():
    timed_out = PuterBrowserTransport(FakeSocketIO(raises=TimeoutError("ack timeout")), sid="s")
    with pytest.raises(PuterBrowserError, match="browser Puter call failed"):
        timed_out.call(system="s", payload={}, frames=[], max_tokens=10)

    malformed = PuterBrowserTransport(FakeSocketIO(response="not-a-dict"), sid="s")
    with pytest.raises(PuterBrowserError, match="malformed browser response"):
        malformed.call(system="s", payload={}, frames=[], max_tokens=10)

    empty = PuterBrowserTransport(FakeSocketIO(response={"ok": True, "text": "   "}), sid="s")
    with pytest.raises(PuterBrowserError, match="empty Puter response"):
        empty.call(system="s", payload={}, frames=[], max_tokens=10)


# ── Integration: browser model inside the creative editor ────────────────────

def test_editor_without_model_uses_local_planner_with_honest_warning(tmp_path, monkeypatch):
    """CLI context (no browser bridge): local feature planner, no crash."""
    import app.agent.short_form_editor as editor_module

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 4,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
            "activity": None,
        }]}, []

    def render(inputs, destination, options, callback):
        import pathlib as _p
        _p.Path(destination).write_bytes(b"rendered")
        return destination

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    from app.editing.editor import EditOptions

    editor = ShortFormCreativeEditor(
        model=None,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    # local planner analyzes real files; stub its per-source analyzer instead
    from app.ai.local_editing import LocalShortFormEditingModel
    monkeypatch.setattr(LocalShortFormEditingModel, "_analyze_source", lambda self, path: {
        "path": str(path), "motion": [0.4] * 8, "audio": [0.3] * 8, "activity": [0.5] * 8,
    })
    artifact = editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=3, bgm_track=None, game="none"),
    )
    assert artifact["planning_mode"] == "local_feature_fallback"
    assert any("RemoteJSONModelError" in w or "local features" in w for w in artifact["warnings"])


def test_bridge_backed_model_drives_full_plan_flow(tmp_path, monkeypatch):
    """A Puter-bridge model (fake socket) produces a hosted-mode artifact."""
    import app.agent.short_form_editor as editor_module

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")
    plan_json = json.dumps({
        "platform": "youtube_shorts", "strategy": "kill montage", "target_duration": 3,
        "shots": [{"source_index": 0, "start": 0.5, "end": 3.5, "role": "hook", "transition": "cut"}],
        "music_requirements": {}, "sound_design": [],
    })
    fake = FakeSocketIO(response={"ok": True, "text": f"Sure! Here is the plan:\n```json\n{plan_json}\n```"})
    transport = PuterBrowserTransport(fake, sid="s")
    model = RemoteJSONModel(transport.call, model_name="puter:qwen/qwen3.7-plus")

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 6,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
        }]}, [{"source_index": 0, "time": 0.5, "data_url": "data:image/jpeg;base64,AA"}]

    def render(inputs, destination, options, callback):
        import pathlib as _p
        _p.Path(destination).write_bytes(b"rendered")
        return destination

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    from app.editing.editor import EditOptions

    editor = ShortFormCreativeEditor(
        model=model,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    artifact = editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=3, bgm_track=None, game="none"),
    )
    assert artifact["planning_mode"] == "hosted_creative_model"
    assert artifact["model_version"] == "puter:qwen/qwen3.7-plus"
    assert artifact["strategy"] == "kill montage"
    assert fake.requests and fake.requests[0]["payload"]["kind"] == "creative_json"
    # frames reached the browser as data URLs
    assert fake.requests[0]["payload"]["frames"][0]["data_url"].startswith("data:image/jpeg")
