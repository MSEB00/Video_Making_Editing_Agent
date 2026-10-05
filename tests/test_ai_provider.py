"""Tests for the keyless creative-AI layer: RemoteJSONModel + web-chat paste workflow."""
import json

import pytest

from app.ai.prompts import PLAN_SYSTEM, REVIEW_SYSTEM, REVISE_SYSTEM, MUSIC_SYSTEM
from app.ai.remote_json_model import RemoteJSONModel, RemoteJSONModelError
from app.ai.manual_plan_model import ManualPlanModel
from app.agent.short_form_editor import ShortFormCreativeEditor


# ── RemoteJSONModel (generic transport interface) ────────────────────────────

def test_remote_model_sends_prompts_payload_frames_and_parses():
    captured = {}

    def call(**kwargs):
        captured.update(kwargs)
        return '```json\n{"platform": "youtube_shorts", "shots": []}\n```'

    model = RemoteJSONModel(call=call, model_name="web-chat")
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


def test_truncated_remote_response_raises_with_snippet():
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return '{ "track_id": "20'  # job-86 style truncation

    model = RemoteJSONModel(call=call, model_name="web-chat")
    with pytest.raises(RemoteJSONModelError, match="unparseable JSON"):
        model.rank_music({}, {"music_requirements": {}}, [])
    assert calls and calls[0]["max_tokens"] == 1400
    assert calls[0]["system"].startswith("Select the Jamendo candidate")


# ── ManualPlanModel (paste workflow) sanity ──────────────────────────────────

def test_manual_model_is_remote_interface_compatible():
    model = ManualPlanModel('{"platform": "youtube_shorts"}')
    assert model.create_plan()["platform"] == "youtube_shorts"
    # critic delegates to the local measured reviewer (no second round-trip)
    review = model.review_render(
        {"shots": []},
        {"sources": [{"width": 1080, "height": 1920, "has_audio": True, "duration": 10}]},
        [{"time": 1.0}],
    )
    assert review["needs_revision"] is False


# ── Integration through create_edit ──────────────────────────────────────────

def test_editor_without_model_uses_local_planner_with_honest_warning(tmp_path, monkeypatch):
    """No web-chat plan supplied: local feature planner, no crash."""
    import app.agent.short_form_editor as editor_module

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 4,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
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
    assert any("RemoteJSONModelError" in w for w in artifact["warnings"])


def test_web_chat_model_drives_full_plan_flow(tmp_path, monkeypatch):
    """A web-chat reply (prose + fences) delivered via transport produces a
    hosted-mode artifact with captions, emphasis and provenance."""
    import app.agent.short_form_editor as editor_module

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")
    plan_json = json.dumps({
        "platform": "youtube_shorts", "strategy": "kill montage", "target_duration": 3,
        "shots": [{"source_index": 0, "start": 0.5, "end": 3.5, "role": "hook",
                   "transition": "cut", "caption": "CLUTCH", "visual_emphasis": "punch_zoom"}],
        "music_requirements": {}, "sound_design": [],
    })

    def web_chat_call(**kwargs):
        # chat.qwen.ai-style reply: prose + fenced JSON
        assert kwargs["system"] == PLAN_SYSTEM
        assert kwargs["frames"] and kwargs["frames"][0]["data_url"].startswith("data:image/jpeg")
        return f"Sure! Here is your plan:\n```json\n{plan_json}\n```\nHope this helps!"

    model = RemoteJSONModel(web_chat_call, model_name="chat-web-manual")

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
    assert artifact["model_version"] == "chat-web-manual"
    assert artifact["strategy"] == "kill montage"
    # string visual_emphasis from the web model is wrapped, not character-split
    assert artifact["timeline"][0]["visual_emphasis"] == ["punch_zoom"]
    assert artifact["captions"] == [{"time": 0.0, "text": "CLUTCH"}]
