"""Robustness tests: LLM JSON salvage, seeded fallback variation, BGM retention."""
import json
import pathlib

import pytest

from app.ai.creative_editor import ShortFormEditingModel, _parse_json_object
from app.ai.local_editing import LocalShortFormEditingModel
from app.editing.editor import EditOptions
from app.agent.short_form_editor import ShortFormCreativeEditor
import app.agent.short_form_editor as editor_module


# ── JSON salvage from hosted-model responses ─────────────────────────────────

def test_parse_json_object_handles_real_llm_shapes():
    assert _parse_json_object('{"a": 1}') == {"a": 1}
    assert _parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert _parse_json_object('```\n{"a": 3}\n```') == {"a": 3}
    assert _parse_json_object('Here is the plan:\n{"a": {"b": [1,2]}}\nHope it helps!') == {"a": {"b": [1, 2]}}
    assert _parse_json_object('{"text": "braces } inside { strings"}') == {"text": "braces } inside { strings"}
    assert _parse_json_object("sorry, I cannot") is None
    assert _parse_json_object("[1, 2, 3]") is None  # JSON but not an object
    assert _parse_json_object("") is None
    assert _parse_json_object(None) is None


class _StubClient:
    def __init__(self, content):
        completions = type("Completions", (), {"create": lambda self, **kw: type(
            "R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]})()})()
        self.chat = type("Chat", (), {"completions": completions})()


def test_json_call_salvages_fenced_plan_and_reports_garbage(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    fenced = ShortFormEditingModel(client=_StubClient('```json\n{"platform": "youtube_shorts"}\n```'), model="m")
    assert fenced.create_plan({}, [], {}, "youtube_shorts", 10)["platform"] == "youtube_shorts"

    garbage = ShortFormEditingModel(client=_StubClient("I cannot help with that."), model="m")
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        garbage.create_plan({}, [], {}, "youtube_shorts", 10)


# ── Seeded variation in the local fallback planner ───────────────────────────

def _tie_model():
    activity = [0.15] * 80
    for i in range(10, 20):
        activity[i] = 0.9   # bump A
    for i in range(60, 70):
        activity[i] = 0.9   # bump B (equal height → near-tie windows)
    model = LocalShortFormEditingModel()
    model._analyze_source = lambda path: {
        "path": path, "motion": activity, "audio": activity, "activity": activity,
    }
    return model


def test_local_planner_same_seed_is_deterministic():
    model = _tie_model()
    a = model.create_plan([pathlib.Path("a.mp4")], {"sources": []}, "youtube_shorts", 8, seed=1)
    b = model.create_plan([pathlib.Path("a.mp4")], {"sources": []}, "youtube_shorts", 8, seed=1)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_local_planner_different_seeds_vary_without_seed_lockstep():
    model = _tie_model()
    plans = {
        seed: [(s["start"], s["end"]) for s in
               model.create_plan([pathlib.Path("a.mp4")], {"sources": []}, "youtube_shorts", 8, seed=seed)["shots"]]
        for seed in (1, 777)
    }
    assert plans[1] != plans[777]  # near-ties resolve differently per seed
    # unseeded stays fully deterministic (backwards compatible)
    u1 = model.create_plan([pathlib.Path("a.mp4")], {"sources": []}, "youtube_shorts", 8)
    u2 = model.create_plan([pathlib.Path("a.mp4")], {"sources": []}, "youtube_shorts", 8)
    assert json.dumps(u1, sort_keys=True) == json.dumps(u2, sort_keys=True)


# ── BGM: default runs must keep the planner's music request ──────────────────

class _RecordingMusic:
    def __init__(self, client_id="jamendo-key"):
        self.client_id = client_id
        self.search_calls = []

    def search(self, requirements, limit=20):
        self.search_calls.append(requirements)
        return []


def _run_create_edit(tmp_path, monkeypatch, bgm_track, music):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")

    class Model:
        def create_plan(self, **kwargs):
            return {
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 3,
                "shots": [{"source_index": 0, "start": 0, "end": 3, "role": "action", "transition": "cut"}],
                "music_requirements": {"search": "energetic electronic", "tags": ["energetic"]},
                "sound_design": [],
            }

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 10,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
        }]}, []

    def render(inputs, destination, options, callback):
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    editor = ShortFormCreativeEditor(
        model=Model(), music_provider=music,
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    return editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=3, bgm_track=bgm_track),
    )


def test_default_bgm_none_keeps_planner_music_request(tmp_path, monkeypatch):
    music = _RecordingMusic()
    artifact = _run_create_edit(tmp_path, monkeypatch, None, music)
    # The regression: bgm_track=None used to wipe music_requirements before search.
    assert len(music.search_calls) >= 1  # search + broader retry when empty
    assert music.search_calls[0].search == "energetic electronic"
    assert artifact["music_mix"]["enabled"] is False  # search ran, no candidates returned


def test_explicit_bgm_off_disables_music(tmp_path, monkeypatch):
    music = _RecordingMusic()
    _run_create_edit(tmp_path, monkeypatch, "none", music)
    assert music.search_calls == []
