"""Robustness tests: LLM JSON salvage, seeded fallback variation, BGM retention."""
import json
import pathlib
import tempfile
from click.testing import CliRunner

import pytest


@pytest.fixture()
def fresh_db(monkeypatch):
    from app.storage.db import init_db, reset_engine
    db_path = pathlib.Path(tempfile.mkdtemp()) / "cli.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    yield db_path
    reset_engine()

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


# ── Job-86 class: truncated music ranking must not kill the job ──────────────

class _SequenceClient:
    """Stub returning a scripted sequence of (content, finish_reason)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                content, finish = outer.responses.pop(0)
                message = type("M", (), {"content": content})()
                choice = type("C", (), {"message": message, "finish_reason": finish})()
                return type("R", (), {"choices": [choice]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


def test_json_call_retries_on_length_truncation(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = _SequenceClient([
        ('{ "track_id": "20', "length"),                     # job-86 style truncation
        ('{"track_id": "2060076", "rationale": "fits"}', "stop"),
    ])
    model = ShortFormEditingModel(client=client, model="m")
    result = model.rank_music({}, {"music_requirements": {}}, [])
    assert result["track_id"] == "2060076"
    assert len(client.calls) == 2
    assert client.calls[1]["max_tokens"] == client.calls[0]["max_tokens"] * 4  # budget raised


def test_json_call_gives_up_after_second_truncation(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = _SequenceClient([
        ('{ "track_id": "2', "length"),
        ('{ "track_id": "20', "length"),
    ])
    model = ShortFormEditingModel(client=client, model="m")
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        model.rank_music({}, {"music_requirements": {}}, [])


def test_music_ranking_failure_degrades_to_no_bgm(tmp_path, monkeypatch):
    class FailingRanker:
        def create_plan(self, **kwargs):
            return {
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 3,
                "shots": [{"source_index": 0, "start": 0, "end": 3, "role": "action", "transition": "cut"}],
                "music_requirements": {"search": "energetic"}, "sound_design": [],
            }

        def rank_music(self, media_context, plan, candidates):
            raise RuntimeError("Creative AI returned unparseable JSON; response started: '{ \"track_id\": \"20'")

    class OneTrackMusic(_RecordingMusic):
        def search(self, requirements, limit=20):
            self.search_calls.append(requirements)
            return [{"id": "t1", "duration": 60, "title": "Track", "artist": "A",
                     "download_allowed": True, "license_url": "https://x/y"}]

        def analyze_audio(self, audio_url, sample_seconds=12):
            return {"bpm": 140.0, "bpm_confidence": 0.7, "sections": []}

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 10,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
        }]}, []

    renders = []

    def render(inputs, destination, options, callback):
        renders.append(destination)
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    editor = ShortFormCreativeEditor(
        model=FailingRanker(), music_provider=OneTrackMusic(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    artifact = editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=3, bgm_track=None),
    )
    assert len(renders) == 1                       # job completed despite ranker failure
    assert artifact["music_mix"]["enabled"] is False
    assert any("Music ranking failed" in w for w in artifact["warnings"])


def test_review_failure_ships_render_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "true")

    class Model:
        def create_plan(self, **kwargs):
            return {
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 2,
                "shots": [{"source_index": 0, "start": 0, "end": 2, "role": "action", "transition": "cut"}],
                "music_requirements": {}, "sound_design": [],
            }

        def review_render(self, plan, context, frames):
            raise RuntimeError("Creative AI returned unparseable JSON (review truncated)")

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": pathlib.Path(paths[0]).name, "duration": 2,
            "width": 1080, "height": 1920, "has_audio": True, "audio_mean_db": -30,
        }]}, [{"time": 0.5, "data_url": "data:image/jpeg;base64,eA=="}]

    renders = []

    def render(inputs, destination, options, callback):
        renders.append(destination)
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    editor = ShortFormCreativeEditor(
        model=Model(),
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    artifact = editor.create_edit(
        [tmp_path / "g.mp4"], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=2, bgm_track=None),
    )
    assert len(renders) == 1
    assert artifact["revision_count"] == 0
    assert any("Render review failed" in w for w in artifact["warnings"])


# ── Chronological shot assembly (recorded-timeline consistency) ──────────────

def test_apply_chronological_order_sorts_by_source_then_position():
    from app.editing.edit_plan import EditPlan

    plan = EditPlan.from_dict({
        "platform": "youtube_shorts", "strategy": "s", "target_duration": 20,
        "shots": [
            {"source_index": 2, "start": 0, "end": 3},
            {"source_index": 0, "start": 5, "end": 8},
            {"source_index": 1, "start": 0, "end": 2},
            {"source_index": 0, "start": 1, "end": 4},
        ],
    }, source_count=3)
    changed = ShortFormCreativeEditor._apply_chronological_order(plan)
    assert changed is True
    assert [(s.source_index, s.start) for s in plan.shots] == [(0, 1.0), (0, 5.0), (1, 0.0), (2, 0.0)]
    # second call is a no-op
    assert ShortFormCreativeEditor._apply_chronological_order(plan) is False


def _chron_run(tmp_path, monkeypatch, chronological_option, plan_shots):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    sources_meta = [
        {"source_index": i, "filename": f"c{i}.mp4", "duration": 10,
         "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25}
        for i in range(3)
    ]
    src = tmp_path / "clips"
    src.mkdir(exist_ok=True)
    files = []
    for i in range(3):
        f = src / f"c{i}.mp4"
        f.write_bytes(b"x")
        files.append(f)

    class Model:
        def create_plan(self, **kwargs):
            assert "chronological_assembly" in kwargs["user_preferences"]
            return {
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 20,
                "shots": plan_shots, "music_requirements": {}, "sound_design": [],
            }

    def analyze(paths, max_sources=12):
        return {"sources": [dict(m) for m in sources_meta[:len(paths)]]}, []

    def render(inputs, destination, options, callback):
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    editor = ShortFormCreativeEditor(
        model=Model(),
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    return editor.create_edit(
        files, tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=20, bgm_track=None, game="none",
                    chronological_order=chronological_option),
    )


SHUFFLED = [
    {"source_index": 2, "start": 0, "end": 3, "role": "hook", "transition": "cut"},
    {"source_index": 0, "start": 1, "end": 4, "role": "action", "transition": "cut"},
    {"source_index": 1, "start": 0, "end": 2, "role": "payoff", "transition": "cut"},
]


def test_default_assembly_is_chronological(tmp_path, monkeypatch):
    artifact = _chron_run(tmp_path, monkeypatch, None, SHUFFLED)
    assert artifact["shot_order"] == "chronological"
    assert artifact["clip_order"] == [0, 1, 2]


def test_impact_order_preserves_planner_sequence(tmp_path, monkeypatch):
    artifact = _chron_run(tmp_path, monkeypatch, False, SHUFFLED)
    assert artifact["shot_order"] == "planner"
    assert artifact["clip_order"] == [2, 0, 1]


def test_cli_edit_order_flag_maps_to_options(tmp_path, monkeypatch, fresh_db):
    import main as main_module
    from app.orchestrator import orchestrator as orchestrator_module

    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(exist_ok=True)
    (clip_dir / "a.mp4").write_bytes(b"clip")
    captured = {}

    def fake_orchestrate(job_id, options=None, progress_callback=None):
        captured["options"] = options
        out = tmp_path / f"job_{job_id}_final.mp4"
        out.write_bytes(b"r")
        return out

    monkeypatch.setattr(orchestrator_module, "orchestrate_job", fake_orchestrate)
    runner = CliRunner()
    r1 = runner.invoke(main_module.cli, ["edit", str(clip_dir), "--order", "impact"])
    assert r1.exit_code == 0, r1.output
    assert captured["options"].chronological_order is False
    r2 = runner.invoke(main_module.cli, ["edit", str(clip_dir), "--order", "chronological"])
    assert r2.exit_code == 0, r2.output
    assert captured["options"].chronological_order is True
    r3 = runner.invoke(main_module.cli, ["edit", str(clip_dir)])
    assert r3.exit_code == 0, r3.output
    assert captured["options"].chronological_order is None  # auto → config decides
