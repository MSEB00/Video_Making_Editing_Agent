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

from app.ai.prompts import parse_json_object as _parse_json_object
from app.ai.remote_json_model import RemoteJSONModel, RemoteJSONModelError
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


def test_remote_model_salvages_fenced_plan_and_reports_garbage():
    fenced = RemoteJSONModel(call=lambda **kw: '```json\n{"platform": "youtube_shorts"}\n```')
    assert fenced.create_plan({}, [], {}, "youtube_shorts", 10)["platform"] == "youtube_shorts"

    garbage = RemoteJSONModel(call=lambda **kw: "I cannot help with that.")
    with pytest.raises(RemoteJSONModelError, match="unparseable JSON"):
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
    model = Model()
    editor = ShortFormCreativeEditor(
        model=model, music_provider=music,
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


# ── Job-86 class: truncated model output must never kill the job ─────────────

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

    def fake_orchestrate(job_id, options=None, progress_callback=None, model=None):
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


# ── Full-session coverage (every indexed clip appears) ───────────────────────

def _coverage_editor():
    return ShortFormCreativeEditor(
        model=object(),
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=lambda *a, **k: None,
    )


def _coverage_plan():
    from app.editing.edit_plan import EditPlan
    return EditPlan.from_dict({
        "platform": "youtube_shorts", "strategy": "s", "target_duration": 12,
        "shots": [{"source_index": 0, "start": 0, "end": 4, "role": "hook", "transition": "cut"}],
    }, source_count=3)


def test_coverage_constructs_event_anchored_shots_for_missing_sources():
    editor = _coverage_editor()
    plan = _coverage_plan()
    sources = [{"source_index": i, "duration": 10.0} for i in range(3)]
    events = {2: [{"kind": "kill", "start": 6.0, "end": 6.5, "confidence": 0.9}]}
    warnings = []
    report = editor._enforce_source_coverage(plan, sources, events, 12.0, warnings)
    assert report["mode"] == "all" and report["constructed_shots"] == 2
    assert report["sources_used"] == 3
    by_source = {shot.source_index: shot for shot in plan.shots}
    # src2 constructed shot anchors to the kill: end = 6.5 + 1.0 hold
    assert by_source[2].end == pytest.approx(7.5, abs=0.01)
    assert by_source[2].start <= 6.0  # event fully inside
    # src1 has no events → centered window
    assert by_source[1].start == pytest.approx(3.0, abs=0.01)
    total = sum(shot.end - shot.start for shot in plan.shots)
    assert total <= 12.01 and warnings == []


def test_coverage_extends_duration_honestly_when_minimum_forced():
    editor = _coverage_editor()
    plan = _coverage_plan()
    sources = [{"source_index": i, "duration": 10.0} for i in range(3)]
    warnings = []
    report = editor._enforce_source_coverage(plan, sources, {}, 2.0, warnings)
    # 3 shots x 1.0s minimum = 3s > requested 2s → honest extension + warning
    assert plan.target_duration == pytest.approx(report["final_duration"], abs=0.01)
    assert plan.target_duration >= 3.0
    assert any("Full-session coverage" in w for w in warnings)


def test_coverage_trims_buildup_not_payoff_when_rescaling():
    editor = _coverage_editor()
    from app.editing.edit_plan import EditPlan
    plan = EditPlan.from_dict({
        "platform": "youtube_shorts", "strategy": "s", "target_duration": 6,
        "shots": [{"source_index": 0, "start": 0, "end": 8, "role": "hook", "transition": "cut"}],
    }, source_count=1)
    sources = [{"source_index": 0, "duration": 10.0}]
    events = {0: [{"kind": "kill", "start": 7.0, "end": 7.4, "confidence": 0.9}]}
    editor._enforce_source_coverage(plan, sources, events, 6.0, [])
    shot = plan.shots[0]
    assert shot.end == pytest.approx(8.0, abs=0.01)   # payoff end untouched
    assert shot.start == pytest.approx(2.0, abs=0.01)  # buildup trimmed instead
    assert shot.end - shot.start == pytest.approx(6.0, abs=0.01)


def test_create_edit_coverage_reaches_every_source(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    src = tmp_path / "clips"
    src.mkdir()
    files = []
    for i in range(3):
        f = src / f"c{i}.mp4"
        f.write_bytes(b"x")
        files.append(f)

    class Model:
        def __init__(self):
            self.coverage_prefs = []

        def create_plan(self, **kwargs):
            self.coverage_prefs.append(kwargs["user_preferences"]["full_session_coverage"])
            return {  # planner only uses source 0 (the old failure mode)
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 12,
                "shots": [{"source_index": 0, "start": 1, "end": 5, "role": "hook", "transition": "cut"}],
                "music_requirements": {}, "sound_design": [],
            }

    def analyze(paths, max_sources=12):
        return {"sources": [
            {"source_index": i, "filename": f"c{i}.mp4", "duration": 10,
             "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25}
            for i in range(len(paths))
        ]}, []

    def render(inputs, destination, options, callback):
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    model = Model()
    editor = ShortFormCreativeEditor(
        model=model,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    artifact = editor.create_edit(
        files, tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=12, bgm_track=None, game="none"),
    )
    assert artifact["source_coverage"]["mode"] == "all"
    assert artifact["source_coverage"]["sources_used"] == 3
    assert artifact["source_coverage"]["constructed_shots"] == 2
    assert artifact["clip_order"] == [0, 1, 2]  # chronological + complete
    total = sum(shot["duration"] for shot in artifact["timeline"])
    assert total <= 12.01

    # opt-out: planner subset preserved
    artifact2 = editor.create_edit(
        files, tmp_path / "out2.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=12, bgm_track=None, game="none", full_session_coverage=False),
    )
    assert artifact2["source_coverage"]["mode"] == "selected"
    assert artifact2["clip_order"] == [0]
    assert model.coverage_prefs == [True, False]  # config default ON, explicit opt-out OFF


def test_cli_source_clips_flag_maps_to_options(tmp_path, monkeypatch, fresh_db):
    import main as main_module
    from app.orchestrator import orchestrator as orchestrator_module

    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(exist_ok=True)
    (clip_dir / "a.mp4").write_bytes(b"clip")
    captured = {}

    def fake_orchestrate(job_id, options=None, progress_callback=None, model=None):
        captured["options"] = options
        out = tmp_path / f"job_{job_id}_final.mp4"
        out.write_bytes(b"r")
        return out

    monkeypatch.setattr(orchestrator_module, "orchestrate_job", fake_orchestrate)
    runner = CliRunner()
    r1 = runner.invoke(main_module.cli, ["edit", str(clip_dir), "--source-clips", "all"])
    assert r1.exit_code == 0, r1.output
    assert captured["options"].full_session_coverage is True
    r2 = runner.invoke(main_module.cli, ["edit", str(clip_dir), "--source-clips", "selected"])
    assert r2.exit_code == 0, r2.output
    assert captured["options"].full_session_coverage is False
    r3 = runner.invoke(main_module.cli, ["edit", str(clip_dir)])
    assert r3.exit_code == 0, r3.output
    assert captured["options"].full_session_coverage is None  # auto → config default
