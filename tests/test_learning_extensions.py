"""Tests for the learning-system extensions: conditional probabilities, style
clusters, curriculum research, semantic SFX ranking, configurable revision
loop, and the new CLI commands (edit / evaluate / research --inspect)."""
import json
import pathlib
import tempfile

import pytest
from click.testing import CliRunner

from app.agent.short_form_editor import ShortFormCreativeEditor, _revision_pass_limit
import app.agent.short_form_editor as editor_module
from app.audio.sfx_library import SfxLibrary
from app.editing.editor import EditOptions
from app.editing.style_learner import EditingStyleLearner
from app.storage.db import init_db
from app.training.collector import suggest_next_queries
from app.training.reference_pipeline import ReferenceTrainingPipeline, _shot_pacing_statistics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_examples(path: pathlib.Path, examples: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in examples), encoding="utf-8")


def _two_style_examples() -> list[dict]:
    examples = []
    for i in range(4):
        examples.append({
            "reference_id": f"fast_{i}", "platform": "youtube_shorts",
            "creator_group": f"creator_f{i}", "style_tags": ["montage"],
            "features": {"cut_density": 1.8 + i * 0.05, "average_shot_duration": 0.9,
                         "caption_density": 0.7, "audio_intensity": 0.85},
            "moments": [{"moment": {"visual_intensity": 0.9, "speech_present": False},
                         "observed_edit": {"cut": 1.0, "zoom": 1.0, "sfx": 0.0}}],
        })
    for i in range(4):
        examples.append({
            "reference_id": f"slow_{i}", "platform": "youtube_shorts",
            "creator_group": f"creator_s{i}", "style_tags": ["cinematic"],
            "features": {"cut_density": 0.2, "average_shot_duration": 5.5,
                         "caption_density": 0.05, "audio_intensity": 0.3},
            "moments": [{"moment": {"visual_intensity": 0.15, "speech_present": True},
                         "observed_edit": {"cut": 0.0, "zoom": 0.0, "caption": 1.0}}],
        })
    return examples


@pytest.fixture()
def fresh_db(monkeypatch):
    from app.storage.db import reset_engine
    db_path = pathlib.Path(tempfile.mkdtemp()) / "cli.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    yield db_path
    reset_engine()


# ── §10 conditional edit probabilities ───────────────────────────────────────

def test_conditional_probabilities_separate_contexts(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    _write_examples(examples_path, _two_style_examples())
    profile = EditingStyleLearner(examples_path, tmp_path / "missing.json").learn("youtube_shorts")
    cep = profile["conditional_edit_probabilities"]
    assert cep["visual_intensity=high"]["zoom"]["p_observed"] > 0.8
    assert cep["visual_intensity=low"]["zoom"]["p_observed"] < 0.2
    # "no effect" is a first-class learned outcome
    assert cep["visual_intensity=low"]["zoom"]["not_observed"] == 4
    assert cep["speech_present=true"]["caption"]["p_observed"] > 0.8
    # Laplace smoothing keeps probabilities strictly inside (0, 1)
    assert 0.0 < cep["all"]["cut"]["p_observed"] < 1.0


# ── §12 style clusters ────────────────────────────────────────────────────────

def test_style_clusters_discover_two_groups(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    _write_examples(examples_path, _two_style_examples())
    profile = EditingStyleLearner(examples_path, tmp_path / "missing.json").learn("youtube_shorts")
    clusters = profile["style_clusters"]
    assert clusters["status"] == "learned"
    assert clusters["cluster_count"] == 2
    assert clusters["mean_silhouette"] > 0.5
    tags = {tuple(sorted(cluster["dominant_tags"])) for cluster in clusters["clusters"]}
    assert tags == {("montage",), ("cinematic",)}
    for cluster in clusters["clusters"]:
        assert cluster["size"] >= 2
        assert cluster["label"]
        assert cluster["member_reference_ids"]


def test_style_clusters_honest_when_insufficient(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    _write_examples(examples_path, _two_style_examples()[:5])
    profile = EditingStyleLearner(examples_path, tmp_path / "missing.json").learn("youtube_shorts")
    clusters = profile["style_clusters"]
    assert clusters["status"] == "insufficient_references"
    assert clusters["clusters"] == []


# ── §7 pacing statistics ──────────────────────────────────────────────────────

def test_shot_pacing_statistics():
    std, cv = _shot_pacing_statistics([2.0, 7.0, 12.0], 20.0)
    assert std == pytest.approx(2.449, abs=0.01)
    assert cv == pytest.approx(0.4899, abs=0.001)
    assert _shot_pacing_statistics([], 10.0) == (None, None)
    assert _shot_pacing_statistics([5.0], 0.0) == (None, None)


# ── §14 curriculum research ───────────────────────────────────────────────────

def test_suggest_next_queries_prefers_underrepresented_categories():
    config = {"query_groups": {
        "valorant": ["v1", "v2"], "cinematic": ["c1"], "funny": ["f1"],
    }}
    pool = [{"category": "valorant"}] * 6
    queries, offset = suggest_next_queries(config, pool, {"query_offset": 0}, 2)
    assert "v1" not in queries and "v2" not in queries
    assert set(queries) == {"c1", "f1"}
    assert offset == 2


def test_suggest_next_queries_round_robins_on_empty_pool():
    config = {"query_groups": {"valorant": ["query one"], "fps": ["query two"]}}
    first, offset1 = suggest_next_queries(config, [], {}, 1)
    assert first == ["query one"] and offset1 == 1
    second, offset2 = suggest_next_queries(
        config, [{"category": "valorant"}], {"query_offset": offset1}, 1
    )
    assert second == ["query two"] and offset2 == 0


# ── §18 semantic SFX retrieval ───────────────────────────────────────────────

def test_sfx_rank_candidates_is_semantic_and_allows_no_match(tmp_path):
    library = SfxLibrary(tmp_path)
    ranked = library.rank_candidates("soft subtle transition rise")
    assert ranked and ranked[0]["filename"] == "original_sweep.wav"
    ranked = library.rank_candidates("hard digital glitch stutter pulse")
    assert ranked and ranked[0]["filename"] == "original_glitch.wav"
    ranked = library.rank_candidates("orchestral choir cathedral")
    assert ranked == []
    with_intensity = library.rank_candidates("impact hit", intensity=0.8)
    assert with_intensity[0]["filename"] == "original_impact.wav"


def test_sfx_sidecars_gain_semantic_metadata(tmp_path):
    library = SfxLibrary(tmp_path)
    library.list_assets()
    sidecar = tmp_path / "original_impact.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data.pop("semantic")
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    assets = {item["filename"]: item for item in library.list_assets()}
    assert assets["original_impact.wav"]["type"] == "impact"
    assert assets["original_impact.wav"]["intensity"] == 0.75
    assert "hit" in assets["original_impact.wav"]["tags"]


# ── §22 configurable revision loop ───────────────────────────────────────────

def test_revision_pass_limit_is_configurable_and_clamped(monkeypatch):
    monkeypatch.delenv("MAX_REVISION_PASSES", raising=False)
    assert _revision_pass_limit({"max_revision_passes": 1}) == 1
    monkeypatch.setenv("MAX_REVISION_PASSES", "2")
    assert _revision_pass_limit({}) == 2
    monkeypatch.setenv("MAX_REVISION_PASSES", "99")
    assert _revision_pass_limit({}) == 3
    monkeypatch.setenv("MAX_REVISION_PASSES", "garbage")
    assert _revision_pass_limit({}) == 1


def _creative_editor_env(tmp_path, monkeypatch, model):
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "result.mp4"
    raw_plan = {
        "platform": "youtube_shorts", "strategy": "continuous", "target_duration": 2,
        "shots": [{"source_index": 0, "start": 0, "end": 2, "role": "action", "transition": "cut"}],
        "music_requirements": {}, "sound_design": [],
    }
    model.raw_plan = raw_plan

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
        model=model,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {"training_status": "no_promoted_model"}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    return editor, [source], output, renders


def test_two_revision_passes_run_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "true")
    monkeypatch.setenv("MAX_REVISION_PASSES", "2")

    class Model:
        review_count = 0

        def create_plan(self, **kwargs):
            return self.raw_plan

        def review_render(self, plan, context, frames):
            self.review_count += 1
            if self.review_count <= 2:
                return {"needs_revision": True, "critique": f"Issue {self.review_count}.",
                        "revision_request": "Fix it."}
            return {"needs_revision": False, "critique": "Clean.", "revision_request": ""}

        def revise_plan(self, **kwargs):
            return self.raw_plan

    model = Model()
    editor, sources, output, renders = _creative_editor_env(tmp_path, monkeypatch, model)
    artifact = editor.create_edit(
        sources, output,
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=2, bgm_track=None),
    )
    assert len(renders) == 3
    assert model.review_count == 3
    assert artifact["revision_count"] == 2
    assert artifact["review"]["needs_revision"] is False
    assert artifact["review"]["revision_pass"] == 2
    assert artifact["review"]["previous_critique"] == "Issue 2."


def test_revision_loop_stops_early_when_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "true")
    monkeypatch.setenv("MAX_REVISION_PASSES", "3")

    class Model:
        review_count = 0

        def create_plan(self, **kwargs):
            return self.raw_plan

        def review_render(self, plan, context, frames):
            self.review_count += 1
            return {"needs_revision": False, "critique": "Clean.", "revision_request": ""}

        def revise_plan(self, **kwargs):
            raise AssertionError("must not revise a clean render")

    model = Model()
    editor, sources, output, renders = _creative_editor_env(tmp_path, monkeypatch, model)
    artifact = editor.create_edit(
        sources, output,
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=2, bgm_track=None),
    )
    assert len(renders) == 1
    assert artifact["revision_count"] == 0


# ── §13/§9 training artifacts: patterns.jsonl ────────────────────────────────

def test_train_candidate_writes_patterns_digest(tmp_path):
    root = tmp_path / "training"
    pipeline = ReferenceTrainingPipeline(root)
    examples = []
    for i in range(8):
        fast = i % 2 == 0
        examples.append({
            "reference_id": f"ref_{i}", "platform": "youtube_shorts",
            "creator_group": f"creator_{i}", "style_tags": ["montage"] if fast else ["cinematic"],
            "features": {
                "cut_density": 1.7 if fast else 0.25,
                "average_shot_duration": 1.0 if fast else 5.2,
                "audio_intensity": 0.8 if fast else 0.3,
            },
            "moments": [{"moment": {"visual_intensity": 0.9 if fast else 0.2},
                         "observed_edit": {"zoom": 1.0 if fast else 0.0}}],
        })
    pipeline.features.mkdir(parents=True, exist_ok=True)
    _write_examples(pipeline.examples_path, examples)

    result = pipeline.train_candidate()
    assert result["status"] in {"promoted", "rejected"}
    version = result["version"]
    patterns_path = pipeline.patterns / f"{version}_patterns.jsonl"
    assert patterns_path.is_file()
    records = [json.loads(line) for line in patterns_path.read_text(encoding="utf-8").splitlines()]
    types = {record["type"] for record in records}
    assert "conditional_edit_probability" in types
    zoom_probs = [
        record for record in records
        if record["type"] == "conditional_edit_probability"
        and record["decision"] == "zoom" and record["context"] == "visual_intensity=high"
        and record["platform"] == "youtube_shorts"
    ]
    assert zoom_probs and zoom_probs[0]["p_observed"] > 0.5


# ── §29 CLI: evaluate / research --inspect / edit ────────────────────────────

def test_cli_evaluate_reports_honestly_without_model(tmp_path, monkeypatch):
    import main as main_module
    real_pipeline = ReferenceTrainingPipeline

    def factory(*args, **kwargs):
        kwargs.setdefault("training_root", tmp_path / "training")
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr("app.training.reference_pipeline.ReferenceTrainingPipeline", factory)
    result = CliRunner().invoke(main_module.cli, ["evaluate"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dataset"]["reference_count"] == 0
    assert report["model"]["status"] == "no_active_model"
    assert "train" in report["model"]["hint"]


def test_cli_research_inspect_needs_no_api(tmp_path, monkeypatch):
    import main as main_module
    from app.research import video_observer as observer_module

    real_store = observer_module.VideoObservationStore

    def store_factory(**kwargs):
        kwargs.setdefault("root", tmp_path / "research")
        return real_store(**kwargs)

    monkeypatch.setattr(observer_module, "VideoObservationStore", store_factory)
    result = CliRunner().invoke(main_module.cli, ["research", "--inspect"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["research_dataset"]["video_count"] == 0
    assert report["research_dataset"]["media_stored"] is False
    assert "candidate_pool" in report


def test_cli_research_requires_topic_unless_inspect():
    import main as main_module
    result = CliRunner().invoke(main_module.cli, ["research"])
    assert result.exit_code != 0
    assert "--topic" in result.output


def test_cli_edit_runs_creative_pipeline(tmp_path, monkeypatch, fresh_db):
    import main as main_module
    from app.orchestrator import orchestrator as orchestrator_module

    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    (clip_dir / "a.mp4").write_bytes(b"clip")

    captured = {}

    def fake_orchestrate(job_id, options=None, progress_callback=None):
        captured["job_id"] = job_id
        captured["options"] = options
        output = tmp_path / "output" / f"job_{job_id}_final.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered")
        output.with_suffix(".edit-plan.json").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(orchestrator_module, "orchestrate_job", fake_orchestrate)
    result = CliRunner().invoke(main_module.cli, [
        "edit", str(clip_dir), "--platform", "youtube_shorts",
        "--request", "fast montage", "--duration", "30",
    ])
    assert result.exit_code == 0, result.output
    assert "Creative job" in result.output
    assert "Edit plan artifact" in result.output
    options = captured["options"]
    assert options.creative_mode is True
    assert options.aspect_ratio == "9:16"
    assert options.target_duration == 30
    assert options.creative_request == "fast montage"
    # job metadata persisted with creative_mode for provenance
    from app.storage.db import SessionLocal
    from app.storage.models import Job
    db = SessionLocal()
    job = db.get(Job, captured["job_id"])
    metadata = json.loads(job.extra_metadata)
    db.close()
    assert metadata["creative_mode"] is True
    assert metadata["style"] == "CREATIVE_AI"
