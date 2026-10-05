"""Tests for the learned editing-policy subsystem (grammar → dataset → policy → planner)."""
import json
import pathlib
import subprocess

from app.learning.grammar import EMBEDDING_KEYS, label_segments, sequence_stats
from app.learning.moment_graph import build_moment_graph
from app.learning.policy_inference import PolicyInference
from app.learning.policy_model import PolicyTrainer
from app.learning.sequence_dataset import SequenceDataset
from app.learning.sequence_extractor import AutomaticReferenceAnalyzer
from app.learning.structure_search import generate_structures, score_and_select


# ── helpers ───────────────────────────────────────────────────────────────────

MONTAGE_PROFILE = (0.85, 0.45, 0.55, 0.6, 0.7, 0.8, 0.95, 0.5)


def _montage_segments(count=8, rising=True):
    """Short shots shaped like a real montage: strong hook, dip, build,
    late payoff, resolve."""
    segments = []
    for i in range(count):
        intensity = MONTAGE_PROFILE[i % len(MONTAGE_PROFILE)] if rising else 0.6
        segments.append({
            "t": i * 1.4, "duration": 1.4,
            "motion": min(1.0, intensity), "audio": min(1.0, intensity * 0.9),
            "silence_ratio": 0.0,
        })
    return segments


def _synthetic_record(reference_id, creator, rising=True):
    labeled = label_segments(_montage_segments(rising=rising))
    stats = sequence_stats(labeled, 1.4 * len(labeled))
    return {
        "reference_id": reference_id, "platform": "youtube_shorts",
        "style_tags": ["montage"], "creator_group": creator, "category": "gaming",
        "rights_basis": "licensed", "duration": round(1.4 * len(labeled), 3),
        "has_audio": True, "sequence": labeled, "stats": stats,
        "analyzer_version": "test", "source": {},
    }


def _trained_policy(tmp_path, count=8):
    records = [_synthetic_record(f"ref_{i}", f"creator_{i}") for i in range(count)]
    trainer = PolicyTrainer(tmp_path / "policies")
    dataset_info = {"dataset_version": "grammar_v001", "sequence_count": count}
    result = trainer.train_candidate(records[:6], records[6:], dataset_info)
    return trainer, result, records


# ── grammar labeling ──────────────────────────────────────────────────────────

def test_label_segments_infers_structure_from_shape():
    segments = _montage_segments(count=8, rising=True)
    assert len(segments) == 8
    labeled = label_segments(segments)
    tokens = [s["token"] for s in labeled]
    assert tokens[0] == "HOOK"                 # short + intense opening
    assert "PAYOFF" in tokens                  # global intensity max labeled payoff
    assert tokens[-1] == "END"                 # resolve after the peak
    assert tokens.count("PAYOFF") == 1
    assert any(t in ("BUILD", "ESCALATE", "PEAK") for t in tokens)
    assert "CONTEXT" in tokens                 # the post-hook dip


def test_label_segments_silence_and_hold():
    segments = [
        {"t": 0.0, "duration": 2.0, "motion": 0.7, "audio": 0.6, "silence_ratio": 0.0},
        {"t": 2.0, "duration": 2.0, "motion": 0.1, "audio": 0.0, "silence_ratio": 0.9},
        {"t": 4.0, "duration": 8.0, "motion": 0.2, "audio": 0.2, "silence_ratio": 0.1},
        {"t": 12.0, "duration": 1.5, "motion": 0.8, "audio": 0.7, "silence_ratio": 0.0},
    ]
    tokens = [s["token"] for s in label_segments(segments)]
    assert "SILENCE" in tokens
    assert "HOLD" in tokens


def test_sequence_stats_embedding_shape():
    stats = sequence_stats(label_segments(_montage_segments()), 11.2)
    assert len(stats["embedding"]) == len(EMBEDDING_KEYS)
    assert stats["cut_rate"] > 0.4
    assert stats["payoff_position"] is not None


# ── extractor on real media ───────────────────────────────────────────────────

def test_analyzer_extracts_grammar_from_real_file(tmp_path):
    video = tmp_path / "ref.mp4"
    inputs = []
    colors = ["red", "blue", "yellow", "green", "purple", "orange"]
    for color in colors:
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=25:d=1.2"]
    fc = "".join(f"[{i}:v]" for i in range(len(colors))) + f"concat=n={len(colors)}:v=1:a=0[v]"
    subprocess.run([
        "ffmpeg", "-y", *inputs, "-f", "lavfi", "-i", "sine=frequency=440:duration=7.2",
        "-filter_complex", fc, "-map", "[v]", "-map", f"{len(colors)}:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(video),
    ], check=True, capture_output=True)

    record = AutomaticReferenceAnalyzer().analyze_file(video, {"reference_id": "test_ref", "creator_group": "c1"})
    assert record is not None
    assert len(record["sequence"]) >= 4          # scene detection finds most color cuts
    assert record["stats"]["cut_rate"] > 0.3     # genuinely fast-cut reference
    assert len(record["stats"]["embedding"]) == len(EMBEDDING_KEYS)
    assert record["has_audio"] is True
    assert all(seg["audio"] is not None for seg in record["sequence"])


def test_analyzer_rejects_unusable_files(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a video")
    assert AutomaticReferenceAnalyzer().analyze_file(bogus) is None


# ── dataset ───────────────────────────────────────────────────────────────────

def test_dataset_dedupes_and_snapshots(tmp_path):
    dataset = SequenceDataset(tmp_path / "grammar")
    dataset.add(_synthetic_record("ref_a", "c1"))
    dataset.add(_synthetic_record("ref_a", "c1"))   # same id → deduped
    dataset.add(_synthetic_record("ref_b", "c2"))
    assert len(dataset.records()) == 2
    info = dataset.snapshot()
    assert info["dataset_version"] == "grammar_v001"
    assert info["sequence_count"] == 2
    assert (tmp_path / "grammar" / "dataset_grammar_v001.jsonl").is_file()
    splits = dataset.creator_split()
    assert set(splits) == {"train", "validation", "test"}


# ── policy training + promotion ──────────────────────────────────────────────

def test_policy_trains_and_reports_honestly(tmp_path):
    trainer, result, _ = _trained_policy(tmp_path, count=8)
    assert result["status"] in {"promoted", "rejected"}
    validation = result["validation"]
    assert validation["held_out_logprob_per_token"] > validation["uniform_order_baseline_logprob"]
    artifact = json.loads((tmp_path / "policies" / f"{result['version']}.json").read_text(encoding="utf-8"))
    assert artifact["stage"] == "stage1_statistics_retrieval"
    assert "caption_presence" in artifact["unknown_channels"]   # honest unknowns
    assert artifact["model"]["token_transitions"]["BUILD"]
    assert artifact["model"]["reference_embeddings"]


def test_policy_refuses_small_dataset(tmp_path):
    trainer = PolicyTrainer(tmp_path / "policies")
    result = trainer.train_candidate(
        [_synthetic_record("r1", "c1")], [], {"dataset_version": "grammar_v001"}
    )
    assert result["status"] == "insufficient_sequences"
    assert result["learning_state"] == "BOOTSTRAP"


# ── THE behavior test: learned policy beats continuous (spec §22) ────────────

def _candidate(name, shots, sources=4, target=30.0):
    return {"name": name, "shots": shots, "source_count": sources, "target_duration": target}


def test_policy_scores_montage_above_continuous_run(tmp_path):
    trainer, result, _ = _trained_policy(tmp_path, count=8)
    assert result["status"] == "promoted"
    inference = PolicyInference(trainer.load_active())

    montage = _candidate("chronological_montage", [
        {"source_index": i % 4, "start": 1.0 + j, "end": 3.4 + j,
         "intensity": 0.4 + 0.08 * j, "has_event": j % 2 == 0}
        for i, j in enumerate(range(8))
    ])
    continuous = _candidate("continuous_run", [
        {"source_index": 0, "start": 2.0, "end": 30.0, "intensity": 0.55, "has_event": True},
    ])
    score_montage = inference.score_structure(montage)
    score_continuous = inference.score_structure(continuous)
    assert score_montage["composite"] > score_continuous["composite"]
    # the continuous option remains scoreable (available, not banned)
    assert score_continuous["composite"] > 0.0
    # and the win comes from learned pacing/hook/duration fit, not activity
    assert score_continuous["components"]["pacing_fit"] < score_montage["components"]["pacing_fit"]
    assert score_continuous["components"]["shot_duration_fit"] < score_montage["components"]["shot_duration_fit"]


def test_retrieval_finds_similar_references(tmp_path):
    trainer, result, records = _trained_policy(tmp_path, count=8)
    inference = PolicyInference(trainer.load_active())
    similar = inference.retrieve_similar(records[0]["stats"], k=3)
    assert similar and similar[0]["similarity"] > 0.9  # nearest is itself/its twins


# ── moment graph + structure search ──────────────────────────────────────────

def test_moment_graph_builds_event_motion_and_run_moments():
    sources = [
        {"source_index": 0, "duration": 20.0},
        {"source_index": 1, "duration": 15.0},
    ]
    features = [
        {"activity": [0.2] * 20 + [0.9] * 8 + [0.2] * 12},   # burst at t≈10-14
        {"activity": [0.3] * 30},
    ]
    events = {0: [{"kind": "kill", "start": 10.0, "end": 11.0, "confidence": 0.9}]}
    moments = build_moment_graph(sources, features, events, target_duration=30)
    kinds = {m["kind"] for m in moments}
    assert "event" in kinds and "run" in kinds
    event_moment = next(m for m in moments if m["kind"] == "event")
    assert event_moment["start"] <= 8.5 and event_moment["end"] >= 11.5  # buildup + hold inside
    assert event_moment["intensity"] >= 0.8


def test_structure_search_generates_hypotheses_and_policy_selects(tmp_path):
    trainer, result, _ = _trained_policy(tmp_path, count=8)
    inference = PolicyInference(trainer.load_active())
    sources = [{"source_index": i, "duration": 20.0} for i in range(4)]
    features = [{"activity": [0.3] * 20 + [0.8] * 10 + [0.3] * 10} for _ in range(4)]
    events = {i: [{"kind": "kill", "start": 10.0, "end": 11.0, "confidence": 0.9}] for i in range(4)}
    moments = build_moment_graph(sources, features, events, target_duration=30)
    candidates = generate_structures(moments, 4, 30.0, learned_shot_duration=2.5)
    names = {c["name"] for c in candidates}
    assert {"chronological_montage", "impact_first", "cold_open_then_chronology",
            "escalation_payoff_last", "continuous_run"} <= names
    scored, ranked, selected = score_and_select(candidates, inference, None, seed=3)
    assert ranked[0]["name"] != "continuous_run"     # learned policy deprioritizes it
    assert any(c["name"] == "continuous_run" for c in scored)  # but it stays available
    assert selected is not None


# ── planner + critic ──────────────────────────────────────────────────────────

def _planner_env(tmp_path, trained=True):
    from app.learning.policy_planner import LearnedPolicyPlanner
    planner = LearnedPolicyPlanner(grammar_root=tmp_path / "grammar", policy_root=tmp_path / "policies")
    if trained:
        trainer, result, _ = _trained_policy(tmp_path, count=8)
        planner.policy = trainer.load_active()
        planner.inference = PolicyInference(planner.policy)
        planner.state = "VALIDATED"
        planner.model = f"policy:{planner.policy['version']}"
    return planner


def _planner_context(tmp_path, planner, monkeypatch):
    sources = [{"source_index": i, "duration": 20.0, "audio_mean_db": -22,
                "has_audio": True, "gameplay_events": [
                    {"kind": "kill", "start": 9.0 + i, "end": 10.0 + i, "confidence": 0.9}]}
               for i in range(3)]
    monkeypatch.setattr(type(planner), "_analyze_sources", staticmethod(lambda paths: [
        {"path": str(p), "motion": [0.3] * 18 + [0.9] * 4 + [0.3] * 18,
         "audio": [0.4] * 40, "activity": [0.3] * 18 + [0.9] * 4 + [0.3] * 18}
        for p in paths
    ]))
    return {"sources": sources}, [str(tmp_path / f"c{i}.mp4") for i in range(3)]


def test_planner_produces_learned_multi_shot_plan(tmp_path, monkeypatch):
    planner = _planner_env(tmp_path, trained=True)
    context, paths = _planner_context(tmp_path, planner, monkeypatch)
    plan = planner.create_plan(
        media_context=context, frames=[], style_profile={}, platform="youtube_shorts",
        target_duration=15, user_preferences={"source_paths": paths}, seed=2,
    )
    assert len(plan["shots"]) >= 3
    assert plan["strategy"] != "continuous_run"
    assert all(shot["caption"] is None for shot in plan["shots"])       # no fabricated captions
    assert plan["sound_design"] == []                                    # honest NONE
    assert plan["music_requirements"]                                    # derived, present
    diag = planner.diagnostics
    assert diag["state"] == "VALIDATED"
    assert diag["policy_influence"] == 1.0
    assert diag["candidate_structure_count"] >= 4
    assert diag["selected_structure"] == plan["strategy"]


def test_planner_untrained_reports_zero_influence(tmp_path, monkeypatch):
    planner = _planner_env(tmp_path, trained=False)
    context, paths = _planner_context(tmp_path, planner, monkeypatch)
    plan = planner.create_plan(
        media_context=context, frames=[], style_profile={}, platform="youtube_shorts",
        target_duration=15, user_preferences={"source_paths": paths}, seed=1,
    )
    assert planner.diagnostics["policy_influence"] == 0.0
    assert planner.diagnostics["state"] in {"NO_DATA", "BOOTSTRAP", "LEARNING"}
    assert plan["shots"]


def test_critic_flags_raw_gameplay_and_revises_to_other_structure(tmp_path, monkeypatch):
    planner = _planner_env(tmp_path, trained=True)
    context, paths = _planner_context(tmp_path, planner, monkeypatch)
    planner.create_plan(media_context=context, frames=[], style_profile={},
                        platform="youtube_shorts", target_duration=15,
                        user_preferences={"source_paths": paths}, seed=2)
    # the job-82..91 failure mode: one long continuous shot
    bad_plan = {
        "strategy": "continuous_run",
        "shots": [{"source_index": 0, "start": 0, "end": 30, "role": "action",
                   "transition": "cut", "caption": None, "visual_emphasis": []}],
        "sound_design": [],
    }
    review = planner.review_render(
        bad_plan,
        {"sources": [{"has_audio": True, "duration": 30, "width": 1080, "height": 1920}]},
        [{"time": 1.0}],
    )
    assert review["structured"]["raw_gameplay_like"] is True
    assert "structure_too_continuous" in review["failure_tags"]
    assert review["critic_score"] < 0.7
    revised = planner.revise_plan(bad_plan, review)
    assert revised["strategy"] != "continuous_run"   # diagnosis-driven revision
    assert len(revised["shots"]) >= 2


# ── integration: priority chain in create_edit ───────────────────────────────

def test_create_edit_uses_learned_policy_when_trained(tmp_path, monkeypatch):
    import app.agent.short_form_editor as editor_module
    import app.learning.policy_planner as planner_module
    from app.agent.short_form_editor import ShortFormCreativeEditor
    from app.editing.editor import EditOptions

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    monkeypatch.setattr(planner_module, "get_learning_state", lambda *a, **k: "VALIDATED")

    class FakePolicyPlanner:
        planning_brain = "learned_policy"
        model = "policy:policy_v009"
        diagnostics = {"state": "VALIDATED", "policy_version": "policy_v009",
                       "selected_structure": "chronological_montage",
                       "candidate_structure_count": 5, "policy_influence": 1.0}

        def create_plan(self, **kwargs):
            assert kwargs["user_preferences"]["source_paths"]  # paths must be injected
            return {
                "platform": "youtube_shorts", "strategy": "chronological_montage",
                "target_duration": 4, "music_requirements": {},
                "shots": [
                    {"source_index": 0, "start": 0, "end": 2, "role": "hook", "transition": "cut"},
                    {"source_index": 0, "start": 2, "end": 4, "role": "payoff", "transition": "cut"},
                ],
                "sound_design": [],
            }

    monkeypatch.setattr(planner_module, "LearnedPolicyPlanner", FakePolicyPlanner)

    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")

    def analyze(paths, max_sources=12):
        return {"sources": [{"source_index": 0, "filename": "gameplay.mp4", "duration": 8,
                             "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25}]}, []

    def render(inputs, destination, options, callback):
        pathlib.Path(destination).write_bytes(b"rendered")
        return destination

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    editor = ShortFormCreativeEditor(
        model=None,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=render,
    )
    artifact = editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=4, bgm_track=None, game="none"),
    )
    assert artifact["planning_mode"] == "learned_policy"
    assert artifact["model_version"] == "policy:policy_v009"
    assert artifact["learning"]["policy_version"] == "policy_v009"
    assert artifact["strategy"] == "chronological_montage"
    assert not any("local features" in w for w in artifact["warnings"])
