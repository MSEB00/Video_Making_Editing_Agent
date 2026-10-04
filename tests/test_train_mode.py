"""Tests for train-mode (self-training loop) and feedback (human validation) CLI."""
import json
import os
import pathlib
import tempfile

import pytest
from click.testing import CliRunner


@pytest.fixture()
def tmp_pipeline(monkeypatch):
    """Point ReferenceTrainingPipeline at a throwaway root."""
    import app.training.reference_pipeline as rp_module

    real = rp_module.ReferenceTrainingPipeline
    root = pathlib.Path(tempfile.mkdtemp()) / "training"

    def factory(*args, **kwargs):
        kwargs.setdefault("training_root", root)
        return real(*args, **kwargs)

    monkeypatch.setattr(rp_module, "ReferenceTrainingPipeline", factory)
    return root


@pytest.fixture()
def tmp_db(monkeypatch):
    from app.storage.db import init_db, reset_engine

    db_path = pathlib.Path(tempfile.mkdtemp()) / "fb.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    yield db_path
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_engine()


def _report(output: str) -> dict:
    return json.loads(output[output.index("{"):])


# ── train-mode ────────────────────────────────────────────────────────────────

def test_train_mode_offline_trains_and_reports(tmp_pipeline, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    import main as main_module

    result = CliRunner().invoke(main_module.cli, ["train-mode", "--offline"])
    assert result.exit_code == 0, result.output
    report = _report(result.output)["training_mode_report"]
    assert report["offline"] is True
    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert stages["downloads"]["skipped"] == "offline mode"
    assert stages["dataset"]["reference_count"] == 0
    assert stages["train"]["status"] == "insufficient_references"  # honest, never faked
    assert stages["model"]["status"] == "no_active_model"


def test_train_mode_rotates_archive_queries_and_skips_keyless_providers(tmp_pipeline, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_DATA_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    import app.research.licensed_media_provider as provider_module
    import main as main_module

    captured_queries = []

    class FakeArchive:
        def __init__(self, *a, **k):
            pass

        def collect(self, limit=6, queries=None, **kwargs):
            captured_queries.append(list(queries))
            return {"imported": [{"file": f"ref_{len(captured_queries)}.mp4"}], "skipped": []}

    monkeypatch.setattr(provider_module, "InternetArchiveProvider", FakeArchive)

    first = CliRunner().invoke(main_module.cli, ["train-mode", "--limit", "3"])
    assert first.exit_code == 0, first.output
    report = _report(first.output)["training_mode_report"]
    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert stages["internet_archive"]["imported"] == 1
    assert stages["pexels"]["skipped"] == "PEXELS_API_KEY not set"
    assert stages["youtube_metadata_pool"]["skipped"] == "YOUTUBE_DATA_API_KEY not set"

    state = json.loads((tmp_pipeline / "licensed_state.json").read_text(encoding="utf-8"))
    assert state["query_offset"] == 3 and state["runs"] == 1

    second = CliRunner().invoke(main_module.cli, ["train-mode", "--limit", "3", "--no-youtube-pool"])
    assert second.exit_code == 0, second.output
    assert len(captured_queries) == 2
    assert not set(captured_queries[0]) & set(captured_queries[1])  # fresh queries each run
    state = json.loads((tmp_pipeline / "licensed_state.json").read_text(encoding="utf-8"))
    assert state["query_offset"] == 6 and state["runs"] == 2


# ── feedback ──────────────────────────────────────────────────────────────────

def _completed_job(tmp_path):
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    output = tmp_path / "job_1_final.mp4"
    output.write_bytes(b"video")
    output.with_suffix(".edit-plan.json").write_text(
        json.dumps({"strategy": "clutch montage", "platform": "youtube_shorts"}), encoding="utf-8"
    )
    db = SessionLocal()
    job = Job(status="completed", input_path=str(tmp_path), output_path=str(output))
    db.add(job)
    db.commit()
    job_id = job.id
    db.close()
    return job_id


def test_feedback_records_validation_with_tag_filtering(tmp_pipeline, tmp_db, tmp_path):
    import main as main_module

    job_id = _completed_job(tmp_path)
    result = CliRunner().invoke(main_module.cli, [
        "feedback", str(job_id), "--rating", "1",
        "--tags", "hook_good, kills_well_aligned, made_up_tag",
        "--note", "kills land nicely, keep the hold length",
    ])
    assert result.exit_code == 0, result.output
    report = _report(result.output)
    assert report["status"] == "recorded"
    assert report["tags"] == ["hook_good", "kills_well_aligned"]
    assert report["ignored_tags"] == ["made_up_tag"]

    lines = (tmp_pipeline / "feedback" / "edits.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["rating"] == 1
    assert record["tags"] == ["hook_good", "kills_well_aligned"]
    assert record["context"]["strategy"] == "clutch montage"
    assert record["notes"].startswith("kills land nicely")


def test_feedback_rejects_unknown_or_incomplete_jobs(tmp_pipeline, tmp_db, tmp_path):
    import main as main_module

    missing = CliRunner().invoke(main_module.cli, ["feedback", "999", "--rating", "1"])
    assert missing.exit_code != 0
    assert "not found" in missing.output

    from app.storage.db import SessionLocal
    from app.storage.models import Job
    db = SessionLocal()
    queued = Job(status="queued", input_path=str(tmp_path))
    db.add(queued)
    db.commit()
    queued_id = queued.id
    db.close()

    rejected = CliRunner().invoke(main_module.cli, ["feedback", str(queued_id), "--rating", "-1"])
    assert rejected.exit_code != 0
    assert "completed edits" in rejected.output
