"""Tests for the web-chat manual planning workflow (plan-request + --plan-file)."""
import json
import pathlib
import subprocess

import pytest
from click.testing import CliRunner


@pytest.fixture()
def fresh_db(monkeypatch):
    from app.storage.db import init_db, reset_engine
    import tempfile
    db_path = pathlib.Path(tempfile.mkdtemp()) / "cli.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    yield db_path
    reset_engine()


@pytest.fixture()
def two_clips(tmp_path):
    clip_dir = tmp_path / "session"
    clip_dir.mkdir()
    for i in range(2):
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc2=s=320x240:r=25:d=4",
            "-f", "lavfi", "-i", f"sine=frequency={300+i*100}:duration=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(clip_dir / f"clip_{i}.mp4"),
        ], check=True, capture_output=True)
    return clip_dir


# ── ManualPlanModel ───────────────────────────────────────────────────────────

def test_manual_plan_model_parses_chat_reply_shapes():
    from app.ai.manual_plan_model import ManualPlanModel
    from app.ai.remote_json_model import RemoteJSONModelError

    fenced = ManualPlanModel('Sure! Here is your plan:\n```json\n{"platform": "youtube_shorts", "shots": []}\n```')
    assert fenced.create_plan()["platform"] == "youtube_shorts"

    garbage = ManualPlanModel("I cannot help with that request.")
    with pytest.raises(RemoteJSONModelError, match="no JSON object"):
        garbage.create_plan()


def test_manual_plan_model_delegates_review_and_keeps_plan_on_revise():
    from app.ai.manual_plan_model import ManualPlanModel

    model = ManualPlanModel("{}")
    plan = {"shots": [{"source_index": 0}]}
    assert model.revise_plan(plan) is plan
    review = model.review_render(
        plan,
        {"sources": [{"width": 1080, "height": 1920, "has_audio": True, "duration": 10}]},
        [{"time": 1.0}],
    )
    assert review["needs_revision"] is False
    assert model.model == "chat-web-manual"


# ── plan-request CLI ──────────────────────────────────────────────────────────

def test_plan_request_writes_prompt_and_frames(tmp_path, two_clips, monkeypatch):
    import main as main_module

    monkeypatch.chdir(tmp_path)  # temp/ai lands in tmp, not the repo
    result = CliRunner().invoke(main_module.cli, [
        "plan-request", str(two_clips), "--duration", "20",
        "--request", "kill montage in match order",
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output[result.output.index("{"):])
    assert report["status"] == "request_written"
    assert report["sources_analyzed"] == 2

    request_path = pathlib.Path(report["request_file"])
    assert request_path.is_file()
    content = request_path.read_text(encoding="utf-8")
    assert "=== SYSTEM ===" in content and "=== CONTEXT ===" in content
    assert "context-aware short-form gaming editor" in content      # plan system prompt
    assert "kill montage in match order" in content                # user request embedded
    ctx_start = content.index("=== CONTEXT ===") + len("=== CONTEXT ===")
    ctx_end = content.index("=== FRAME IMAGES") if "=== FRAME IMAGES" in content else content.index("=== AFTER")
    context = json.loads(content[ctx_start:ctx_end].strip())
    assert len(context["media"]["sources"]) == 2
    assert context["user_preferences"]["full_session_coverage"] is True
    assert report["frame_images"] and pathlib.Path(report["frame_images"][0]).is_file()
    assert str(report["response_file_to_create"]).endswith(".txt")
    assert any("--plan-file" in step for step in report["instructions"])


# ── edit --plan-file CLI ──────────────────────────────────────────────────────

def test_edit_plan_file_builds_manual_model(tmp_path, monkeypatch, fresh_db):
    import main as main_module
    from app.ai.manual_plan_model import ManualPlanModel
    from app.orchestrator import orchestrator as orchestrator_module

    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    (clip_dir / "a.mp4").write_bytes(b"clip")
    (clip_dir / "b.mp4").write_bytes(b"clip")
    plan_file = tmp_path / "reply.txt"
    plan_file.write_text('```json\n{"platform": "youtube_shorts", "shots": []}\n```', encoding="utf-8")

    captured = {}

    def fake_orchestrate(job_id, options=None, progress_callback=None, model=None):
        captured["model"] = model
        out = tmp_path / f"job_{job_id}_final.mp4"
        out.write_bytes(b"r")
        return out

    monkeypatch.setattr(orchestrator_module, "orchestrate_job", fake_orchestrate)
    result = CliRunner().invoke(main_module.cli, [
        "edit", str(clip_dir), "--plan-file", str(plan_file), "--duration", "20",
    ])
    assert result.exit_code == 0, result.output
    assert "Using web-chat AI plan" in result.output
    model = captured["model"]
    assert isinstance(model, ManualPlanModel)
    assert model.create_plan()["platform"] == "youtube_shorts"
    assert len(model._source_paths) == 2  # sources available for local music ranking


# ── drawtext graceful degradation ────────────────────────────────────────────

def _fake_render_env(tmp_path, monkeypatch, drawtext_available):
    """Deterministic render environment: no real probing, no real encoding."""
    import app.editing.editor as editor

    monkeypatch.setattr(editor, "_DRAWTEXT_AVAILABLE", drawtext_available)
    monkeypatch.setattr(editor, "get_duration", lambda p: 3.0)
    monkeypatch.setattr(editor, "probe", lambda p: {"streams": [{"codec_type": "audio"}]})
    monkeypatch.setattr(editor, "get_audio_stream", lambda d: {"codec_type": "audio"})

    captured = {}

    def fake_run(cmd, *a, **kw):
        if "-filter_complex" in cmd:
            captured["fc"] = cmd[cmd.index("-filter_complex") + 1]
            pathlib.Path(cmd[-1]).write_bytes(b"fake video")
        return type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(editor.subprocess, "run", fake_run)
    return editor, captured


def test_captions_degrade_when_drawtext_missing(tmp_path, monkeypatch):
    editor, captured = _fake_render_env(tmp_path, monkeypatch, drawtext_available=False)
    notified = []
    out = tmp_path / "out.mp4"
    result = editor.render_edited_video(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"], out,
        editor.EditOptions(variation_seed=5, aspect_ratio="9:16",
                           planned_captions=["EARLY WORK", "CLOSE IT OUT"],
                           transition_sequence=["fade"]),
        progress_callback=lambda icon, msg: notified.append(msg),
    )
    assert result == out and out.is_file()          # render survives
    assert "drawtext" not in captured["fc"]          # captions stripped from graph
    assert any("drawtext" in message for message in notified)  # honest notice


def test_captions_render_when_drawtext_available(tmp_path, monkeypatch):
    editor, captured = _fake_render_env(tmp_path, monkeypatch, drawtext_available=True)
    out = tmp_path / "out.mp4"
    editor.render_edited_video(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"], out,
        editor.EditOptions(variation_seed=5, aspect_ratio="9:16",
                           planned_captions=["EARLY WORK", None],
                           transition_sequence=["fade"]),
    )
    assert captured["fc"].count("drawtext") == 1     # only the non-None caption
    assert "EARLY WORK" in captured["fc"]
