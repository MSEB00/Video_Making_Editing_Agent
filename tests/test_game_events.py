"""Tests for gameplay event detection and kill-aligned shot snapping."""
import pathlib
import subprocess

import pytest

from app.agent.short_form_editor import ShortFormCreativeEditor
import app.agent.short_form_editor as editor_module
from app.analysis.games import get_event_detector
from app.analysis.games.valorant import ValorantEventDetector
from app.ai.local_editing import LocalShortFormEditingModel  # noqa: F401  (import guard)
from app.editing.edit_plan import EditPlan, PlannedShot
from app.editing.editor import EditOptions


def _make_feed_video(path: pathlib.Path, with_entries: bool) -> None:
    """640x360 static field; a white 'kill-feed entry' box appears in the
    top-right HUD region during t=3-7 and t=9-11 when with_entries."""
    inputs = [
        "-f", "lavfi", "-i", "color=c=0x336633:s=640x360:r=25:d=12",
    ]
    extra = []
    if with_entries:
        inputs += ["-f", "lavfi", "-i", "color=c=white:s=120x40:r=25:d=12"]
        extra = [
            "-filter_complex",
            "[0:v][1:v]overlay=x=470:y=12:enable='between(t,3,7)+between(t,9,11)'[out]",
            "-map", "[out]",
        ]
    cmd = [
        "ffmpeg", "-y", *inputs, *extra,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def feed_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("events") / "feed.mp4"
    _make_feed_video(path, with_entries=True)
    return path


@pytest.fixture(scope="module")
def quiet_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("events") / "quiet.mp4"
    _make_feed_video(path, with_entries=False)
    return path


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_returns_valorant_detector_and_none_for_unknown():
    detector = get_event_detector("valorant")
    assert isinstance(detector, ValorantEventDetector)
    assert get_event_detector("chess") is None
    assert get_event_detector("") is None


# ── Detection on synthetic footage ───────────────────────────────────────────

def test_detector_finds_kill_feed_bursts(feed_video):
    events = ValorantEventDetector({}).detect(feed_video)
    assert len(events) == 2
    first, second = events
    assert first["kind"] == "kill"
    assert 2.5 <= first["start"] <= 3.5
    assert 6.75 <= first["end"] <= 7.75
    assert 8.5 <= second["start"] <= 9.5
    assert all(0 < event["confidence"] <= 1 for event in events)
    assert all(event["peak_structure"] > 0 for event in events)


def test_detector_reports_no_events_on_quiet_footage(quiet_video):
    assert ValorantEventDetector({}).detect(quiet_video) == []


def test_detector_handles_missing_file(tmp_path):
    assert ValorantEventDetector({}).detect(tmp_path / "absent.mp4") == []


# ── Edit plan string-emphasis fix ────────────────────────────────────────────

def test_visual_emphasis_string_is_wrapped_not_split():
    shot = PlannedShot.from_dict(
        {"source_index": 0, "start": 0, "end": 2, "visual_emphasis": "punch_zoom"},
        source_count=1,
    )
    assert shot.visual_emphasis == ["punch_zoom"]
    shot = PlannedShot.from_dict(
        {"source_index": 0, "start": 0, "end": 2, "visual_emphasis": ["punch_zoom", "freeze"]},
        source_count=1,
    )
    assert shot.visual_emphasis == ["punch_zoom", "freeze"]


# ── Deterministic shot alignment ─────────────────────────────────────────────

def _editor() -> ShortFormCreativeEditor:
    return ShortFormCreativeEditor(
        model=object(),
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=type("Style", (), {"learn": lambda self, platform: {}})(),
        sfx_library=type("Sfx", (), {"list_assets": lambda self: []})(),
        renderer=lambda *a, **k: None,
    )


def _plan(start: float, end: float) -> EditPlan:
    return EditPlan.from_dict({
        "platform": "youtube_shorts", "strategy": "test", "target_duration": end - start,
        "shots": [{"source_index": 0, "start": start, "end": end, "role": "action", "transition": "cut"}],
    }, source_count=1)


def test_alignment_places_event_as_payoff_with_hold():
    editor = _editor()
    plan = _plan(6.0, 10.0)
    events = {0: [{"kind": "kill", "start": 7.0, "end": 7.5, "confidence": 0.9}]}
    sources = [{"source_index": 0, "duration": 12.0}]
    alignment = editor._align_plan_to_events(plan, events, sources)
    shot = plan.shots[0]
    # event end (7.5) + hold (1.0) => shot ends 8.5; length preserved => starts 4.5
    assert shot.end == pytest.approx(8.5, abs=0.01)
    assert shot.start == pytest.approx(4.5, abs=0.01)
    assert shot.start <= 7.0 and shot.end >= 7.5 + 0.9  # event fully inside + hold
    assert alignment[0]["event_end"] == 7.5


def test_alignment_never_cuts_mid_event():
    editor = _editor()
    plan = _plan(0.0, 4.0)
    # event straddles the planned cut point at 4.0
    events = {0: [{"kind": "kill", "start": 3.0, "end": 5.5, "confidence": 0.8}]}
    sources = [{"source_index": 0, "duration": 10.0}]
    editor._align_plan_to_events(plan, events, sources)
    shot = plan.shots[0]
    assert shot.start <= 3.0
    assert shot.end >= 6.0  # event end + hold fully inside


def test_alignment_respects_max_shift_and_no_events():
    editor = _editor()
    plan = _plan(0.0, 4.0)
    far_events = {0: [{"kind": "kill", "start": 20.0, "end": 20.5, "confidence": 0.9}]}
    sources = [{"source_index": 0, "duration": 30.0}]
    assert editor._align_plan_to_events(plan, far_events, sources) == {}
    assert plan.shots[0].start == 0.0 and plan.shots[0].end == 4.0
    assert editor._align_plan_to_events(plan, {}, sources) == {}


# ── End-to-end integration through create_edit ───────────────────────────────

def test_create_edit_snaps_plan_and_records_events_in_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "result.mp4"

    raw_plan = {
        "platform": "youtube_shorts", "strategy": "montage", "target_duration": 4,
        "shots": [{"source_index": 0, "start": 0.0, "end": 4.0, "role": "action", "transition": "cut"}],
        "music_requirements": {}, "sound_design": [],
    }

    class Model:
        def create_plan(self, **kwargs):
            # the hosted planner receives detected events in the media context
            assert kwargs["media_context"]["sources"][0]["gameplay_events"], "events must reach the planner"
            return raw_plan

    fake_events = [{"kind": "kill", "start": 7.0, "end": 7.5, "confidence": 0.9}]

    class FakeDetector:
        def detect(self, path):
            return fake_events

    import app.analysis.games as games_module
    monkeypatch.setattr(games_module, "get_event_detector", lambda game: FakeDetector())

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": pathlib.Path(paths[0]).name, "duration": 12.0,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
        }]}, [{"time": 0.5, "data_url": "data:image/jpeg;base64,eA=="}]

    def render(inputs, destination, options, callback):
        # renderer must receive the snapped window via planned_segments
        assert options.planned_segments == [(pytest.approx(4.5, abs=0.01), pytest.approx(4.0, abs=0.01))]
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
        [source], output,
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=4, bgm_track=None, game="valorant"),
    )
    timeline_shot = artifact["timeline"][0]
    assert timeline_shot["source_start"] == pytest.approx(4.5, abs=0.01)
    assert timeline_shot["source_end"] == pytest.approx(8.5, abs=0.01)
    assert timeline_shot["aligned_event"]["event_start"] == 7.0
    assert artifact["gameplay_events"]["0"][0]["start"] == 7.0
    assert artifact["event_alignment"]["0"]["event_confidence"] == 0.9


# ── Self-calibration (permissive retry pass) ─────────────────────────────────

def test_relaxed_retry_finds_subtle_events(feed_video):
    # Strict threshold deliberately above the burst magnitude: the configured
    # pass finds nothing, the automatic permissive pass recovers the events.
    detector = ValorantEventDetector({"event_detection": {"min_structure_delta": 150.0}})
    events = detector.detect(feed_video)
    assert len(events) == 2
    assert all(event.get("relaxed_pass") is True for event in events)
    assert all(event["confidence"] <= 0.76 for event in events)  # discounted (0.95 * 0.8)
    assert 2.5 <= events[0]["start"] <= 3.5


def test_relaxed_retry_still_reports_nothing_when_truly_quiet(quiet_video):
    detector = ValorantEventDetector({"event_detection": {"min_structure_delta": 150.0}})
    assert detector.detect(quiet_video) == []
