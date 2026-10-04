import pathlib

from app.ai.local_editing import LocalShortFormEditingModel


def test_local_fallback_prefers_continuity_and_can_decline_unsynchronized_music():
    model = LocalShortFormEditingModel()
    model._analyze_source = lambda path: {
        "path": path,
        "motion": [0.4] * 80,
        "audio": [0.3] * 80,
        "activity": [0.37] * 80,
    }

    plan = model.create_plan(
        [pathlib.Path("capture.mp4")],
        {"sources": []},
        "youtube_shorts",
        35,
    )

    assert plan["strategy"] == "continuous_sequence"
    assert len(plan["shots"]) == 1
    assert plan["shots"][0]["start"] == 0
    assert plan["shots"][0]["end"] == 35

    music = model.rank_music({}, plan, [{
        "id": "track-1",
        "duration": 120,
        "download_allowed": True,
        "license_url": "https://example.org/license",
        "audio_features": {"bpm_confidence": 0.22},
    }])

    assert music["track_id"] is None
    assert "preserve the original gameplay audio" in music["rationale"]