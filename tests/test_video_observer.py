import json

import pytest

from app.research.video_observer import ResearchValidationError, VideoObservationStore
from app.training.reference_pipeline import ReferenceTrainingPipeline


def test_player_observation_stores_notes_but_never_media_or_training_data(tmp_path):
    store = VideoObservationStore(tmp_path / "research")
    video = store.register_videos([{
        "video_id": "abcDEF_1234",
        "title": "Gaming reference",
        "channel": "Public channel",
        "duration_seconds": 30,
        "discovered_at": "2026-10-01T10:00:00+00:00",
    }])[0]
    session = store.create_session(video["video_id"])

    observation = store.add_observation({
        "session_id": session["session_id"],
        "start_seconds": 5.2,
        "end_seconds": 8.1,
        "structure_label": "buildup",
        "editing_decisions": {"cut": "observed", "sfx": "not_observed"},
        "context": {"visual_intensity": 0.7, "speech_present": False},
        "confidence": 0.8,
        "note": "Hold the shot through the reveal.",
    })

    assert observation["source"]["media_saved"] is False
    assert observation["training_eligible"] is False
    assert observation["start_seconds"] == 5.2
    assert store.observations_for(session["session_id"]) == [observation]
    completed = store.finish_session(session["session_id"])
    assert completed["status"] == "completed"
    assert store.stats()["training_eligible_youtube_observations"] == 0
    assert store.stats()["media_stored"] is False


def test_player_observation_validates_time_and_feature_ranges(tmp_path):
    store = VideoObservationStore(tmp_path / "research")
    store.register_videos([{"video_id": "abcDEF_1234", "duration_seconds": 10}])
    session = store.create_session("abcDEF_1234")

    with pytest.raises(ResearchValidationError, match="timestamps"):
        store.add_observation({
            "session_id": session["session_id"],
            "start_seconds": 9,
            "end_seconds": 11,
            "note": "out of range",
        })
    with pytest.raises(ResearchValidationError, match="visual_intensity"):
        store.add_observation({
            "session_id": session["session_id"],
            "start_seconds": 1,
            "end_seconds": 2,
            "note": "invalid score",
            "context": {"visual_intensity": 1.2},
        })


def test_discovery_metadata_expires_after_thirty_days(tmp_path):
    store = VideoObservationStore(tmp_path / "research")
    expired = [{
        "video_id": "abcDEF_1234",
        "discovered_at": "2026-01-01T00:00:00+00:00",
        "research_status": "studied",
        "observation_status": "observed",
    }]
    store._write_jsonl(store.videos_path, expired)

    videos = store.list_videos()

    assert videos == []
    assert store.videos_path.read_text(encoding="utf-8") == ""


def test_only_approved_and_rights_confirmed_observations_enter_training(tmp_path):
    root = tmp_path / "training"
    store = VideoObservationStore(root / "research", allow_derived_observations=True)
    store.register_videos([{
        "video_id": "abcDEF_1234",
        "channel_id": "creator-1",
        "duration_seconds": 20,
        "license": "creativeCommon",
        "category": "fps",
    }])
    session = store.create_session("abcDEF_1234")
    store.add_observation({
        "session_id": session["session_id"],
        "start_seconds": 4,
        "end_seconds": 7,
        "structure_label": "buildup",
        "editing_decisions": {"zoom": "observed", "sfx": "not_observed"},
        "context": {"visual_intensity": 0.8},
        "rights_confirmed": True,
        "note": "Zoom held through a visual escalation.",
    })

    pipeline = ReferenceTrainingPipeline(root)
    examples = pipeline._read_examples()

    assert len(examples) == 1
    assert examples[0]["creator_group"] == "creator-1"
    assert examples[0]["provenance"]["media_saved"] is False
    assert examples[0]["moments"][0]["moment"]["visual_intensity"] == 0.8
    assert examples[0]["moments"][0]["observed_edit"] == {"zoom": 1.0, "sfx": 0.0}


def test_unapproved_or_unconfirmed_observations_never_enter_training(tmp_path):
    root = tmp_path / "training"
    store = VideoObservationStore(root / "research", allow_derived_observations=False)
    store.register_videos([{
        "video_id": "abcDEF_1234",
        "duration_seconds": 20,
        "license": "creativeCommon",
    }])
    session = store.create_session("abcDEF_1234")
    store.add_observation({
        "session_id": session["session_id"],
        "start_seconds": 1,
        "end_seconds": 2,
        "structure_label": "hook",
        "rights_confirmed": True,
        "note": "Human note.",
    })

    assert ReferenceTrainingPipeline(root)._read_examples() == []