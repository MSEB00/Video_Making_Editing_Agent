import dashboard.app as dashboard
from app.research.video_observer import VideoObservationStore


def test_research_flow_uses_metadata_and_manual_player_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_DERIVED_OBSERVATIONS_APPROVED", "true")
    store = VideoObservationStore(tmp_path / "research")
    monkeypatch.setattr(dashboard, "_research_store", lambda: store)

    class FakeResearchAgent:
        def discover(self, **kwargs):
            assert kwargs["license_filter"] == "creativeCommon"
            assert kwargs["strategies"] == ["cinematic gaming Shorts"]
            return [{
                "video_id": "abcDEF_1234",
                "title": "A gaming short",
                "channel": "Creator",
                "duration_seconds": 20,
                "license": "creativeCommon",
                "discovery_queries": kwargs["strategies"],
            }]

    monkeypatch.setattr(dashboard, "YouTubeResearchAgent", FakeResearchAgent)
    client = dashboard.app.test_client()

    discovered = client.post("/api/research/discover", json={
        "topic": "cinematic gaming Shorts",
        "limit": 3,
    })

    assert discovered.status_code == 200
    assert discovered.json["found"] == 1
    assert discovered.json["videos"][0]["media_stored"] is False
    session_response = client.post("/api/research/session", json={"video_id": "abcDEF_1234"})
    session_id = session_response.json["session_id"]
    observation_response = client.post("/api/research/observations", json={
        "session_id": session_id,
        "start_seconds": 3.2,
        "end_seconds": 6.5,
        "structure_label": "buildup",
        "editing_decisions": {"cut": "not_observed", "bgm": "observed"},
        "note": "The music rises before the held shot resolves.",
        "confidence": 0.7,
    })

    assert observation_response.status_code == 201
    assert observation_response.json["observation"]["training_eligible"] is False
    details = client.get(f"/api/research/session/{session_id}")
    assert len(details.json["observations"]) == 1
    assert client.post(f"/api/research/session/{session_id}/finish").json["status"] == "completed"
    assert client.get("/api/research/state").json["youtube_player_observations_trainable"] is False


def test_research_routes_reject_non_loopback_clients():
    client = dashboard.app.test_client()

    response = client.get("/research", environ_base={"REMOTE_ADDR": "192.0.2.10"})

    assert response.status_code == 403


def test_research_discovery_rejects_non_object_payloads():
    client = dashboard.app.test_client()

    response = client.post("/api/research/discover", json=["unexpected"])

    assert response.status_code == 400


def test_youtube_player_annotations_are_disabled_without_explicit_approval(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_DERIVED_OBSERVATIONS_APPROVED", raising=False)
    store = VideoObservationStore(tmp_path / "research")
    client = dashboard.app.test_client()
    video = store.register_videos([{"video_id": "xYzABC_12345", "duration_seconds": 15}])[0]

    response = client.post("/api/research/session", json={"video_id": video["video_id"]})

    assert response.status_code == 403
    assert "written YouTube approval" in response.json["error"]