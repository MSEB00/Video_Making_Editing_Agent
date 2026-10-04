import os, tempfile, pathlib, subprocess, random, json, threading
import pytest
import app.agent.short_form_editor as creative_agent_module
from app.audio.jamendo_provider import JamendoMusicProvider, MusicRequirements
from app.audio.sfx_library import SfxLibrary
from app.orchestrator.orchestrator import orchestrate_job
import dashboard.app as dashboard_app_module
from app.storage.db import init_db, SessionLocal
from app.storage.models import Job
from app.publishing.publisher import publish_video
from app.ai.creative_editor import CreativeAIConfigurationError, ShortFormEditingModel, _select_prompt_frames
from app.agent.short_form_editor import ShortFormCreativeEditor
from app.editing.edit_plan import EditPlan
from app.editing.editor import EditOptions, _choose_clip_segment, _choose_transition_type, _select_source_clips, _split_single_source_shots
from app.editing.style_learner import EditingStyleLearner
from app.research.youtube_research_agent import YouTubeResearchAgent
from app.training.collector import collect_candidate_pool, curate_candidates
from app.training.dataset_quality import dataset_quality_report, split_by_creator
from app.training.reference_pipeline import ReferenceTrainingPipeline
from dashboard.app import _parse_editing_options, _parse_intent

def _create_color_video(path: pathlib.Path, color: str):
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s=320x240:d=2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@pytest.fixture(scope="function")
def fresh_db():
    # Use a temporary SQLite DB file
    from app.storage.db import reset_engine
    db_path = pathlib.Path(tempfile.mkdtemp()) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    reset_engine()
    init_db()
    yield
    # cleanup
    reset_engine()
    os.environ.pop("DATABASE_URL", None)
    if db_path.exists():
        db_path.unlink()

def test_orchestrate_job_creates_output(fresh_db):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = pathlib.Path(tmpdir)
        # create two dummy videos
        _create_color_video(input_dir / "a.mp4", "red")
        _create_color_video(input_dir / "b.mp4", "blue")
        # add job record
        session = SessionLocal()
        job = Job(status="queued", input_path=str(input_dir), extra_metadata="{}")
        session.add(job)
        session.commit()
        job_id = job.id
        # run orchestrator
        output_path = orchestrate_job(job_id)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        # ensure DB updated
        refreshed = session.get(Job, job_id)
        assert refreshed.output_path == str(output_path)
        session.close()


def test_shorts_request_uses_fast_vertical_preset():
    options = _parse_editing_options("make a YouTube Shorts video")

    assert options["aspect_ratio"] == "9:16"
    assert options["transition_type"] == "random"
    assert options["randomize_clips"] is True
    assert options["max_clips"] == 10
    assert options["max_clip_duration"] == 6.0


def test_shorts_request_keeps_requested_transition():
    options = _parse_editing_options("make a Shorts video with wipe transitions")

    assert options["transition_type"] == "wipeleft"
    assert options["max_clip_duration"] == 6.0


def test_shorts_request_can_explicitly_disable_transitions():
    options = _parse_editing_options("make Shorts with direct cuts only")

    assert options["transition_type"] == "none"


def test_transition_sfx_can_be_disabled_by_request():
    options = _parse_editing_options("make a Shorts edit with no SFX")

    assert options["sfx_preference"] is False


def test_publish_command_targets_platforms_and_optional_job():
    intent, params = _parse_intent("publish job 12 to YouTube and Instagram")

    assert intent == "publish"
    assert params == {"platforms": ["youtube", "instagram"], "job_id": 12}


def test_publish_reports_missing_platform_setup_without_uploading(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(tmp_path / "missing-client.json"))
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_USER_ID", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    results = publish_video(tmp_path / "not-uploaded.mp4", ["youtube", "instagram"])

    assert not results["youtube"]["success"]
    assert "YOUTUBE_CLIENT_SECRETS_FILE" in results["youtube"]["message"]
    assert not results["instagram"]["success"]
    assert "PUBLIC_BASE_URL" in results["instagram"]["message"]


def test_random_clip_selection_varies_by_seed_and_is_reproducible():
    clips = [pathlib.Path(f"clip_{index}.mp4") for index in range(8)]
    first = _select_source_clips(clips, EditOptions(max_clips=4, randomize_clips=True), random.Random(17))
    repeat = _select_source_clips(clips, EditOptions(max_clips=4, randomize_clips=True), random.Random(17))
    next_edit = _select_source_clips(clips, EditOptions(max_clips=4, randomize_clips=True), random.Random(18))

    assert first == repeat
    assert first != next_edit
    assert len(first) == 4


def test_random_clip_segment_varies_start_and_duration():
    options = EditOptions(max_clip_duration=6, randomize_clips=True)
    first = _choose_clip_segment(20, options, random.Random(21))
    repeat = _choose_clip_segment(20, options, random.Random(21))
    next_edit = _choose_clip_segment(20, options, random.Random(22))

    assert first == repeat
    assert first != next_edit
    assert 0 < first[0] < 14
    assert 3 <= first[1] <= 6


def test_single_long_recording_becomes_non_overlapping_shots():
    source = pathlib.Path("long_gameplay.mp4")
    options = EditOptions(max_clips=8, max_clip_duration=6, randomize_clips=True)

    shots = _split_single_source_shots(source, 30, options, random.Random(23))
    repeat = _split_single_source_shots(source, 30, options, random.Random(23))

    assert len(shots) >= 2
    assert shots == repeat
    ordered = sorted(shots, key=lambda shot: shot[1])
    assert all(
        current[1] >= previous[1] + previous[2]
        for previous, current in zip(ordered, ordered[1:])
    )
    assert all(shot[0] == source and shot[2] <= 6 for shot in shots)


def test_random_transitions_vary_without_adjacent_repeats():
    rng = random.Random(29)
    transitions = []
    previous = None
    for _ in range(8):
        current = _choose_transition_type("random", rng, previous)
        transitions.append(current)
        previous = current

    assert len(set(transitions)) > 1
    assert all(first != second for first, second in zip(transitions, transitions[1:]))


def test_random_edit_can_request_trending_audio():
    options = _parse_editing_options("make random clips with latest audio")

    assert options["randomize_clips"] is True
    assert options["transition_type"] == "random"
    assert options["aspect_ratio"] == "9:16"
    assert options["bgm_track"] == "trending licensed Jamendo catalog"


def test_jamendo_download_filters_and_saves_attribution(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, payload=None):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

        def iter_content(self, chunk_size):
            yield b"licensed-audio"

    calls = []
    track = {
        "id": "123",
        "name": "Weekly Highlight",
        "artist_name": "Creator",
        "shareurl": "https://www.jamendo.com/track/123",
        "license_ccurl": "https://creativecommons.org/licenses/by/4.0/",
        "audiodownload_allowed": True,
        "audiodownload": "https://audio.example/123.mp3",
        "audio": "https://audio.example/123-preview.mp3",
    }
    restricted_track = dict(track, id="456", audiodownload_allowed=False)

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/tracks/"):
            return FakeResponse({"headers": {"status": "success"}, "results": [track, restricted_track]})
        return FakeResponse()

    class FakeHTTP:
        get = staticmethod(fake_get)

    provider = JamendoMusicProvider(client_id="test-client", cache_dir=tmp_path / "cache", http=FakeHTTP)
    candidates = provider.search(MusicRequirements(
        search="high energy instrumental",
        fuzzytags=("electronic", "intense"),
        speed=("high",),
        instrumental=True,
        duration_min=30,
        duration_max=120,
    ))
    track_path, license_record = provider.download_track(candidates[0], tmp_path / "assets")

    assert len(candidates) == 1
    assert track_path == tmp_path / "assets" / "jamendo_123.mp3"
    assert track_path.read_bytes() == b"licensed-audio"
    assert license_record["artist"] == "Creator"
    assert license_record["license_url"] == track["license_ccurl"]
    assert license_record["commercial_use"] == "verify_with_provider_for_intended_use"
    assert license_record["download_allowed"] is True
    saved_license = json.loads((tmp_path / "assets" / "jamendo_123.license.json").read_text(encoding="utf-8"))
    assert saved_license["data"] == license_record
    api_params = calls[0][1]["params"]
    assert api_params["order"] == "popularity_week"
    assert api_params["content_id_free"] == "true"
    assert api_params["ccnc"] == "false"
    assert api_params["search"] == "high energy instrumental"
    assert api_params["vocalinstrumental"] == "instrumental"


def test_style_learner_aggregates_only_matching_platform_examples(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    examples = [{"video": f"example-{index}", "platform": "youtube_shorts", "duration": 8, "segments": [
            {"start": 0, "end": 2, "type": "hook", "cut_density": 0.8 if index == 0 else 0.6},
            {"start": 2, "end": 6, "type": "payoff", "cut_density": 0.4 if index == 0 else 0.6},
        ]} for index in range(5)] + [
        {"video": "example-2", "platform": "instagram_reels", "duration": 5, "segments": [
            {"start": 0, "end": 5, "type": "setup", "cut_density": 0.1},
        ]},
    ]
    examples_path.write_text("\n".join(json.dumps(item) for item in examples), encoding="utf-8")

    profile = EditingStyleLearner(examples_path).learn("youtube_shorts")

    assert profile["example_count"] == 5
    assert profile["learned_averages"]["average_shot_duration"] == 3.0
    assert profile["learned_averages"]["cut_density"] == 0.6


def test_style_learner_does_not_learn_from_one_reference(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    example = {"reference_id": "same-reference", "platform": "youtube_shorts", "features": {"cut_density": 1}}
    examples_path.write_text(json.dumps(example) + "\n" + json.dumps(example), encoding="utf-8")

    profile = EditingStyleLearner(examples_path, tmp_path / "missing-active.json").learn("youtube_shorts")

    assert profile["example_count"] == 1
    assert profile["training_status"] == "insufficient_references"
    assert profile["learned_averages"] == {}


def test_runtime_style_learner_requires_a_promoted_model(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    examples = [
        {"reference_id": f"ref-{index}", "platform": "youtube_shorts", "features": {"cut_density": 0.5}}
        for index in range(5)
    ]
    examples_path.write_text("\n".join(json.dumps(item) for item in examples), encoding="utf-8")

    profile = EditingStyleLearner(
        examples_path, tmp_path / "missing-active.json", promoted_only=True
    ).learn("youtube_shorts")

    assert profile["training_status"] == "no_promoted_model"
    assert profile["learned_averages"] == {}


def test_prompt_frames_cover_distinct_sources_before_repeating_sources():
    frames = [
        {"source_index": source_index, "time": timestamp}
        for source_index in range(17)
        for timestamp in (0.18, 0.62)
    ]

    selected = _select_prompt_frames(frames)

    assert len(selected) == 16
    assert len({frame["source_index"] for frame in selected}) == 16
    assert selected[0]["source_index"] == 0
    assert selected[-1]["source_index"] == 16
    assert all(frame["time"] == 0.62 for frame in selected)


def test_youtube_research_uses_multiple_queries_and_limits_channel_concentration(tmp_path):
    query_calls = []

    class FakeRequest:
        def __init__(self, result):
            self.result = result

        def execute(self):
            return self.result

    class FakeSearch:
        def list(self, **kwargs):
            query_calls.append(kwargs["q"])
            assert kwargs["videoLicense"] == "creativeCommon"
            assert kwargs["order"] == "viewCount"
            items = {
                "Valorant clutch": [
                    {"id": {"videoId": "v1"}, "snippet": {"channelId": "creator-a", "channelTitle": "A", "title": "Clutch", "publishedAt": "2026-09-30T00:00:00Z"}},
                    {"id": {"videoId": "v2"}, "snippet": {"channelId": "creator-b", "channelTitle": "B", "title": "Ace", "publishedAt": "2026-09-29T00:00:00Z"}},
                ],
                "gaming meme": [
                    {"id": {"videoId": "v3"}, "snippet": {"channelId": "creator-a", "channelTitle": "A", "title": "Meme", "publishedAt": "2026-09-28T00:00:00Z"}},
                    {"id": {"videoId": "v4"}, "snippet": {"channelId": "creator-c", "channelTitle": "C", "title": "Funny", "publishedAt": "2026-09-27T00:00:00Z"}},
                ],
            }
            return FakeRequest({"items": items[kwargs["q"]]})

    class FakeVideos:
        def list(self, **kwargs):
            assert kwargs["part"] == "snippet,contentDetails,status,statistics"
            ids = kwargs["id"].split(",")
            return FakeRequest({"items": [
                {
                    "id": video_id,
                    "snippet": {"categoryId": "20"},
                    "contentDetails": {"duration": "PT27.4S"},
                    "status": {"license": "creativeCommon"},
                    "statistics": {"viewCount": "12345"},
                } for video_id in ids
            ]})

    class FakeClient:
        def search(self):
            return FakeSearch()

        def videos(self):
            return FakeVideos()

    references = YouTubeResearchAgent(
        client=FakeClient(), metadata_dir=tmp_path, max_per_channel=1
    ).discover(
        strategies=["Valorant clutch", "gaming meme"],
        order="viewCount",
        lookback_days=30,
    )

    assert query_calls == ["Valorant clutch", "gaming meme"]
    assert {item["video_id"] for item in references} == {"v1", "v2", "v4"}
    assert all(item["duration_seconds"] == 27.4 for item in references)
    assert all(item["category_id"] == "20" for item in references)
    assert all(item["license"] == "creativeCommon" for item in references)
    assert all(item["license_filter"] == "creativeCommon" for item in references)
    assert all(item["view_count"] == 12345 for item in references)
    assert all(item["discovery_mode"] == "trending" for item in references)
    assert all("description" in item for item in references)
    assert all(item["media_downloaded"] is False for item in references)
    assert len(list(tmp_path.glob("discovery_*.json"))) == 1


def test_youtube_research_periodic_loop_stops_cleanly():
    stop_event = threading.Event()
    agent = YouTubeResearchAgent(client=object())
    calls = []

    def discover(**kwargs):
        calls.append(kwargs)
        stop_event.set()

    agent.discover = discover
    agent.run_periodically(interval_seconds=1, stop_event=stop_event, strategies=["gaming shorts"])

    assert calls == [{"strategies": ["gaming shorts"]}]


def test_candidate_curator_filters_license_deduplicates_and_caps_channels():
    candidates = [
        {"video_id": "a1", "channel_id": "creator-a", "license": "creativeCommon", "title": "Valorant clutch", "view_count": 1000, "duration_seconds": 20, "discovery_queries": ["valorant clutch"]},
        {"video_id": "a1", "channel_id": "creator-a", "license": "creativeCommon", "title": "Valorant clutch", "view_count": 1000, "duration_seconds": 20, "discovery_queries": ["valorant clutch"]},
        {"video_id": "a2", "channel_id": "creator-a", "license": "creativeCommon", "title": "Valorant montage", "view_count": 900, "duration_seconds": 20, "discovery_queries": ["valorant montage"]},
        {"video_id": "b1", "channel_id": "creator-b", "license": "creativeCommon", "title": "FPS highlights", "view_count": 800, "duration_seconds": 30, "discovery_queries": ["fps highlights"]},
        {"video_id": "x1", "channel_id": "creator-x", "license": "youtube", "title": "Valorant clutch", "view_count": 50000, "duration_seconds": 15, "discovery_queries": ["valorant clutch"]},
    ]

    selected = curate_candidates(
        candidates,
        {"valorant clutch": "valorant", "valorant montage": "valorant", "fps highlights": "fps"},
        max_per_channel=1,
        target_size=10,
        max_category_share=0.5,
    )

    assert {item["video_id"] for item in selected} == {"a1", "b1"}
    assert all(item["media_acquired"] is False for item in selected)
    assert all(item["attribution_required"] is True for item in selected)


def test_candidate_collection_rotates_queries_and_resumes_metadata_pool(tmp_path):
    config_path = tmp_path / "queries.yaml"
    config_path.write_text(
        "dataset:\n"
        "  searches_per_run: 1\n"
        "  max_videos_per_query: 5\n"
        "  max_videos_per_channel: 10\n"
        "  target_dataset_size: 10\n"
        "  max_category_share: 0.5\n"
        "  license: creativeCommon\n"
        "  order: viewCount\n"
        "  lookback_days: 30\n"
        "query_groups:\n"
        "  valorant:\n"
        "    - query one\n"
        "  fps:\n"
        "    - query two\n",
        encoding="utf-8",
    )

    class FakeAgent:
        calls = []

        def discover(self, **kwargs):
            self.calls.append(kwargs)
            query = kwargs["strategies"][0]
            return [{
                "video_id": query,
                "channel_id": f"channel-{query}",
                "channel": query,
                "title": query,
                "description": query,
                "view_count": 100,
                "duration_seconds": 20,
                "license": "creativeCommon",
                "discovery_queries": [query],
                "media_downloaded": False,
            }]

    agent = FakeAgent()
    candidates_path = tmp_path / "candidates.json"
    state_path = tmp_path / "collection_state.json"
    first = collect_candidate_pool(
        config_path, candidates_path, state_path, agent=agent
    )
    second = collect_candidate_pool(
        config_path, candidates_path, state_path, agent=agent
    )

    assert first["queries_this_run"] == ["query one"]
    assert second["queries_this_run"] == ["query two"]
    assert second["candidate_count"] == 2
    assert {item["video_id"] for item in second["items"]} == {"query one", "query two"}
    assert json.loads(state_path.read_text(encoding="utf-8"))["query_offset"] == 0
    assert all(call["license_filter"] == "creativeCommon" for call in agent.calls)


def test_creator_split_is_70_15_15_without_channel_leakage():
    examples = [
        {"reference_id": f"ref-{index}", "creator_group": f"creator-{index}"}
        for index in range(100)
    ]

    partitions = split_by_creator(examples)

    assert {name: len(items) for name, items in partitions.items()} == {
        "train": 70, "validation": 15, "test": 15
    }
    creator_sets = [
        {item["creator_group"] for item in partitions[name]}
        for name in ("train", "validation", "test")
    ]
    assert not creator_sets[0] & creator_sets[1]
    assert not creator_sets[0] & creator_sets[2]
    assert not creator_sets[1] & creator_sets[2]


def test_reference_import_requires_rights_basis_and_records_derived_features(tmp_path, monkeypatch):
    source = tmp_path / "reference.mp4"
    source.write_bytes(b"owned reference media")
    pipeline = ReferenceTrainingPipeline(tmp_path / "training")
    monkeypatch.setattr("app.training.reference_pipeline.probe", lambda path: {
        "format": {"duration": "20.0"},
        "streams": [
            {"codec_type": "video", "width": 320, "height": 240, "avg_frame_rate": "30/1", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    })
    monkeypatch.setattr(ReferenceTrainingPipeline, "_scene_cut_times", staticmethod(lambda path: [2.0, 7.0, 12.0]))
    monkeypatch.setattr(ReferenceTrainingPipeline, "_audio_levels", staticmethod(lambda path: (-18.0, -6.0)))
    monkeypatch.setattr(ReferenceTrainingPipeline, "_silence_ratio", staticmethod(lambda path, duration: 0.1))

    with pytest.raises(ValueError, match="rights_basis"):
        pipeline.import_local_video(source, rights_basis="youtube_discovery")

    example = pipeline.import_local_video(
        source,
        rights_basis="user_owned",
        platform="youtube_shorts",
        style_tags=["competitive"],
        category="valorant",
        source_url="https://example.org/licensed/reference.mp4",
        license_url="https://example.org/license",
        annotations={"features": {"caption_density": 0.3}},
    )

    assert example["features"]["cut_density"] == 0.15
    assert example["features"]["average_shot_duration"] == 5.0
    assert example["features"]["audio_intensity"] == 0.7
    assert example["features"]["caption_density"] == 0.3
    assert example["features"]["fps"] == 30.0
    # shots are [2, 5, 5, 8] seconds → std ≈ 2.449, CV ≈ 0.4899
    assert example["features"]["shot_duration_std"] == pytest.approx(2.449, abs=0.01)
    assert example["features"]["pacing_irregularity"] == pytest.approx(0.4899, abs=0.001)
    assert example["features"]["silence_ratio"] == 0.1
    assert example["features"]["audio_peak_db"] == -6.0
    assert example["features"]["audio_dynamic_range_db"] == 12.0
    assert example["sha256"] == example["reference_id"]
    manifest = json.loads((pipeline.root / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["total_items"] == 1
    assert manifest["items"][0]["source"]["license_url"] == "https://example.org/license"
    assert manifest["items"][0]["source"]["attribution_required"] is True
    assert manifest["items"][0]["file"].startswith("raw/valorant/")
    assert pathlib.Path(example["source_path"]).is_file()
    report = dataset_quality_report(pipeline.root)
    assert report["total_videos"] == 1
    assert report["corrupt_file_count"] == 0
    assert report["average_duration_seconds"] == 20.0


def test_reference_training_versions_aggregate_and_feedback_only_offline(tmp_path):
    pipeline = ReferenceTrainingPipeline(tmp_path / "training")
    examples = []
    for index in range(20):
        style = "competitive" if index % 2 else "cinematic"
        intensity = 1.0 if style == "competitive" else 0.0
        examples.append({
            "reference_id": f"reference-{index}",
            "platform": "youtube_shorts",
            "style_tags": [style],
            "creator_group": f"creator-{index}",
            "features": {
                "duration": 45.0 if intensity else 15.0,
                "cut_density": 0.9 if intensity else 0.1,
            },
            "moments": [{
                "moment": {"visual_intensity": intensity},
                "observed_edit": {"zoom": intensity},
            }],
        })
    pipeline.examples_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.examples_path.write_text("\n".join(json.dumps(item) for item in examples), encoding="utf-8")
    pipeline.record_feedback("job-7", 1, ["hook_good", "captions_good"])

    result = pipeline.train_candidate(platforms=("youtube_shorts",))
    active = json.loads((pipeline.patterns / "active.json").read_text(encoding="utf-8"))
    profile = EditingStyleLearner(
        pipeline.examples_path, pipeline.patterns / "active.json"
    ).learn("youtube_shorts")

    assert result["status"] == "promoted"
    assert result["validation_mae"] < result["baseline_mae"]
    assert active["dataset"]["validation_count"] == 3
    assert active["dataset"]["test_count"] == 3
    split = json.loads((pipeline.datasets / f"splits_{result['version']}.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_reference_ids"])
    validation_ids = set(split["validation_reference_ids"])
    test_ids = set(split["test_reference_ids"])
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids
    assert active["feedback_signals"]["hook_good"]["positive"] == 1
    assert profile["example_count"] == 14
    assert profile["model_version"] == result["version"]
    assert any(item["moment_feature"] == "visual_intensity" for item in profile["relationships"])


def test_dashboard_feedback_accepts_only_completed_edits(fresh_db, tmp_path, monkeypatch):
    session = SessionLocal()
    output_path = tmp_path / "completed.mp4"
    job = Job(status="completed", input_path="input", output_path=str(output_path), extra_metadata="{}")
    session.add(job)
    session.commit()
    job_id = job.id
    session.close()
    recorded = {}

    class FakeTrainingPipeline:
        def record_feedback(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(dashboard_app_module, "ReferenceTrainingPipeline", FakeTrainingPipeline)
    response = dashboard_app_module.app.test_client().post("/api/feedback", json={
        "edit_id": job_id,
        "rating": -1,
        "tags": ["too_fast", "unrecognized"],
    })

    assert response.status_code == 200
    assert recorded["rating"] == -1
    assert recorded["tags"] == ["too_fast"]
    assert recorded["context"]["job_id"] == job_id


def test_ai_creative_mode_requires_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(CreativeAIConfigurationError, match="OPENAI_API_KEY"):
        ShortFormEditingModel()


def test_ai_creative_mode_gemini_requires_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(CreativeAIConfigurationError, match="GEMINI_API_KEY"):
        ShortFormEditingModel()


def test_sfx_library_provides_original_described_assets(tmp_path):
    assets = SfxLibrary(tmp_path).list_assets()

    assert {item["filename"] for item in assets} == {
        "original_sweep.wav", "original_impact.wav", "original_glitch.wav"
    }
    assert all(item["origin"] == "generated locally by Gaming Video Agent" for item in assets)
    assert all(item["duration"] > 0 for item in assets)


def test_ai_selected_music_can_nudge_transition_timing_to_beats():
    plan = EditPlan.from_dict({
        "platform": "youtube_shorts",
        "strategy": "short payoff",
        "target_duration": 7,
        "shots": [
            {"source_index": 0, "start": 0, "end": 3, "role": "setup", "transition": "cut"},
            {"source_index": 0, "start": 4, "end": 7, "role": "payoff", "transition": "fade"},
        ],
    }, source_count=1)
    track = {
        "audio_features": {"bpm_estimate": 120, "bpm_confidence": 0.9},
        "selection": {"section_start": 0, "beat_sync_strength": 1.0},
    }

    ShortFormCreativeEditor._align_plan_to_music(plan, track, 0.2, [{"duration": 10}])

    assert plan.shots[0].end == pytest.approx(3.1)
    assert plan.music_mix["beat_alignment"]["boundaries"] == [1]


def test_creative_agent_plans_selects_music_reviews_and_revises(tmp_path, monkeypatch):
    source = tmp_path / "gameplay.mp4"
    output = tmp_path / "final.mp4"
    sfx_path = tmp_path / "soft_hit.wav"
    sfx_path.write_bytes(b"sfx")
    raw_plan = {
        "platform": "youtube_shorts",
        "strategy": "contextual clutch payoff",
        "rationale": "The final segment contains the strongest payoff.",
        "target_duration": 8,
        "alternatives": [{"strategy": "rapid montage", "rationale": "More action density."}],
        "shots": [
            {"source_index": 0, "start": 0, "end": 3, "role": "hook", "transition": "cut", "caption": "Wait for it"},
            {"source_index": 0, "start": 5, "end": 9, "role": "payoff", "transition": "fade", "visual_emphasis": ["punch_zoom"]},
        ],
        "music_requirements": {"search": "tense electronic instrumental", "speed": ["high"], "instrumental": True},
        "music_mix": {},
        "sound_design": [{"time": 5.8, "asset_filename": "soft_hit.wav", "level": 0.2}],
        "ending": "Cut after the payoff.",
    }

    class FakeModel:
        def create_plan(self, **kwargs):
            assert kwargs["platform"] == "youtube_shorts"
            assert kwargs["available_sfx"] == [{"filename": "soft_hit.wav", "format": "wav", "duration": 0.5}]
            return raw_plan

        def rank_music(self, media_context, plan, candidates):
            assert candidates[0]["audio_features"]["bpm_estimate"] == 128
            return {"track_id": "music-1", "section_start": 4, "volume": 0.24,
                    "duck_under_original_audio": True, "rationale": "Fits the footage energy."}

        def review_render(self, plan, review_context, frames):
            return {"needs_revision": True, "critique": "Caption arrives too early.", "revision_request": "Move the hook caption later."}

        def revise_plan(self, **kwargs):
            revised = json.loads(json.dumps(raw_plan))
            revised["shots"][0]["caption"] = "Watch the final seconds"
            return revised

    class FakeMusicProvider:
        client_id = "test-client"

        def search(self, requirements, limit):
            assert "tense" in requirements.search
            return [{"id": "music-1", "title": "Night Drive", "artist": "Artist", "duration": 120,
                     "download_allowed": True, "download_url": "https://audio.example/track.mp3",
                     "audio_url": "https://audio.example/preview.mp3", "audio_features": {}}]

        def analyze_audio(self, audio_url):
            return {"bpm_estimate": 128, "energy_variability": 0.2}

        def download_track(self, track, destination):
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "chosen.mp3"
            path.write_bytes(b"audio")
            return path, {"provider": "jamendo", "track_id": track["id"], "license_url": "https://license.example/by"}

    class FakeStyleLearner:
        def learn(self, platform):
            return {"platform": platform, "example_count": 0, "learned_averages": {}}

    class FakeSfxLibrary:
        def list_assets(self):
            return [{"filename": "soft_hit.wav", "format": "wav", "duration": 0.5, "path": str(sfx_path)}]

    render_calls = []

    def fake_analyze(paths, max_sources=12):
        return {"sources": [{"source_index": 0, "filename": pathlib.Path(paths[0]).name, "duration": 10}]}, []

    def fake_render(inputs, destination, options, progress_callback):
        render_calls.append((list(inputs), options))
        pathlib.Path(destination).write_bytes(b"render")
        return pathlib.Path(destination)

    monkeypatch.setattr(creative_agent_module, "analyze_sources", fake_analyze)
    agent = ShortFormCreativeEditor(
        model=FakeModel(), music_provider=FakeMusicProvider(),
        style_learner=FakeStyleLearner(), sfx_library=FakeSfxLibrary(), renderer=fake_render,
    )
    artifact = agent.create_edit(
        [source], output,
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16", bgm_track="auto"),
    )

    assert len(render_calls) == 2
    assert render_calls[0][1].planned_segments == [(0.0, 3.0), (5.0, 4.0)]
    assert render_calls[0][1].transition_sequence == ["fade"]
    assert render_calls[0][1].bgm_start_seconds == 4
    assert render_calls[0][1].duck_bgm_to_original_audio is True
    assert render_calls[0][1].sfx_clips == [(sfx_path, 5.8, 0.2)]
    assert artifact["version"] == 2
    assert artifact["music_license"]["track_id"] == "music-1"
    assert output.with_suffix(".edit-plan.json").is_file()
