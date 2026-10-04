"""Tests for automatic licensed-reference collection and YouTube metadata priors."""
import json
import pathlib
import subprocess

import pytest
import requests

from app.research.licensed_media_provider import LicensedMediaError, PexelsVideoProvider
from app.research.metadata_priors import youtube_duration_priors
from app.training.reference_pipeline import ReferenceTrainingPipeline


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload=None, chunks=None, headers=None, status=200):
        self._payload = payload
        self._chunks = chunks or []
        self.headers = headers or {}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("boom")

    def iter_content(self, chunk_size=1024):
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _search_payload():
    def video(vid, user_id, user_name):
        return {
            "id": vid,
            "url": f"https://www.pexels.com/video/{vid}/",
            "duration": 8,
            "user": {"id": user_id, "name": user_name},
            "video_files": [{"link": f"https://cdn.pexels.test/{vid}.mp4", "height": 720, "width": 1280}],
        }
    return {"videos": [video(1, 11, "Alice"), video(2, 11, "Alice"), video(3, 22, "Bob")]}


class FakeSession:
    def __init__(self, file_bytes_by_id):
        self.file_bytes_by_id = file_bytes_by_id
        self.search_calls = []

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        if "videos/search" in url:
            self.search_calls.append(params)
            return FakeResponse(payload=_search_payload())
        video_id = pathlib.Path(url.split("?")[0]).stem  # e.g. "1" from .../1.mp4
        payload = self.file_bytes_by_id.get(video_id, b"")
        return FakeResponse(
            chunks=[payload],
            headers={"content-length": str(len(payload))},
        )


def _make_tiny_mp4(path: pathlib.Path, color: str, seconds: int = 2) -> bytes:
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=25:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path.read_bytes()


@pytest.fixture(scope="module")
def mp4_bytes_by_id(tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    # Distinct bytes per video id so sha256-based dedup keeps them separate.
    return {
        "1": _make_tiny_mp4(media / "v1.mp4", "red", 2),
        "2": _make_tiny_mp4(media / "v2.mp4", "red", 2),
        "3": _make_tiny_mp4(media / "v3.mp4", "blue", 3),
    }


# ── Provider ──────────────────────────────────────────────────────────────────

def test_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(LicensedMediaError, match="PEXELS_API_KEY"):
        PexelsVideoProvider()


def test_collect_imports_with_diversity_cap_and_license_records(tmp_path, mp4_bytes_by_id, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    session = FakeSession(mp4_bytes_by_id)
    pipeline = ReferenceTrainingPipeline(tmp_path / "training")
    provider = PexelsVideoProvider(
        inbox=tmp_path / "inbox",
        license_dir=tmp_path / "licenses",
        session=session,
        max_per_uploader=1,
        config_path=tmp_path / "absent-config.yaml",
    )
    result = provider.collect(
        limit=8, queries=["esports gaming highlight"],
        category="gaming", pipeline=pipeline,
    )

    # Alice has 2 videos but cap is 1 → 2 imported (Alice+Bob), 1 skipped
    assert len(result["imported"]) == 2
    assert {item["uploader"] for item in result["imported"]} == {"Alice", "Bob"}
    assert result["skipped"] and result["skipped"][0]["reason"] == "uploader_diversity_cap"

    examples = pipeline._read_examples()
    assert len(examples) == 2
    assert all(example["rights_basis"] == "licensed" for example in examples)
    assert {example["creator_group"] for example in examples} == {"pexels_11", "pexels_22"}
    assert all(example["license_name"] == "Pexels License" for example in examples)
    assert sorted(example["features"]["duration"] for example in examples) == [2.0, 3.0]

    license_records = list((tmp_path / "licenses").glob("*.license.json"))
    assert len(license_records) == 2
    record = json.loads(license_records[0].read_text(encoding="utf-8"))
    assert record["media_from_youtube_or_instagram"] is False
    assert record["acquisition"] == "official_provider_api_download"
    assert session.search_calls and session.search_calls[0]["query"] == "esports gaming highlight"


def test_collect_skips_oversized_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    session = FakeSession({})
    session.get = lambda url, **kw: (
        FakeResponse(payload=_search_payload()) if "videos/search" in url
        else FakeResponse(chunks=[b"x" * 10], headers={"content-length": str(999 * 1024 * 1024)})
    )
    provider = PexelsVideoProvider(
        inbox=tmp_path / "inbox", license_dir=tmp_path / "licenses",
        session=session, max_file_mb=1, config_path=tmp_path / "absent.yaml",
    )
    result = provider.collect(limit=4, queries=["q"], pipeline=ReferenceTrainingPipeline(tmp_path / "training"))
    assert result["imported"] == []
    assert all(item["reason"] == "download_failed_or_too_large" for item in result["skipped"])


# ── YouTube metadata priors ───────────────────────────────────────────────────

def test_duration_priors_summarize_pool(tmp_path):
    pool = {"items": [
        {"category": "valorant", "duration_seconds": 20},
        {"category": "valorant", "duration_seconds": 30},
        {"category": "valorant", "duration_seconds": 40},
        {"category": "valorant", "duration_seconds": 50},
        {"category": "funny", "duration_seconds": 10},
        {"category": "funny", "duration_seconds": 999},   # > short-form max → excluded
        {"category": "broken", "duration_seconds": None},  # invalid → excluded
    ]}
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    priors = youtube_duration_priors(path)
    assert priors["source"] == "youtube_data_api_metadata_only"
    assert priors["media_accessed"] is False
    valorant = priors["by_category"]["valorant"]
    assert valorant == {"count": 4, "median_seconds": 35.0, "p25_seconds": 27.5,
                        "p75_seconds": 42.5, "min_seconds": 20.0, "max_seconds": 50.0}
    assert "funny" not in priors["by_category"]  # only 1 valid value < 3 minimum
    assert priors["overall"]["count"] == 5


def test_duration_priors_honest_when_missing(tmp_path):
    assert youtube_duration_priors(tmp_path / "absent.json") == {}


# ── CLI + planner integration ─────────────────────────────────────────────────

def test_cli_collect_references_reports_and_requires_key(monkeypatch):
    from click.testing import CliRunner
    import main as main_module
    import app.research.licensed_media_provider as provider_module

    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    result = CliRunner().invoke(main_module.cli, ["collect-references"])
    assert result.exit_code != 0
    assert "PEXELS_API_KEY" in result.output

    class FakeProvider:
        def __init__(self, *a, **k):
            pass

        def collect(self, **kwargs):
            return {"imported": [{"reference_id": "abc123", "uploader": "Alice"}], "skipped": []}

    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setattr(provider_module, "PexelsVideoProvider", FakeProvider)
    result = CliRunner().invoke(main_module.cli, ["collect-references", "--limit", "4"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output[result.output.index("{"):])
    assert report["downloaded_and_imported"] == 1
    assert report["youtube_or_instagram_media_downloaded"] is False


def test_hosted_planner_receives_metadata_priors(tmp_path, monkeypatch):
    import app.agent.short_form_editor as editor_module
    from app.agent.short_form_editor import ShortFormCreativeEditor
    from app.editing.editor import EditOptions

    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "false")
    source = tmp_path / "gameplay.mp4"
    source.write_bytes(b"source")
    captured = {}

    class Model:
        def create_plan(self, **kwargs):
            captured.update(kwargs)
            return {
                "platform": "youtube_shorts", "strategy": "s", "target_duration": 2,
                "shots": [{"source_index": 0, "start": 0, "end": 2, "role": "action", "transition": "cut"}],
                "music_requirements": {}, "sound_design": [],
            }

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": "gameplay.mp4", "duration": 2,
            "width": 1920, "height": 1080, "has_audio": True, "audio_mean_db": -25,
        }]}, []

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
    artifact = editor.create_edit(
        [source], tmp_path / "out.mp4",
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16",
                    target_duration=2, bgm_track=None, game="none"),
    )
    assert "metadata_priors" in captured
    assert isinstance(captured["metadata_priors"], dict)
    assert artifact["planning_mode"] == "hosted_creative_model"
