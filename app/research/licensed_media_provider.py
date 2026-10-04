"""Acquire legally downloadable, licensed reference media for offline training.

Why this exists: YouTube and Instagram expose NO official frame/audio access —
only metadata. Unofficial extraction violates their terms of service and this
project's policy. This module instead acquires stock media whose license
explicitly permits download and reuse (currently: Pexels License), so the
reference training pipeline can learn editing features automatically and
legally — no human annotation required.

Learned features from stock references generalize short-form pacing/rhythm
priors; they are combined at planning time with YouTube metadata priors and
(with approval-gated manual observations) YouTube research knowledge.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Any, Optional

import requests

from app.utilities.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INBOX = PROJECT_ROOT / "training" / "licensed_inbox"
DEFAULT_LICENSE_DIR = PROJECT_ROOT / "training" / "licensed_licenses"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "training_queries.yaml"

PEXELS_LICENSE_NAME = "Pexels License"
PEXELS_LICENSE_URL = "https://www.pexels.com/license/"

DEFAULT_QUERIES = (
    "esports gaming highlight",
    "sports highlights montage",
    "action montage fast cuts",
    "travel cinematic montage",
    "dance vertical montage",
)


class LicensedMediaError(RuntimeError):
    pass


class PexelsVideoProvider:
    """Search and download Pexels videos (their license permits download/reuse).

    All network I/O goes through an injectable session for testability.
    Downloads are capped in size, deduplicated by video id, and every file
    gets a persisted license record before it enters the training dataset.
    """

    def __init__(
        self,
        api_key: str | None = None,
        inbox: pathlib.Path | None = None,
        license_dir: pathlib.Path | None = None,
        session: Any = None,
        max_file_mb: int = 60,
        max_per_uploader: int = 2,
        config_path: pathlib.Path | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("PEXELS_API_KEY") or "").strip()
        if not self.api_key:
            raise LicensedMediaError(
                "PEXELS_API_KEY is not set. Get a free key at https://www.pexels.com/api/ "
                "and add PEXELS_API_KEY=... to your .env file."
            )
        self.inbox = pathlib.Path(inbox or DEFAULT_INBOX)
        self.license_dir = pathlib.Path(license_dir or DEFAULT_LICENSE_DIR)
        self.session = session or requests.Session()
        self.max_file_mb = max(1, int(max_file_mb))
        self.max_per_uploader = max(1, int(max_per_uploader))
        self.settings = self._load_settings(pathlib.Path(config_path or DEFAULT_CONFIG))

    @staticmethod
    def _load_settings(path: pathlib.Path) -> dict[str, Any]:
        try:
            import yaml

            with path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            section = data.get("licensed_media") if isinstance(data, dict) else None
            return section if isinstance(section, dict) else {}
        except (OSError, ValueError):
            return {}

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, per_page: int = 8, orientation: str = "portrait") -> list[dict[str, Any]]:
        response = self.session.get(
            "https://api.pexels.com/videos/search",
            params={
                "query": query[:120],
                "per_page": max(1, min(80, int(per_page))),
                "orientation": orientation,
            },
            headers={"Authorization": self.api_key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        videos = payload.get("videos") if isinstance(payload, dict) else None
        return [self._normalize(item, query) for item in (videos or []) if isinstance(item, dict)]

    @staticmethod
    def _normalize(raw: dict[str, Any], query: str) -> dict[str, Any]:
        files = [f for f in (raw.get("video_files") or []) if isinstance(f, dict) and f.get("link")]
        best: Optional[dict[str, Any]] = None
        for candidate in files:
            height = int(candidate.get("height") or 0)
            if best is None or abs(height - 720) < abs(int(best.get("height") or 0) - 720):
                best = candidate
        user = raw.get("user") or {}
        return {
            "provider": "pexels",
            "video_id": str(raw.get("id") or ""),
            "page_url": str(raw.get("url") or ""),
            "download_url": str((best or {}).get("link") or ""),
            "width": (best or {}).get("width"),
            "height": (best or {}).get("height"),
            "duration_seconds": raw.get("duration"),
            "uploader": str(user.get("name") or "unknown"),
            "uploader_id": str(user.get("id") or user.get("name") or "unknown"),
            "query": query,
            "license_name": PEXELS_LICENSE_NAME,
            "license_url": PEXELS_LICENSE_URL,
        }

    # ── Collect + import ────────────────────────────────────────────────────

    def collect(
        self,
        limit: int = 8,
        queries: list[str] | None = None,
        orientation: str = "portrait",
        category: str = "gaming",
        platform: str = "all",
        style_tags: list[str] | None = None,
        pipeline: Any = None,
    ) -> dict[str, Any]:
        """Search, download, license-record and import up to *limit* references.

        Returns {"imported": [...], "skipped": [...]} with honest reasons for
        every skip. Never fabricates: a failed download is a skip, not data.
        """
        from app.training.reference_pipeline import ReferenceTrainingPipeline

        pipeline = pipeline or ReferenceTrainingPipeline()
        queries = [str(q).strip() for q in (queries or self.settings.get("default_queries") or DEFAULT_QUERIES) if str(q).strip()]
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.license_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for query in queries:
            try:
                results = self.search(query, per_page=max(4, (limit * 2) // max(1, len(queries)) + 2), orientation=orientation)
            except requests.RequestException as exc:
                log.warning("Licensed media search failed for %r: %s", query, exc)
                continue
            for item in results:
                if item["video_id"] and item["video_id"] not in seen_ids:
                    seen_ids.add(item["video_id"])
                    candidates.append(item)

        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        uploader_counts: dict[str, int] = {}
        for candidate in candidates:
            if len(imported) >= max(1, int(limit)):
                break
            uploader_key = f"pexels_{candidate['uploader_id']}"
            if uploader_counts.get(uploader_key, 0) >= self.max_per_uploader:
                skipped.append({"video_id": candidate["video_id"], "reason": "uploader_diversity_cap"})
                continue
            if not candidate["download_url"]:
                skipped.append({"video_id": candidate["video_id"], "reason": "no_download_url"})
                continue
            path = self._download(candidate)
            if path is None:
                skipped.append({"video_id": candidate["video_id"], "reason": "download_failed_or_too_large"})
                continue
            uploader_counts[uploader_key] = uploader_counts.get(uploader_key, 0) + 1
            self._write_license_record(candidate, path)
            tags = style_tags or ["stock_reference", *candidate["query"].split()[:2]]
            try:
                example = pipeline.import_local_video(
                    path,
                    rights_basis="licensed",
                    platform=platform,
                    style_tags=[str(tag)[:40] for tag in tags][:6],
                    creator_group=uploader_key,
                    category=category,
                    source_url=candidate["page_url"],
                    license_name=candidate["license_name"],
                    license_url=candidate["license_url"],
                    attribution_required=False,
                    source_metadata={
                        "platform": "pexels",
                        "video_id": candidate["video_id"],
                        "title": f"pexels_{candidate['video_id']}",
                        "channel": candidate["uploader"],
                        "channel_id": candidate["uploader_id"],
                        "query": candidate["query"],
                    },
                )
            except (ValueError, OSError, RuntimeError) as exc:
                skipped.append({"video_id": candidate["video_id"], "reason": f"import_failed: {type(exc).__name__}"})
                continue
            features = example.get("features", {})
            imported.append({
                "reference_id": str(example.get("reference_id", ""))[:16],
                "uploader": candidate["uploader"],
                "query": candidate["query"],
                "duration": features.get("duration"),
                "cut_density": features.get("cut_density"),
                "file": path.name,
            })
            log.info(
                "[DATASET] Imported licensed reference %s (uploader=%s, cut_density=%s)",
                path.name, candidate["uploader"], features.get("cut_density"),
            )
        return {"imported": imported, "skipped": skipped}

    # ── Download + license records ─────────────────────────────────────────

    def _download(self, candidate: dict[str, Any]) -> Optional[pathlib.Path]:
        destination = self.inbox / f"pexels_{candidate['video_id']}.mp4"
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        max_bytes = self.max_file_mb * 1024 * 1024
        temporary = destination.with_suffix(".part")
        try:
            with self.session.get(candidate["download_url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                declared = int(response.headers.get("content-length") or 0)
                if declared > max_bytes:
                    return None
                written = 0
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            stream.close()
                            temporary.unlink(missing_ok=True)
                            return None
                        stream.write(chunk)
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError):
            temporary.unlink(missing_ok=True)
            return None

    def _write_license_record(self, candidate: dict[str, Any], path: pathlib.Path) -> None:
        record = {
            **candidate,
            "local_file": str(path),
            "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "acquisition": "official_provider_api_download",
            "media_from_youtube_or_instagram": False,
        }
        target = self.license_dir / f"pexels_{candidate['video_id']}.license.json"
        target.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")
