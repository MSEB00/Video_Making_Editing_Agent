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
import re
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
    "sports highlights montage",
    "action montage fast cuts",
    "skateboard montage edit",
    "dance vertical edit",
    "car drift montage",
    "travel cinematic transitions",
)

# Only true short-form references teach short-form grammar; long VODs would
# pollute the dataset with cut_density ~= 0 "style".
SHORT_FORM_MAX_SECONDS = 180.0


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
            "rights_basis": "licensed",
            "attribution_required": False,
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
        queries = [str(q).strip() for q in (queries or self.default_queries()) if str(q).strip()]
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
            uploader_key = f"{candidate['provider']}_{_safe_name(candidate['uploader_id'])}"
            if uploader_counts.get(uploader_key, 0) >= self.max_per_uploader:
                skipped.append({"video_id": candidate["video_id"], "reason": "uploader_diversity_cap"})
                continue
            if not candidate["download_url"]:
                skipped.append({"video_id": candidate["video_id"], "reason": "no_download_url"})
                continue
            duration = candidate.get("duration_seconds")
            if isinstance(duration, (int, float)) and duration > SHORT_FORM_MAX_SECONDS:
                skipped.append({"video_id": candidate["video_id"], "reason": "duration_not_short_form"})
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
                    rights_basis=candidate.get("rights_basis", "licensed"),
                    platform=platform,
                    style_tags=[str(tag)[:40] for tag in tags][:6],
                    creator_group=uploader_key,
                    category=category,
                    source_url=candidate["page_url"],
                    license_name=candidate["license_name"],
                    license_url=candidate["license_url"],
                    attribution_required=bool(candidate.get("attribution_required", False)),
                    source_metadata={
                        "platform": candidate["provider"],
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

    def default_queries(self) -> tuple[str, ...] | list[str]:
        return self.settings.get("default_queries") or DEFAULT_QUERIES

    def _download(self, candidate: dict[str, Any]) -> Optional[pathlib.Path]:
        destination = self.inbox / f"{candidate['provider']}_{_safe_name(candidate['video_id'])}.mp4"
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
        target = self.license_dir / f"{candidate['provider']}_{_safe_name(candidate['video_id'])}.license.json"
        target.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:80]


def _rights_from_license_url(url: Any) -> Optional[tuple[str, str]]:
    """Classify a license URL into (rights_basis, license_name); None = reject."""
    text = str(url or "").strip().lower()
    if not text:
        return None
    if "publicdomain" in text or "/zero/" in text or "pd.mark" in text:
        return "public_domain", "Public Domain / CC0"
    if "creativecommons.org" in text:
        return "licensed", "Creative Commons (see license_url)"
    return None


ARCHIVE_DEFAULT_QUERIES = (
    "valorant gameplay",
    "esports montage",
    "gaming montage",
    "counter strike highlights",
    "overwatch play montage",
)

# Rotating bank for 'train-mode': each run picks the next slice so repeated
# runs keep discovering NEW items instead of re-hitting dedup on the same
# top search results.
ARCHIVE_QUERY_BANK = (
    "valorant gameplay",
    "esports montage",
    "gaming montage",
    "counter strike highlights",
    "overwatch play montage",
    "fortnite montage",
    "apex legends montage",
    "csgo highlights",
    "call of duty montage",
    "mlg montage",
    "pubg highlights",
    "rocket league montage",
    "gaming funny moments",
    "speedrun highlights edit",
    "halo gameplay montage",
)


class InternetArchiveProvider(PexelsVideoProvider):
    """archive.org provider: official open APIs, CC/PD-licensed items only.

    No API key required. Only items whose search metadata carries a
    Creative Commons or Public Domain license URL are accepted, and only
    small short-form .mp4 derivative files pass (the size cap doubles as a
    long-VOD guard). Unknown licenses are rejected, never assumed.
    """

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        kwargs.pop("api_key", None)
        self.api_key = ""
        self.inbox = pathlib.Path(kwargs.pop("inbox", None) or DEFAULT_INBOX)
        self.license_dir = pathlib.Path(kwargs.pop("license_dir", None) or DEFAULT_LICENSE_DIR)
        self.session = kwargs.pop("session", None) or requests.Session()
        self.max_file_mb = max(1, int(kwargs.pop("max_file_mb", 60)))
        self.max_per_uploader = max(1, int(kwargs.pop("max_per_uploader", 2)))
        self.settings = self._load_settings(pathlib.Path(kwargs.pop("config_path", None) or DEFAULT_CONFIG))

    def default_queries(self) -> tuple[str, ...] | list[str]:
        return self.settings.get("archive_queries") or ARCHIVE_DEFAULT_QUERIES

    def search(self, query: str, per_page: int = 8, orientation: str = "portrait") -> list[dict[str, Any]]:
        response = self.session.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": (
                    f'({query[:100]}) AND mediatype:(movies) AND '
                    "(licenseurl:(*creativecommons*) OR licenseurl:(*publicdomain*))"
                ),
                "fl[]": ["identifier", "title", "creator", "licenseurl"],
                "rows": max(1, min(50, int(per_page) * 2)),
                "page": 1,
                "output": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json() or {}
        docs = ((payload.get("response") or {}).get("docs")) or []
        candidates = []
        for doc in docs:
            if not isinstance(doc, dict) or not doc.get("identifier"):
                continue
            rights = _rights_from_license_url(doc.get("licenseurl"))
            if rights is None:
                continue  # unknown license → reject, never assume
            resolved = self._resolve_short_form_mp4(str(doc["identifier"]))
            if resolved is None:
                continue
            creator = str(doc.get("creator") or doc["identifier"])[:60]
            candidates.append({
                "provider": "internet_archive",
                "video_id": str(doc["identifier"]),
                "page_url": f"https://archive.org/details/{doc['identifier']}",
                "download_url": resolved["download_url"],
                "duration_seconds": resolved["length_seconds"],
                "uploader": creator,
                "uploader_id": creator,
                "query": query,
                "license_name": rights[1],
                "license_url": str(doc.get("licenseurl")),
                "rights_basis": rights[0],
                "attribution_required": rights[0] == "licensed",
            })
            if len(candidates) >= max(1, int(per_page)):
                break
        return candidates

    def _resolve_short_form_mp4(self, identifier: str) -> Optional[dict[str, Any]]:
        """Pick the smallest short-form .mp4 file from an item's metadata."""
        try:
            response = self.session.get(f"https://archive.org/metadata/{identifier}", timeout=30)
            response.raise_for_status()
            meta = response.json() or {}
        except (requests.RequestException, ValueError):
            return None
        best: Optional[tuple[int, str, Optional[float]]] = None
        max_bytes = self.max_file_mb * 1024 * 1024
        for entry in meta.get("files") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "")
            if not name.lower().endswith(".mp4"):
                continue
            try:
                size = int(float(entry.get("size") or 0))
            except (TypeError, ValueError):
                continue
            if size <= 0 or size > max_bytes:
                continue  # also rejects giant VOD derivatives
            try:
                length = float(entry.get("length"))
            except (TypeError, ValueError):
                length = None
            if length is not None and length > SHORT_FORM_MAX_SECONDS:
                continue
            if best is None or size < best[0]:
                best = (size, name, length)
        if best is None:
            return None
        return {
            "download_url": f"https://archive.org/download/{identifier}/{best[1]}",
            "length_seconds": best[2],
        }
