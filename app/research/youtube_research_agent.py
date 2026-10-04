"""Discover short-form gaming references using the official YouTube Data API."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import threading
from typing import Any


SEARCH_STRATEGIES = (
    "gaming shorts",
    "FPS gaming Shorts",
    "competitive gaming Shorts",
    "esports highlights shorts",
    "gaming montage",
    "gaming reaction Shorts",
    "gaming story clips",
    "cinematic gaming Shorts",
    "funny gaming Shorts",
    "Valorant clutch Shorts",
)


class YouTubeResearchAgent:
    """Collect permitted public metadata only; never download or extract media."""

    def __init__(
        self,
        api_key: str | None = None,
        client: Any = None,
        metadata_dir: pathlib.Path | None = None,
        max_per_channel: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_DATA_API_KEY") or os.getenv("YOUTUBE_API_KEY")
        self.client = client
        project_root = pathlib.Path(__file__).resolve().parents[2]
        self.metadata_dir = pathlib.Path(metadata_dir or project_root / "training" / "youtube" / "metadata")
        self.max_per_channel = max(1, max_per_channel)

    def discover(
        self,
        results_per_query: int = 25,
        strategies: tuple[str, ...] | list[str] = SEARCH_STRATEGIES,
        published_after: str | None = None,
        license_filter: str = "creativeCommon",
        order: str = "date",
        lookback_days: int = 90,
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        if license_filter not in {"any", "youtube", "creativeCommon"}:
            raise ValueError("license_filter must be any, youtube, or creativeCommon.")
        if order not in {"date", "viewCount"}:
            raise ValueError("order must be date or viewCount.")
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least one.")
        per_query = max(1, min(50, int(results_per_query)))
        after = published_after or (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)
        ).isoformat()
        records: dict[str, dict[str, Any]] = {}
        channel_counts: dict[str, int] = {}
        discovered_at = dt.datetime.now(dt.timezone.utc).isoformat()
        for query in strategies:
            response = self._execute(client.search().list(
                part="snippet",
                type="video",
                q=query,
                maxResults=per_query,
                order=order,
                publishedAfter=after,
                videoDuration="short",
                **({} if license_filter == "any" else {"videoLicense": license_filter}),
            ))
            video_ids = [
                (item.get("id") or {}).get("videoId")
                for item in response.get("items", [])
                if (item.get("id") or {}).get("videoId")
            ]
            video_details = self._video_details(client, video_ids)
            for item in response.get("items", []):
                video_id = (item.get("id") or {}).get("videoId")
                snippet = item.get("snippet") or {}
                channel_id = snippet.get("channelId")
                if not video_id or not channel_id:
                    continue
                if video_id in records:
                    if query not in records[video_id]["discovery_queries"]:
                        records[video_id]["discovery_queries"].append(query)
                    continue
                if channel_counts.get(channel_id, 0) >= self.max_per_channel:
                    continue
                records[video_id] = {
                    "platform": "youtube",
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "channel_id": channel_id,
                    "description": str(snippet.get("description", ""))[:5000],
                    "published_at": snippet.get("publishedAt"),
                    "duration_seconds": video_details.get(video_id, {}).get("duration_seconds"),
                    "category_id": video_details.get(video_id, {}).get("category_id") or snippet.get("categoryId"),
                    "license": video_details.get(video_id, {}).get("license"),
                    "license_filter": license_filter,
                    "view_count": video_details.get(video_id, {}).get("view_count"),
                    "like_count": video_details.get(video_id, {}).get("like_count"),
                    "comment_count": video_details.get(video_id, {}).get("comment_count"),
                    "thumbnail_url": self._thumbnail_url(snippet),
                    "discovery_queries": [query],
                    "discovery_mode": "trending" if order == "viewCount" else "recent",
                    "discovered_at": discovered_at,
                    "media_downloaded": False,
                }
                channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1
        result = list(records.values())
        self._save_metadata(result, list(strategies), after, order)
        return result

    def run_periodically(
        self,
        interval_seconds: int = 24 * 60 * 60,
        stop_event: threading.Event | None = None,
        **discover_options: Any,
    ) -> None:
        """Repeat discovery until stopped; intended for an explicit background worker."""
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be at least one second.")
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                self.discover(**discover_options)
            except Exception:
                logging.getLogger(__name__).exception("YouTube metadata discovery failed")
            if stop.wait(interval_seconds):
                return

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError("Set YOUTUBE_DATA_API_KEY to enable YouTube metadata discovery.")
        from googleapiclient.discovery import build

        self.client = build("youtube", "v3", developerKey=self.api_key, cache_discovery=False)
        return self.client

    def _save_metadata(self, records: list[dict[str, Any]], queries: list[str], after: str, order: str) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(days=30)
        for cached_path in self.metadata_dir.glob("discovery_*.json"):
            try:
                modified = dt.datetime.fromtimestamp(cached_path.stat().st_mtime, dt.timezone.utc)
                if modified < cutoff:
                    cached_path.unlink()
            except OSError:
                continue
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        path = self.metadata_dir / f"discovery_{stamp}.json"
        path.write_text(json.dumps({
            "discovered_at": now.isoformat(),
            "expires_at": (now + dt.timedelta(days=30)).isoformat(),
            "published_after": after,
            "order": order,
            "queries": queries,
            "results": records,
        }, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def _video_details(client: Any, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        details: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(video_ids), 50):
            response = YouTubeResearchAgent._execute(client.videos().list(
                part="snippet,contentDetails,status,statistics", id=",".join(video_ids[offset:offset + 50])
            ))
            for item in response.get("items", []):
                duration = item.get("contentDetails", {}).get("duration", "")
                seconds = _duration_seconds(duration)
                snippet = item.get("snippet", {})
                details[str(item.get("id"))] = {
                    "duration_seconds": seconds,
                    "category_id": snippet.get("categoryId"),
                    "license": item.get("status", {}).get("license"),
                    "view_count": _optional_int(item.get("statistics", {}).get("viewCount")),
                    "like_count": _optional_int(item.get("statistics", {}).get("likeCount")),
                    "comment_count": _optional_int(item.get("statistics", {}).get("commentCount")),
                }
        return details

    @staticmethod
    def _execute(request: Any) -> dict[str, Any]:
        try:
            return request.execute(num_retries=3)
        except TypeError as exc:
            if "num_retries" not in str(exc):
                raise
            return request.execute()

    @staticmethod
    def _thumbnail_url(snippet: dict[str, Any]) -> str | None:
        thumbnails = snippet.get("thumbnails") or {}
        selected = thumbnails.get("high") or thumbnails.get("medium") or thumbnails.get("default") or {}
        return selected.get("url")


def _duration_seconds(value: str) -> float | None:
    import re

    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value)
    if not match:
        return None
    hours, minutes, seconds = (float(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


YouTubeResearchProvider = YouTubeResearchAgent