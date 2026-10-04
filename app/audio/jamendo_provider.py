"""Jamendo API v3 search, metadata caching, and permitted audio downloads."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from app.utilities.ffmpeg_utils import get_ffmpeg_path


JAMENDO_API = "https://api.jamendo.com/v3.0"
DEFAULT_CACHE = pathlib.Path(__file__).resolve().parents[2] / "data" / "jamendo_cache"


@dataclass(frozen=True)
class MusicRequirements:
    search: str
    tags: tuple[str, ...] = ()
    fuzzytags: tuple[str, ...] = ()
    speed: tuple[str, ...] = ()
    instrumental: Optional[bool] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    content_id_free: bool = True


class JamendoMusicProvider:
    def __init__(
        self,
        client_id: Optional[str] = None,
        cache_dir: pathlib.Path = DEFAULT_CACHE,
        http: Any = requests,
        cache_ttl_seconds: int = 21600,
    ) -> None:
        self.client_id = client_id or os.getenv("JAMENDO_CLIENT_ID")
        self.cache_dir = pathlib.Path(cache_dir)
        self.http = http
        self.cache_ttl_seconds = cache_ttl_seconds

    def search(self, requirements: MusicRequirements, limit: int = 20) -> list[dict[str, Any]]:
        if not self.client_id:
            raise RuntimeError("Set JAMENDO_CLIENT_ID to enable Jamendo music search.")
        params = self._query_params(requirements, limit)
        cache_key = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"search_{cache_key}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        response = self.http.get(f"{JAMENDO_API}/tracks/", params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if payload.get("headers", {}).get("status") != "success":
            message = payload.get("headers", {}).get("error_message", "Jamendo search failed")
            raise RuntimeError(message)

        tracks = []
        for raw_track in payload.get("results", []):
            if not _is_true(raw_track.get("audiodownload_allowed")) or not raw_track.get("audiodownload"):
                continue
            track = self._normalize_track(raw_track)
            self._write_track_cache(track)
            tracks.append(track)
        self._write_cache(cache_path, tracks)
        return tracks

    def get_metadata(self, track_id: str) -> Optional[dict[str, Any]]:
        cached = self._read_cache(self.cache_dir / f"track_{track_id}.json")
        if cached is not None:
            return cached
        if not self.client_id:
            raise RuntimeError("Set JAMENDO_CLIENT_ID to retrieve Jamendo track metadata.")
        response = self.http.get(
            f"{JAMENDO_API}/tracks/",
            params={
                "client_id": self.client_id,
                "format": "json",
                "id": track_id,
                "include": "licenses+musicinfo+stats",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            return None
        track = self._normalize_track(results[0])
        self._write_track_cache(track)
        return track

    def download_track(
        self,
        track: dict[str, Any],
        destination_dir: pathlib.Path,
        intended_use: str = "social_video",
    ) -> tuple[pathlib.Path, dict[str, Any]]:
        if not track.get("download_allowed") or not track.get("download_url"):
            raise RuntimeError("Jamendo did not permit downloading this track.")
        destination_dir = pathlib.Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(track["id"]))
        suffix = ".mp3"
        audio_path = destination_dir / f"jamendo_{safe_id}{suffix}"
        if not audio_path.exists():
            response = self.http.get(track["download_url"], stream=True, timeout=30)
            response.raise_for_status()
            temporary_path = audio_path.with_suffix(audio_path.suffix + ".part")
            total_bytes = 0
            try:
                with temporary_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > 30 * 1024 * 1024:
                            raise RuntimeError("Jamendo track exceeded the 30 MB download limit.")
                        output_file.write(chunk)
                temporary_path.replace(audio_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        license_record = {
            "provider": "jamendo",
            "track_id": str(track["id"]),
            "title": track.get("title", "Unknown title"),
            "artist": track.get("artist", "Unknown artist"),
            "license_url": track.get("license_url"),
            "license_type": track.get("license_type"),
            "download_allowed": True,
            "commercial_use": "verify_with_provider_for_intended_use",
            "attribution_required": track.get("attribution_required", True),
            "intended_use": intended_use,
            "warning": (
                "Jamendo API/download permission does not establish commercial or social-platform rights. "
                "Verify the track license or obtain a Jamendo Pro license before monetized use."
            ),
            "source_url": track.get("source_url"),
        }
        self._write_cache(destination_dir / f"jamendo_{safe_id}.license.json", license_record)
        return audio_path, license_record

    def analyze_audio(self, audio_url: str, sample_seconds: int = 12) -> dict[str, Any]:
        """Estimate loudness dynamics and beat periodicity from a short stream sample."""
        if not audio_url.startswith("https://"):
            return {}
        try:
            result = subprocess.run(
                [
                    get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", audio_url,
                    "-t", str(sample_seconds), "-vn", "-ac", "1", "-ar", "200",
                    "-f", "f32le", "pipe:1",
                ],
                capture_output=True,
                timeout=25,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode != 0 or not result.stdout:
            return {}

        samples = [item[0] for item in struct.iter_unpack("<f", result.stdout[:len(result.stdout) // 4 * 4])]
        window = 20
        envelope = [
            math.sqrt(sum(value * value for value in samples[index:index + window]) / window)
            for index in range(0, len(samples) - window + 1, window)
        ]
        if len(envelope) < 30:
            return {"sampled_seconds": sample_seconds, "bpm_estimate": None}
        mean_energy = sum(envelope) / len(envelope)
        variance = sum((value - mean_energy) ** 2 for value in envelope) / len(envelope)
        centered = [value - mean_energy for value in envelope]
        autocorrelation = {}
        for lag in range(3, min(11, len(centered) // 3)):
            score = sum(centered[index] * centered[index - lag] for index in range(lag, len(centered)))
            autocorrelation[lag] = score
        best_lag = max(autocorrelation, key=autocorrelation.get) if autocorrelation else None
        peak = max(autocorrelation.values(), default=0.0)
        confidence = peak / (sum(abs(value) for value in autocorrelation.values()) or 1.0)
        return {
            "sampled_seconds": sample_seconds,
            "mean_rms": round(mean_energy, 5),
            "energy_variability": round(math.sqrt(variance), 5),
            "bpm_estimate": round(600 / best_lag, 1) if best_lag else None,
            "bpm_confidence": round(confidence, 3) if best_lag else 0.0,
        }

    def _query_params(self, requirements: MusicRequirements, limit: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "client_id": self.client_id,
            "format": "json",
            "limit": max(1, min(limit, 50)),
            "order": "popularity_week",
            "include": "licenses+musicinfo+stats",
            "audioformat": "mp32",
            "audiodlformat": "mp32",
            "content_id_free": str(requirements.content_id_free).lower(),
            "ccnc": "false",
            "ccnd": "false",
            "ccsa": "false",
        }
        if requirements.search:
            params["search"] = requirements.search[:120]
        if requirements.tags:
            params["tags"] = "+".join(_clean_tag(tag) for tag in requirements.tags if _clean_tag(tag))
        if requirements.fuzzytags:
            params["fuzzytags"] = "+".join(_clean_tag(tag) for tag in requirements.fuzzytags if _clean_tag(tag))
        valid_speeds = {"verylow", "low", "medium", "high", "veryhigh"}
        speeds = [speed for speed in requirements.speed if speed in valid_speeds]
        if speeds:
            params["speed"] = "+".join(speeds)
        if requirements.instrumental is not None:
            params["vocalinstrumental"] = "instrumental" if requirements.instrumental else "vocal"
        if requirements.duration_min is not None and requirements.duration_max is not None:
            params["durationbetween"] = f"{max(1, requirements.duration_min)}_{max(requirements.duration_min, requirements.duration_max)}"
        return params

    def _normalize_track(self, raw: dict[str, Any]) -> dict[str, Any]:
        licenses = raw.get("licenses")
        if isinstance(licenses, list):
            license_info = licenses[0] if licenses else {}
        elif isinstance(licenses, dict):
            license_info = licenses
        else:
            license_info = {}
        license_url = raw.get("license_ccurl") or license_info.get("license_ccurl") or license_info.get("url")
        license_type = raw.get("license_ccname") or license_info.get("license_ccname") or license_info.get("name")
        return {
            "id": str(raw.get("id", "")),
            "title": raw.get("name", "Unknown title"),
            "artist": raw.get("artist_name", "Unknown artist"),
            "duration": _as_float(raw.get("duration")),
            "audio_url": raw.get("audio"),
            "download_url": raw.get("audiodownload"),
            "download_allowed": _is_true(raw.get("audiodownload_allowed")) and bool(raw.get("audiodownload")),
            "license_url": license_url,
            "license_type": license_type,
            "attribution_required": bool(license_url),
            "source_url": raw.get("shareurl"),
            "musicinfo": raw.get("musicinfo", {}),
            "stats": raw.get("stats", {}),
            "audio_features": {},
        }

    def _read_cache(self, cache_path: pathlib.Path) -> Optional[Any]:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if time.time() - payload["cached_at"] > self.cache_ttl_seconds:
                return None
            return payload["data"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, cache_path: pathlib.Path, data: Any) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"cached_at": time.time(), "data": data}, indent=2), encoding="utf-8")

    def _write_track_cache(self, track: dict[str, Any]) -> None:
        self._write_cache(self.cache_dir / f"track_{track['id']}.json", track)


def _is_true(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}


def _clean_tag(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", value.strip().lower())


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
