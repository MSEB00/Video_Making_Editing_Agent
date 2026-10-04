"""Persist human-authored observations without copying embedded video media."""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import re
import threading
import uuid
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_ROOT = PROJECT_ROOT / "training" / "research"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
DECISION_VALUES = {"observed", "not_observed", "uncertain", "not_applicable"}


class ResearchValidationError(ValueError):
    pass


class VideoObservationStore:
    """Store public metadata and explicit researcher annotations as JSONL."""

    _write_lock = threading.Lock()

    def __init__(
        self,
        root: pathlib.Path = DEFAULT_RESEARCH_ROOT,
        allow_derived_observations: bool = False,
    ) -> None:
        self.root = pathlib.Path(root)
        self.allow_derived_observations = allow_derived_observations
        self.videos_path = self.root / "videos.jsonl"
        self.sessions_path = self.root / "sessions.jsonl"
        self.observations_path = self.root / "observations.jsonl"
        self.metadata_path = self.root / "metadata.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._purge_expired()
        self._write_metadata()

    def register_videos(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = dt.datetime.now(dt.timezone.utc)
        existing = {
            item["video_id"]: item
            for item in self._read_jsonl(self.videos_path)
            if _not_expired(item.get("metadata_expires_at"), now)
        }
        registered = []
        for raw in videos:
            video_id = str(raw.get("video_id") or "").strip()
            if not VIDEO_ID_PATTERN.fullmatch(video_id):
                continue
            discovered_at = _bounded_text(raw.get("discovered_at"), 80) or now.isoformat()
            metadata_expires_at = _expiration_for(discovered_at) or (now + dt.timedelta(days=30)).isoformat()
            item = {
                "video_id": video_id,
                "title": _bounded_text(raw.get("title"), 400),
                "channel": _bounded_text(raw.get("channel") or raw.get("channel_title"), 200),
                "channel_id": _bounded_text(raw.get("channel_id"), 120),
                "duration_seconds": _optional_duration(raw.get("duration_seconds")),
                "published_at": _bounded_text(raw.get("published_at"), 80),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "discovery_topics": _string_list(raw.get("discovery_queries") or raw.get("topics"), 12, 120),
                "category": _bounded_text(raw.get("category"), 80) or "gaming",
                "license": _bounded_text(raw.get("license"), 80),
                "license_url": _validated_http_url(raw.get("license_url")),
                "research_status": "discovered",
                "observation_status": "not_observed",
                "discovered_at": discovered_at,
                "metadata_expires_at": metadata_expires_at,
                "studied_at": None,
                "media_acquired": False,
                "media_stored": False,
            }
            prior = existing.get(video_id, {})
            item.update({
                key: prior[key]
                for key in ("research_status", "observation_status", "studied_at")
                if prior.get(key) is not None
            })
            existing[video_id] = item
            registered.append(item)
        self._write_jsonl(self.videos_path, list(existing.values()))
        self._write_metadata()
        return registered

    def list_videos(self, limit: int = 100) -> list[dict[str, Any]]:
        now = dt.datetime.now(dt.timezone.utc)
        all_items = self._read_jsonl(self.videos_path)
        items = [item for item in all_items if _not_expired(item.get("metadata_expires_at"), now)]
        if len(items) != len(all_items):
            self._write_jsonl(self.videos_path, items)
        return list(reversed(items[-max(1, min(500, int(limit))):]))

    def create_session(self, video_id: str) -> dict[str, Any]:
        video = self.get_video(video_id)
        if video is None:
            raise ResearchValidationError("Register the video before starting a research session.")
        session = {
            "session_id": uuid.uuid4().hex,
            "video_id": video_id,
            "platform": "youtube",
            "started_at": _now(),
            "finished_at": None,
            "status": "in_progress",
            "player": "official_youtube_iframe",
            "media_captured": False,
        }
        self._append_jsonl(self.sessions_path, session)
        self._update_video(video_id, research_status="in_progress", studied_at=session["started_at"])
        self._write_metadata()
        return session

    def finish_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise ResearchValidationError("Research session was not found.")
        if session["status"] == "completed":
            return session
        session = {**session, "status": "completed", "finished_at": _now()}
        self._append_jsonl(self.sessions_path, session)
        video = self.get_video(session["video_id"])
        if video:
            self._update_video(
                session["video_id"],
                research_status="studied",
                observation_status="observed" if self.observations_for(session_id) else "no_observations",
                studied_at=session["finished_at"],
            )
        self._write_metadata()
        return session

    def add_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = _bounded_text(payload.get("session_id"), 64)
        session = self.get_session(session_id)
        if session is None or session["status"] != "in_progress":
            raise ResearchValidationError("An active research session is required.")
        video = self.get_video(session["video_id"])
        if video is None:
            raise ResearchValidationError("The session's video metadata is missing.")

        start = _finite_number(payload.get("start_seconds"), "start_seconds")
        end = _finite_number(payload.get("end_seconds"), "end_seconds")
        duration = video.get("duration_seconds")
        if start < 0 or end < start or (duration is not None and end > duration + 0.5):
            raise ResearchValidationError("Observation timestamps must be within the video duration.")

        context = payload.get("context") or {}
        if not isinstance(context, dict):
            raise ResearchValidationError("context must be an object.")
        editing = payload.get("editing_decisions") or {}
        if not isinstance(editing, dict):
            raise ResearchValidationError("editing_decisions must be an object.")
        allowed_decisions = {"cut", "zoom", "speed_change", "transition", "caption", "sfx", "bgm"}
        if set(editing) - allowed_decisions or any(
            not isinstance(value, str) or value not in DECISION_VALUES
            for value in editing.values()
        ):
            raise ResearchValidationError("Editing decisions contain an unsupported field or value.")

        observation = {
            "observation_id": uuid.uuid4().hex,
            "session_id": session_id,
            "source": {
                "platform": "youtube",
                "video_id": session["video_id"],
                "url": video["url"],
                "observed_with": "human_using_embedded_player",
                "media_saved": False,
            },
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "structure_label": _bounded_text(payload.get("structure_label"), 80) or "unclassified",
            "content_context": {
                "visual_intensity": _optional_unit_interval(context.get("visual_intensity"), "visual_intensity"),
                "audio_intensity": _optional_unit_interval(context.get("audio_intensity"), "audio_intensity"),
                "speech_present": _optional_bool(context.get("speech_present"), "speech_present"),
                "gameplay_context": _bounded_text(context.get("gameplay_context"), 120),
                "mood": _bounded_text(context.get("mood"), 80),
            },
            "editing_decisions": dict(editing),
            "researcher_note": _bounded_text(payload.get("note"), 1000),
            "confidence": _optional_unit_interval(payload.get("confidence"), "confidence"),
            "observation_source": "human_authored",
            "observed_at": _now(),
            "training_eligible": False,
            "training_eligibility_reason": (
                "YouTube player observations are excluded from model training by default; "
                "use independently rights-cleared local references for training."
            ),
        }
        if not observation["researcher_note"] and not any(editing.values()) and observation["structure_label"] == "unclassified":
            raise ResearchValidationError("Add a note, structure label, or editing decision before saving.")
        rights_confirmed = _optional_bool(payload.get("rights_confirmed"), "rights_confirmed") is True
        license_cleared = video.get("license") == "creativeCommon"
        training_eligible = self.allow_derived_observations and rights_confirmed and license_cleared
        observation["rights_confirmation"] = {
            "creator_and_audio_rights_confirmed_by_operator": rights_confirmed,
            "youtube_derived_data_approval_configured": self.allow_derived_observations,
            "discovery_license": video.get("license"),
        }
        observation["training_eligible"] = training_eligible
        if training_eligible:
            observation["training_eligibility_reason"] = "Operator-confirmed creator/audio rights and configured written YouTube approval."
        self._append_jsonl(self.observations_path, observation)
        self._update_video(session["video_id"], observation_status="observed")
        self._write_metadata()
        return observation

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        return next((item for item in self._read_jsonl(self.videos_path) if item.get("video_id") == video_id), None)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        matches = [item for item in self._read_jsonl(self.sessions_path) if item.get("session_id") == session_id]
        return matches[-1] if matches else None

    def observations_for(self, session_id: str) -> list[dict[str, Any]]:
        return [item for item in self._read_jsonl(self.observations_path) if item.get("session_id") == session_id]

    def observations(self, limit: int = 5000) -> list[dict[str, Any]]:
        return self._read_jsonl(self.observations_path)[-max(1, min(20000, int(limit))):]

    def stats(self) -> dict[str, Any]:
        videos = self._read_jsonl(self.videos_path)
        sessions = self._read_jsonl(self.sessions_path)
        latest_sessions = {item["session_id"]: item for item in sessions if item.get("session_id")}
        observations = self._read_jsonl(self.observations_path)
        eligible_count = sum(item.get("training_eligible") is True for item in observations)
        return {
            "video_count": len(videos),
            "session_count": len(latest_sessions),
            "completed_sessions": sum(item.get("status") == "completed" for item in latest_sessions.values()),
            "observation_count": len(observations),
            "training_eligible_youtube_observations": eligible_count,
            "media_stored": False,
        }

    def _update_video(self, video_id: str, **changes: Any) -> None:
        videos = self._read_jsonl(self.videos_path)
        for item in videos:
            if item.get("video_id") == video_id:
                item.update(changes)
        self._write_jsonl(self.videos_path, videos)

    def _write_metadata(self) -> None:
        value = {
            "dataset_name": "human_authored_short_form_research",
            "dataset_version": "v001",
            "updated_at": _now(),
            "video_count": len(self._read_jsonl(self.videos_path)),
            "session_count": len({item.get("session_id") for item in self._read_jsonl(self.sessions_path) if item.get("session_id")}),
            "observation_count": len(self._read_jsonl(self.observations_path)),
            "youtube_media_captured": False,
            "youtube_observations_training_eligible": any(
                item.get("training_eligible") is True
                for item in self._read_jsonl(self.observations_path)
            ),
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        with self._write_lock:
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
            temporary.replace(self.metadata_path)

    def _purge_expired(self) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
        sessions = [
            item for item in self._read_jsonl(self.sessions_path)
            if _not_older_than(item.get("started_at"), cutoff)
        ]
        session_ids = {item.get("session_id") for item in sessions}
        observations = [
            item for item in self._read_jsonl(self.observations_path)
            if item.get("session_id") in session_ids and _not_older_than(item.get("observed_at"), cutoff)
        ]
        if len(sessions) != len(self._read_jsonl(self.sessions_path)):
            self._write_jsonl(self.sessions_path, sessions)
        if len(observations) != len(self._read_jsonl(self.observations_path)):
            self._write_jsonl(self.observations_path, observations)

    @staticmethod
    def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    @classmethod
    def _append_jsonl(cls, path: pathlib.Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
        with cls._write_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)

    @classmethod
    def _write_jsonl(cls, path: pathlib.Path, values: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = "".join(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n" for value in values)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with cls._write_lock:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _string_list(value: Any, count_limit: int, text_limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_bounded_text(item, text_limit) for item in value[:count_limit] if _bounded_text(item, text_limit)]


def _validated_http_url(value: Any) -> str | None:
    text = _bounded_text(value, 500)
    return text if text.startswith(("https://", "http://")) else None


def _not_expired(value: Any, now: dt.datetime) -> bool:
    try:
        expires_at = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
    return expires_at > now


def _not_older_than(value: Any, cutoff: dt.datetime) -> bool:
    try:
        timestamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp >= cutoff


def _expiration_for(value: str) -> str | None:
    try:
        timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return (timestamp + dt.timedelta(days=30)).isoformat()


def _optional_duration(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if math.isfinite(number) and 0 < number <= 86400 else None


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ResearchValidationError(f"{name} must be finite.")
    return number


def _optional_unit_interval(value: Any, name: str) -> float | None:
    if value in (None, ""):
        return None
    number = _finite_number(value, name)
    if not 0 <= number <= 1:
        raise ResearchValidationError(f"{name} must be between 0 and 1.")
    return round(number, 3)


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ResearchValidationError(f"{name} must be a boolean or null.")