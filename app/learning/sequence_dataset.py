"""Editing-grammar sequence dataset (JSONL) with snapshots and splits."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

from app.learning.sequence_extractor import ANALYZER_VERSION
from app.training.dataset_quality import split_by_creator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GRAMMAR_ROOT = PROJECT_ROOT / "training" / "grammar"


class SequenceDataset:
    """Stores one grammar sequence per reference (deduped by reference_id)."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = pathlib.Path(root or DEFAULT_GRAMMAR_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequences_path = self.root / "sequences.jsonl"
        self.state_path = self.root / "dataset_state.json"

    # ── records ─────────────────────────────────────────────────────────────

    def records(self) -> list[dict[str, Any]]:
        try:
            lines = self.sequences_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("reference_id"):
                out.append(value)
        # newest wins per reference_id
        latest: dict[str, dict[str, Any]] = {}
        for record in out:
            latest[str(record["reference_id"])] = record
        return list(latest.values())

    def add(self, record: dict[str, Any]) -> None:
        if not record.get("reference_id"):
            raise ValueError("grammar record requires reference_id")
        with self.sequences_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    # ── versioning ──────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"snapshot_counter": 0}

    def snapshot(self) -> dict[str, Any]:
        """Freeze the current dataset as grammar_vNNN; returns snapshot info."""
        records = self.records()
        state = self.state()
        counter = int(state.get("snapshot_counter", 0)) + 1
        version = f"grammar_v{counter:03d}"
        digest = hashlib.sha256(
            json.dumps(sorted(str(r["reference_id"]) for r in records)).encode("utf-8")
        ).hexdigest()[:12]
        snapshot_path = self.root / f"dataset_{version}.jsonl"
        with snapshot_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        info = {
            "dataset_version": version,
            "snapshot_digest": digest,
            "sequence_count": len(records),
            "creator_count": len({str(r.get("creator_group") or r["reference_id"]) for r in records}),
            "analyzer_version": ANALYZER_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        state.update({"snapshot_counter": counter, "last_snapshot": info})
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return info

    # ── splits ──────────────────────────────────────────────────────────────

    def creator_split(self) -> dict[str, list[dict[str, Any]]]:
        """Leak-free train/validation/test split by creator group."""
        return split_by_creator(self.records())
