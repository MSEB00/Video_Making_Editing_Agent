"""Editing-policy learner (Stage 1: learned statistics + retrieval).

Learns TEMPORAL editing behavior from grammar sequences:
  - token transition probabilities  P(next | current)         (spec §5)
  - context-conditioned action probabilities
        P(cut_after | intensity bucket, position bucket)      (spec §5/§36)
  - per-token shot-duration distributions                     (hold behavior)
  - pacing / hook / payoff-position / intensity-curve profiles (spec §14)
  - reference embeddings for style retrieval                  (spec §15)

Stage gating is honest (spec §6): this Stage-1 model trains on small data;
sequence-neural stages are declared but NOT faked — the artifact reports
which stage produced it and why.

Validation: held-out per-token log-likelihood vs a uniform-order baseline
and vs the previously promoted policy. Promotion only on real improvement.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Optional

from app.learning.grammar import EMBEDDING_KEYS, TOKENS

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_POLICY_ROOT = PROJECT_ROOT / "training" / "grammar" / "policies"

MIN_SEQUENCES = 5
MIN_CREATORS = 3
STAGE = "stage1_statistics_retrieval"
STAGE_THRESHOLDS = {
    "stage1_statistics_retrieval": (0, "learned transition statistics + retrieval"),
    "stage2_sequence_learner": (50, "declared; not implemented — no fake models"),
    "stage3_neural_policy": (500, "declared; not implemented — no fake models"),
}
UNKNOWN_CHANNELS = (
    "caption_presence", "caption_rhythm", "sfx_density", "bgm_presence",
    "transition_types", "visual_effect_types", "speech_content",
)


def _bucket(value: Optional[float], cuts: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    return "low" if value < cuts[0] else "medium" if value < cuts[1] else "high"


def _tercile_cuts(values: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        return (1.0 / 3.0, 2.0 / 3.0)
    ordered = sorted(values)
    return (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3])


class PolicyTrainer:
    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = pathlib.Path(root or DEFAULT_POLICY_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.root / "active.json"

    # ── training ────────────────────────────────────────────────────────────

    def train_candidate(
        self,
        train_records: list[dict[str, Any]],
        held_out_records: list[dict[str, Any]],
        dataset_info: dict[str, Any],
    ) -> dict[str, Any]:
        all_records = train_records + held_out_records
        creators = {str(r.get("creator_group") or r.get("reference_id")) for r in all_records}
        if len(all_records) < MIN_SEQUENCES:
            return {"status": "insufficient_sequences", "sequence_count": len(all_records),
                    "minimum": MIN_SEQUENCES, "learning_state": "BOOTSTRAP"}
        if len(creators) < MIN_CREATORS:
            return {"status": "insufficient_creator_diversity", "creator_count": len(creators),
                    "minimum": MIN_CREATORS, "learning_state": "BOOTSTRAP"}

        model = self._fit(train_records)
        held_out_score = self._held_out_logprob(model, held_out_records)
        baseline_score = self._baseline_logprob(held_out_records)
        previous = self.load_active()
        previous_score = (
            self._held_out_logprob(previous.get("model", {}), held_out_records)
            if previous else None
        )

        versions = sorted(self.root.glob("policy_v*.json"))
        version = f"policy_v{len(versions) + 1:03d}"
        beats_baseline = held_out_score > baseline_score
        beats_previous = previous_score is None or held_out_score >= previous_score
        promoted = bool(held_out_records) and beats_baseline and beats_previous

        artifact = {
            "version": version,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": STAGE,
            "stage_note": STAGE_THRESHOLDS[STAGE][1],
            "dataset": {
                **dataset_info,
                "training_sequences": len(train_records),
                "held_out_sequences": len(held_out_records),
                "creator_count": len(creators),
            },
            "validation": {
                "held_out_logprob_per_token": round(held_out_score, 4),
                "uniform_order_baseline_logprob": round(baseline_score, 4),
                "previous_policy_logprob": (round(previous_score, 4) if previous_score is not None else None),
                "beats_baseline": beats_baseline,
                "beats_previous": beats_previous,
                "promoted": promoted,
            },
            "previous_version": (previous or {}).get("version"),
            "unknown_channels": list(UNKNOWN_CHANNELS),
            "model": model,
        }
        candidate_path = self.root / f"{version}.json"
        candidate_path.write_text(json.dumps(artifact, indent=1, ensure_ascii=True), encoding="utf-8")
        if promoted:
            self.active_path.write_text(json.dumps(artifact, indent=1, ensure_ascii=True), encoding="utf-8")
        return {
            "status": "promoted" if promoted else "rejected",
            "version": version,
            "path": str(candidate_path),
            "learning_state": "VALIDATED" if promoted else "LEARNING",
            "validation": artifact["validation"],
            "sequence_count": len(all_records),
        }

    def load_active(self) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # ── fitting ─────────────────────────────────────────────────────────────

    def _fit(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        transitions: dict[str, Counter] = {token: Counter() for token in TOKENS}
        durations: dict[str, list[float]] = defaultdict(list)
        cut_after: dict[tuple[str, str], list[float]] = defaultdict(list)
        cut_rates, shot_durations, payoff_positions, hook_durations, hook_intensities = [], [], [], [], []
        curves: list[list[float]] = []
        silence_users = 0
        embeddings: list[dict[str, Any]] = []

        for record in records:
            sequence = record.get("sequence") or []
            stats = record.get("stats") or {}
            for seg in sequence:
                token = seg.get("token")
                if token in transitions:
                    durations[token].append(float(seg.get("duration") or 0.0))
                    key = (
                        _bucket(seg.get("intensity"), (1 / 3, 2 / 3)),
                        _bucket(seg.get("position"), (1 / 3, 2 / 3)),
                    )
                    cut_after[key].append(1.0 if seg.get("cut_after") else 0.0)
            for prev, nxt in zip(sequence, sequence[1:]):
                if prev.get("token") in transitions and nxt.get("token") in TOKENS:
                    transitions[prev["token"]][nxt["token"]] += 1
            for key in ("cut_rate",):
                if isinstance(stats.get(key), (int, float)):
                    cut_rates.append(float(stats[key]))
            if isinstance(stats.get("mean_shot_duration"), (int, float)):
                shot_durations.append(float(stats["mean_shot_duration"]))
            if isinstance(stats.get("payoff_position"), (int, float)):
                payoff_positions.append(float(stats["payoff_position"]))
            if isinstance(stats.get("hook_duration"), (int, float)):
                hook_durations.append(float(stats["hook_duration"]))
            if isinstance(stats.get("hook_intensity"), (int, float)):
                hook_intensities.append(float(stats["hook_intensity"]))
            curve = [v for v in (stats.get("intensity_curve") or []) if isinstance(v, (int, float))]
            if curve:
                curves.append(curve)
            if isinstance(stats.get("silence_ratio_mean"), (int, float)) and float(stats["silence_ratio_mean"]) > 0.05:
                silence_users += 1
            if stats.get("embedding"):
                embeddings.append({
                    "reference_id": record.get("reference_id"),
                    "style_tags": record.get("style_tags"),
                    "creator_group": record.get("creator_group"),
                    "embedding": stats["embedding"],
                    "cut_rate": stats.get("cut_rate"),
                    "payoff_position": stats.get("payoff_position"),
                })

        token_transitions = {}
        for token, counter in transitions.items():
            total = sum(counter.values())
            token_transitions[token] = {
                nxt: {"n": count, "p": round((count + 1) / (total + len(TOKENS)), 4)}
                for nxt, count in counter.items()
            } or {nxt: {"n": 0, "p": round(1.0 / len(TOKENS), 4)} for nxt in TOKENS}
            if not counter:
                token_transitions[token] = {nxt: {"n": 0, "p": round(1.0 / len(TOKENS), 4)} for nxt in TOKENS}

        context_actions = {
            "cut_after": {
                f"{intensity}_{position}": {
                    "p": round(statistics.mean(values), 3), "n": len(values),
                }
                for (intensity, position), values in sorted(cut_after.items())
            },
        }
        curve_mean = None
        if curves:
            width = max(len(c) for c in curves)
            curve_mean = [
                round(statistics.mean([c[i] for c in curves if len(c) > i]), 4)
                for i in range(width)
            ]

        def _dist(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"mean": None, "std": None, "n": 0}
            return {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
                "n": len(values),
            }

        return {
            "token_transitions": token_transitions,
            "token_durations": {token: _dist(values) for token, values in sorted(durations.items())},
            "context_actions": context_actions,
            "cut_rate": _dist(cut_rates),
            "mean_shot_duration": _dist(shot_durations),
            "payoff_position": _dist(payoff_positions),
            "hook_duration": _dist(hook_durations),
            "hook_intensity": _dist(hook_intensities),
            "intensity_curve_mean": curve_mean,
            "silence_use_rate": round(silence_users / len(records), 3) if records else None,
            "reference_embeddings": embeddings,
            "embedding_keys": list(EMBEDDING_KEYS),
        }

    # ── validation ──────────────────────────────────────────────────────────

    def _held_out_logprob(self, model: dict[str, Any], records: list[dict[str, Any]]) -> float:
        transitions = (model or {}).get("token_transitions") or {}
        total, count = 0.0, 0
        for record in records:
            sequence = record.get("sequence") or []
            for prev, nxt in zip(sequence, sequence[1:]):
                table = transitions.get(str(prev.get("token"))) or {}
                entry = table.get(str(nxt.get("token")))
                p = float(entry.get("p", 1.0 / len(TOKENS))) if entry else 1.0 / len(TOKENS)
                total += math.log(max(p, 1e-9))
                count += 1
        return total / count if count else float("-inf")

    @staticmethod
    def _baseline_logprob(records: list[dict[str, Any]]) -> float:
        count = sum(max(0, len(r.get("sequence") or []) - 1) for r in records)
        return math.log(1.0 / len(TOKENS)) if count else float("-inf")


def learning_state(dataset_count: int, trainer: PolicyTrainer) -> str:
    """Explicit state machine (spec §29): honest about what has been learned."""
    if dataset_count <= 0:
        return "NO_DATA"
    if dataset_count < MIN_SEQUENCES:
        return "BOOTSTRAP"
    active = trainer.load_active()
    if active is None:
        return "LEARNING"
    used = (active.get("usage") or {}).get("edits_used_in", 0)
    validation = active.get("validation") or {}
    if used > 0:
        return "ACTIVE"
    return "VALIDATED" if validation.get("promoted") else "LEARNING"


def mark_policy_used(root: pathlib.Path | None = None, job_id: Optional[int] = None) -> None:
    """Record that the active policy drove a real edit (LEARNING → ACTIVE)."""
    trainer = PolicyTrainer(root)
    active = trainer.load_active()
    if not active:
        return
    usage = active.setdefault("usage", {"edits_used_in": 0, "jobs": []})
    usage["edits_used_in"] = int(usage.get("edits_used_in", 0)) + 1
    if job_id is not None:
        usage.setdefault("jobs", []).append(job_id)
        usage["jobs"] = usage["jobs"][-50:]
    trainer.active_path.write_text(json.dumps(active, indent=1, ensure_ascii=True), encoding="utf-8")
