"""Structure hypothesis generation + learned scoring (spec §10/§22).

Generates multiple editorial hypotheses from the moment graph — including
the legacy continuous run, which stays available but is no longer the
automatic winner — then scores each with the learned policy. Selection is
by learned evidence, with seeded jitter so near-ties vary between runs.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from app.learning.policy_inference import PolicyInference


def _budget_shots(moments: list[dict[str, Any]], target_duration: float,
                  shot_budget: float) -> list[dict[str, Any]]:
    """Take moments until the duration budget is filled (shots trimmed to budget)."""
    shots: list[dict[str, Any]] = []
    remaining = target_duration
    for moment in moments:
        if remaining < 1.0:
            break
        duration = min(moment["duration"], remaining, shot_budget)
        if duration < 1.0:
            continue
        end = moment["start"] + duration
        shots.append({
            "source_index": moment["source_index"],
            "start": moment["start"],
            "end": round(min(end, moment["end"]), 3),
            "intensity": moment.get("intensity"),
            "has_event": moment.get("kind") == "event",
            "moment_id": moment.get("id"),
        })
        remaining -= duration
    return shots


def generate_structures(
    moments: list[dict[str, Any]],
    source_count: int,
    target_duration: float,
    learned_shot_duration: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Build the candidate hypothesis set. No hypothesis is privileged."""
    shot_budget = float(learned_shot_duration or 4.5)
    events = [m for m in moments if m["kind"] == "event"]
    motions = [m for m in moments if m["kind"] == "motion"]
    runs = [m for m in moments if m["kind"] == "run"]
    highlights = sorted(events + motions, key=lambda m: -(m.get("intensity") or 0.0))
    chronological = sorted(events + motions, key=lambda m: (m["source_index"], m["start"]))
    candidates: list[dict[str, Any]] = []

    def add(name: str, ordered: list[dict[str, Any]]) -> None:
        shots = _budget_shots(ordered, target_duration, shot_budget)
        if shots:
            candidates.append({"name": name, "shots": shots,
                               "source_count": source_count,
                               "target_duration": target_duration})

    if chronological:
        add("chronological_montage", chronological)
    if highlights:
        add("impact_first", highlights)
        strongest = highlights[0]
        rest = [m for m in chronological if m["id"] != strongest["id"]]
        cold_open = [dict(strongest, duration=min(strongest["duration"], 2.5))] + rest
        add("cold_open_then_chronology", cold_open)
        add("escalation_payoff_last", sorted(events + motions,
            key=lambda m: (m.get("intensity") or 0.0, m["source_index"], m["start"])))
    if events and len({m["source_index"] for m in events}) > 1:
        # round-robin across sources (multi-source montage), each source chronological
        by_source: dict[int, list[dict[str, Any]]] = {}
        for moment in sorted(events + motions, key=lambda m: (m["source_index"], m["start"])):
            by_source.setdefault(moment["source_index"], []).append(moment)
        round_robin: list[dict[str, Any]] = []
        depth = 0
        while True:
            layer = [items[depth] for items in by_source.values() if len(items) > depth]
            if not layer:
                break
            round_robin.extend(layer)
            depth += 1
        add("multi_source_montage", round_robin)
    if runs:
        best_run = max(runs, key=lambda m: (m.get("intensity") or 0.0))
        add("continuous_run", [best_run])
    return candidates


def score_and_select(
    candidates: list[dict[str, Any]],
    inference: Optional[PolicyInference],
    gameplay_stats: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
    retrieval_weight: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[int]]:
    """Score every candidate; return (scored, ranked, selected_index).

    Without a trained policy, scoring degrades to honest measurement-only
    components (diversity/event coverage) and the selection reason is
    reported as untrained — never disguised as learned.
    """
    rng = random.Random(seed)
    retrieved = []
    if inference is not None and gameplay_stats:
        retrieved = inference.retrieve_similar(gameplay_stats)
    retrieved_cut_rates = [r["cut_rate"] for r in retrieved if isinstance(r.get("cut_rate"), (int, float))]

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        if inference is not None:
            result = inference.score_structure(candidate)
            composite = result["composite"]
            if retrieved_cut_rates:
                import statistics as _stats
                agreement = max(0.0, 1.0 - abs(result.get("cut_rate", 0.0) - _stats.mean(retrieved_cut_rates))
                                / max(0.25, _stats.mean(retrieved_cut_rates)))
                composite = round((1.0 - retrieval_weight) * composite + retrieval_weight * agreement, 4)
                result["retrieval_agreement"] = round(agreement, 4)
        else:
            shots = candidate["shots"]
            result = {
                "composite": round(
                    0.5 * (len({s["source_index"] for s in shots}) / max(1, candidate["source_count"]))
                    + 0.5 * (sum(1 for s in shots if s["has_event"]) / len(shots)), 4),
                "components": {"untrained": True},
                "tokens": [],
            }
        jitter = 1.0 + (rng.random() - 0.5) * 0.02
        scored.append({**candidate, "score": result, "jittered": round(result["composite"] * jitter, 6)})
    ranked = sorted(scored, key=lambda item: item["jittered"], reverse=True)
    selected = ranked[0] if ranked else None
    selected_index = scored.index(selected) if selected is not None else None
    return scored, ranked, selected_index
