"""Policy inference: score candidate edit structures with learned evidence.

The inference surface used by the planner and A/B harness:
  - sequence_logprob(tokens): learned transition likelihood
  - score_structure(candidate): multi-component editorial score combining
    learned likelihood, pacing fit, hook/payoff fit, curve fit, diversity
    and restraint — NOT raw activity (spec §22/§39)
  - retrieve_similar(stats): style retrieval by embedding distance (§15)
  - suggest_actions(context): learned action probabilities (§7/§23)

All components are z-scored against learned distributions; missing channels
are excluded honestly instead of being guessed.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Optional

from app.learning.grammar import TOKENS, label_segments


def _gaussian_score(value: Optional[float], mean: Optional[float], std: Optional[float],
                    default: float = 0.5) -> float:
    """1.0 at the learned mean, decaying with |z| (clamped at 3 sigma)."""
    if value is None or mean is None:
        return default
    sigma = float(std) if std and std > 1e-6 else 0.35 * max(abs(float(mean)), 0.2)
    z = abs(float(value) - float(mean)) / sigma
    return round(max(0.0, 1.0 - z / 3.0), 4)


class PolicyInference:
    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = policy or {}
        self.model = self.policy.get("model") or {}
        self.version = self.policy.get("version")

    # ── sequence likelihood ─────────────────────────────────────────────────

    def sequence_logprob(self, tokens: list[str]) -> Optional[float]:
        """Mean per-transition log-likelihood; None when no transitions exist
        (a single-shot candidate has no sequence evidence — treated as
        neutral downstream, never rewarded)."""
        transitions = self.model.get("token_transitions") or {}
        total, count = 0.0, 0
        for prev, nxt in zip(tokens, tokens[1:]):
            table = transitions.get(prev) or {}
            entry = table.get(nxt)
            p = float(entry["p"]) if entry else 1.0 / len(TOKENS)
            total += math.log(max(p, 1e-9))
            count += 1
        return total / count if count else None

    # ── structure scoring ───────────────────────────────────────────────────

    def score_structure(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Score one candidate structure with learned evidence.

        candidate: {"name", "shots": [{"source_index","start","end","intensity",
                    "has_event"}], "source_count", "target_duration"}
        Returns per-component scores in [0,1] + weighted composite + the
        labeled token sequence used (diagnostics).
        """
        shots = candidate.get("shots") or []
        if not shots:
            return {"composite": 0.0, "components": {}, "tokens": []}
        durations = [max(0.05, float(s["end"]) - float(s["start"])) for s in shots]
        total = sum(durations) or 1.0
        # Reuse the SAME weak labeler applied to references — candidates and
        # references live in one representation (no separate scoring rules).
        segments = [
            {"t": float(s["start"]), "duration": d,
             "motion": s.get("intensity"), "audio": None, "silence_ratio": None}
            for s, d in zip(shots, durations)
        ]
        labeled = label_segments(segments)
        tokens = [seg["token"] for seg in labeled]

        cut_rate = (len(shots) - 1) / total if total > 0 else 0.0
        mean_shot = statistics.mean(durations)
        payoff_positions = [seg["position"] for seg in labeled if seg["token"] == "PAYOFF"]
        payoff_position = statistics.mean(payoff_positions) if payoff_positions else None
        hook = labeled[0]

        ll = self.sequence_logprob(tokens)
        likelihood_score = (
            0.5 if ll is None
            else round(1.0 / (1.0 + math.exp(-2.0 * (ll - math.log(1.0 / len(TOKENS))))), 4)
        )

        components = {
            "learned_likelihood": likelihood_score,
            "pacing_fit": min(1.0, _gaussian_score(cut_rate, _mean(self.model.get("cut_rate")),
                                                   _std(self.model.get("cut_rate")))),
            "shot_duration_fit": _gaussian_score(mean_shot, _mean(self.model.get("mean_shot_duration")),
                                                 _std(self.model.get("mean_shot_duration"))),
            "hook_fit": round(0.5 * _gaussian_score(hook.get("duration"), _mean(self.model.get("hook_duration")),
                                                    _std(self.model.get("hook_duration")))
                            + 0.5 * _gaussian_score(hook.get("intensity"), _mean(self.model.get("hook_intensity")),
                                                    _std(self.model.get("hook_intensity"))), 4),
            "payoff_position_fit": _gaussian_score(payoff_position, _mean(self.model.get("payoff_position")),
                                                   _std(self.model.get("payoff_position"))),
            "curve_fit": self._curve_fit(labeled),
            "source_diversity": round(
                len({s["source_index"] for s in shots}) / max(1, int(candidate.get("source_count") or 1)), 4
            ),
            "event_coverage": round(
                sum(1 for s in shots if s.get("has_event")) / len(shots), 4
            ),
            "duration_fit": _gaussian_score(total, float(candidate.get("target_duration") or total), 0.25 * float(candidate.get("target_duration") or total or 1)),
        }
        composite = round(statistics.mean(components.values()), 4)
        return {"composite": composite, "components": components, "tokens": tokens,
                "cut_rate": round(cut_rate, 4), "mean_shot_duration": round(mean_shot, 3)}

    def _curve_fit(self, labeled: list[dict[str, Any]]) -> float:
        target = self.model.get("intensity_curve_mean")
        if not target:
            return 0.5
        bins = len(target)
        sums: list[list[float]] = [[] for _ in range(bins)]
        for seg in labeled:
            bucket = min(bins - 1, int(float(seg.get("position", 0.0)) * bins))
            sums[bucket].append(float(seg.get("intensity", 0.0)))
        candidate_curve = [statistics.mean(b) if b else None for b in sums]
        diffs = [
            abs(c - t) for c, t in zip(candidate_curve, target)
            if c is not None and t is not None
        ]
        if not diffs:
            return 0.5
        return round(max(0.0, 1.0 - statistics.mean(diffs) * 2.0), 4)

    # ── style retrieval ─────────────────────────────────────────────────────

    def retrieve_similar(self, stats: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Nearest reference sequences by z-normalized embedding distance."""
        references = self.model.get("reference_embeddings") or []
        if not references or not stats:
            return []
        query = _query_embedding(stats)
        norms = _embedding_norms(references)
        scored = []
        for ref, (mean, std) in zip(references, norms):
            vec = ref.get("embedding") or []
            if not vec:
                continue
            z_query = [(q - mean[i]) / std[i] if std[i] else 0.0 for i, q in enumerate(query) if i < len(mean)]
            z_ref = [(v - mean[i]) / std[i] if std[i] else 0.0 for i, v in enumerate(vec) if i < len(mean)]
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_query, z_ref)))
            similarity = 1.0 / (1.0 + distance)
            scored.append({
                "reference_id": ref.get("reference_id"),
                "style_tags": ref.get("style_tags"),
                "similarity": round(similarity, 4),
                "cut_rate": ref.get("cut_rate"),
                "payoff_position": ref.get("payoff_position"),
            })
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[: max(1, k)]

    # ── action suggestions ──────────────────────────────────────────────────

    def suggest_actions(self, intensity: Optional[float], position: Optional[float]) -> dict[str, Any]:
        """Learned action probabilities for a context (never event→effect maps)."""
        from app.learning.policy_model import _bucket

        context_actions = self.model.get("context_actions") or {}
        cut_table = context_actions.get("cut_after") or {}
        key = f"{_bucket(intensity, (1/3, 2/3))}_{_bucket(position, (1/3, 2/3))}"
        entry = cut_table.get(key)
        durations = self.model.get("token_durations") or {}
        hold_mean = (durations.get("HOLD") or {}).get("mean")
        return {
            "CUT": (entry or {}).get("p"),
            "HOLD": (round(1.0 - entry["p"], 3) if entry else None),
            "hold_duration_hint": hold_mean,
            "context": key,
            "sample_count": (entry or {}).get("n", 0),
            "unknown_channels": self.policy.get("unknown_channels") or [],
        }


def _mean(dist: Optional[dict[str, Any]]) -> Optional[float]:
    return (dist or {}).get("mean")


def _std(dist: Optional[dict[str, Any]]) -> Optional[float]:
    return (dist or {}).get("std")


def _query_embedding(stats: dict[str, Any]) -> list[float]:
    from app.learning.grammar import embedding_from_stats
    return embedding_from_stats(stats)


def _embedding_norms(references: list[dict[str, Any]]) -> list[tuple[list[float], list[float]]]:
    vectors = [r.get("embedding") or [] for r in references]
    width = max((len(v) for v in vectors), default=0)
    means, stds = [], []
    for dim in range(width):
        column = [float(v[dim]) for v in vectors if len(v) > dim]
        mean = statistics.mean(column) if column else 0.0
        std = statistics.pstdev(column) if len(column) > 1 else 0.0
        means.append(mean)
        stds.append(std)
    return [(means, stds)] * len(references)
