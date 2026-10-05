"""Learned editing-policy planner — the primary local creative brain.

Priority chain in ShortFormCreativeEditor becomes (spec §46):
    injected remote plan (web-chat)  →  LearnedPolicyPlanner (this module,
    when a policy is trained)  →  local_feature_fallback (safety net, with
    an explicit reason).

The planner composes measured evidence (kill events, motion, audio) into a
candidate moment graph, generates structural hypotheses, and lets the LEARNED
policy score them — replacing "highest activity wins". Music requirements are
derived per-edit from the selected structure's measured shape and the learned
pacing distribution; caption/SFX/emphasis channels that references cannot
teach yet are decided as NONE and reported honestly (spec §23–26).
"""
from __future__ import annotations

import statistics
from typing import Any, Optional

from app.learning.grammar import label_segments
from app.learning.moment_graph import build_moment_graph
from app.learning.policy_inference import PolicyInference
from app.learning.policy_model import PolicyTrainer, learning_state
from app.learning.sequence_dataset import SequenceDataset
from app.learning.structure_search import generate_structures, score_and_select
from app.utilities.logger import get_logger

log = get_logger(__name__)


def get_learning_state(grammar_root=None, policy_root=None) -> str:
    dataset = SequenceDataset(grammar_root)
    trainer = PolicyTrainer(policy_root)
    return learning_state(len(dataset.records()), trainer)


class LearnedPolicyPlanner:
    """Drop-in creative model backed by the learned editing policy."""

    planning_brain = "learned_policy"

    def __init__(self, grammar_root=None, policy_root=None) -> None:
        self.dataset = SequenceDataset(grammar_root)
        self.trainer = PolicyTrainer(policy_root)
        policy = self.trainer.load_active()
        self.policy = policy
        self.inference = PolicyInference(policy) if policy else None
        self.state = learning_state(len(self.dataset.records()), self.trainer)
        self.model = f"policy:{(policy or {}).get('version', 'untrained')}"
        self.diagnostics: dict[str, Any] = {"state": self.state}
        self._ranked: list[dict[str, Any]] = []
        self._last_selected_index: Optional[int] = None
        self._last_kwargs: dict[str, Any] = {}
        self._features: list[dict[str, Any]] = []

    # ── planning ────────────────────────────────────────────────────────────

    def create_plan(
        self,
        media_context: dict[str, Any],
        frames: list[dict[str, Any]],
        style_profile: dict[str, Any],
        platform: str,
        target_duration: int,
        user_request: str = "",
        user_preferences: Optional[dict[str, Any]] = None,
        available_sfx: Optional[list[dict[str, Any]]] = None,
        metadata_priors: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        import pathlib

        preferences = user_preferences or {}
        sources = media_context.get("sources") or []
        source_paths = [pathlib.Path(p) for p in (preferences.get("source_paths") or [])]

        # measured motion/audio activity per source (same analyzer the safety
        # fallback uses — measurement, not policy)
        self._features = self._analyze_sources(source_paths[: len(sources)])

        events_by_source: dict[int, list[dict[str, Any]]] = {}
        for source in sources:
            index = int(source.get("source_index", 0))
            events = source.get("gameplay_events") or []
            if events:
                events_by_source[index] = events

        moments = build_moment_graph(sources, self._features, events_by_source, float(target_duration))
        if not moments:
            raise RuntimeError("Policy planner found no candidate moments in the analyzed footage.")

        learned_shot = None
        if self.inference is not None:
            learned_shot = (self.inference.model.get("mean_shot_duration") or {}).get("mean")
        candidates = generate_structures(
            moments, len(sources), float(target_duration), learned_shot_duration=learned_shot
        )
        gameplay_stats = self._gameplay_stats(sources)
        scored, ranked, selected_index = score_and_select(
            candidates, self.inference, gameplay_stats, seed=seed
        )
        if not ranked:
            raise RuntimeError("Policy planner generated no viable structures.")
        selected = ranked[0]
        self._ranked = ranked
        self._last_selected_index = 0
        self._last_kwargs = {
            "platform": platform, "target_duration": float(target_duration),
            "user_request": user_request, "sources": sources,
        }

        plan = self._plan_from_candidate(selected, platform, float(target_duration), user_request)
        retrieved = (
            self.inference.retrieve_similar(gameplay_stats)
            if self.inference is not None else []
        )
        self.diagnostics = {
            "state": self.state,
            "policy_version": (self.policy or {}).get("version"),
            "dataset_version": ((self.policy or {}).get("dataset") or {}).get("dataset_version"),
            "retrieved_references": [r["reference_id"] for r in retrieved],
            "retrieved_count": len(retrieved),
            "policy_influence": 1.0 if self.inference is not None else 0.0,
            "candidate_structure_count": len(ranked),
            "selected_structure": selected["name"],
            "selected_score": selected.get("score", {}).get("composite"),
            "score_breakdown": selected.get("score", {}).get("components"),
            "runner_up": (ranked[1]["name"] if len(ranked) > 1 else None),
            "decision_count": len(plan["shots"]) + 2,  # shots + music + sfx decisions
            "no_effect_decisions": sum(
                1 for s in plan["shots"] if not s["visual_emphasis"] and not s["caption"]
            ) + (1 if not plan["sound_design"] else 0),
            "learned_action_sample": (
                self.inference.suggest_actions(0.7, 0.5) if self.inference else None
            ),
            "unknown_channels": (self.policy or {}).get("unknown_channels"),
        }
        log.info(
            "[POLICY] selected structure '%s' (score %.3f, %d candidates, state=%s)",
            selected["name"], selected.get("score", {}).get("composite", 0.0),
            len(ranked), self.state,
        )
        return plan

    def _plan_from_candidate(self, candidate: dict[str, Any], platform: str,
                             target_duration: float, user_request: str) -> dict[str, Any]:
        segments = [
            {"t": shot["start"], "duration": shot["end"] - shot["start"],
             "motion": shot.get("intensity"), "audio": None, "silence_ratio": None}
            for shot in candidate["shots"]
        ]
        labeled = label_segments(segments)
        role_map = {"HOOK": "hook", "PAYOFF": "payoff", "PEAK": "payoff", "END": "ending"}
        shots = []
        for shot, seg in zip(candidate["shots"], labeled):
            shots.append({
                "source_index": shot["source_index"],
                "start": round(float(shot["start"]), 3),
                "end": round(float(shot["end"]), 3),
                "role": role_map.get(seg["token"], "action"),
                "transition": "cut",  # transition types are an unknown channel — honest default
                "caption": None,      # captions require justification we don't fabricate
                "visual_emphasis": [],  # emphasis channel unknown from references → NONE
            })
        stats = {
            "cut_rate": candidate.get("score", {}).get("cut_rate"),
            "mean_shot_duration": candidate.get("score", {}).get("mean_shot_duration"),
        }
        return {
            "platform": platform,
            "strategy": candidate["name"],
            "rationale": (
                f"Learned policy selected '{candidate['name']}' from "
                f"{self.diagnostics.get('candidate_structure_count', 0) if self.diagnostics else 0} "
                f"structural hypotheses by learned transition likelihood, pacing fit, hook and "
                f"payoff-position fit against {((self.policy or {}).get('dataset') or {}).get('sequence_count', 0)} "
                f"reference sequences. User request: {user_request[:200] or '(none)'}."
            ),
            "target_duration": round(sum(s["end"] - s["start"] for s in shots), 3) or target_duration,
            "alternatives": [
                {"strategy": item["name"], "score": item.get("score", {}).get("composite")}
                for item in (self._ranked or [])[1:4]
            ],
            "shots": shots,
            "music_requirements": self._derive_music_requirements(stats, target_duration, labeled),
            "music_mix": {},
            "sound_design": [],
            "ending": "Resolve on the final selected moment.",
            "policy_diagnostics": {
                "state": self.state,
                "policy_version": (self.policy or {}).get("version"),
            },
        }

    def _derive_music_requirements(self, stats: dict[str, Any], target_duration: float,
                                   labeled: list[dict[str, Any]]) -> dict[str, Any]:
        """Music requirements derived per-edit from measured structure + learned pacing.

        Not a fixed genre map: tempo follows where this edit's cut rate sits in
        the LEARNED reference distribution; mood follows this edit's intensity
        curve shape. BGM-free edits remain possible (empty requirements when
        the structure is silence-driven).
        """
        cut_rate = stats.get("cut_rate")
        learned = (self.inference.model.get("cut_rate") if self.inference else None) or {}
        mean, std = learned.get("mean"), learned.get("std") or 0.3
        if cut_rate is None or mean is None:
            speed = ["medium", "high"]
        else:
            z = (cut_rate - mean) / max(std, 0.05)
            speed = ["high", "veryhigh"] if z > 0.5 else ["low", "medium"] if z < -0.5 else ["medium", "high"]
        intensities = [seg["intensity"] for seg in labeled] or [0.5]
        first_half = statistics.mean(intensities[: max(1, len(intensities) // 2)])
        second_half = statistics.mean(intensities[max(1, len(intensities) // 2):])
        level = statistics.mean(intensities)
        silence_share = (self.inference.model.get("silence_use_rate") if self.inference else None) or 0.0
        if level < 0.25 and silence_share > 0.3:
            return {}  # learned references use silence; this edit is quiet → NO BGM
        if second_half - first_half > 0.12:
            search, tags = "driving instrumental with a rising build", ["cinematic", "instrumental", "build"]
        elif first_half - second_half > 0.12:
            search, tags = "instrumental that resolves from intense to calm", ["instrumental", "downtempo"]
        elif level >= 0.55:
            search, tags = "energetic electronic instrumental with driving rhythm", ["energetic", "electronic", "instrumental"]
        else:
            search, tags = "restrained atmospheric instrumental with gradual build", ["ambient", "instrumental"]
        return {
            "search": search,
            "tags": tags,
            "speed": speed,
            "instrumental": True,
            "duration_min": max(20, int(target_duration * 0.7)),
            "duration_max": max(600, int(target_duration * 3)),
        }

    # ── music ranking (delegates to measured local ranker) ─────────────────

    def rank_music(self, media_context: dict[str, Any], plan: dict[str, Any],
                   candidates: list[dict[str, Any]]) -> dict[str, Any]:
        from app.ai.local_editing import LocalShortFormEditingModel

        local = LocalShortFormEditingModel()
        local.features = self._features
        return local.rank_music(media_context, plan, candidates)

    # ── structured policy-aware critic (spec §38/§19) ──────────────────────

    def review_render(self, plan: dict[str, Any], review_context: dict[str, Any],
                      frames: list[dict[str, Any]]) -> dict[str, Any]:
        shots = plan.get("shots") or []
        render_source = (review_context.get("sources") or [{}])[0]
        model = (self.inference.model if self.inference else {}) or {}
        metrics: dict[str, Any] = {}
        failure_tags: list[str] = []

        total = sum(max(0.05, float(s["end"]) - float(s["start"])) for s in shots) or 1.0
        cut_rate = max(0, len(shots) - 1) / total
        learned_cut = model.get("cut_rate") or {}
        learned_shot = model.get("mean_shot_duration") or {}

        # raw-gameplay likeness (the job-82..91 failure mode) — measured, not vibes
        raw_like = False
        if learned_cut.get("mean") is not None:
            threshold = max(0.05, learned_cut["mean"] - 1.5 * float(learned_cut.get("std") or learned_cut["mean"] * 0.5))
            raw_like = cut_rate < threshold
        if len(shots) == 1 and total > 8.0:
            raw_like = True
        metrics["raw_gameplay_like"] = raw_like
        if raw_like:
            failure_tags += ["raw_gameplay_like", "cut_density_too_low", "structure_too_continuous"]

        from app.learning.policy_inference import _gaussian_score
        metrics["pacing"] = _gaussian_score(cut_rate, learned_cut.get("mean"), learned_cut.get("std"))
        metrics["hook"] = _gaussian_score(
            shots[0]["end"] - shots[0]["start"] if shots else None,
            (model.get("hook_duration") or {}).get("mean"),
            (model.get("hook_duration") or {}).get("std"),
        )
        payoff_positions = [
            (i + 0.5) / len(shots) for i, s in enumerate(shots) if s.get("role") == "payoff"
        ]
        metrics["payoff_placement"] = _gaussian_score(
            statistics.mean(payoff_positions) if payoff_positions else None,
            (model.get("payoff_position") or {}).get("mean"),
            (model.get("payoff_position") or {}).get("std"),
        )
        long_threshold = 2.0 * float(learned_shot.get("mean") or 5.0)
        dead = sum(1 for s in shots if (s["end"] - s["start"]) > long_threshold)
        metrics["dead_time_share"] = round(dead / len(shots), 3) if shots else 0.0
        if metrics["dead_time_share"] > 0.3:
            failure_tags.append("dead_time")
        if metrics["pacing"] < 0.4:
            failure_tags.append("pacing_off_reference")
        if metrics["hook"] < 0.4:
            failure_tags.append("weak_hook")

        effects = len(plan.get("sound_design") or []) + sum(
            1 for s in shots if s.get("caption") or s.get("visual_emphasis")
        )
        per_minute = effects / (total / 60.0) if total else 0.0
        metrics["effect_restraint"] = round(max(0.0, 1.0 - max(0.0, per_minute - 6.0) / 6.0), 3)
        if metrics["effect_restraint"] < 0.5:
            failure_tags.append("over_edited")
        switches = sum(1 for a, b in zip(shots, shots[1:]) if a["source_index"] != b["source_index"])
        ratio = switches / max(1, len(shots) - 1) if len(shots) > 1 else 0.0
        metrics["coherence"] = round(1.0 - abs(ratio - 0.55) / 0.55, 3)

        output_ok = bool(render_source.get("has_audio")) and float(render_source.get("duration") or 0) > 0
        metrics["output_valid"] = output_ok
        if not output_ok:
            failure_tags.append("invalid_render")
        if not frames:
            failure_tags.append("no_review_frames")

        numeric = [v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "dead_time_share"]
        critic_score = round(statistics.mean(numeric), 3) if numeric else 0.0
        better_exists = any(
            item["name"] != plan.get("strategy")
            and item.get("score", {}).get("composite", 0) > (plan.get("policy_score") or 0) + 0.05
            for item in self._ranked
        )
        needs_revision = bool(output_ok) and critic_score < 0.45 and better_exists and bool(self._ranked)
        critique_parts = [f"{key}={value}" for key, value in metrics.items() if not isinstance(value, bool)]
        return {
            "needs_revision": needs_revision,
            "critique": "Structured policy critique: " + ", ".join(critique_parts) +
                        (f"; flags: {','.join(failure_tags)}" if failure_tags else ""),
            "revision_request": (
                f"Switch structure away from '{plan.get('strategy')}'; diagnosis: {','.join(failure_tags) or 'low composite'}."
                if needs_revision else ""
            ),
            "structured": metrics,
            "failure_tags": failure_tags,
            "critic_score": critic_score,
        }

    def revise_plan(self, plan: dict[str, Any], critique: dict[str, Any] | None = None,
                    **_: Any) -> dict[str, Any]:
        """Deterministic diagnosis-driven revision: take the next-best structure."""
        current = plan.get("strategy")
        for item in self._ranked:
            if item["name"] != current:
                kwargs = self._last_kwargs
                return self._plan_from_candidate(
                    item, kwargs.get("platform", "youtube_shorts"),
                    float(kwargs.get("target_duration") or 30), kwargs.get("user_request", ""),
                )
        return plan

    # ── measurement helpers ─────────────────────────────────────────────────

    @staticmethod
    def _analyze_sources(source_paths: list) -> list[dict[str, Any]]:
        from app.ai.local_editing import LocalShortFormEditingModel

        analyzer = LocalShortFormEditingModel()
        features = []
        for path in source_paths:
            try:
                features.append(analyzer._analyze_source(path))
            except Exception:
                features.append({"path": str(path), "motion": [], "audio": [], "activity": []})
        return features

    @staticmethod
    def _gameplay_stats(sources: list[dict[str, Any]]) -> dict[str, Any]:
        audio_values = []
        for source in sources:
            db = source.get("audio_mean_db")
            if isinstance(db, (int, float)):
                audio_values.append(max(0.0, min(1.0, (float(db) + 60.0) / 60.0)))
        durations = [float(s.get("duration") or 0.0) for s in sources]
        return {
            "mean_intensity": None,   # raw-footage intensity is per-moment, not global
            "audio_energy_mean": round(statistics.mean(audio_values), 4) if audio_values else None,
            "segment_count": len(sources),
            "mean_shot_duration": round(statistics.mean(durations), 3) if durations else None,
        }
