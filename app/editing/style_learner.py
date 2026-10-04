"""Learn aggregate short-form editing characteristics from annotated timelines."""
from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any


DEFAULT_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "data" / "editing_examples.jsonl"
LEARNED_FEATURES = (
    "cut_density",
    "caption_density",
    "visual_emphasis",
    "dead_space_tolerance",
    "early_payoff_seconds",
    "average_shot_duration",
    "effect_density",
    "sfx_density",
    "audio_intensity",
)
MIN_REFERENCE_EXAMPLES = 5


class EditingStyleLearner:
    """Summarize generalized timeline patterns without retaining creator identity."""

    def __init__(
        self,
        examples_path: pathlib.Path | None = None,
        active_model_path: pathlib.Path | None = None,
        promoted_only: bool = False,
    ) -> None:
        self.examples_path = pathlib.Path(
            examples_path or os.getenv("EDITING_EXAMPLES_PATH") or DEFAULT_EXAMPLES
        )
        project_root = pathlib.Path(__file__).resolve().parents[2]
        self.active_model_path = pathlib.Path(
            active_model_path or project_root / "training" / "models" / "active.json"
        )
        self.legacy_active_model_path = (
            project_root / "training" / "youtube" / "patterns" / "active.json"
            if active_model_path is None else None
        )
        self.promoted_only = promoted_only

    def learn(self, platform: str) -> dict[str, Any]:
        promoted = self._load_promoted_profile(platform)
        if promoted is not None:
            return promoted
        if self.promoted_only:
            profile = self._empty_profile(platform)
            profile["training_status"] = "no_promoted_model"
            return profile

        examples: dict[str, dict[str, Any]] = {}
        if not self.examples_path.is_file():
            return self._empty_profile(platform)

        with self.examples_path.open("r", encoding="utf-8") as examples_file:
            for line in examples_file:
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(example, dict):
                    continue
                if example.get("platform") not in (platform, "all"):
                    continue
                reference_id = str(example.get("reference_id") or example.get("video") or _stable_id(example))
                examples.setdefault(reference_id, example)

        if len(examples) < MIN_REFERENCE_EXAMPLES:
            profile = self._empty_profile(platform)
            profile.update({
                "example_count": len(examples),
                "training_status": "insufficient_references",
                "minimum_examples": MIN_REFERENCE_EXAMPLES,
            })
            return profile
        return self.build_profile(platform, list(examples.values()))

    def build_profile(self, platform: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build per-video aggregates; references with many segments do not dominate."""
        unique = {
            str(item.get("reference_id") or item.get("video") or _stable_id(item)): item
            for item in examples
            if item.get("platform") in (platform, "all")
        }
        if len(unique) < MIN_REFERENCE_EXAMPLES:
            profile = self._empty_profile(platform)
            profile.update({
                "example_count": len(unique),
                "training_status": "insufficient_references",
                "minimum_examples": MIN_REFERENCE_EXAMPLES,
            })
            return profile

        by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
        per_video: list[dict[str, Any]] = []
        creators: set[str] = set()
        for reference_id, example in unique.items():
            creator = example.get("creator_group")
            if creator:
                creators.add(str(creator))
            style_tags = example.get("style_tags") or ["general"]
            if isinstance(style_tags, str):
                style_tags = [style_tags]
            features = dict(example.get("features") or {})
            legacy_features = self._legacy_features(example)
            for name, value in legacy_features.items():
                features.setdefault(name, value)
            if "average_shot_duration" not in features:
                durations = self._shot_durations(example)
                if durations:
                    features["average_shot_duration"] = statistics.mean(durations)
            video = {
                "reference_id": reference_id,
                "features": features,
                "moments": example.get("moments", []),
                "style_tags": [str(tag) for tag in style_tags],
            }
            per_video.append(video)
            for style in style_tags:
                by_style[str(style)].append(video)

        measurements: dict[str, list[float]] = defaultdict(list)
        for video in per_video:
            for name, value in video["features"].items():
                numeric = _number(value)
                if numeric is not None:
                    measurements[name].append(numeric)
        profile = {
            "platform": platform,
            "example_count": len(unique),
            "creator_count": len(creators),
            "training_status": "learned",
            "learned_averages": {
                feature: round(statistics.mean(values), 3)
                for feature, values in measurements.items() if values
            },
            "style_profiles": {
                style: self._average_features(items)
                for style, items in sorted(by_style.items())
            },
            "relationships": self._learn_relationships(per_video),
            "conditional_edit_probabilities": self._learn_conditional_probabilities(per_video),
            "style_clusters": self._discover_style_clusters(per_video),
        }
        return profile

    @staticmethod
    def _empty_profile(platform: str) -> dict[str, Any]:
        return {
            "platform": platform,
            "example_count": 0,
            "training_status": "no_references",
            "learned_averages": {},
            "style_profiles": {},
            "relationships": [],
        }

    def _load_promoted_profile(self, platform: str) -> dict[str, Any] | None:
        model_paths = [self.active_model_path]
        if self.legacy_active_model_path:
            model_paths.append(self.legacy_active_model_path)
        for model_path in model_paths:
            if not model_path.is_file():
                continue
            try:
                model = json.loads(model_path.read_text(encoding="utf-8"))
                profile = model.get("profiles", {}).get(platform)
                if isinstance(profile, dict):
                    return {
                        **profile,
                        "model_version": model.get("version"),
                        "feedback_signals": model.get("feedback_signals", {}),
                    }
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
        return None

    @staticmethod
    def _legacy_features(example: dict[str, Any]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for segment in example.get("segments", []):
            for feature in LEARNED_FEATURES:
                value = _number(segment.get(feature))
                if value is not None:
                    values[feature].append(value)
            start, end = _number(segment.get("start")), _number(segment.get("end"))
            if start is not None and end is not None and end > start:
                values["average_shot_duration"].append(end - start)
            if segment.get("type") == "payoff" and start is not None:
                values["early_payoff_seconds"].append(start)
        return {name: statistics.mean(items) for name, items in values.items() if items}

    @staticmethod
    def _shot_durations(example: dict[str, Any]) -> list[float]:
        durations = []
        for segment in example.get("segments", []):
            start, end = _number(segment.get("start")), _number(segment.get("end"))
            if start is not None and end is not None and end > start:
                durations.append(end - start)
        return durations

    @staticmethod
    def _average_features(videos: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for video in videos:
            for name, value in video["features"].items():
                numeric = _number(value)
                if numeric is not None:
                    values[name].append(numeric)
        return {
            name: round(statistics.mean(items), 3)
            for name, items in values.items() if items
        }

    # ── Context-conditional editing distributions (§ generalized patterns) ──

    CONTEXT_AXES = (("visual_intensity", 0.34, 0.67), ("audio_intensity", 0.34, 0.67))

    @classmethod
    def _learn_conditional_probabilities(cls, videos: list[dict[str, Any]]) -> dict[str, Any]:
        """Estimate P(edit decision | content context) from annotated moments.

        Buckets context features (visual/audio intensity, speech presence) and
        counts observed vs not_observed editing decisions per bucket. Uses
        Laplace smoothing; 'not_observed' (NO EFFECT) is a first-class outcome.
        """
        counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"observed": 0, "not_observed": 0})
        )
        for video in videos:
            for moment in video.get("moments", []):
                context = moment.get("moment", {}) or {}
                edits = moment.get("observed_edit", {}) or {}
                decisions = {
                    name: (1.0 if value >= 0.5 else 0.0)
                    for name, value in edits.items()
                    if _number(value) is not None
                }
                if not decisions:
                    continue
                keys = ["all"]
                for axis, low_cut, high_cut in cls.CONTEXT_AXES:
                    value = _number(context.get(axis))
                    if value is not None:
                        bucket = "low" if value < low_cut else "medium" if value < high_cut else "high"
                        keys.append(f"{axis}={bucket}")
                speech = context.get("speech_present")
                if isinstance(speech, bool):
                    keys.append(f"speech_present={'true' if speech else 'false'}")
                elif _number(speech) is not None:
                    keys.append(f"speech_present={'true' if float(speech) >= 0.5 else 'false'}")
                for key in keys:
                    for name, outcome in decisions.items():
                        state = "observed" if outcome >= 0.5 else "not_observed"
                        counts[key][name][state] += 1
        result: dict[str, Any] = {}
        for key, decision_counts in sorted(counts.items()):
            result[key] = {
                decision: {
                    "p_observed": round(
                        (states["observed"] + 1) / (states["observed"] + states["not_observed"] + 2), 3
                    ),
                    "observed": states["observed"],
                    "not_observed": states["not_observed"],
                    "samples": states["observed"] + states["not_observed"],
                }
                for decision, states in sorted(decision_counts.items())
            }
        return result

    # ── Data-driven style cluster discovery (§ multiple styles) ──────────────

    CLUSTER_MIN_REFERENCES = 6
    CLUSTER_MIN_SIZE = 2
    CLUSTER_MAX_K = 4

    @classmethod
    def _discover_style_clusters(cls, videos: list[dict[str, Any]]) -> dict[str, Any]:
        """Deterministic k-means over normalized per-video feature vectors.

        Labels are derived from each centroid's largest deviations from the
        global mean (no predefined style vocabulary). Returns an honest
        'insufficient' status when the dataset cannot support clustering.
        """
        n = len(videos)
        if n < cls.CLUSTER_MIN_REFERENCES:
            return {
                "status": "insufficient_references",
                "reference_count": n,
                "minimum_references": cls.CLUSTER_MIN_REFERENCES,
                "clusters": [],
            }
        names, matrix = cls._feature_matrix(videos)
        if not names:
            return {"status": "no_numeric_features", "reference_count": n, "clusters": []}
        normalized, means, spreads = _normalize_columns(matrix)
        best: tuple[float, int, list[int], float] | None = None  # silhouette, k, assignment, inertia
        for k in range(2, min(cls.CLUSTER_MAX_K, n // cls.CLUSTER_MIN_SIZE) + 1):
            assignment, inertia = _kmeans(normalized, k)
            sizes = [assignment.count(cluster) for cluster in range(k)]
            if min(sizes) < cls.CLUSTER_MIN_SIZE:
                continue
            silhouette = _mean_silhouette(normalized, assignment)
            # Highest silhouette wins; ties prefer the simpler (smaller k) model.
            if best is None or (silhouette, -k) > (best[0], -best[1]):
                best = (silhouette, k, assignment, inertia)
        if best is None:
            return {"status": "insufficient_diversity", "reference_count": n, "clusters": []}
        silhouette, k, assignment, inertia = best
        score = inertia / (n * len(names))
        clusters = []
        for cluster in range(k):
            member_rows = [row for index, row in enumerate(matrix) if assignment[index] == cluster]
            centroid = [statistics.mean(column) for column in zip(*member_rows)]
            centroid_norm = [
                (value - means[dim]) / spreads[dim] if spreads[dim] else 0.0
                for dim, value in enumerate(centroid)
            ]
            deviations = sorted(
                (
                    {"feature": names[dim], "centroid": round(centroid[dim], 3),
                     "deviation": round(centroid_norm[dim], 3),
                     "direction": "high" if centroid_norm[dim] >= 0 else "low"}
                    for dim in range(len(names))
                ),
                key=lambda item: abs(item["deviation"]),
                reverse=True,
            )[:3]
            tag_counts: Counter[str] = Counter()
            for index, video in enumerate(videos):
                if assignment[index] == cluster:
                    tag_counts.update(video.get("style_tags") or ["general"])
            clusters.append({
                "cluster_id": cluster,
                "size": len(member_rows),
                "label": ", ".join(f"{item['direction']} {item['feature']}" for item in deviations) or "mixed",
                "centroid": {names[dim]: round(value, 3) for dim, value in enumerate(centroid)},
                "distinctive_features": deviations,
                "dominant_tags": [tag for tag, _ in tag_counts.most_common(3)],
                "member_reference_ids": sorted(
                    str(videos[index].get("reference_id") or index)
                    for index in range(n) if assignment[index] == cluster
                ),
            })
        return {
            "status": "learned",
            "reference_count": n,
            "cluster_count": k,
            "mean_silhouette": round(silhouette, 3),
            "normalized_inertia": round(score, 4),
            "features_used": names,
            "clusters": clusters,
        }

    @staticmethod
    def _feature_matrix(videos: list[dict[str, Any]]) -> tuple[list[str], list[list[float]]]:
        """Build a numeric matrix from features present in >=70% of references."""
        presence: Counter[str] = Counter()
        for video in videos:
            for name, value in (video.get("features") or {}).items():
                if _number(value) is not None and not isinstance(value, bool):
                    presence[name] += 1
        threshold = max(2, int(len(videos) * 0.7))
        names = sorted(name for name, count in presence.items() if count >= threshold)
        if not names:
            return [], []
        column_values = {
            name: [_number((video.get("features") or {}).get(name)) for video in videos]
            for name in names
        }
        means = {
            name: statistics.mean([v for v in values if v is not None])
            for name, values in column_values.items()
            if any(v is not None for v in values)
        }
        matrix = [
            [
                column_values[name][index] if column_values[name][index] is not None
                else means.get(name, 0.0)
                for name in names
            ]
            for index in range(len(videos))
        ]
        return names, matrix

    @staticmethod
    def _learn_relationships(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for video in videos:
            for moment in video["moments"]:
                inputs = moment.get("moment", {})
                edits = moment.get("observed_edit", {})
                for input_name, input_value in inputs.items():
                    for edit_name, edit_value in edits.items():
                        left, right = _number(input_value), _number(edit_value)
                        if left is not None and right is not None:
                            pairs[(input_name, edit_name)].append((left, right))
        relationships = []
        for (input_name, edit_name), values in pairs.items():
            if len(values) < 3:
                continue
            left_values, right_values = zip(*values)
            correlation = _correlation(left_values, right_values)
            if correlation is not None:
                relationships.append({
                    "moment_feature": input_name,
                    "editing_response": edit_name,
                    "sample_count": len(values),
                    "correlation": round(correlation, 3),
                })
        return sorted(relationships, key=lambda item: abs(item["correlation"]), reverse=True)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_id(example: dict[str, Any]) -> str:
    return str(hash(json.dumps(example, sort_keys=True, ensure_ascii=True)))


def _correlation(left: Any, right: Any) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    try:
        return statistics.correlation(left, right)
    except (statistics.StatisticsError, ValueError):
        return None


def _normalize_columns(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Min-max normalize columns; returns (normalized rows, means, spreads)."""
    if not matrix:
        return [], [], []
    transposed = list(zip(*matrix))
    means: list[float] = []
    spreads: list[float] = []
    normalized_columns: list[list[float]] = []
    for column in transposed:
        low, high = min(column), max(column)
        mean = statistics.mean(column)
        spread = high - low
        means.append(mean)
        spreads.append(spread)
        normalized_columns.append(
            [(value - low) / spread if spread else 0.0 for value in column]
        )
    normalized = [list(row) for row in zip(*normalized_columns)]
    return normalized, means, spreads


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _kmeans(rows: list[list[float]], k: int, max_iterations: int = 50) -> tuple[list[int], float]:
    """Deterministic k-means: farthest-point init + Lloyd iterations.

    Returns (cluster assignment per row, total squared inertia).
    """
    n = len(rows)
    mean_point = [statistics.mean(column) for column in zip(*rows)]
    first = max(range(n), key=lambda index: _squared_distance(rows[index], mean_point))
    centers = [rows[first]]
    for _ in range(1, k):
        farthest = max(
            range(n),
            key=lambda index: min(_squared_distance(rows[index], center) for center in centers),
        )
        centers.append(rows[farthest])
    assignment = [0] * n
    for _ in range(max_iterations):
        changed = False
        for index, row in enumerate(rows):
            nearest = min(range(len(centers)), key=lambda c: _squared_distance(row, centers[c]))
            if nearest != assignment[index]:
                assignment[index] = nearest
                changed = True
        if not changed:
            break
        for cluster in range(len(centers)):
            members = [rows[index] for index in range(n) if assignment[index] == cluster]
            if members:
                centers[cluster] = [statistics.mean(column) for column in zip(*members)]
    inertia = sum(_squared_distance(rows[index], centers[assignment[index]]) for index in range(n))
    return assignment, inertia


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(_squared_distance(left, right))


def _mean_silhouette(rows: list[list[float]], assignment: list[int]) -> float:
    """Mean silhouette coefficient across all points (Euclidean)."""
    n = len(rows)
    k = max(assignment) + 1
    if k < 2 or n < 3:
        return -1.0
    members = {cluster: [index for index in range(n) if assignment[index] == cluster] for cluster in range(k)}
    total = 0.0
    for index in range(n):
        cluster = assignment[index]
        same = [other for other in members[cluster] if other != index]
        if not same:
            continue
        a = sum(_euclidean(rows[index], rows[other]) for other in same) / len(same)
        b = min(
            (
                sum(_euclidean(rows[index], rows[other]) for other in members[other_cluster])
                / len(members[other_cluster])
            )
            for other_cluster in range(k)
            if other_cluster != cluster and members[other_cluster]
        )
        denominator = max(a, b)
        total += (b - a) / denominator if denominator > 0 else 0.0
    return total / n
