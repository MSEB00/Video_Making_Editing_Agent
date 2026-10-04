"""Learn aggregate short-form editing characteristics from annotated timelines."""
from __future__ import annotations

import json
import os
import pathlib
import statistics
from collections import defaultdict
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
        for example in unique.values():
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
            video = {"features": features, "moments": example.get("moments", [])}
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
