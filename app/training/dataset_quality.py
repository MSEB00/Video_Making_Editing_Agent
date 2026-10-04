"""Offline dataset checks and deterministic creator-separated split helpers."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import Counter
from typing import Any


SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def split_by_creator(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        creator = str(example.get("creator_group") or example.get("reference_id") or "")
        groups.setdefault(creator, []).append(example)
    total = len(examples)
    desired = _target_counts(total)
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_RATIOS}
    counts = {name: 0 for name in SPLIT_RATIOS}
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
    )
    for _, items in sorted_groups:
        destination = max(
            SPLIT_RATIOS,
            key=lambda name: (desired[name] - counts[name], SPLIT_RATIOS[name]),
        )
        result[destination].extend(items)
        counts[destination] += len(items)
    return result


def dataset_quality_report(
    root: pathlib.Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = pathlib.Path(root)
    manifest_path = root / "dataset.json"
    if manifest is None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {"items": []}
    items = manifest.get("items", [])
    categories = Counter()
    creators = Counter()
    licenses = Counter()
    durations = []
    widths = []
    heights = []
    views = []
    formats = Counter()
    seen_hashes: set[str] = set()
    duplicate_count = 0
    corrupt_count = 0
    missing_metadata = 0
    missing_files = 0
    for item in items:
        source = item.get("source", {})
        categories[str(source.get("category") or "unknown")] += 1
        creators[str(source.get("creator") or "unknown")] += 1
        licenses[str(source.get("license") or source.get("rights_basis") or "unknown")] += 1
        digest = str(item.get("sha256") or "")
        if digest and digest in seen_hashes:
            duplicate_count += 1
        elif digest:
            seen_hashes.add(digest)
        file_path = root / str(item.get("file") or "")
        if not file_path.is_file():
            missing_files += 1
        elif digest and _sha256(file_path) != digest:
            corrupt_count += 1
        metadata_path = root / str(item.get("metadata_path") or "")
        if not metadata_path.is_file():
            missing_metadata += 1
        feature_path = root / str(item.get("features_path") or "")
        try:
            features = json.loads(feature_path.read_text(encoding="utf-8")).get("features", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            features = {}
            missing_metadata += 1
        for field, output in (("duration", durations), ("width", widths), ("height", heights)):
            try:
                value = float(features[field])
                if math.isfinite(value) and value > 0:
                    output.append(value)
            except (KeyError, TypeError, ValueError):
                continue
        video_format = features.get("format")
        if video_format:
            formats[str(video_format)] += 1
        try:
            count = source.get("view_count")
            if count is not None:
                views.append(float(count))
        except (TypeError, ValueError):
            pass
    total = len(items)
    return {
        "dataset_name": manifest.get("dataset_name", "short_form_gaming_cc"),
        "dataset_version": manifest.get("version", "v001"),
        "total_videos": total,
        "videos_per_category": dict(categories),
        "videos_per_creator": dict(creators),
        "average_duration_seconds": round(sum(durations) / len(durations), 3) if durations else None,
        "average_width": round(sum(widths) / len(widths), 1) if widths else None,
        "average_height": round(sum(heights) / len(heights), 1) if heights else None,
        "format_distribution": dict(formats),
        "average_view_count": round(sum(views) / len(views), 1) if views else None,
        "license_distribution": dict(licenses),
        "duplicate_count": duplicate_count,
        "corrupt_file_count": corrupt_count,
        "missing_file_count": missing_files,
        "missing_metadata_count": missing_metadata,
        "category_imbalance": len(categories) > 1 and max(categories.values(), default=0) / max(1, total) > 0.7,
    }


def write_dataset_report(root: pathlib.Path) -> dict[str, Any]:
    root = pathlib.Path(root)
    report = dataset_quality_report(root)
    destination = root / "dataset_report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def _target_counts(total: int) -> dict[str, int]:
    counts = {name: int(total * ratio) for name, ratio in SPLIT_RATIOS.items()}
    remainder = total - sum(counts.values())
    for name in sorted(SPLIT_RATIOS, key=SPLIT_RATIOS.get, reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()