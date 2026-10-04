"""Build creator-separated local JSONL partitions from imported references."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.dataset_quality import split_by_creator
from app.training.reference_pipeline import ReferenceTrainingPipeline


def main() -> int:
    pipeline = ReferenceTrainingPipeline()
    examples = pipeline._read_examples()
    unique = {item.get("reference_id"): item for item in examples if item.get("reference_id")}
    partitions = split_by_creator(list(unique.values()))
    output = pipeline.datasets / "dataset_v001"
    output.mkdir(parents=True, exist_ok=True)
    for name, items in partitions.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in items), encoding="utf-8")
    split_manifest = {
        "dataset_version": "v001",
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "counts": {name: len(items) for name, items in partitions.items()},
        "creator_groups": {
            name: sorted({str(item.get("creator_group") or item.get("reference_id")) for item in items})
            for name, items in partitions.items()
        },
        "files": {name: f"datasets/dataset_v001/{name}.jsonl" for name in partitions},
    }
    (pipeline.datasets / "splits_v001.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(split_manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())