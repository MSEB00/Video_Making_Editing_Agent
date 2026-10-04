"""Report held-out test metrics for the newest locally trained model."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.reference_pipeline import ReferenceTrainingPipeline


def main() -> int:
    pipeline = ReferenceTrainingPipeline()
    versions = sorted(pipeline.patterns.glob("style_model_v*.json"))
    if not versions:
        print(json.dumps({"status": "no_trained_model"}, indent=2))
        return 2
    candidate = json.loads(versions[-1].read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "evaluated",
        "version": candidate.get("version"),
        "test_count": candidate.get("dataset", {}).get("test_count", 0),
        "test_mean_absolute_error": candidate.get("validation", {}).get("test_mean_absolute_error"),
        "promoted": (pipeline.patterns / "active.json").is_file()
        and json.loads((pipeline.patterns / "active.json").read_text(encoding="utf-8")).get("version") == candidate.get("version"),
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())