"""Train an offline editing-style candidate from the local reference dataset."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.reference_pipeline import ReferenceTrainingPipeline


def main() -> int:
    result = ReferenceTrainingPipeline().train_candidate()
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result.get("status") in {"promoted", "rejected"} else 2


if __name__ == "__main__":
    raise SystemExit(main())