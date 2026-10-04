"""Create a local quality report for the imported training dataset."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.dataset_quality import write_dataset_report


def main() -> int:
    report = write_dataset_report(ROOT / "training")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())