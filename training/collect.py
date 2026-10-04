"""Collect and curate YouTube metadata candidates; never downloads media."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.training.collector import collect_candidate_pool


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=pathlib.Path, default=ROOT / "config" / "training_queries.yaml")
    parser.add_argument("--target", type=int)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--max-results", type=int)
    parser.add_argument("--license", choices=("creativeCommon", "youtube", "any"))
    parser.add_argument("--order", choices=("viewCount", "date"))
    parser.add_argument("--dry-run", action="store_true", help="Only discover and display metadata; acquisition is never performed by this command.")
    args = parser.parse_args()
    pool = collect_candidate_pool(
        config_path=args.queries,
        query_limit=args.query_limit,
        results_per_query=args.max_results,
        target_size=args.target,
        license_filter=args.license,
        order=args.order,
    )
    print(json.dumps({
        "mode": "metadata-only dry run" if args.dry_run else "metadata-only collection",
        "candidate_count": len(pool["items"]),
        "queries": pool["queries_this_run"],
        "candidates": [{
            "title": item.get("title"),
            "channel": item.get("channel"),
            "views": item.get("view_count"),
            "license": item.get("license"),
            "category": item.get("category"),
            "curation_score": item.get("curation_score"),
            "acquired": False,
            "url": item.get("url"),
        } for item in pool["items"]],
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())