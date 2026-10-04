"""Duration/format priors derived from collected YouTube research metadata.

Metadata-only by design: aggregates duration distributions from the
discovered candidate pool (training/candidates.json, populated via the
official YouTube Data API). No media is accessed, stored, or analyzed.
Gives planners real online short-form context (e.g. "valorant shorts
median 34 s, p25-p75 22-51 s") without ever touching copyrighted content.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = PROJECT_ROOT / "training" / "candidates.json"
SHORT_FORM_MAX_SECONDS = 180.0


def _summarize(values: list[float]) -> dict[str, float] | None:
    if len(values) < 3:
        return None
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "count": len(ordered),
        "median_seconds": round(statistics.median(ordered), 1),
        "p25_seconds": round(quartiles[0], 1),
        "p75_seconds": round(quartiles[2], 1),
        "min_seconds": round(ordered[0], 1),
        "max_seconds": round(ordered[-1], 1),
    }


def youtube_duration_priors(candidates_path: pathlib.Path | None = None) -> dict[str, Any]:
    """Return per-category and overall duration priors from the candidate pool.

    Empty dict when no pool exists — callers must treat "no priors" as a
    valid outcome and never fabricate statistics.
    """
    path = pathlib.Path(candidates_path or DEFAULT_CANDIDATES)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    by_category: dict[str, list[float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        duration = item.get("duration_seconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            continue
        if duration <= 0 or duration > SHORT_FORM_MAX_SECONDS:
            continue
        by_category.setdefault(str(item.get("category") or "gaming"), []).append(float(duration))
    by_category_summary = {
        category: summary
        for category, summary in (
            (category, _summarize(values)) for category, values in sorted(by_category.items())
        )
        if summary
    }
    overall = _summarize([value for values in by_category.values() for value in values])
    if not by_category_summary and overall is None:
        return {}
    priors: dict[str, Any] = {
        "source": "youtube_data_api_metadata_only",
        "media_accessed": False,
        "by_category": by_category_summary,
    }
    if overall is not None:
        priors["overall"] = overall
    return priors
