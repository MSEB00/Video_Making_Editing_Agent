"""Quota-aware, metadata-only curation of Creative Commons YouTube candidates."""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
from typing import Any

import yaml

from app.research.youtube_research_agent import YouTubeResearchAgent


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "training_queries.yaml"
DEFAULT_CANDIDATES = PROJECT_ROOT / "training" / "candidates.json"
DEFAULT_STATE = PROJECT_ROOT / "training" / "collection_state.json"


def load_collection_config(path: pathlib.Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with pathlib.Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config.get("query_groups"), dict) or not config["query_groups"]:
        raise ValueError("training query config must define at least one query group.")
    return config


def curate_candidates(
    candidates: list[dict[str, Any]],
    query_categories: dict[str, str],
    max_per_channel: int,
    target_size: int,
    max_category_share: float = 0.35,
) -> list[dict[str, Any]]:
    """Deduplicate, score and cap by creator/category without treating views as quality."""
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        video_id = str(candidate.get("video_id") or "")
        if not video_id or candidate.get("license") != "creativeCommon":
            continue
        current = unique.get(video_id)
        if current is None:
            unique[video_id] = dict(candidate)
        else:
            queries = sorted(set(
                current.get("discovery_queries", []) + candidate.get("discovery_queries", [])
            ))
            current.update(candidate)
            current["discovery_queries"] = queries

    max_views = max((int(item.get("view_count") or 0) for item in unique.values()), default=0)
    ranked = []
    for item in unique.values():
        queries = item.get("discovery_queries") or []
        category_votes: dict[str, int] = {}
        for query in queries:
            category = query_categories.get(query, "gaming")
            category_votes[category] = category_votes.get(category, 0) + 1
        category = max(category_votes, key=category_votes.get) if category_votes else "gaming"
        query_text = " ".join(queries).lower()
        content_text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        query_terms = {term for term in query_text.split() if len(term) > 2}
        content_terms = {term.strip("#.,!?()[]{}\"'") for term in content_text.split()}
        relevance = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
        views = int(item.get("view_count") or 0)
        view_signal = math.log1p(views) / math.log1p(max_views) if max_views else 0.0
        duration = item.get("duration_seconds")
        format_signal = max(0.0, 1.0 - float(duration) / 240.0) if duration else 0.0
        item.update({
            "category": category,
            "query": queries[0] if queries else None,
            "relevance_score": round(relevance, 4),
            "view_signal": round(view_signal, 4),
            "format_signal": round(format_signal, 4),
            "curation_score": round(0.55 * relevance + 0.30 * view_signal + 0.15 * format_signal, 4),
            "license_url": None,
            "attribution_required": True,
            "media_acquired": False,
        })
        ranked.append(item)

    ranked.sort(key=lambda item: (item["curation_score"], item.get("view_count") or 0), reverse=True)
    category_cap = max(1, math.ceil(target_size * max_category_share))
    channel_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    selected = []
    for item in ranked:
        channel = str(item.get("channel_id") or "")
        category = str(item["category"])
        if channel_counts.get(channel, 0) >= max(1, max_per_channel):
            continue
        if category_counts.get(category, 0) >= category_cap:
            continue
        selected.append(item)
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(selected) >= target_size:
            break
    return selected


def suggest_next_queries(
    config: dict[str, Any],
    pool_items: list[dict[str, Any]],
    state: dict[str, Any],
    limit: int,
) -> tuple[list[str], int]:
    """Deficit-driven curriculum: prefer categories underrepresented in the pool.

    Counts current per-category representation, targets a uniform share
    across query groups, and picks the queries from the most underrepresented
    groups first. Ties fall back to the classic round-robin rotation so an
    empty pool cycles queries deterministically. Returns (queries, new_offset).
    """
    groups: dict[str, list[str]] = config["query_groups"]
    query_categories = {query: category for category, queries in groups.items() for query in queries}
    all_queries = list(query_categories)
    if not all_queries:
        return [], 0
    offset = int(state.get("query_offset", 0)) % len(all_queries)
    rotated = [all_queries[(offset + index) % len(all_queries)] for index in range(len(all_queries))]
    rotation_rank = {query: index for index, query in enumerate(rotated)}

    counts = {name: 0 for name in groups}
    for item in pool_items:
        category = str(item.get("category") or "gaming")
        if category in counts:
            counts[category] += 1

    picks = {name: 0 for name in groups}
    selected: list[str] = []
    limit = max(1, min(int(limit), len(all_queries)))
    while len(selected) < limit:
        target_share = (sum(counts.values()) + len(selected)) / len(groups)
        best: str | None = None
        best_key: tuple[float, int] | None = None
        for query in rotated:
            if query in selected:
                continue
            group = query_categories[query]
            deficit = target_share - counts[group] - picks[group]
            key = (deficit, -rotation_rank[query])
            if best_key is None or key > best_key:
                best_key = key
                best = query
        if best is None:
            break
        selected.append(best)
        picks[query_categories[best]] += 1
    return selected, (offset + len(selected)) % len(all_queries)


def collect_candidate_pool(
    config_path: pathlib.Path = DEFAULT_CONFIG,
    candidates_path: pathlib.Path = DEFAULT_CANDIDATES,
    state_path: pathlib.Path = DEFAULT_STATE,
    agent: YouTubeResearchAgent | None = None,
    query_limit: int | None = None,
    results_per_query: int | None = None,
    target_size: int | None = None,
    license_filter: str | None = None,
    order: str | None = None,
) -> dict[str, Any]:
    config = load_collection_config(config_path)
    settings = config.get("dataset", {})
    groups: dict[str, list[str]] = config["query_groups"]
    query_categories = {
        query: category
        for category, queries in groups.items()
        for query in queries
    }
    state_path = pathlib.Path(state_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    run_limit = max(1, query_limit or int(settings.get("searches_per_run", 2)))

    # Load and prune the prior pool FIRST so the curriculum can see which
    # categories are already represented.
    candidates_path = pathlib.Path(candidates_path)
    prior_pool = _read_pool(candidates_path)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=30)
    prior_items = []
    for item in prior_pool.get("items", []):
        timestamp = item.get("discovered_at") or prior_pool.get("updated_at")
        try:
            discovered = dt.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if discovered.tzinfo is None:
            discovered = discovered.replace(tzinfo=dt.timezone.utc)
        if discovered >= cutoff:
            prior_items.append(item)

    selected_queries, next_offset = suggest_next_queries(config, prior_items, state, run_limit)
    per_query = max(1, min(
        50,
        results_per_query or int(settings.get("max_videos_per_query", 50)),
    ))
    max_per_channel = max(1, int(settings.get("max_videos_per_channel", 10)))
    target_size = max(1, int(target_size or settings.get("target_dataset_size", 500)))

    discoveries = (agent or YouTubeResearchAgent(max_per_channel=max_per_channel)).discover(
        results_per_query=per_query,
        strategies=selected_queries,
        license_filter=license_filter or str(settings.get("license", "creativeCommon")),
        order=order or str(settings.get("order", "viewCount")),
        lookback_days=max(1, int(settings.get("lookback_days", 90))),
    )
    curated = curate_candidates(
        prior_items + discoveries,
        query_categories,
        max_per_channel,
        target_size,
        float(settings.get("max_category_share", 0.35)),
    )
    now_text = now.isoformat()
    for item in curated:
        item.setdefault("discovered_at", now_text)
    pool = {
        "dataset_name": str(settings.get("name", "short_form_gaming_cc")),
        "dataset_version": str(settings.get("version", "v001")),
        "updated_at": now_text,
        "expires_at": (now + dt.timedelta(days=30)).isoformat(),
        "candidate_count": len(curated),
        "queries_this_run": selected_queries,
        "target_dataset_size": target_size,
        "items": curated,
    }
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(json.dumps(pool, indent=2, ensure_ascii=True), encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "query_offset": next_offset,
        "last_queries": selected_queries,
        "updated_at": now_text,
        "candidate_count": len(curated),
        "category_counts": {
            category: sum(1 for item in curated if str(item.get("category")) == category)
            for category in groups
        },
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    return pool


def _read_items(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("items", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def _read_pool(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}