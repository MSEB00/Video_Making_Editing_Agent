"""Candidate moment graph over raw gameplay (spec §8).

Instead of "pick clips, then assemble", detection + measured motion produce
a graph of candidate MOMENTS, each with features and role options:

    moment kinds:
      event    — anchored on a detected gameplay event (kill), with buildup
                 before and hold after already inside the window
      motion   — measured high-activity window (no event)
      run      — best continuous passage (the legacy option, still a citizen)

Structures are hypotheses assembled from moments; the learned policy scores
them. Moments carry no creative decisions — only measurements + provenance.
"""
from __future__ import annotations

from typing import Any, Optional

EVENT_BUILDUP_SECONDS = 2.0
EVENT_HOLD_SECONDS = 1.0
MOTION_WINDOW_SECONDS = 2.5
MOTION_STRIDE_SECONDS = 0.5
MAX_MOTION_MOMENTS_PER_SOURCE = 2


def build_moment_graph(
    sources: list[dict[str, Any]],
    motion_features: list[dict[str, Any]],
    events_by_source: dict[int, list[dict[str, Any]]],
    target_duration: float,
) -> list[dict[str, Any]]:
    """Build candidate moments for every analyzed source."""
    moments: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        try:
            duration = float(source.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        if duration < 1.0:
            continue
        activity = []
        if index < len(motion_features) and isinstance(motion_features[index], dict):
            activity = [float(v) for v in (motion_features[index].get("activity") or [])]

        taken: list[tuple[float, float]] = []

        # 1) event-anchored moments (buildup + payoff + hold inside window)
        for event in events_by_source.get(int(source.get("source_index", index)), []):
            try:
                e_start = float(event.get("start", 0.0))
                e_end = float(event.get("end") or e_start)
            except (TypeError, ValueError):
                continue
            start = max(0.0, e_start - EVENT_BUILDUP_SECONDS)
            end = min(duration - 0.05, e_end + EVENT_HOLD_SECONDS)
            if end - start < 1.0:
                continue
            moments.append({
                "id": f"s{index}_ev{len(moments)}",
                "source_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "intensity": round(_window_peak(activity, start, end), 4),
                "kind": "event",
                "event_confidence": event.get("confidence"),
                "event_kind": event.get("kind"),
            })
            taken.append((start, end))

        # 2) measured motion moments (excluding event windows)
        if activity:
            window = int(MOTION_WINDOW_SECONDS * 2)  # activity sampled at 2 Hz
            stride = max(1, int(MOTION_STRIDE_SECONDS * 2))
            scored = []
            for offset in range(0, max(1, len(activity) - window + 1), stride):
                values = activity[offset:offset + window]
                if len(values) < window // 2:
                    continue
                start = offset / 2.0
                end = start + len(values) / 2.0
                if any(start < t_end and end > t_start for t_start, t_end in taken):
                    continue
                mean = sum(values) / len(values)
                peak = max(values)
                scored.append((0.7 * mean + 0.3 * peak, start, end, mean))
            scored.sort(reverse=True)
            for score, start, end, mean in scored[:MAX_MOTION_MOMENTS_PER_SOURCE]:
                moments.append({
                    "id": f"s{index}_mo{len(moments)}",
                    "source_index": index,
                    "start": round(start, 3),
                    "end": round(min(end, duration - 0.05), 3),
                    "duration": round(min(end, duration - 0.05) - start, 3),
                    "intensity": round(min(1.0, mean), 4),
                    "kind": "motion",
                })

        # 3) best continuous run (legacy option stays available, never forced)
        run_length = min(max(3.0, target_duration * 0.8), duration - 0.05)
        if activity and run_length >= 2.0:
            window = int(run_length * 2)
            best: Optional[tuple[float, float]] = None
            for offset in range(0, max(1, len(activity) - window + 1), max(1, window // 8)):
                values = activity[offset:offset + window]
                mean = sum(values) / len(values) if values else 0.0
                if best is None or mean > best[0]:
                    best = (mean, offset / 2.0)
            if best is not None:
                start = best[1]
                moments.append({
                    "id": f"s{index}_run",
                    "source_index": index,
                    "start": round(start, 3),
                    "end": round(min(start + run_length, duration - 0.05), 3),
                    "duration": round(min(run_length, duration - 0.05 - start), 3),
                    "intensity": round(min(1.0, best[0]), 4),
                    "kind": "run",
                })
    return moments


def _window_peak(activity: list[float], start: float, end: float) -> float:
    if not activity:
        return 0.5  # unknown motion → neutral, never fabricated high/low
    w0, w1 = int(start * 2), max(int(start * 2) + 1, int(end * 2))
    window = activity[w0:min(w1, len(activity))]
    return min(1.0, max(window)) if window else 0.0
