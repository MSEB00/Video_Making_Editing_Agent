"""Editing-grammar representation: timeline tokens + weak-supervision labeling.

A reference video becomes a SEQUENCE of editing states (tokens) derived from
measured signals (scene cuts, motion, audio energy, silence, position) — the
labels are inferred per-video from its own distributions (percentiles), not
from fixed templates like "hook = first 1 second".

Tokens are a shared alphabet for:
  - the sequence dataset (training/grammar/sequences.jsonl)
  - the policy learner (transition statistics, duration distributions)
  - structure hypotheses (candidate assemblies of gameplay moments)

Honest scope: caption/SFX/BGM presence is NOT reliably extractable from
reference media without heavier tooling; those channels are marked unknown
(None) rather than fabricated.
"""
from __future__ import annotations

import statistics
from typing import Any, Optional

# Action/state vocabulary (spec §7). These are ACTIONS the policy may take or
# observe — never event->effect mappings.
TOKENS = (
    "HOOK",      # opening moment engineered to retain
    "CONTEXT",   # low-intensity orientation
    "BUILD",     # rising intensity run
    "ESCALATE",  # steep rise, high intensity
    "PEAK",      # local intensity maximum (not the global one)
    "PAYOFF",    # global intensity maximum / reward moment
    "HOLD",      # deliberately sustained shot (low intensity, long)
    "RECOVER",   # post-peak decay
    "SILENCE",   # audio-dropout segment (used as a device)
    "END",       # closing segment
)

ACTION_VOCABULARY = (
    "HOLD", "CUT", "SHORTEN", "EXTEND", "REORDER", "REPLAY",
    "SPEED_UP", "SLOW_DOWN", "FREEZE", "ZOOM", "REFRAME", "SHAKE",
    "TRANSITION", "CAPTION", "NO_CAPTION", "SFX", "NO_SFX",
    "BGM", "NO_BGM", "DUCK_AUDIO", "SILENCE", "AUDIO_EMPHASIS", "NONE",
)


def _percentile_rank(values: list[float], value: float) -> float:
    """Rank of value within values, in [0, 1]."""
    if not values:
        return 0.5
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return (below + 0.5 * equal) / len(values)


def label_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign grammar tokens to measured segments of ONE video.

    Each segment: {"t", "duration", "motion", "audio", "silence_ratio"} with
    motion/audio in [0, 1] (or None when unknown). Labeling is inferred from
    the video's own percentile distributions and structural position.
    """
    if not segments:
        return []
    n = len(segments)
    motions = [float(s.get("motion") or 0.0) for s in segments]
    durations = [float(s.get("duration") or 0.0) for s in segments]
    median_duration = statistics.median(durations) if durations else 0.0

    # composite intensity per segment (audio optional → motion-only fallback)
    intensities = []
    for seg in segments:
        motion = float(seg.get("motion") or 0.0)
        audio = seg.get("audio")
        intensity = 0.6 * motion + 0.4 * float(audio) if audio is not None else motion
        intensities.append(round(min(1.0, max(0.0, intensity)), 4))

    global_peak_index = max(range(n), key=lambda i: intensities[i]) if n else 0
    labeled: list[dict[str, Any]] = []
    for index, seg in enumerate(segments):
        intensity = intensities[index]
        rank = _percentile_rank(intensities, intensity)
        duration = float(seg.get("duration") or 0.0)
        silence_ratio = seg.get("silence_ratio")
        position = index / max(1, n - 1) if n > 1 else 0.5
        prev_intensity = intensities[index - 1] if index > 0 else None
        next_intensity = intensities[index + 1] if index + 1 < n else None

        if silence_ratio is not None and float(silence_ratio) > 0.6 and rank < 0.5:
            token = "SILENCE"
        elif index == global_peak_index and n > 1:
            # the global intensity maximum IS the payoff — even when it opens
            # the video (cold open) or closes it (peak ending)
            token = "PAYOFF"
        elif index == n - 1 and n > 1:
            token = "END"
        elif index == 0 and rank >= 0.55 and (n == 1 or duration <= max(1.5, 1.25 * median_duration)):
            token = "HOOK"
        elif index == 0:
            token = "HOOK" if rank >= 0.4 else "CONTEXT"
        elif rank >= 0.8:
            token = "PEAK" if (next_intensity is not None and next_intensity >= intensity) else "ESCALATE"
        elif prev_intensity is not None and intensity > prev_intensity and rank >= 0.5:
            token = "BUILD"
        elif prev_intensity is not None and intensity < prev_intensity and rank < 0.5 and position > 0.5:
            token = "RECOVER"
        elif median_duration > 0 and duration >= 2.0 * median_duration and rank < 0.45:
            token = "HOLD"
        elif rank < 0.3:
            token = "CONTEXT"
        else:
            token = "BUILD" if (next_intensity is not None and next_intensity > intensity) else "CONTEXT"

        labeled.append({
            "index": index,
            "t": round(float(seg.get("t") or 0.0), 3),
            "duration": round(duration, 3),
            "token": token,
            "intensity": intensity,
            "motion": round(motions[index], 4),
            "audio": (round(float(seg["audio"]), 4) if seg.get("audio") is not None else None),
            "silence_ratio": (round(float(silence_ratio), 3) if silence_ratio is not None else None),
            "position": round(position, 3),
            "cut_after": index < n - 1,
        })
    return labeled


def intensity_curve(labeled: list[dict[str, Any]], bins: int = 10) -> list[Optional[float]]:
    """Mean intensity across normalized timeline position (None bins = unknown)."""
    if not labeled:
        return [None] * bins
    sums: list[list[float]] = [[] for _ in range(bins)]
    for seg in labeled:
        bucket = min(bins - 1, int(float(seg.get("position", 0.0)) * bins))
        sums[bucket].append(float(seg.get("intensity", 0.0)))
    return [round(statistics.mean(b), 4) if b else None for b in sums]


def sequence_stats(labeled: list[dict[str, Any]], total_duration: float) -> dict[str, Any]:
    """Compact learned-vector for one sequence (retrieval embedding + stats)."""
    if not labeled:
        return {}
    durations = [float(s["duration"]) for s in labeled if s.get("duration")]
    intensities = [float(s["intensity"]) for s in labeled]
    payoff_positions = [
        float(s["position"]) for s in labeled if s["token"] == "PAYOFF"
    ]
    audio_values = [float(s["audio"]) for s in labeled if s.get("audio") is not None]
    silence_values = [float(s["silence_ratio"]) for s in labeled if s.get("silence_ratio") is not None]
    first = labeled[0]
    stats = {
        "segment_count": len(labeled),
        "cut_rate": round((len(labeled) - 1) / total_duration, 4) if total_duration > 0 else None,
        "mean_shot_duration": round(statistics.mean(durations), 3) if durations else None,
        "shot_duration_std": round(statistics.stdev(durations), 3) if len(durations) >= 2 else 0.0,
        "mean_intensity": round(statistics.mean(intensities), 4),
        "intensity_std": round(statistics.pstdev(intensities), 4),
        "hook_duration": round(float(first.get("duration") or 0.0), 3),
        "hook_intensity": round(float(first.get("intensity") or 0.0), 4),
        "payoff_position": round(statistics.mean(payoff_positions), 3) if payoff_positions else None,
        "silence_ratio_mean": round(statistics.mean(silence_values), 3) if silence_values else None,
        "audio_energy_mean": round(statistics.mean(audio_values), 4) if audio_values else None,
        "intensity_curve": intensity_curve(labeled),
        "token_histogram": {token: sum(1 for s in labeled if s["token"] == token) for token in TOKENS},
    }
    stats["embedding"] = embedding_from_stats(stats)
    return stats


EMBEDDING_KEYS = (
    "cut_rate", "mean_shot_duration", "shot_duration_std", "mean_intensity",
    "intensity_std", "hook_duration", "hook_intensity", "payoff_position",
    "silence_ratio_mean", "audio_energy_mean",
)


def embedding_from_stats(stats: dict[str, Any]) -> list[float]:
    """Fixed-order feature vector for retrieval (missing → 0.0, flagged separately)."""
    vector = []
    for key in EMBEDDING_KEYS:
        value = stats.get(key)
        vector.append(round(float(value), 4) if isinstance(value, (int, float)) else 0.0)
    return vector
