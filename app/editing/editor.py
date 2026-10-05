"""
app/editing/editor.py
---------------------
Full-fledged video editing engine for the Gaming Video Agent.
Supports:
  - Dynamic transitions (video xfade + delay/amix audio crossfades) with multiple styles:
    fade, wipeleft, wiperight, slideleft, slideright, circlecrop, dissolve, fadeblack, random
  - Background music (BGM) mixing with volume balancing, looping, and fade-in/fade-out
    - Aspect ratio conversions (16:9 YouTube, 9:16 YouTube Shorts / Instagram Reels)
  - Color grading enhancement (vibrancy & contrast boost for gaming)
  - Hardware accelerated rendering (NVENC/QSV) with automatic CPU fallback
"""
from __future__ import annotations

import json
import math
import os
import random
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict

from app.audio.bgm_manager import get_track_by_name_or_genre
from app.utilities.ffmpeg_utils import get_duration, probe, get_audio_stream, get_ffmpeg_path
from app.utilities.logger import get_logger

log = get_logger(__name__)

SUPPORTED_TRANSITIONS = [
    "fade", "wipeleft", "wiperight", "slideleft", "slideright",
    "circlecrop", "dissolve", "fadeblack", "fadewhite", "smoothleft", "smoothright"
]

@dataclass
class EditOptions:
    transition_type: str = "random"        # 'fade', 'wipeleft', 'slideleft', 'circlecrop', 'dissolve', 'random', 'none'
    transition_duration: float = 0.75      # duration of transition in seconds
    bgm_track: Optional[str] = None         # explicit local asset or AI-selected audio path
    bgm_volume: float = 0.30               # volume ratio (0.0 to 1.0)
    aspect_ratio: str = "16:9"             # '16:9' or '9:16'
    color_grade: bool = True               # boost saturation and contrast
    max_clips: Optional[int] = None        # limit total clips
    max_clip_duration: Optional[float] = None # trim clip duration for fast montages
    title_text: Optional[str] = None       # optional title overlay
    randomize_clips: bool = False
    variation_seed: Optional[int] = None
    sfx_preference: Optional[bool] = None
    creative_mode: bool = False
    creative_request: str = ""
    game: str = "valorant"
    platform: str = "youtube_shorts"
    target_duration: int = 45
    planned_segments: Optional[List[tuple[float, float]]] = None
    transition_sequence: Optional[List[str]] = None
    planned_captions: Optional[List[Optional[str]]] = None
    planned_emphasis: Optional[List[List[str]]] = None
    bgm_audio_path: Optional[pathlib.Path] = None
    bgm_start_seconds: float = 0.0
    duck_bgm_to_original_audio: bool = False
    sfx_clips: Optional[List[tuple[pathlib.Path, float, float]]] = None
    audio_normalization_target_lufs: float = -16.0
    planned_audio_gains_db: Optional[List[float]] = None
    chronological_order: Optional[bool] = None   # None = use creative_editing.chronological_shot_order
    full_session_coverage: Optional[bool] = None  # None = use creative_editing.full_session_coverage


def _select_source_clips(input_files: List[pathlib.Path], options: EditOptions, rng: random.Random) -> List[pathlib.Path]:
    selected = list(input_files)
    if options.randomize_clips:
        rng.shuffle(selected)
    if options.max_clips and len(selected) > options.max_clips:
        selected = selected[:options.max_clips]
    return selected


def _choose_clip_segment(source_duration: float, options: EditOptions, rng: random.Random) -> tuple[float, float]:
    segment_duration = min(source_duration, options.max_clip_duration) if options.max_clip_duration else source_duration
    if options.randomize_clips and options.max_clip_duration and source_duration > segment_duration:
        lower_bound = min(segment_duration, max(2.0, segment_duration * 0.5))
        segment_duration = rng.uniform(lower_bound, segment_duration)
    segment_duration = max(segment_duration, 1.0)
    latest_start = max(0.0, source_duration - segment_duration)
    start_time = rng.uniform(0.0, latest_start) if options.randomize_clips and latest_start else 0.0
    return start_time, segment_duration


def _split_single_source_shots(
    source: pathlib.Path,
    source_duration: float,
    options: EditOptions,
    rng: random.Random
) -> List[tuple[pathlib.Path, float, float]]:
    if not options.randomize_clips or not options.max_clip_duration or source_duration <= options.max_clip_duration:
        return []

    min_shot_duration = min(3.0, options.max_clip_duration)
    shot_count = min(
        options.max_clips or 8,
        max(2, int(source_duration / max(2.0, options.max_clip_duration * 0.75)))
    )
    shot_count = min(shot_count, int(source_duration / min_shot_duration))
    if shot_count < 2:
        return []

    bucket_duration = source_duration / shot_count
    shots = []
    for index in range(shot_count):
        bucket_start = index * bucket_duration
        max_duration = min(options.max_clip_duration, bucket_duration)
        min_duration = min(max_duration, max(1.5, max_duration * 0.55))
        duration = rng.uniform(min_duration, max_duration)
        start = bucket_start + rng.uniform(0.0, bucket_duration - duration)
        shots.append((source, start, duration))
    return shots


def _choose_transition_type(
    transition_type: str,
    rng: random.Random,
    previous_transition: Optional[str] = None
) -> str:
    if transition_type != "random":
        return transition_type if transition_type in SUPPORTED_TRANSITIONS else "fade"
    choices = [name for name in SUPPORTED_TRANSITIONS if name != previous_transition]
    return rng.choice(choices or SUPPORTED_TRANSITIONS)


def _escape_drawtext(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("'", ":", ",", ";", "%", "[", "]"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _drawtext_font_option() -> str:
    font_path = os.getenv("FFMPEG_FONT_FILE")
    if not font_path:
        for candidate in (pathlib.Path("C:/Windows/Fonts/arial.ttf"), pathlib.Path("C:/Windows/Fonts/segoeui.ttf")):
            if candidate.is_file():
                font_path = str(candidate)
                break
    if not font_path or not pathlib.Path(font_path).is_file():
        return ""
    escaped_path = str(pathlib.Path(font_path)).replace("\\", "/").replace(":", "\\:")
    return f"fontfile='{escaped_path}':"


def render_edited_video(
    input_files: List[pathlib.Path],
    output_path: pathlib.Path,
    options: Optional[EditOptions] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> pathlib.Path:
    """
    Renders multiple gameplay clips into a polished, professional video
    with transitions, background music, aspect ratio formatting, and color grading.
    """
    if not options:
        options = EditOptions()

    if not input_files:
        raise ValueError("No input files provided for rendering.")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _notify(icon: str, msg: str):
        if progress_callback:
            try:
                progress_callback(icon, msg)
            except Exception:
                pass
        try:
            log.info(msg)
        except Exception:
            pass

    # 1. Clip selection & duration inspection
    rng = random.Random(options.variation_seed)
    planned_segments = options.planned_segments
    if planned_segments is not None:
        if len(planned_segments) != len(input_files):
            raise ValueError("Planned segment count must match the ordered input file list.")
        selected_clips = list(input_files)
        preset_shots = [
            (path, float(segment[0]), float(segment[1]))
            for path, segment in zip(selected_clips, planned_segments)
        ]
    else:
        selected_clips = _select_source_clips(input_files, options, rng)
        preset_shots = []
    if planned_segments is None and len(selected_clips) == 1:
        try:
            preset_shots = _split_single_source_shots(
                selected_clips[0], get_duration(selected_clips[0]), options, rng
            )
        except Exception:
            preset_shots = []
        if preset_shots:
            selected_clips = [shot[0] for shot in preset_shots]

    _notify("🔍", f"Analyzing {len(selected_clips)} clip(s)...")

    clip_durations: List[float] = []
    clip_starts: List[float] = []
    has_audio_list: List[bool] = []

    for idx, p in enumerate(selected_clips):
        try:
            d = get_duration(p)
            if preset_shots:
                start_time, segment_duration = preset_shots[idx][1:]
            else:
                start_time, segment_duration = _choose_clip_segment(d, options, rng)
            clip_starts.append(start_time)
            clip_durations.append(segment_duration)
        except Exception:
            clip_starts.append(0.0)
            clip_durations.append(5.0)

        # Check if clip has an audio stream
        try:
            probe_data = probe(p)
            has_audio = get_audio_stream(probe_data) is not None
        except Exception:
            has_audio = True
        has_audio_list.append(has_audio)

    num_clips = len(selected_clips)
    _notify("🎬", f"Total raw duration: {round(sum(clip_durations), 1)}s across {num_clips} clip(s)")

    # 2. Resolve BGM track
    bgm_path = None
    if options.bgm_track and options.bgm_track.lower() not in ("none", "false", "off"):
        bgm_path = get_track_by_name_or_genre(options.bgm_track)
        if bgm_path and bgm_path.exists():
            _notify("🎵", f"Mixing BGM track: {bgm_path.stem.replace('_', ' ').title()} (Vol: {int(options.bgm_volume*100)}%)")
        else:
            bgm_path = None

    # 3. Handle Transitions
    tr_type = options.transition_type.lower()
    if options.transition_sequence is not None:
        use_transitions = num_clips > 1 and any(
            transition.lower() not in ("none", "cut", "off", "false")
            for transition in options.transition_sequence
        )
    else:
        use_transitions = (num_clips > 1) and (tr_type not in ("none", "cut", "off", "false"))

    # Cap transition duration so it doesn't exceed clip lengths
    min_dur = min(clip_durations)
    t_dur = min(options.transition_duration, min_dur * 0.45)
    t_dur = max(0.2, round(t_dur, 2))

    if use_transitions:
        _notify("✨", f"Applying '{tr_type}' transitions ({t_dur}s) between clips")

    # 4. Construct FFmpeg Inputs and Filter Complex
    cmd_inputs = []
    for p in selected_clips:
        cmd_inputs.extend(["-i", str(p.resolve())])

    bgm_index = None
    selected_bgm_path = options.bgm_audio_path
    if selected_bgm_path and selected_bgm_path.is_file():
        bgm_path = selected_bgm_path
    elif options.creative_mode:
        bgm_path = None
    if bgm_path:
        bgm_index = len(selected_clips)
        if options.bgm_start_seconds > 0:
            cmd_inputs.extend(["-ss", str(options.bgm_start_seconds)])
        cmd_inputs.extend(["-i", str(bgm_path.resolve())])

    valid_sfx_clips = [
        (pathlib.Path(path), max(0.0, float(start)), max(0.0, min(1.0, float(volume))))
        for path, start, volume in (options.sfx_clips or [])
        if pathlib.Path(path).is_file()
    ]
    sfx_input_start = len(selected_clips) + (1 if bgm_index is not None else 0)
    for path, _, _ in valid_sfx_clips:
        cmd_inputs.extend(["-i", str(path.resolve())])

    filter_lines = []
    audio_lines: List[str] = []  # audio-only mirror of the graph, reused for the loudnorm measurement pass

    # Format / Scale each clip into standard stream
    is_vertical = (options.aspect_ratio == "9:16")
    w, h = (1080, 1920) if is_vertical else (1920, 1080)

    for i in range(num_clips):
        # Trimming if max_clip_duration is set
        needs_trim = planned_segments is not None or options.max_clip_duration or clip_starts[i] > 0
        trim_filter = (
            f"trim=start={clip_starts[i]}:duration={clip_durations[i]},setpts=PTS-STARTPTS,"
            if needs_trim else ""
        )

        if is_vertical:
            punch_zoom = options.planned_emphasis and i < len(options.planned_emphasis) and any(
                "punch_zoom" in item.lower() for item in options.planned_emphasis[i]
            )
            # Crop to 9:16 at source resolution FIRST, then scale up to the
            # target canvas. The previous scale-to-cover-then-crop order built
            # a huge intermediate frame (e.g. 3687x2074 from a 16:9 source —
            # >3x the final pixel count, upscaled before cropping), which
            # wasted RAM (OOM on constrained machines) and encode time.
            # min(iw, ih*9/16) / min(ih, iw*16/9) keeps the crop inside the
            # frame for sources that are already vertical, without distortion.
            # NOTE: color grading is applied once after the transition chain
            # (step 6), mirroring the landscape path — per-branch eq/unsharp
            # nodes multiply filter-graph memory without visual benefit.
            zoom_div = "/1.08" if punch_zoom else ""
            portrait_scale = (
                f"crop=w='trunc(min(iw,ih*9/16){zoom_div}/2)*2':"
                f"h='trunc(min(ih,iw*16/9){zoom_div}/2)*2',"
                f"scale={w}:{h},"
            )
            filter_lines.append(
                f"[{i}:v]{trim_filter}{portrait_scale}setsar=1,fps=60,"
                f"settb=AVTB,format=yuv420p[v_in_{i}];"
            )
        else:
            punch_zoom = options.planned_emphasis and i < len(options.planned_emphasis) and any(
                "punch_zoom" in item.lower() for item in options.planned_emphasis[i]
            )
            landscape_scale = (
                f"scale={round(w * 1.08)}:{round(h * 1.08)}:force_original_aspect_ratio=increase,crop={w}:{h},"
                if punch_zoom else
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
            )
            filter_lines.append(
                f"[{i}:v]{trim_filter}{landscape_scale}setsar=1,fps=60,settb=AVTB,format=yuv420p[v_in_{i}];"
            )

        # Audio formatting or silent audio fallback
        if has_audio_list[i]:
            atrim_filter = (
                f"atrim=start={clip_starts[i]}:duration={clip_durations[i]},asetpts=PTS-STARTPTS,"
                if needs_trim else ""
            )
            gain_db = 0.0
            if options.planned_audio_gains_db and i < len(options.planned_audio_gains_db):
                gain_db = max(-12.0, min(24.0, float(options.planned_audio_gains_db[i])))
            gain_filter = f"volume={gain_db}dB," if gain_db else ""
            audio_lines.append(
                f"[{i}:a]{atrim_filter}{gain_filter}aformat=sample_rates=44100:channel_layouts=stereo[a_in_{i}];"
            )
        else:
            audio_lines.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{clip_durations[i]}[a_in_{i}];")

    # 5. Chain transitions
    final_video_label = "[v_in_0]"
    final_audio_label = "[a_in_0]"
    current_time = clip_durations[0]
    shot_output_starts = [0.0]
    previous_transition = None

    if use_transitions:
        # boundary_fades[k] = audio crossfade duration across the boundary
        # between clips k-1 and k (absent for hard cuts).
        boundary_fades: Dict[int, float] = {}
        for k in range(1, num_clips):
            planned_transition = (
                options.transition_sequence[k - 1]
                if options.transition_sequence and k - 1 < len(options.transition_sequence)
                else tr_type
            )
            if planned_transition.lower() in ("none", "cut", "off", "false"):
                v_out = f"[v_cut_{k}]"
                filter_lines.append(
                    f"{final_video_label}[v_in_{k}]concat=n=2:v=1:a=0{v_out};"
                )
                shot_output_starts.append(current_time)
                current_time += clip_durations[k]
                final_video_label = v_out
                previous_transition = None
                continue

            cur_transition = _choose_transition_type(planned_transition, rng, previous_transition)
            previous_transition = cur_transition

            offset = max(0.1, current_time - t_dur)
            shot_output_starts.append(offset)
            current_time = offset + clip_durations[k]
            boundary_fades[k] = t_dur

            v_out = f"[v_trans_{k}]"
            filter_lines.append(
                f"{final_video_label}[v_in_{k}]xfade=transition={cur_transition}:duration={t_dur}:offset={round(offset, 3)}{v_out};"
            )
            final_video_label = v_out

        # Assemble the audio timeline as ONE bounded-memory mixdown: apply
        # crossfade envelopes at overlapping boundaries, place every clip at
        # its exact output offset with adelay, then sum with a single amix.
        # (Chained acrossfade nodes buffer the entire preceding stream at
        # every level; with 3+ clips that grows unbounded and can OOM-kill
        # ffmpeg before a single frame is encoded.)
        for i in range(num_clips):
            chain = f"[a_in_{i}]"
            fade_out = boundary_fades.get(i + 1)
            if fade_out:
                fade_out = min(fade_out, max(0.05, clip_durations[i] * 0.45))
                chain += (
                    f"afade=t=out:st={round(max(0.0, clip_durations[i] - fade_out), 3)}"
                    f":d={round(fade_out, 3)},"
                )
            fade_in = boundary_fades.get(i)
            if fade_in:
                fade_in = min(fade_in, max(0.05, clip_durations[i] * 0.45))
                chain += f"afade=t=in:st=0:d={round(fade_in, 3)},"
            delay_ms = int(round(shot_output_starts[i] * 1000))
            chain += f"adelay={delay_ms}|{delay_ms}[a_placed_{i}];"
            audio_lines.append(chain)
        placed_labels = "".join(f"[a_placed_{i}]" for i in range(num_clips))
        audio_lines.append(
            f"{placed_labels}amix=inputs={num_clips}:duration=longest:normalize=0,"
            f"asetpts=PTS-STARTPTS[a_timeline];"
        )
        final_audio_label = "[a_timeline]"
    elif num_clips > 1:
        # Simple concat without xfade
        v_inputs = "".join(f"[v_in_{i}]" for i in range(num_clips))
        a_inputs = "".join(f"[a_in_{i}]" for i in range(num_clips))
        filter_lines.append(f"{v_inputs}concat=n={num_clips}:v=1:a=0[v_concat];")
        audio_lines.append(f"{a_inputs}concat=n={num_clips}:v=0:a=1[a_concat];")
        final_video_label = "[v_concat]"
        final_audio_label = "[a_concat]"
        current_time = sum(clip_durations)
        shot_output_starts = []
        elapsed = 0.0
        for duration in clip_durations:
            shot_output_starts.append(elapsed)
            elapsed += duration

    # 6. Color grading & visual enhancement
    # Applied once on the combined stream (never per input branch): parallel
    # grade nodes feeding xfade caused unbounded filter-graph memory growth.
    if options.color_grade:
        grade_chain = (
            "eq=saturation=1.18:contrast=1.08,unsharp=5:5:0.6:3:3:0"
            if is_vertical else
            "eq=saturation=1.12:contrast=1.04"
        )
        filter_lines.append(f"{final_video_label}{grade_chain}[v_colored];")
        final_video_label = "[v_colored]"

    # Optional Title Text
    if options.title_text:
        safe_title = options.title_text.replace("'", "").replace(":", "")
        filter_lines.append(
            f"{final_video_label}drawtext={_drawtext_font_option()}text='{safe_title}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=80:"
            f"box=1:boxcolor=black@0.5:boxborderw=10:enable='between(t,0,4)'[v_titled];"
        )
        final_video_label = "[v_titled]"

    if options.planned_captions:
        for index, caption in enumerate(options.planned_captions[:num_clips]):
            if not caption:
                continue
            start = shot_output_starts[index] if index < len(shot_output_starts) else 0.0
            end = min(current_time, start + clip_durations[index])
            safe_caption = _escape_drawtext(caption)
            caption_y = "h-text_h-260" if is_vertical else "h-text_h-72"
            filter_lines.append(
                f"{final_video_label}drawtext={_drawtext_font_option()}text='{safe_caption}':fontsize=58:fontcolor=white:"
                f"borderw=4:bordercolor=black:x=(w-text_w)/2:y={caption_y}:"
                f"enable='between(t,{start:.3f},{end:.3f})'[v_caption_{index}];"
            )
            final_video_label = f"[v_caption_{index}]"

    # 7. Mix BGM if enabled
    if bgm_index is not None:
        fade_out_st = max(0.5, current_time - 1.5)
        audio_lines.append(
            f"[{bgm_index}:a]aloop=loop=-1:size=2e+09,volume={options.bgm_volume},"
            f"afade=t=in:ss=0:d=0.8,afade=t=out:st={round(fade_out_st, 2)}:d=1.5[bgm_fx];"
        )
        if options.duck_bgm_to_original_audio:
            audio_lines.append(
                f"{final_audio_label}asplit=2[program_audio][duck_key];"
                f"[bgm_fx][duck_key]sidechaincompress=threshold=0.025:ratio=3:attack=80:release=500[bgm_ducked];"
                "[program_audio][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[a_mixed];"
            )
        else:
            audio_lines.append(
                f"{final_audio_label}[bgm_fx]amix=inputs=2:duration=first:dropout_transition=2[a_mixed];"
            )
        final_audio_label = "[a_mixed]"

    if valid_sfx_clips:
        sfx_labels = []
        for index, (path, start, volume) in enumerate(valid_sfx_clips):
            input_index = sfx_input_start + index
            delay_ms = round(start * 1000)
            sfx_label = f"[planned_sfx_{index}]"
            audio_lines.append(
                f"[{input_index}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume={volume},adelay={delay_ms}|{delay_ms}{sfx_label};"
            )
            sfx_labels.append(sfx_label)
        audio_lines.append(
            f"{final_audio_label}{''.join(sfx_labels)}"
            f"amix=inputs={len(sfx_labels) + 1}:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95[a_with_planned_sfx];"
        )
        final_audio_label = "[a_with_planned_sfx]"

    loudnorm_pending: Optional[tuple] = None
    if options.creative_mode:
        target_lufs = max(-30.0, min(-8.0, float(options.audio_normalization_target_lufs)))
        # Two-pass loudnorm is applied after the graph is assembled (see
        # _measure_loudnorm below): dynamic single-pass loudnorm proved
        # unbounded in memory when combined with a slower video encoder and
        # is non-deterministic across review/revision re-renders.
        loudnorm_pending = (final_audio_label, target_lufs)

    # Optional graph throttle for memory-constrained machines. When the filter
    # graph runs far ahead of a slow encoder, ffmpeg's in-flight frame queue
    # grows (≈3 MB per 1080x1920 frame) and can exhaust RAM. Setting
    # FFMPEG_GRAPH_THROTTLE=<speed> (e.g. 1.0 = realtime) bounds the backlog.
    # Off by default: hardware-encoder machines render faster unthrottled.
    throttle = os.getenv("FFMPEG_GRAPH_THROTTLE", "").strip()
    if throttle:
        try:
            throttle_speed = float(throttle)
        except ValueError:
            throttle_speed = 0.0
            log.warning("Ignoring invalid FFMPEG_GRAPH_THROTTLE=%r", throttle)
        if throttle_speed > 0:
            filter_lines.append(f"{final_video_label}realtime=speed={throttle_speed}[v_throttled];")
            final_video_label = "[v_throttled]"

    # Force 4:2:0 chroma subsampling on the final video chain. Without this,
    # libx264 may negotiate the filter graph up to the High 4:4:4 Predictive
    # profile (~3x frame memory, poor player support, disallowed for optimal
    # YouTube processing).
    filter_lines.append(f"{final_video_label}format=yuv420p[v_compat];")
    final_video_label = "[v_compat]"

    # Strip trailing semicolon from last filter line; keep the audio-only
    # graph separate so the loudnorm measurement pass can run it cheaply
    # (ffmpeg requires every declared graph output to be mapped).
    audio_filter_str = "".join(audio_lines).rstrip(";")
    filter_complex_str = ("".join(filter_lines) + "".join(audio_lines)).rstrip(";")

    if loudnorm_pending is not None:
        pre_label, target_lufs = loudnorm_pending
        filter_complex_str, final_audio_label = _append_two_pass_loudnorm(
            cmd_inputs, filter_complex_str, audio_filter_str, pre_label, target_lufs
        )

    _notify("⚡", "Rendering final video (FFmpeg processing)...")

    # Try hardware encoder first (NVENC), fallback to libx264.
    # The CPU preset adapts to core count: on weak machines 'fast' cannot keep
    # up with a 1080p60 filter graph, and ffmpeg's in-flight frame queue then
    # grows until the process is OOM-killed. Override via env if needed.
    cpu_preset = os.getenv("FFMPEG_X264_PRESET", "").strip().lower()
    if not cpu_preset:
        cores = os.cpu_count() or 4
        cpu_preset = "ultrafast" if cores <= 2 else "veryfast" if cores <= 4 else "fast"
    cpu_crf = os.getenv("FFMPEG_X264_CRF", "19").strip() or "19"
    encoders_to_try = [
        (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"], "NVIDIA NVENC"),
        (["-c:v", "libx264", "-preset", cpu_preset, "-crf", cpu_crf], f"libx264 CPU ({cpu_preset})"),
    ]

    render_success = False
    last_error = ""

    for enc_args, enc_name in encoders_to_try:
        cmd = [
            get_ffmpeg_path(), "-y",
            *cmd_inputs,
            "-filter_complex", filter_complex_str,
            "-map", final_video_label,
            "-map", final_audio_label,
            *enc_args,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path)
        ]
        log.debug(f"Running render with {enc_name}")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode == 0:
            render_success = True
            log.info(f"Rendered successfully using {enc_name}")
            break
        else:
            last_error = res.stderr[-500:]
            log.warning(f"{enc_name} failed, trying next encoder: {last_error}")

    if not render_success:
        raise RuntimeError(f"FFmpeg render failed: {last_error}")

    _notify("✅", f"Video render verified! Output: {output_path.name}")
    return output_path


def _measure_loudnorm(
    cmd_inputs: List[str],
    audio_filter: str,
    pre_label: str,
    target_lufs: float,
) -> Optional[Dict[str, float]]:
    """Run the audio-only measurement pass of two-pass loudnorm.

    Executes ONLY the audio half of the filter graph with `-f null` (video
    branches are excluded entirely: ffmpeg requires every declared graph
    output to be mapped). Cheap, fast, and memory-safe. Returns the measured
    values or None when measurement fails.
    """
    measure_fc = (
        f"{audio_filter};"
        f"{pre_label}loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json[a_measure]"
    )
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats", "-y",
        *cmd_inputs,
        "-filter_complex", measure_fc,
        "-map", "[a_measure]",
        "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        log.warning("Loudnorm measurement pass failed (rc=%s)", res.returncode)
        return None
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        measured = {
            "measured_I": float(data["input_i"]),
            "measured_TP": float(data["input_tp"]),
            "measured_LRA": float(data["input_lra"]),
            "measured_thresh": float(data["input_thresh"]),
            "offset": float(data["target_offset"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(math.isfinite(value) for value in measured.values()):
        return None  # e.g. pure-silence streams report -inf
    return measured


def _append_two_pass_loudnorm(
    cmd_inputs: List[str],
    filter_complex: str,
    audio_filter: str,
    pre_label: str,
    target_lufs: float,
) -> tuple[str, str]:
    """Append linear (two-pass) loudnorm to the audio chain.

    Linear mode preserves dynamics, is deterministic across re-renders (the
    review/revise loop relies on this), and avoids the unbounded buffering
    dynamic mode exhibits when the video encoder is the slower stream.
    Falls back to single-pass dynamic normalization if measurement fails.
    Returns (updated filter_complex, new final audio label).
    """
    measured = _measure_loudnorm(cmd_inputs, audio_filter, pre_label, target_lufs)
    if measured is not None:
        loudnorm_args = (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
            f":measured_I={measured['measured_I']}"
            f":measured_TP={measured['measured_TP']}"
            f":measured_LRA={measured['measured_LRA']}"
            f":measured_thresh={measured['measured_thresh']}"
            f":offset={measured['offset']}"
            f":linear=true"
        )
        log.info(
            "Loudnorm measured input at %.1f LUFS / TP %.1f dB; applying linear normalization to %.1f LUFS",
            measured["measured_I"], measured["measured_TP"], target_lufs,
        )
    else:
        loudnorm_args = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        log.warning("Loudnorm measurement unavailable; falling back to single-pass dynamic mode.")
    updated = (
        f"{filter_complex};"
        f"{pre_label}{loudnorm_args},"
        f"aformat=sample_rates=44100:channel_layouts=stereo[a_normalized]"
    )
    return updated, "[a_normalized]"
