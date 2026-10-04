"""
app/editing/editor.py
---------------------
Full-fledged video editing engine for the Gaming Video Agent.
Supports:
  - Dynamic transitions (xfade / acrossfade) with multiple styles:
    fade, wipeleft, wiperight, slideleft, slideright, circlecrop, dissolve, fadeblack, random
  - Background music (BGM) mixing with volume balancing, looping, and fade-in/fade-out
    - Aspect ratio conversions (16:9 YouTube, 9:16 YouTube Shorts / Instagram Reels)
  - Color grading enhancement (vibrancy & contrast boost for gaming)
  - Hardware accelerated rendering (NVENC/QSV) with automatic CPU fallback
"""
from __future__ import annotations

import os
import random
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

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
            portrait_grade = "eq=saturation=1.18:contrast=1.08,unsharp=5:5:0.6:3:3:0," if options.color_grade else ""
            punch_zoom = options.planned_emphasis and i < len(options.planned_emphasis) and any(
                "punch_zoom" in item.lower() for item in options.planned_emphasis[i]
            )
            portrait_scale = (
                f"scale={round(w * 1.08)}:{round(h * 1.08)}:force_original_aspect_ratio=increase,crop={w}:{h},"
                if punch_zoom else
                f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            )
            filter_lines.append(
                f"[{i}:v]{trim_filter}{portrait_scale}setsar=1,fps=60,{portrait_grade}"
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
            filter_lines.append(
                f"[{i}:a]{atrim_filter}{gain_filter}aformat=sample_rates=44100:channel_layouts=stereo[a_in_{i}];"
            )
        else:
            filter_lines.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{clip_durations[i]}[a_in_{i}];")

    # 5. Chain transitions
    final_video_label = "[v_in_0]"
    final_audio_label = "[a_in_0]"
    current_time = clip_durations[0]
    shot_output_starts = [0.0]
    previous_transition = None

    if use_transitions:
        for k in range(1, num_clips):
            planned_transition = (
                options.transition_sequence[k - 1]
                if options.transition_sequence and k - 1 < len(options.transition_sequence)
                else tr_type
            )
            if planned_transition.lower() in ("none", "cut", "off", "false"):
                v_out = f"[v_cut_{k}]"
                a_out = f"[a_cut_{k}]"
                filter_lines.append(
                    f"{final_video_label}[v_in_{k}]concat=n=2:v=1:a=0{v_out};"
                )
                filter_lines.append(
                    f"{final_audio_label}[a_in_{k}]concat=n=2:v=0:a=1{a_out};"
                )
                shot_output_starts.append(current_time)
                current_time += clip_durations[k]
                final_video_label = v_out
                final_audio_label = a_out
                previous_transition = None
                continue

            cur_transition = _choose_transition_type(planned_transition, rng, previous_transition)
            previous_transition = cur_transition

            offset = max(0.1, current_time - t_dur)
            shot_output_starts.append(offset)
            current_time = offset + clip_durations[k]

            v_out = f"[v_trans_{k}]"
            a_out = f"[a_trans_{k}]"

            filter_lines.append(
                f"{final_video_label}[v_in_{k}]xfade=transition={cur_transition}:duration={t_dur}:offset={round(offset, 3)}{v_out};"
            )
            filter_lines.append(
                f"{final_audio_label}[a_in_{k}]acrossfade=d={t_dur}{a_out};"
            )
            final_video_label = v_out
            final_audio_label = a_out
    elif num_clips > 1:
        # Simple concat without xfade
        v_inputs = "".join(f"[v_in_{i}]" for i in range(num_clips))
        a_inputs = "".join(f"[a_in_{i}]" for i in range(num_clips))
        filter_lines.append(f"{v_inputs}concat=n={num_clips}:v=1:a=0[v_concat];")
        filter_lines.append(f"{a_inputs}concat=n={num_clips}:v=0:a=1[a_concat];")
        final_video_label = "[v_concat]"
        final_audio_label = "[a_concat]"
        current_time = sum(clip_durations)
        shot_output_starts = []
        elapsed = 0.0
        for duration in clip_durations:
            shot_output_starts.append(elapsed)
            elapsed += duration

    # 6. Color grading & visual enhancement
    if options.color_grade and not is_vertical:
        filter_lines.append(f"{final_video_label}eq=saturation=1.12:contrast=1.04[v_colored];")
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
        filter_lines.append(
            f"[{bgm_index}:a]aloop=loop=-1:size=2e+09,volume={options.bgm_volume},"
            f"afade=t=in:ss=0:d=0.8,afade=t=out:st={round(fade_out_st, 2)}:d=1.5[bgm_fx];"
        )
        if options.duck_bgm_to_original_audio:
            filter_lines.append(
                f"{final_audio_label}asplit=2[program_audio][duck_key];"
                f"[bgm_fx][duck_key]sidechaincompress=threshold=0.025:ratio=3:attack=80:release=500[bgm_ducked];"
                "[program_audio][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[a_mixed];"
            )
        else:
            filter_lines.append(
                f"{final_audio_label}[bgm_fx]amix=inputs=2:duration=first:dropout_transition=2[a_mixed];"
            )
        final_audio_label = "[a_mixed]"

    if valid_sfx_clips:
        sfx_labels = []
        for index, (path, start, volume) in enumerate(valid_sfx_clips):
            input_index = sfx_input_start + index
            delay_ms = round(start * 1000)
            sfx_label = f"[planned_sfx_{index}]"
            filter_lines.append(
                f"[{input_index}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume={volume},adelay={delay_ms}|{delay_ms}{sfx_label};"
            )
            sfx_labels.append(sfx_label)
        filter_lines.append(
            f"{final_audio_label}{''.join(sfx_labels)}"
            f"amix=inputs={len(sfx_labels) + 1}:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95[a_with_planned_sfx];"
        )
        final_audio_label = "[a_with_planned_sfx]"

    if options.creative_mode:
        target_lufs = max(-30.0, min(-8.0, float(options.audio_normalization_target_lufs)))
        filter_lines.append(
            f"{final_audio_label}loudnorm=I={target_lufs}:TP=-1.5:LRA=11[a_normalized];"
        )
        final_audio_label = "[a_normalized]"

    # Strip trailing semicolon from last filter line
    filter_complex_str = "".join(filter_lines).rstrip(";")

    _notify("⚡", "Rendering final video (FFmpeg processing)...")

    # Try hardware encoder first (NVENC), fallback to libx264
    encoders_to_try = [
        (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"], "NVIDIA NVENC"),
        (["-c:v", "libx264", "-preset", "fast", "-crf", "19"], "libx264 CPU")
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

