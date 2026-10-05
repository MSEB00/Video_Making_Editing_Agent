"""
main.py
--------
CLI entry point for the Gaming Video Agent.
Provides commands to initialise the database and to start a processing job.
"""
from __future__ import annotations

import sys
import json
import os
import pathlib
import datetime as dt
import time

import click
from typing import Any

from dotenv import load_dotenv

from app.config.config_loader import load_config
from app.utilities.logger import get_logger

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

log = get_logger(__name__)

@click.group()
def cli() -> None:
    """Top level command group."""
    pass


@cli.command(name="plan-request")
@click.argument('input_path', type=click.Path(exists=True, file_okay=False), metavar='CLIP_DIR')
@click.option('--platform', default='youtube_shorts', show_default=True)
@click.option('--game', default='valorant', show_default=True)
@click.option('--request', default='', help='Free-text creative brief for the chat AI.')
@click.option('--duration', 'target_duration', default=40, type=click.IntRange(5, 180), show_default=True)
@click.option('--max-frames', default=6, type=click.IntRange(0, 12), show_default=True,
              help='Sample frames exported as images (attach them to the chat for vision planning).')
def plan_request(input_path: str, platform: str, game: str, request: str,
                 target_duration: int, max_frames: int) -> None:
    """Export the creative-planning prompt for a WEB CHAT AI (chat.qwen.ai, ChatGPT, ...).

    Analyzes the clip folder exactly like the agent would (media facts, kill
    events, learned style profile, SFX inventory, YouTube metadata priors),
    then writes a paste-ready prompt file plus optional frame images under
    temp/ai/. Paste into any chat AI, save its JSON reply to a file, then:

        python main.py edit CLIP_DIR --plan-file <response file>
    """
    import base64

    from app.analysis.games import get_event_detector
    from app.analysis.media_context import analyze_sources
    from app.ai.prompts import PLAN_SYSTEM, build_plan_context, select_prompt_frames
    from app.audio.sfx_library import SfxLibrary
    from app.editing.style_learner import EditingStyleLearner
    from app.research.metadata_priors import youtube_duration_priors

    cfg = load_config()
    settings = (cfg or {}).get("creative_editing", {})
    clip_dir = pathlib.Path(input_path)
    exts = {".mp4", ".mov", ".mkv"}
    files = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise click.ClickException(f"No video clips found in {clip_dir}")
    max_sources = max(1, int(settings.get("max_source_videos", 12)))
    click.echo(f"Analyzing {min(len(files), max_sources)} clip(s)...")
    media_context, frames = analyze_sources(files, max_sources=max_sources)

    detector = get_event_detector(game)
    if detector is not None:
        click.echo(f"Scanning for {game} gameplay events...")
        for index, source in enumerate(media_context.get("sources") or []):
            if index >= len(files):
                break
            try:
                events = detector.detect(files[index])
            except Exception:
                events = []
            source["gameplay_events"] = [
                {"kind": e.get("kind"), "start": e.get("start"),
                 "end": e.get("end", e.get("start")), "confidence": e.get("confidence")}
                for e in events
            ]

    platform_norm = platform.strip().lower()
    style_profile = EditingStyleLearner(promoted_only=True).learn(platform_norm)
    sfx_assets = SfxLibrary().list_assets()
    try:
        priors = youtube_duration_priors()
    except Exception:
        priors = {}
    preferences = {
        "game": game,
        "requested_style": request,
        "music_preference": None,
        "sfx_preference": None,
        "user_selected_aspect_ratio": "9:16" if platform_norm in VERTICAL_PLATFORMS else "16:9",
        "chronological_assembly": bool(settings.get("chronological_shot_order", True)),
        "full_session_coverage": bool(settings.get("full_session_coverage", True)),
    }
    context = build_plan_context(
        media_context=media_context,
        style_profile=style_profile,
        platform=platform_norm,
        target_duration=target_duration,
        user_request=request,
        user_preferences=preferences,
        available_sfx=[
            {key: item[key] for key in ("filename", "format", "duration", "description",
                                        "type", "mood", "intensity", "tags") if item.get(key) is not None}
            for item in sfx_assets
        ],
        metadata_priors=priors,
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ai_dir = pathlib.Path("temp") / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = ai_dir / f"frames_{stamp}"
    saved_frames: list[str] = []
    for i, frame in enumerate(select_prompt_frames(frames, limit=max_frames)):
        data_url = str(frame.get("data_url") or "")
        if data_url.startswith("data:") and "," in data_url:
            frames_dir.mkdir(parents=True, exist_ok=True)
            target = frames_dir / f"frame_{i:02d}_src{frame.get('source_index')}.jpg"
            try:
                target.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                saved_frames.append(str(target))
            except (ValueError, OSError):
                continue

    request_path = ai_dir / f"request_{stamp}.txt"
    response_path = ai_dir / f"response_{stamp}.txt"
    parts = [
        "You are acting as the creative planner for a gaming short-form editing agent.",
        "Follow the SYSTEM instructions below and produce the edit plan from the CONTEXT JSON.",
        "Return JSON only (code fences are tolerated).",
        "",
        "=== SYSTEM ===",
        PLAN_SYSTEM,
        "",
        "=== CONTEXT ===",
        json.dumps(context, ensure_ascii=True, separators=(",", ":")),
    ]
    if saved_frames:
        parts += ["", "=== FRAME IMAGES (attach these to your chat message) ==="]
        parts += [f"- {p}" for p in saved_frames]
    parts += [
        "",
        "=== AFTER YOU GET THE REPLY ===",
        f"1. Save the AI's FULL reply text to: {response_path}",
        f"2. Run: python main.py edit \"{input_path}\" --plan-file \"{response_path}\" "
        f"--platform {platform} --duration {target_duration} --game {game}",
    ]
    request_path.write_text("\n".join(parts), encoding="utf-8")
    click.echo(json.dumps({
        "status": "request_written",
        "request_file": str(request_path),
        "response_file_to_create": str(response_path),
        "frame_images": saved_frames,
        "sources_analyzed": len(media_context.get("sources") or []),
        "kill_events_detected": sum(
            len(s.get("gameplay_events") or []) for s in (media_context.get("sources") or [])
        ),
        "style_profile_status": style_profile.get("training_status"),
        "instructions": [
            f"1. Copy the whole contents of {request_path}",
            "2. Paste into any web chat AI — chat.qwen.ai or ChatGPT (attach the frame images too, if you can)",
            f"3. Save the AI's full reply as {response_path}",
            f"4. python main.py edit \"{input_path}\" --plan-file \"{response_path}\" "
            f"--platform {platform} --duration {target_duration}",
        ],
    }, indent=2))


@cli.command()
def initdb() -> None:
    """Create / reset the SQLite database schema."""
    from app.storage.db import init_db

    init_db()
    click.echo("Database initialised.")
    log.info("Database initialised via CLI")


STYLE_PRESETS: dict[str, dict] = {
    "FAST_PACED": {
        "transition_type": "random",
        "transition_duration": 0.5,
        "randomize_clips": True,
        "max_clips": 10,
        "max_clip_duration": 4.0,
    },
    "MONTAGE": {
        "transition_type": "random",
        "transition_duration": 0.4,
        "randomize_clips": True,
        "max_clips": 12,
        "max_clip_duration": 2.5,
    },
    "CINEMATIC": {
        "transition_type": "fade",
        "transition_duration": 1.0,
        "randomize_clips": False,
    },
    "SIMPLE": {
        "transition_type": "none",
        "randomize_clips": False,
    },
}

VERTICAL_PLATFORMS = {
    "youtube_shorts", "shorts", "yt_shorts",
    "instagram_reels", "instagram", "reels", "tiktok",
}


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False), metavar='CLIP_DIR')
@click.option('--style', default='FAST_PACED', show_default=True,
              type=click.Choice(sorted(STYLE_PRESETS), case_sensitive=False),
              help='Editing style preset to apply')
@click.option('--platform', default='youtube', show_default=True,
              help='Target platform (youtube, youtube_shorts, instagram_reels, tiktok, ...)')
@click.option('--queue-only', is_flag=True,
              help='Only create the queued job record; do not run the pipeline now.')
def process(input_path: str, style: str, platform: str, queue_only: bool) -> None:
    """Create a processing job for a directory of gameplay clips and run it.

    CLIP_DIR must be a folder containing .mp4/.mov/.mkv clips. The job and its
    resolved edit options are persisted first, then the orchestrator renders
    the final video (unless --queue-only is given).
    """
    _ = load_config()
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    platform_norm = platform.strip().lower()
    style_norm = style.upper()
    preset = dict(STYLE_PRESETS[style_norm])
    preset.update({
        "style": style_norm,
        "platform": platform_norm,
        "aspect_ratio": "9:16" if platform_norm in VERTICAL_PLATFORMS else "16:9",
    })

    log.info(
        "Starting job",
        extra={"input": input_path, "style": style_norm, "platform": platform_norm},
    )
    db = SessionLocal()
    try:
        job = Job(
            status="queued",
            input_path=input_path,
            extra_metadata=json.dumps(preset),
        )
        db.add(job)
        db.commit()
        job_id = job.id
    finally:
        db.close()
    click.echo(f"Job {job_id} queued.")
    log.info("Job queued", extra={"job_id": job_id})

    if queue_only:
        return

    from app.orchestrator.orchestrator import orchestrate_job

    def _progress(icon: str, msg: str) -> None:
        click.echo(f"{icon}  {msg}")

    try:
        output_path = orchestrate_job(job_id, progress_callback=_progress)
    except Exception as exc:
        log.exception("Job failed", extra={"job_id": job_id})
        click.echo(f"Job {job_id} FAILED: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Job {job_id} completed -> {output_path}")
    log.info("Job completed", extra={"job_id": job_id, "output": str(output_path)})


@cli.command()
@click.option('--topic', default=None, help='Configurable short-form research topic or style.')
@click.option('--limit', default=5, type=click.IntRange(1, 10), show_default=True, help='Maximum search results for this explicit query.')
@click.option('--inspect', 'inspect_only', is_flag=True, help='Show research dataset status without calling any API.')
def research(topic: str | None, limit: int, inspect_only: bool) -> None:
    """Discover public YouTube metadata for manual embedded-player research."""
    from app.research.video_observer import VideoObservationStore

    if inspect_only:
        store = VideoObservationStore()
        pool_path = pathlib.Path(__file__).resolve().parent / "training" / "candidates.json"
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pool = {}
        categories: dict[str, int] = {}
        for item in pool.get("items", []) if isinstance(pool, dict) else []:
            key = str(item.get("category") or "gaming")
            categories[key] = categories.get(key, 0) + 1
        click.echo(json.dumps({
            "research_dataset": store.stats(),
            "candidate_pool": {
                "candidate_count": pool.get("candidate_count", 0) if isinstance(pool, dict) else 0,
                "updated_at": pool.get("updated_at") if isinstance(pool, dict) else None,
                "target_dataset_size": pool.get("target_dataset_size") if isinstance(pool, dict) else None,
                "category_counts": categories,
            },
        }, indent=2))
        return

    if not topic:
        raise click.UsageError("--topic is required unless --inspect is given.")

    from app.research.youtube_research_agent import YouTubeResearchAgent

    candidates = YouTubeResearchAgent().discover(
        results_per_query=limit,
        strategies=[topic[:120]],
        license_filter="creativeCommon",
        order="viewCount",
    )
    registered = VideoObservationStore().register_videos(candidates)
    click.echo(json.dumps({
        "status": "metadata_discovered",
        "query": topic[:120],
        "found": len(candidates),
        "registered": len(registered),
        "media_downloaded": False,
        "next_step": "Open /research to watch references and enter manual notes.",
    }, indent=2))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False), metavar='CLIP_DIR')
@click.option('--platform', default='youtube_shorts', show_default=True,
              help='youtube_shorts | instagram_reels | youtube | tiktok')
@click.option('--game', default='valorant', show_default=True, help='Game context hint for the planner.')
@click.option('--request', default='', help='Free-text creative brief for the AI planner.')
@click.option('--duration', 'target_duration', default=45, type=click.IntRange(5, 180), show_default=True,
              help='Target duration in seconds.')
@click.option('--bgm', default=None, help='Music hint (Jamendo search phrase or local asset); "none" disables BGM.')
@click.option('--order', 'shot_order', default='auto', show_default=True,
              type=click.Choice(['auto', 'chronological', 'impact']),
              help='Shot assembly: chronological = recording timeline (config default), '
                   'impact = planner-chosen ordering.')
@click.option('--source-clips', 'source_clips', default='auto', show_default=True,
              type=click.Choice(['auto', 'all', 'selected']),
              help='all = every indexed clip in the folder appears in the video '
                   '(config default); selected = planner picks a subset.')
@click.option('--plan-file', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Use a plan produced by a web chat AI (see plan-request) instead of '
                   'the local planner. The file may contain the raw chat reply (fences/prose ok).')
def edit(input_path: str, platform: str, game: str, request: str, target_duration: int, bgm: str | None,
         shot_order: str, source_clips: str, plan_file: str | None) -> None:
    """Run the full CREATIVE pipeline: AI plan → render → review → revise.

    Planning brains, in order of preference: --plan-file (any web chat AI —
    chat.qwen.ai, ChatGPT; see the plan-request command) or the measured
    local feature planner (default). Produces output/job_N_final.mp4 plus an .edit-plan.json
    artifact with full decision provenance.
    """
    from app.storage.db import SessionLocal
    from app.storage.models import Job
    from app.orchestrator.orchestrator import orchestrate_job
    from app.editing.editor import EditOptions

    _ = load_config()
    platform_norm = platform.strip().lower()
    aspect_ratio = "9:16" if platform_norm in VERTICAL_PLATFORMS else "16:9"
    chronological = {"auto": None, "chronological": True, "impact": False}[shot_order]
    coverage = {"auto": None, "all": True, "selected": False}[source_clips]
    metadata = {
        "style": "CREATIVE_AI",
        "creative_mode": True,
        "creative_request": request,
        "game": game,
        "platform": platform_norm,
        "target_duration": target_duration,
        "aspect_ratio": aspect_ratio,
        "bgm_track": bgm,
        "shot_order": shot_order,
        "source_clips": source_clips,
        "plan_file": plan_file,
    }
    db = SessionLocal()
    try:
        job = Job(status="queued", input_path=input_path, extra_metadata=json.dumps(metadata))
        db.add(job)
        db.commit()
        job_id = job.id
    finally:
        db.close()
    click.echo(f"Creative job {job_id} queued.")
    log.info("Creative job queued", extra={"job_id": job_id, "platform": platform_norm})

    options = EditOptions(
        creative_mode=True,
        creative_request=request,
        game=game,
        platform=platform_norm,
        target_duration=target_duration,
        aspect_ratio=aspect_ratio,
        bgm_track=bgm,
        chronological_order=chronological,
        full_session_coverage=coverage,
        variation_seed=job_id,
    )

    def _progress(icon: str, msg: str) -> None:
        click.echo(f"{icon}  {msg}")

    creative_model = None
    if plan_file:
        from app.ai.manual_plan_model import ManualPlanModel

        plan_text = pathlib.Path(plan_file).read_text(encoding="utf-8", errors="replace")
        exts = {".mp4", ".mov", ".mkv"}
        source_files = sorted(
            p for p in pathlib.Path(input_path).iterdir() if p.suffix.lower() in exts
        )
        creative_model = ManualPlanModel(plan_text, source_files)
        click.echo(f"Using web-chat AI plan from {plan_file}")

    try:
        output_path = orchestrate_job(job_id, options=options, progress_callback=_progress,
                                      model=creative_model)
    except Exception as exc:
        log.exception("Creative job failed", extra={"job_id": job_id})
        click.echo(f"Job {job_id} FAILED: {exc}", err=True)
        sys.exit(1)
    plan_path = pathlib.Path(str(output_path)).with_suffix(".edit-plan.json")
    click.echo(f"Job {job_id} completed -> {output_path}")
    if plan_path.is_file():
        click.echo(f"Edit plan artifact -> {plan_path}")
    log.info("Creative job completed", extra={"job_id": job_id, "output": str(output_path)})


@cli.command()
def evaluate() -> None:
    """Report dataset and active-model statistics without retraining."""
    from app.training.reference_pipeline import ReferenceTrainingPipeline

    pipeline = ReferenceTrainingPipeline()
    examples = pipeline._read_examples()
    unique = {str(item.get("reference_id")): item for item in examples if item.get("reference_id")}
    creators = len({item.get("creator_group") or item.get("reference_id") for item in unique.values()})
    youtube_examples = sum(
        1 for item in unique.values()
        if (item.get("provenance") or {}).get("platform") == "youtube"
    )
    feature_names = sorted({
        name
        for item in unique.values()
        for name, value in (item.get("features") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    active_path = pipeline.patterns / "active.json"
    active = None
    if active_path.is_file():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            active = None
    model_report: dict = {
        "status": "no_active_model",
        "hint": "Import rights-cleared references, then run: python main.py train",
    }
    if active:
        model_report = {
            "status": "active",
            "version": active.get("version"),
            "created_at": active.get("created_at"),
            "validation": active.get("validation"),
            "path": str(active_path),
            "platforms": {
                platform: {
                    "training_status": profile.get("training_status"),
                    "example_count": profile.get("example_count"),
                    "cluster_count": len((profile.get("style_clusters") or {}).get("clusters", [])),
                    "cluster_labels": [
                        cluster.get("label")
                        for cluster in (profile.get("style_clusters") or {}).get("clusters", [])
                    ],
                    "relationship_count": len(profile.get("relationships", [])),
                    "conditional_contexts": len(profile.get("conditional_edit_probabilities", {})),
                }
                for platform, profile in (active.get("profiles") or {}).items()
                if isinstance(profile, dict)
            },
        }
    click.echo(json.dumps({
        "dataset": {
            "reference_count": len(unique),
            "creator_count": creators,
            "local_rights_cleared_examples": len(unique) - youtube_examples,
            "youtube_derived_examples": youtube_examples,
            "feature_count": len(feature_names),
            "features": feature_names,
        },
        "model": model_report,
    }, indent=2))


@cli.command()
def train() -> None:
    """Train/promote from rights-cleared local reference examples only."""
    from app.research.video_observer import VideoObservationStore
    from app.training.reference_pipeline import ReferenceTrainingPipeline

    started = time.perf_counter()
    pipeline = ReferenceTrainingPipeline()
    result = pipeline.train_candidate()
    examples = pipeline._read_examples()
    youtube_observation_count = sum(
        len(item.get("provenance", {}).get("observation_ids", []))
        for item in examples if item.get("provenance", {}).get("platform") == "youtube"
    )
    feature_names = {
        name for item in examples for name, value in (item.get("features") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    research_root = pipeline.root / "research"
    metadata_videos = VideoObservationStore(research_root).stats()["video_count"]
    local_examples = sum(not item.get("provenance", {}).get("platform") == "youtube" for item in examples)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "training_timestamp": timestamp,
        "status": result.get("status"),
        "dataset_version": "v001",
        "dataset_examples": len(examples),
        "rights_cleared_local_examples": local_examples,
        "youtube_metadata_videos": metadata_videos,
        "approved_youtube_observations_used": youtube_observation_count,
        "feature_count": len(feature_names),
        "feature_names": sorted(feature_names),
        "model_type": "aggregate per-video editing-style profile" if result.get("status") != "insufficient_references" else None,
        "training_count": result.get("training_count", 0),
        "validation_count": result.get("validation_count", 0),
        "validation_metrics": {
            key: result[key] for key in ("validation_mae", "baseline_mae", "test_mae") if key in result
        },
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "model_version": result.get("version"),
        "model_path": result.get("path"),
        "limitations": (
            "Insufficient authorized examples; no model was trained or promoted."
            if result.get("status", "").startswith("insufficient") else None
        ),
    }
    pipeline._append_jsonl(pipeline.analysis / "training_runs.jsonl", report)
    click.echo(json.dumps({
        "training_report": report,
        "result": result,
        "training_source": "rights-cleared local references and explicitly approved observations",
    }, indent=2))


@cli.command(name="collect-references")
@click.option('--provider', default='pexels', show_default=True,
              type=click.Choice(['pexels', 'internet_archive']),
              help='pexels (needs free PEXELS_API_KEY) or internet_archive (no key; CC/PD items only).')
@click.option('--limit', default=8, type=click.IntRange(1, 24), show_default=True,
              help='How many licensed references to download and import.')
@click.option('--query', 'queries', multiple=True,
              help='Override default stock queries (repeatable).')
@click.option('--orientation', default='portrait', show_default=True,
              type=click.Choice(['portrait', 'landscape', 'square']))
@click.option('--category', default='gaming', show_default=True,
              type=click.Choice(sorted(['valorant', 'fps', 'gaming', 'esports', 'montage', 'highlights', 'funny'])))
def collect_references(provider: str, limit: int, queries: tuple[str, ...], orientation: str, category: str) -> None:
    """AUTO training-data collection from legally downloadable licensed media.

    Providers: Pexels (license permits download+reuse; free PEXELS_API_KEY in
    .env) and Internet Archive (official open API, no key; only Creative
    Commons / Public Domain items accepted). Downloads short edited videos
    (<=180 s) with per-uploader diversity caps, persists license records, and
    imports them into the training dataset with automatic feature analysis.
    No YouTube/Instagram media is ever downloaded. Follow with:
    python main.py train
    """
    from app.research.licensed_media_provider import (
        InternetArchiveProvider,
        LicensedMediaError,
        PexelsVideoProvider,
    )

    try:
        media_provider = (
            InternetArchiveProvider() if provider == "internet_archive" else PexelsVideoProvider()
        )
    except LicensedMediaError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Collecting up to {limit} licensed reference(s) from {provider}...")
    result = media_provider.collect(
        limit=limit,
        queries=list(queries) or None,
        orientation=orientation,
        category=category,
    )
    click.echo(json.dumps({
        "status": "collected",
        "provider": provider,
        "downloaded_and_imported": len(result["imported"]),
        "skipped": len(result["skipped"]),
        "imported": result["imported"],
        "skip_reasons": result["skipped"],
        "license": "See training/licensed_licenses/ for per-file records",
        "youtube_or_instagram_media_downloaded": False,
        "next_step": "python main.py train   (then: python main.py evaluate)",
    }, indent=2))


@cli.command(name="import-reference")
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False), metavar='VIDEO')
@click.option('--rights-basis', required=True,
              type=click.Choice(['user_owned', 'licensed', 'public_domain', 'explicitly_permitted']),
              help='Your rights basis for using this video as a training reference.')
@click.option('--platform', default='youtube_shorts', show_default=True,
              type=click.Choice(['youtube_shorts', 'instagram_reels', 'gaming_shorts', 'all']))
@click.option('--style-tags', default='', help='Comma-separated style tags, e.g. "montage,aggressive".')
@click.option('--creator-group', default=None, help='Anonymous creator/channel grouping (for leak-free splits).')
@click.option('--category', default='gaming', show_default=True,
              type=click.Choice(sorted(['valorant', 'fps', 'gaming', 'esports', 'montage', 'highlights', 'funny'])))
def import_reference(video_path: str, rights_basis: str, platform: str, style_tags: str,
                     creator_group: str | None, category: str) -> None:
    """Import a RIGHTS-CLEARED video as a local training reference.

    Analyzes the video offline (scene cuts, pacing, loudness, silence) and
    adds it to the training dataset. Use your own edits or properly licensed
    material only — never feed it copyrighted videos you don't have rights
    to. Training (python main.py train) consumes these references.
    """
    from app.training.reference_pipeline import ReferenceTrainingPipeline

    tags = [tag.strip() for tag in style_tags.split(",") if tag.strip()] or None
    pipeline = ReferenceTrainingPipeline()
    try:
        example = pipeline.import_local_video(
            pathlib.Path(video_path),
            rights_basis=rights_basis,
            platform=platform,
            style_tags=tags,
            creator_group=creator_group,
            category=category,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    features = example.get("features", {})
    click.echo(json.dumps({
        "status": "imported",
        "reference_id": example.get("reference_id", "")[:16] + "...",
        "rights_basis": rights_basis,
        "platform": platform,
        "category": category,
        "style_tags": example.get("style_tags"),
        "key_features": {
            key: features.get(key)
            for key in ("duration", "cut_density", "average_shot_duration",
                        "shot_duration_std", "pacing_irregularity",
                        "silence_ratio", "audio_mean_db")
        },
        "next_step": "Add more references (>=5 distinct, >=3 creator groups), then: python main.py train",
    }, indent=2))


@cli.command(name="grammar-extract")
@click.argument('videos', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('--from-references', is_flag=True,
              help='Extract from every imported training reference (training/raw).')
def grammar_extract(videos: tuple[str, ...], from_references: bool) -> None:
    """Extract editing-grammar sequences from reference videos (the TEACHER pass).

    Runs the AutomaticReferenceAnalyzer (scene cuts, motion envelope, audio
    energy, silence → labeled timeline tokens) over legally obtained
    references and appends them to training/grammar/sequences.jsonl.
    Follow with: python main.py train-policy
    """
    from app.learning.sequence_dataset import SequenceDataset
    from app.learning.sequence_extractor import AutomaticReferenceAnalyzer

    dataset = SequenceDataset()
    analyzer = AutomaticReferenceAnalyzer()
    targets: list[tuple[pathlib.Path, dict]] = []
    if from_references:
        from app.training.reference_pipeline import ReferenceTrainingPipeline
        for example in ReferenceTrainingPipeline()._read_examples():
            source_path = example.get("source_path")
            if source_path and pathlib.Path(source_path).is_file():
                targets.append((pathlib.Path(source_path), example))
    targets += [(pathlib.Path(video), {}) for video in videos]
    if not targets:
        raise click.UsageError("No videos given and --from-references found none. "
                               "Import references first (collect-references / import-reference).")
    extracted, skipped = [], []
    for index, (path, meta) in enumerate(targets, start=1):
        click.echo(f"[OBSERVER] Analyzing reference {index}/{len(targets)}: {path.name}")
        record = analyzer.analyze_file(path, meta)
        if record is None:
            skipped.append(str(path))
            continue
        dataset.add(record)
        tokens = [seg["token"] for seg in record["sequence"]]
        extracted.append({
            "reference_id": record["reference_id"][:16],
            "duration": record["duration"],
            "segments": len(tokens),
            "tokens_head": tokens[:8],
            "cut_rate": record["stats"].get("cut_rate"),
        })
    click.echo(json.dumps({
        "status": "extracted",
        "extracted": extracted,
        "skipped": skipped,
        "dataset_sequences": len(dataset.records()),
        "next_step": "python main.py train-policy",
    }, indent=2))


@cli.command(name="train-policy")
def train_policy() -> None:
    """Train/validate/promote the editing policy from the grammar dataset.

    Honest reporting (spec §13): sequence counts, creator counts, held-out
    log-likelihood vs uniform baseline vs previous policy, promotion decision,
    learning state. Never claims training it did not do.
    """
    from app.learning.policy_model import PolicyTrainer, learning_state
    from app.learning.sequence_dataset import SequenceDataset

    dataset = SequenceDataset()
    trainer = PolicyTrainer()
    records = dataset.records()
    if not records:
        click.echo(json.dumps({
            "status": "insufficient_sequences", "sequence_count": 0,
            "learning_state": "NO_DATA",
            "next_step": "python main.py grammar-extract --from-references",
        }, indent=2))
        return
    dataset_info = dataset.snapshot()
    partitions = dataset.creator_split()
    held_out = partitions["validation"] + partitions["test"]
    result = trainer.train_candidate(partitions["train"], held_out, dataset_info)
    result["learning_state"] = learning_state(len(records), trainer)
    result["dataset"] = dataset_info
    click.echo(json.dumps({"training_report": result}, indent=2))


@cli.command(name="policy-status")
def policy_status() -> None:
    """Report the learning state machine, dataset, policy and trajectory stats."""
    from app.learning.policy_model import PolicyTrainer, learning_state
    from app.learning.sequence_dataset import SequenceDataset

    dataset = SequenceDataset()
    trainer = PolicyTrainer()
    records = dataset.records()
    active = trainer.load_active()
    trajectories_path = trainer.root.parent / "trajectories" / "edits.jsonl"
    trajectory_count = 0
    if trajectories_path.is_file():
        trajectory_count = sum(1 for line in trajectories_path.read_text(encoding="utf-8").splitlines() if line.strip())
    click.echo(json.dumps({
        "learning_state": learning_state(len(records), trainer),
        "grammar_sequences": len(records),
        "creators": len({str(r.get("creator_group")) for r in records}),
        "dataset_snapshot": dataset.state().get("last_snapshot"),
        "policy": ({
            "version": active.get("version"),
            "stage": active.get("stage"),
            "validation": active.get("validation"),
            "dataset": active.get("dataset"),
            "used_in_edits": (active.get("usage") or {}).get("edits_used_in", 0),
            "unknown_channels": active.get("unknown_channels"),
        } if active else None),
        "edit_trajectories_recorded": trajectory_count,
    }, indent=2))


@cli.command(name="ab-test")
@click.argument('input_path', type=click.Path(exists=True, file_okay=False), metavar='CLIP_DIR')
@click.option('--baseline', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Prior edit-plan artifact to compare against (e.g. output/job_91_final.edit-plan.json).')
@click.option('--platform', default='youtube_shorts', show_default=True)
@click.option('--duration', 'target_duration', default=40, type=click.IntRange(5, 180), show_default=True)
@click.option('--game', default='valorant', show_default=True)
def ab_test(input_path: str, baseline: str | None, platform: str, target_duration: int, game: str) -> None:
    """A/B: legacy activity planner vs learned policy on the SAME footage.

    Plan-level comparison (no rendering): builds both plans, scores both with
    the learned policy, records a preference entry (spec §37/§42), and prints
    the full comparison — including the exact baseline artifact values when
    --baseline is supplied.
    """
    from app.analysis.games import get_event_detector
    from app.analysis.media_context import analyze_sources
    from app.learning.policy_inference import PolicyInference
    from app.learning.policy_model import PolicyTrainer
    from app.learning.policy_planner import LearnedPolicyPlanner, get_learning_state

    clip_dir = pathlib.Path(input_path)
    exts = {".mp4", ".mov", ".mkv"}
    files = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise click.UsageError(f"No clips in {clip_dir}")
    click.echo(f"Analyzing {len(files)} source(s)...")
    media_context, frames = analyze_sources(files)
    detector = get_event_detector(game)
    if detector is not None:
        for index, source in enumerate(media_context.get("sources") or []):
            if index >= len(files):
                break
            try:
                events = detector.detect(files[index])
            except Exception:
                events = []
            source["gameplay_events"] = [
                {"kind": e.get("kind"), "start": e.get("start"), "end": e.get("end", e.get("start")),
                 "confidence": e.get("confidence")}
                for e in events
            ]

    # Candidate A — legacy activity planner
    from app.ai.local_editing import LocalShortFormEditingModel
    legacy = LocalShortFormEditingModel()
    plan_a = legacy.create_plan(files, media_context, platform, target_duration)

    # Candidate B — learned policy
    state = get_learning_state()
    plan_b = None
    planner_b = None
    if state in ("TRAINED", "VALIDATED", "ACTIVE"):
        planner_b = LearnedPolicyPlanner()
        plan_b = planner_b.create_plan(
            media_context=media_context, frames=frames, style_profile={},
            platform=platform, target_duration=target_duration,
            user_preferences={"source_paths": [str(f) for f in files],
                              "chronological_assembly": True, "full_session_coverage": True},
            seed=1,
        )

    trainer = PolicyTrainer()
    active = trainer.load_active()
    inference = PolicyInference(active) if active else None

    def summarize(plan: dict, label: str) -> dict:
        shots = plan.get("shots") or []
        durations = [float(s["end"]) - float(s["start"]) for s in shots]
        total = sum(durations) or 1.0
        scored = None
        if inference is not None and shots:
            candidate = {
                "name": plan.get("strategy"), "source_count": len(media_context["sources"]),
                "target_duration": target_duration,
                "shots": [{**s, "intensity": None, "has_event": False} for s in shots],
            }
            scored = inference.score_structure(candidate)
        return {
            "candidate": label,
            "strategy": plan.get("strategy"),
            "shot_count": len(shots),
            "durations": [round(d, 2) for d in durations],
            "cut_density": round(max(0, len(shots) - 1) / total, 3),
            "sources_used": sorted({int(s["source_index"]) for s in shots}),
            "captions": sum(1 for s in shots if s.get("caption")),
            "emphasis": sum(1 for s in shots if s.get("visual_emphasis")),
            "music_requested": bool(plan.get("music_requirements")),
            "policy_score": (scored or {}).get("composite"),
            "score_breakdown": (scored or {}).get("components"),
        }

    comparison: dict[str, Any] = {
        "learning_state": state,
        "candidate_a_legacy": summarize(plan_a, "A_legacy_activity"),
    }
    if plan_b is not None:
        comparison["candidate_b_policy"] = summarize(plan_b, "B_learned_policy")
        comparison["candidate_c_policy_retrieval"] = {
            "candidate": "C_policy_plus_retrieval",
            "retrieved_references": (planner_b.diagnostics or {}).get("retrieved_references"),
            "note": "retrieval blending is included in B's score when references match",
        }
        preference = {
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input": str(clip_dir),
            "candidates": {
                "A_legacy_activity": comparison["candidate_a_legacy"]["policy_score"],
                "B_learned_policy": comparison["candidate_b_policy"]["policy_score"],
            },
            "preferred": (
                "B_learned_policy"
                if (comparison["candidate_b_policy"]["policy_score"] or 0)
                >= (comparison["candidate_a_legacy"]["policy_score"] or 0)
                else "A_legacy_activity"
            ),
        }
        pref_path = trainer.root.parent / "trajectories" / "preferences.jsonl"
        pref_path.parent.mkdir(parents=True, exist_ok=True)
        with pref_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(preference, ensure_ascii=True) + "\n")
        comparison["preference_recorded"] = preference
    else:
        comparison["candidate_b_policy"] = {
            "status": "BLOCKED",
            "reason": f"learning_state={state}; run: python main.py grammar-extract --from-references && python main.py train-policy",
        }
    if baseline:
        base = json.loads(pathlib.Path(baseline).read_text(encoding="utf-8"))
        base_timeline = base.get("timeline") or []
        comparison["baseline_artifact"] = {
            "path": str(baseline),
            "planning_mode": base.get("planning_mode"),
            "model_version": base.get("model_version"),
            "strategy": base.get("strategy"),
            "shots": len(base_timeline),
            "durations": [item.get("duration") for item in base_timeline],
            "captions": len(base.get("captions") or []),
            "sfx": len(base.get("sfx") or []),
            "music_enabled": (base.get("music_mix") or {}).get("enabled"),
            "shot_order": base.get("shot_order"),
            "revision_count": base.get("revision_count"),
        }
    click.echo(json.dumps(comparison, indent=2))


@cli.command(name="events")
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False), metavar='VIDEO')
@click.option('--game', default='valorant', show_default=True, help='Game profile with an event detector.')
def events(video_path: str, game: str) -> None:
    """Detect gameplay events (e.g. VALORANT kills) in a video file.

    Prints timestamped events with confidence scores. Use this to verify
    detection against real kills and to tune config/<game>.yaml
    event_detection thresholds for your recording/HUD setup.
    """
    from app.analysis.games import get_event_detector

    detector = get_event_detector(game)
    if detector is None:
        raise click.UsageError(f"No event detector for game {game!r}. Supported: valorant.")
    detected = detector.detect(pathlib.Path(video_path))
    click.echo(json.dumps({
        "video": video_path,
        "game": game,
        "event_count": len(detected),
        "events": detected,
        "hint": (
            "Compare event times against real kills; if detection misses or "
            "over-fires, tune event_detection thresholds in config/<game>.yaml."
        ),
    }, indent=2))


@cli.command(name="train-mode")
@click.option('--limit', default=6, type=click.IntRange(1, 24), show_default=True,
              help='Max new licensed references to download this run.')
@click.option('--offline', is_flag=True,
              help='Skip all downloads; train/evaluate on the existing dataset only.')
@click.option('--with-youtube-pool/--no-youtube-pool', default=True, show_default=True,
              help='Also refresh the YouTube metadata pool (metadata only, needs YOUTUBE_DATA_API_KEY).')
def train_mode(limit: int, offline: bool, with_youtube_pool: bool) -> None:
    """SELF-TRAINING LOOP: legal online collection → dataset → train → evaluate.

    Designed for repeated runs. Each run rotates through a gaming-montage
    query bank for the Internet Archive (CC/PD items only), uses Pexels when
    PEXELS_API_KEY is set, refreshes YouTube metadata priors (metadata only —
    never downloads YouTube/Instagram media), then trains a candidate model
    (promoted only when it beats baseline and the active model) and reports
    honest statistics. Close the loop after each edit with:
    python main.py feedback <job-id> --rating 1 --tags hook_good
    """
    from app.training.reference_pipeline import ReferenceTrainingPipeline

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    stages: list[dict] = []
    pipeline = ReferenceTrainingPipeline()

    if not offline:
        from app.research.licensed_media_provider import (
            ARCHIVE_QUERY_BANK,
            InternetArchiveProvider,
            PexelsVideoProvider,
        )

        # Rotate 3 query-bank entries per run so repeated runs keep finding
        # NEW items instead of re-hitting dedup on identical top results.
        state_path = pipeline.root / "licensed_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        bank = list(ARCHIVE_QUERY_BANK)
        offset = int(state.get("query_offset", 0)) % len(bank)
        picked = [bank[(offset + index) % len(bank)] for index in range(3)]
        click.echo(f"[TRAINING] Internet Archive (CC/PD only), queries: {picked}")
        try:
            result = InternetArchiveProvider().collect(limit=limit, queries=picked)
            stages.append({
                "stage": "internet_archive",
                "imported": len(result["imported"]),
                "skipped": len(result["skipped"]),
                "queries": picked,
                "files": [item["file"] for item in result["imported"]],
            })
        except Exception as exc:
            stages.append({"stage": "internet_archive", "error": type(exc).__name__})
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "query_offset": (offset + 3) % len(bank),
            "runs": int(state.get("runs", 0)) + 1,
            "last_queries": picked,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")

        if os.getenv("PEXELS_API_KEY"):
            click.echo("[TRAINING] Pexels rhythm/grammar references...")
            try:
                result = PexelsVideoProvider().collect(limit=limit)
                stages.append({"stage": "pexels", "imported": len(result["imported"]),
                               "skipped": len(result["skipped"])})
            except Exception as exc:
                stages.append({"stage": "pexels", "error": type(exc).__name__})
        else:
            stages.append({"stage": "pexels", "skipped": "PEXELS_API_KEY not set"})

        if with_youtube_pool:
            if os.getenv("YOUTUBE_DATA_API_KEY") or os.getenv("YOUTUBE_API_KEY"):
                click.echo("[TRAINING] Refreshing YouTube metadata pool (metadata only)...")
                try:
                    from app.training.collector import collect_candidate_pool
                    pool = collect_candidate_pool()
                    stages.append({"stage": "youtube_metadata_pool",
                                   "candidate_count": pool.get("candidate_count"),
                                   "queries_this_run": pool.get("queries_this_run")})
                except Exception as exc:
                    stages.append({"stage": "youtube_metadata_pool", "error": type(exc).__name__})
            else:
                stages.append({"stage": "youtube_metadata_pool",
                               "skipped": "YOUTUBE_DATA_API_KEY not set"})
    else:
        stages.append({"stage": "downloads", "skipped": "offline mode"})

    examples = pipeline._read_examples()
    unique = {str(item.get("reference_id")): item for item in examples if item.get("reference_id")}
    creators = len({item.get("creator_group") or item.get("reference_id") for item in unique.values()})
    stages.append({"stage": "dataset", "reference_count": len(unique), "creator_count": creators})

    click.echo("[TRAINING] Training candidate model...")
    train_result = pipeline.train_candidate()
    stages.append({"stage": "train", **{
        key: train_result.get(key)
        for key in ("status", "version", "example_count", "creator_count",
                    "training_count", "validation_count", "validation_mae",
                    "baseline_mae", "test_mae", "limitations")
        if train_result.get(key) is not None
    }})

    model_stage: dict = {"status": "no_active_model"}
    active_path = pipeline.patterns / "active.json"
    if active_path.is_file():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            model_stage = {"status": "active", "version": active.get("version"),
                           "validation": active.get("validation")}
        except json.JSONDecodeError:
            pass
    stages.append({"stage": "model", **model_stage})

    click.echo(json.dumps({
        "training_mode_report": {
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "offline": offline,
            "stages": stages,
        },
        "next_steps": [
            "python main.py evaluate",
            "python main.py edit input\\<your_session> --platform youtube_shorts --duration 40 --request \"intense clutch montage\"",
            "python main.py feedback <job-id> --rating 1 --tags hook_good,kills_well_aligned",
            "Then run train-mode again — your feedback signals feed the next candidate model.",
        ],
    }, indent=2))


@cli.command()
@click.argument('edit_id')
@click.option('--rating', required=True, type=click.Choice(['1', '-1']),
              help='1 = good edit, -1 = bad edit.')
@click.option('--tags', default='',
              help='Comma-separated: hook_good, kills_well_aligned, kills_misaligned, '
                   'too_fast, too_slow, too_many_effects, bgm_mismatch, captions_good, transitions_bad')
@click.option('--note', default='', help='Free-text note (max 1000 chars).')
def feedback(edit_id: str, rating: str, tags: str, note: str) -> None:
    """Record YOUR validation of a finished edit into the training loop.

    Only completed jobs are accepted. Per-tag positive/negative counts become
    feedback_signals attached to every future candidate model — this is how
    your verdicts steer training without any manual dataset work.
    """
    from app.storage.db import SessionLocal
    from app.storage.models import Job
    from app.training.reference_pipeline import FEEDBACK_TAGS, ReferenceTrainingPipeline

    db = SessionLocal()
    try:
        try:
            job = db.get(Job, int(edit_id))
        except (TypeError, ValueError):
            job = None
        if job is None:
            raise click.ClickException(f"Job {edit_id!r} not found.")
        if job.status != "completed" or not job.output_path:
            raise click.ClickException(
                f"Feedback is only accepted for completed edits (job {job.id} is '{job.status}')."
            )
        context: dict = {"job_id": job.id, "output": pathlib.Path(job.output_path).name}
        plan_path = pathlib.Path(job.output_path).with_suffix(".edit-plan.json")
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                context.update({"strategy": plan.get("strategy"), "platform": plan.get("platform")})
            except (OSError, json.JSONDecodeError):
                pass
    finally:
        db.close()

    requested = [tag.strip() for tag in tags.split(",") if tag.strip()]
    selected = [tag for tag in requested if tag in FEEDBACK_TAGS]
    ignored = [tag for tag in requested if tag not in FEEDBACK_TAGS]
    path = ReferenceTrainingPipeline().record_feedback(
        edit_id=str(edit_id),
        rating=int(rating),
        tags=selected,
        notes=note[:1000],
        context=context,
    )
    click.echo(json.dumps({
        "status": "recorded",
        "edit_id": edit_id,
        "rating": int(rating),
        "tags": selected,
        "ignored_tags": ignored,
        "file": str(path),
        "used_by": "feedback_signals in every future trained model version",
    }, indent=2))


if __name__ == "__main__":
    # Load configuration early so that env overrides are applied
    _ = load_config()
    cli()
