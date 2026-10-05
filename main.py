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
from dotenv import load_dotenv

from app.config.config_loader import load_config
from app.utilities.logger import get_logger

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

log = get_logger(__name__)

@click.group()
def cli() -> None:
    """Top level command group."""
    pass


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
def edit(input_path: str, platform: str, game: str, request: str, target_duration: int, bgm: str | None,
         shot_order: str, source_clips: str) -> None:
    """Run the full CREATIVE pipeline: AI plan → render → review → revise.

    Uses the hosted creative model when configured (GEMINI_API_KEY /
    OPENAI_API_KEY) and falls back to the measured local feature planner
    otherwise. Produces output/job_N_final.mp4 plus an .edit-plan.json
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

    try:
        output_path = orchestrate_job(job_id, options=options, progress_callback=_progress)
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
