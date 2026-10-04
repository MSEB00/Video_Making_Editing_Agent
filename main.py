"""
main.py
--------
CLI entry point for the Gaming Video Agent.
Provides commands to initialise the database and to start a processing job.
"""
from __future__ import annotations

import sys
import json
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
    cfg = load_config()
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
@click.option('--topic', required=True, help='Configurable short-form research topic or style.')
@click.option('--limit', default=5, type=click.IntRange(1, 10), show_default=True, help='Maximum search results for this explicit query.')
def research(topic: str, limit: int) -> None:
    """Discover public YouTube metadata for manual embedded-player research."""
    from app.research.video_observer import VideoObservationStore
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


if __name__ == "__main__":
    # Load configuration early so that env overrides are applied
    _ = load_config()
    cli()
