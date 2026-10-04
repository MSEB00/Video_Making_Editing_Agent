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


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--style', default='FAST_PACED', help='Editing style to apply')
@click.option('--platform', default='youtube', help='Target platform for rendering')
def process(input_path: str, style: str, platform: str) -> None:
    """Create a new processing job for *input_path*.

    This is a placeholder – the full pipeline will be built in later phases.
    """
    cfg = load_config()
    from app.storage.db import get_db

    log.info(
        "Starting job",
        extra={"input": input_path, "style": style, "platform": platform},
    )
    # Minimal DB interaction – create a Job record
    from app.storage.models import Job
    db_gen = get_db()
    db = next(db_gen)
    job = Job(
        status="queued",
        input_path=input_path,
        metadata=f"{{'style':'{style}','platform':'{platform}'}}",
    )
    db.add(job)
    db.commit()
    click.echo(f"Job {job.id} queued.")
    log.info("Job queued", extra={"job_id": job.id})


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
