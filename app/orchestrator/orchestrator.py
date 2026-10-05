"""Orchestrator for the Gaming Video Agent.

Loads a Job, gathers video files from its input directory, records each as a
Media entry (kind='raw'), applies professional editing (transitions, BGM,
aspect ratio formatting, color grading), and stores the final path back on the Job.
"""

from __future__ import annotations

import json
import pathlib
import os
from typing import List, Optional, Callable

from sqlalchemy.orm import Session

from app.storage.db import SessionLocal
from app.storage.models import Job, Media
from app.editing.editor import render_edited_video, EditOptions
from app.utilities.logger import get_logger

log = get_logger(__name__)

def _gather_media_files(input_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return sorted list of video files (mp4, mov, mkv) in *input_dir*.
    Sorting respects the timestamp ordering embedded in folder names.
    """
    exts = {".mp4", ".mov", ".mkv"}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    files.sort()
    log.debug("Found %d video files", len(files), extra={"dir": str(input_dir)})
    return files

def orchestrate_job(
    job_id: int,
    options: Optional[EditOptions] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    model: Optional[object] = None
) -> pathlib.Path:
    """Run the full editing pipeline for a given *job_id*.
    Returns the path to the rendered output video.
    """
    session: Session = SessionLocal()
    job: Job = session.get(Job, job_id)
    if not job:
        session.close()
        raise ValueError(f"Job {job_id} not found")
    input_path = pathlib.Path(job.input_path)
    if not input_path.is_dir():
        session.close()
        raise ValueError(f"Input path {input_path} is not a directory")
    
    media_files = _gather_media_files(input_path)
    if not media_files:
        session.close()
        raise RuntimeError(f"No video files in {input_path}")

    # Resolve edit options
    if options is None:
        opts_dict = {}
        if job.extra_metadata:
            try:
                opts_dict = json.loads(job.extra_metadata)
            except Exception:
                opts_dict = {}
        options = EditOptions(
            transition_type=opts_dict.get("transition_type", "random"),
            transition_duration=float(opts_dict.get("transition_duration", 0.75)),
            bgm_track=opts_dict.get("bgm_track"),
            bgm_volume=float(opts_dict.get("bgm_volume", 0.30)),
            aspect_ratio=opts_dict.get("aspect_ratio", "16:9"),
            color_grade=bool(opts_dict.get("color_grade", True)),
            max_clips=opts_dict.get("max_clips"),
            max_clip_duration=opts_dict.get("max_clip_duration"),
            title_text=opts_dict.get("title_text"),
            randomize_clips=bool(opts_dict.get("randomize_clips", False)),
            variation_seed=opts_dict.get("variation_seed", job.id),
            sfx_preference=opts_dict.get("sfx_preference"),
            creative_mode=bool(opts_dict.get("creative_mode", False)),
            creative_request=str(opts_dict.get("creative_request", "")),
            game=opts_dict.get("game", "valorant"),
            platform=opts_dict.get("platform", "youtube_shorts"),
            target_duration=int(opts_dict.get("target_duration", 45))
        )

    try:
        job.status = "processing"
        session.commit()

        # Record Media entries (avoid duplicates)
        for p in media_files:
            exists = (
                session.query(Media)
                .filter(Media.job_id == job.id, Media.path == str(p))
                .first()
            )
            if not exists:
                session.add(Media(job_id=job.id, kind="raw", path=str(p)))
        session.commit()

        # Determine output location
        output_dir = pathlib.Path(os.getenv("OUTPUT_DIR", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"job_{job.id}_final.mp4"

        if options.creative_mode:
            from app.agent.short_form_editor import ShortFormCreativeEditor

            creative_artifact = ShortFormCreativeEditor(model=model).create_edit(
                input_files=media_files,
                output_path=output_file,
                options=options,
                progress_callback=progress_callback,
            )
            saved_metadata = json.loads(job.extra_metadata) if job.extra_metadata else {}
            saved_metadata["creative_edit"] = creative_artifact
            job.extra_metadata = json.dumps(saved_metadata)
        else:
            render_edited_video(
                input_files=media_files,
                output_path=output_file,
                options=options,
                progress_callback=progress_callback
            )

        # Update job
        job.status = "completed"
        job.output_path = str(output_file)
        session.commit()
        log.info("Job %s completed, output at %s", job.id, output_file)
        return output_file

    except Exception as e:
        job.status = "failed"
        session.commit()
        log.exception("Job %s failed: %s", job.id, e)
        raise e
    finally:
        session.close()
