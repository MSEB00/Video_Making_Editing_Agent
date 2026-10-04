# Gaming Video Agent — Video Making & Editing Agent

An automated pipeline that turns raw gameplay recordings into polished,
platform-ready short-form videos (YouTube Shorts / Instagram Reels / 16:9
YouTube), with AI-assisted creative editing, rights-aware background music,
a reference-video research workflow, and a chat-driven dashboard.

## Pipeline Overview

```
input/<session>/*.mp4  →  Orchestrator  →  Editor (FFmpeg)  →  output/job_N_final.mp4  →  Publisher
                                              │
                          ┌───────────────────┼────────────────────────┐
                          │  classic mode     │  creative mode         │
                          │  style presets    │  AI plan (Gemini/      │
                          │  transitions      │  OpenAI-compatible)    │
                          │  BGM + SFX mix    │  Jamendo music search  │
                          │  color grade      │  learned style profile │
                          │  16:9 / 9:16      │  captions + emphasis   │
                          └───────────────────┴────────────────────────┘
```

## Features

- **CLI + chat dashboard** — run jobs from the terminal or from a Flask/
  SocketIO web UI with live progress steps.
- **Pro editing engine (FFmpeg)** — xfade transitions (11 styles),
  bounded-memory audio crossfade mixdown, color grading, title/caption
  overlays, SFX hits, BGM mixing with ducking, loudness normalization
  (creative mode), NVENC hardware encoding with automatic CPU fallback.
- **Gameplay-event alignment** — a kill-feed detector (FFmpeg region
  sampling + Pillow statistics, no extra dependencies) finds timestamped
  eliminations in VALORANT footage; both the hosted AI planner (events are
  part of its context) and a deterministic post-plan snap place each kill
  as the shot's payoff: buildup before, ~1 s hold after, never cutting
  mid-event. Tunable in `config/valorant.yaml`; disable via
  `creative_editing.align_to_gameplay_events`.
- **Aspect-ratio targeting** — 16:9 (YouTube) and 9:16 (Shorts/Reels/
  TikTok) with crop-first vertical reframing (no upscale-then-crop waste)
  and optional punch-zoom emphasis.
- **AI creative mode** — an OpenAI-compatible planner (Google Gemini by
  default, OpenAI optional) analyses low-res frames + media facts, then
  produces a shot list, captions, emphasis, transition plan, and music
  direction; a render-critique step allows one revision pass.
- **Rights-aware audio** — Jamendo API search/download for licensed BGM
  (with metadata cache), plus a locally generated SFX library. No
  copyrighted audio is ever fetched.
- **YouTube research workflow** — discovers public video *metadata only*
  (never downloads media) via the YouTube Data API with a diversity-aware
  curriculum that preferentially researches underrepresented styles; a
  local-only `/research` page supports manual observation notes; explicitly
  approved observations plus your own rights-cleared references feed an
  offline training pipeline that learns aggregate editing-style profiles
  (cut density, caption density, pacing irregularity, silence ratio, ...),
  context-conditional edit probabilities P(decision | intensity/speech),
  and data-driven style clusters (k-means with silhouette selection).
- **Publishing** — YouTube upload via OAuth (desktop client) and
  Instagram Reels via the Graph API.
- **Job store** — SQLite via SQLAlchemy (`jobs`, `media`, `events`).
- **Structured logging** — colored console + JSON-lines files in `logs/`.

## Requirements

- Python 3.10+ (developed on 3.11)
- FFmpeg + FFprobe on `PATH` (or set `FFMPEG_PATH` / `FFPROBE_PATH`)
- Optional: NVIDIA GPU for NVENC hardware encoding (auto-detected,
  falls back to libx264)

## Setup

```bash
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests + lint tools

# 2. Configure environment
copy .env.example .env          # Windows (Linux/macOS: cp .env.example .env)
# then edit .env — see the comments inside

# 3. Initialise the database
python main.py initdb
```

## Usage

### CLI

```bash
# Render a folder of gameplay clips (16:9, FAST_PACED preset)
python main.py process input\my_session --style FAST_PACED --platform youtube

# Vertical Short with the montage preset
python main.py process input\my_session --style MONTAGE --platform youtube_shorts

# Only create the queued job record, run later from the dashboard
python main.py process input\my_session --queue-only

# Full CREATIVE pipeline: AI plan → render → review → revise (uses the
# hosted model when GEMINI_API_KEY/OPENAI_API_KEY is set, otherwise the
# measured local feature planner). Writes output\job_N_final.mp4 plus an
# .edit-plan.json artifact with full decision provenance.
python main.py edit input\my_session --platform youtube_shorts --duration 30 --request "fast montage, punchy hook"

# Discover reference metadata for a research topic (needs YOUTUBE_DATA_API_KEY)
python main.py research --topic "valorant clutch shorts" --limit 5

# Detect gameplay events (VALORANT kill feed) in a clip — use it to verify
# alignment quality and tune config/valorant.yaml thresholds
python main.py events "input\my_session\VALORANT clip.mp4" --game valorant

# Inspect research dataset + candidate pool status (no API calls)
python main.py research --inspect

# Train/promote the aggregate editing-style model from rights-cleared
# references and explicitly approved observations
python main.py train

# Report dataset + active-model statistics without retraining
python main.py evaluate
```

Style presets: `FAST_PACED` (default), `MONTAGE`, `CINEMATIC`, `SIMPLE`.
Platforms: `youtube` (16:9), `youtube_shorts` / `shorts` / `instagram_reels` /
`reels` / `tiktok` (9:16).

### Dashboard

```bash
python dashboard/app.py
# → http://localhost:5000
```

Chat-driven UI: list input folders, request edits in natural language
("make a Shorts video with wipe transitions and no SFX"), watch live
progress, publish finished jobs ("publish job 2 to YouTube"), and use the
local-only `/research` page for reference-video observations.

### Publishing credentials

- **YouTube**: create an OAuth *Desktop* client in Google Cloud Console,
  download the JSON, and point `YOUTUBE_CLIENT_SECRETS_FILE` at it
  (default `credentials/youtube_client_secret.json`). Never commit this
  file — `.gitignore` already blocks `*client_secret*.json`, `credentials/`
  and `*token*.json`. First upload opens a browser consent flow; the
  resulting token is cached at `YOUTUBE_TOKEN_FILE`.
- **Instagram Reels**: needs a long-lived `INSTAGRAM_ACCESS_TOKEN`, your
  `INSTAGRAM_USER_ID`, and `PUBLIC_BASE_URL` — an HTTPS URL that publicly
  serves this machine's `/output` folder (Instagram fetches the video
  file from there).

## Configuration

- `config/*.yaml` — deep-merged at startup (`app/config/config_loader.py`).
  Environment variables of the form `APP__SECTION__KEY` override nested
  keys. Files: `app.yaml` (core), `editing.yaml` (creative planner
  bounds), `audio.yaml`, `platforms.yaml`, `valorant.yaml` (HUD regions),
  `training_queries.yaml`.
- `.env` — secrets and runtime knobs; see `.env.example` for the full list.

## Project Layout

```
main.py                     CLI entry point (initdb / process / research / train)
app/
  agent/short_form_editor.py    creative-mode end-to-end agent
  ai/creative_editor.py         Gemini/OpenAI-compatible planner + critic
  ai/local_editing.py           offline heuristics (no-API fallback)
  analysis/media_context.py     ffprobe facts + low-res frames for the AI
  audio/bgm_manager.py          BGM library (user assets + Jamendo)
  audio/jamendo_provider.py     Jamendo search/cache/download
  audio/sfx_library.py          locally generated SFX assets
  config/config_loader.py       YAML + env config merge
  editing/editor.py             FFmpeg rendering engine
  editing/edit_plan.py          AI plan → concrete timeline
  editing/style_learner.py      aggregate style profile from examples
  orchestrator/orchestrator.py  job runner (DB ↔ editor ↔ agent)
  publishing/publisher.py       YouTube / Instagram upload
  research/                     YouTube metadata discovery + observations
  storage/                      SQLAlchemy models + session (SQLite)
  training/                     reference analysis, dataset quality, collector
  utilities/                    ffmpeg wrapper, logger, files, hardware probe
config/                     YAML configuration
dashboard/                  Flask + SocketIO chat dashboard & research UI
training/                   dataset build/analyze/train scripts + data
assets/bgm, assets/sfx      audio assets (user-managed / generated)
tests/                      pytest suite (45 tests)
```

## Testing

```bash
pytest tests/ -q
```

Requires FFmpeg on `PATH` (tests render tiny synthetic clips). The suite
covers editor segment/transition selection, the audio mixdown, orchestrator
DB flow, dashboard intent parsing, research routes, the AI provider config,
the SFX library, the training pipeline, conditional-probability learning,
style-cluster discovery, curriculum research, the revision loop, and the
CLI commands (61 tests).

## Rendering notes

- Output is always H.264 **yuv420p** (High profile) — the safe, universal
  format for YouTube/Instagram processing and playback.
- The libx264 fallback preset adapts to CPU core count (≤2 cores:
  `ultrafast`, ≤4: `veryfast`, else `fast`) so the filter graph can never
  outrun the encoder and balloon the frame queue on weak machines.
  Override with `FFMPEG_X264_PRESET` / `FFMPEG_X264_CRF`.
- Audio is assembled as a single bounded mixdown (per-clip fade envelopes
  + `adelay` placement + one `amix`), which scales to any number of clips.
- Creative-mode loudness normalization uses **two-pass linear loudnorm**:
  an audio-only measurement pass (cheap, memory-safe) followed by a
  deterministic `linear=true` application — preserves dynamics and keeps
  re-renders in the review/revise loop bit-stable. Output audio is always
  pinned to 44.1 kHz stereo (single-pass loudnorm otherwise emits 96 kHz).
- `FFMPEG_GRAPH_THROTTLE` (optional) caps filter-graph speed relative to
  realtime for very low-RAM machines; off by default.

## Legal / rights

The research subsystem only ever collects **public metadata** via the
official YouTube Data API and is designed around manual observation; it
never downloads or extracts media from YouTube. Training consumes only
references you affirmatively mark as rights-cleared (`user_owned`,
`licensed`, `public_domain`, `explicitly_permitted`) and observations you
explicitly approve. Background music comes from Jamendo under its license
terms (license metadata is stored alongside each track in `assets/bgm/`).
