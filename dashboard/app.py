import sys, os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import threading
import pathlib
import ipaddress
import secrets
import datetime as dt
from dotenv import load_dotenv
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")

from app.research.youtube_research_agent import SEARCH_STRATEGIES, YouTubeResearchAgent
from app.research.video_observer import ResearchValidationError, VideoObservationStore
from app.training.reference_pipeline import ReferenceTrainingPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(32)
socketio = SocketIO(app, async_mode='threading')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
RESEARCH_CANDIDATES = pathlib.Path(PROJECT_ROOT) / 'training' / 'candidates.json'


def _youtube_derived_observations_approved() -> bool:
    return os.getenv('YOUTUBE_DERIVED_OBSERVATIONS_APPROVED', '').lower() in {'1', 'true', 'yes'}


@app.before_request
def _restrict_research_routes_to_local_clients():
    if request.path == '/research' or request.path.startswith('/api/research/'):
        try:
            remote = ipaddress.ip_address(request.remote_addr or '')
        except ValueError:
            return jsonify({'error': 'Research routes are local-only.'}), 403
        if not remote.is_loopback:
            return jsonify({'error': 'Research routes are local-only.'}), 403

# ---- Helpers ----

def _list_input_folders():
    """Return list of folders inside input/ containing video clips."""
    folders = []
    input_path = pathlib.Path(INPUT_DIR)
    if not input_path.exists():
        return folders
    for entry in sorted(input_path.iterdir()):
        if entry.is_dir():
            vids = [f for f in entry.iterdir() if f.suffix.lower() in ('.mp4', '.mov', '.mkv')]
            total_size_mb = sum(f.stat().st_size for f in vids) / (1024 * 1024) if vids else 0
            folders.append({
                'name': entry.name,
                'path': str(entry.resolve()),
                'videos': len(vids),
                'size_mb': round(total_size_mb, 1)
            })
    return folders

def _parse_editing_options(msg: str, client_opts: dict = None) -> dict:
    """Parse editing instructions (transitions, BGM, aspect ratio) from user text and client UI."""
    low = msg.lower()
    opts = client_opts.copy() if client_opts else {}

    # 1. Transitions
    if any(k in low for k in ["no transition", "without transition", "direct cut", "cut only", "no fade"]):
        opts["transition_type"] = "none"
    elif "wipeleft" in low or "wipe left" in low:
        opts["transition_type"] = "wipeleft"
    elif "wiperight" in low or "wipe right" in low or "wipe" in low:
        opts["transition_type"] = "wipeleft"
    elif "slide" in low:
        opts["transition_type"] = "slideleft"
    elif "circle" in low or "circlecrop" in low:
        opts["transition_type"] = "circlecrop"
    elif "dissolve" in low:
        opts["transition_type"] = "dissolve"
    elif "fadeblack" in low or "fade to black" in low:
        opts["transition_type"] = "fadeblack"
    elif "random" in low:
        opts["transition_type"] = "random"
    elif any(k in low for k in ["transition", "transitions", "xfade", "crossfade", "fade", "smooth"]):
        if "transition_type" not in opts or opts["transition_type"] == "none":
            opts["transition_type"] = "fade"

    # Keep music language as user preference; the creative planner chooses actual tracks.
    if any(k in low for k in ["no bgm", "no music", "without bgm", "without music", "mute music", "game audio only", "original audio"]):
        opts["bgm_track"] = None

    # Volume parsing
    vol_match = re.search(r'(?:vol|volume)\s*(?:at|=)?\s*(\d+)%?', low)
    if vol_match:
        val = int(vol_match.group(1))
        opts["bgm_volume"] = min(1.0, max(0.05, val / 100.0 if val > 1 else val))

    wants_trending_audio = any(term in low for term in ["trending audio", "latest audio", "viral audio", "trending music", "latest music"])
    if wants_trending_audio:
        opts["bgm_track"] = "trending licensed Jamendo catalog"
    if any(term in low for term in ["random edit", "random clips", "shuffle clips", "surprise me"]):
        opts["randomize_clips"] = True
        opts["transition_type"] = "random"
        opts["aspect_ratio"] = "9:16"
        opts["max_clips"] = min(int(opts.get("max_clips", 10)), 10)
        opts["max_clip_duration"] = min(float(opts.get("max_clip_duration", 6)), 6.0)
        opts["transition_duration"] = min(float(opts.get("transition_duration", 0.35)), 0.35)

    if any(term in low for term in ["no sfx", "no sound effects", "without sound effects"]):
        opts["sfx_preference"] = False
    elif "sfx" in low or "sound effect" in low:
        opts["sfx_preference"] = True

    # 3. Format / Aspect Ratio
    if any(k in low for k in ["vertical", "short", "shorts", "reel", "reels", "9:16"]):
        opts["aspect_ratio"] = "9:16"
        opts["randomize_clips"] = True
        opts["max_clips"] = min(int(opts.get("max_clips", 10)), 10)
        opts["max_clip_duration"] = min(float(opts.get("max_clip_duration", 6)), 6.0)
        opts["transition_duration"] = min(float(opts.get("transition_duration", 0.35)), 0.35)
        if opts.get("transition_type") != "none" and not any(k in low for k in ["transition", "transitions", "wipe", "slide", "dissolve", "fade", "circle", "random"]):
            opts["transition_type"] = "random"
        opts["platform"] = "instagram_reels" if any(k in low for k in ["instagram", "insta", "reel", "reels"]) else "youtube_shorts"
    elif any(k in low for k in ["horizontal", "landscape", "youtube", "16:9"]):
        opts["aspect_ratio"] = "16:9"
        opts["platform"] = "youtube_longform"

    # Default fallbacks
    opts.setdefault("transition_type", "random")
    opts.setdefault("bgm_track", "auto")
    opts.setdefault("bgm_volume", 0.30)
    opts.setdefault("aspect_ratio", "16:9")
    opts.setdefault("platform", "youtube_shorts" if opts["aspect_ratio"] == "9:16" else "youtube_longform")
    opts.setdefault("color_grade", True)

    return opts

def _parse_intent(msg: str):
    """Parse user chat message into (intent, params)."""
    low = msg.lower().strip()

    # Greetings
    if re.match(r'^(hi|hello|hey|yo|sup|greetings)\b', low) and len(low.split()) < 3:
        return 'greeting', {}

    # Help
    if any(kw in low for kw in ['help', 'what can you do', 'how do i', 'commands', 'instructions']):
        return 'help', {}

    # List clips
    if any(kw in low for kw in ['list', 'show clips', 'show folders', 'available', 'what clips', 'what folders']):
        return 'list', {}

    # Status / History
    if any(kw in low for kw in ['status', 'jobs', 'history', 'recent', 'queue', 'progress']):
        return 'status', {}

    if re.search(r"\bpublish\b", low):
        platforms = []
        if any(term in low for term in ["youtube", "yt", "shorts"]):
            platforms.append("youtube")
        if any(term in low for term in ["instagram", "insta", "reel", "reels"]):
            platforms.append("instagram")
        job_match = re.search(r"\bjob\s*#?(\d+)\b", low)
        return 'publish', {
            'platforms': platforms,
            'job_id': int(job_match.group(1)) if job_match else None,
        }

    # Make / create / render / edit video (Natural language)
    triggers = ['make', 'create', 'produce', 'edit', 'render', 'build', 'generate', 'video', 'compile', 'stitch', 'merge', 'combine', 'montage', 'highlight', 'transition', 'bgm', 'music']
    if any(kw in low for kw in triggers):
        folders = _list_input_folders()
        if not folders:
            return 'no_input', {}

        # 1. Folder match
        for f in folders:
            if f['name'].lower() in low:
                return 'process', {'folder': f['path'], 'name': f['name']}

        for f in folders:
            parts = re.split(r'[-_]', f['name'].lower())
            if any(p and p in low for p in parts if len(p) > 2):
                return 'process', {'folder': f['path'], 'name': f['name']}

        # Number match
        nums = re.findall(r'\bfolder\s*(\d+)\b|\bvideo\s*(\d+)\b', low)
        if nums:
            n = next((n[0] or n[1] for n in nums if n[0] or n[1]), None)
            if n:
                idx = int(n) - 1
                if 0 <= idx < len(folders):
                    return 'process', {'folder': folders[idx]['path'], 'name': folders[idx]['name']}

        # Default: latest folder with clips
        folders_with_clips = [f for f in folders if f['videos'] > 0]
        chosen = folders_with_clips[-1] if folders_with_clips else folders[-1]
        return 'process_default', {'folder': chosen['path'], 'name': chosen['name']}

    return 'unknown', {}


def _emit_step(sid, icon, text):
    """Emit a live processing step indicator to the client."""
    socketio.emit('agent_step', {'icon': icon, 'msg': text}, to=sid)


def _run_pipeline(sid, folder_path, folder_name=None, edit_opts=None):
    """Run full video assembly in a background worker thread."""
    from app.orchestrator.orchestrator import orchestrate_job
    from app.editing.editor import EditOptions
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    if not folder_name:
        folder_name = os.path.basename(folder_path)

    _emit_step(sid, '📂', f'Selected input: {folder_name}')

    p_folder = pathlib.Path(folder_path)
    if not p_folder.exists() or not p_folder.is_dir():
        _emit_step(sid, '❌', f'Folder path not found: {folder_path}')
        socketio.emit('chat_response', {
            'msg': f'❌ Could not find folder: {folder_path}',
            'type': 'error'
        }, to=sid)
        return

    # Find clips
    vids = [f for f in p_folder.iterdir() if f.suffix.lower() in ('.mp4', '.mov', '.mkv')]
    if not vids:
        _emit_step(sid, '⚠️', 'No video files found in selected folder.')
        socketio.emit('chat_response', {
            'msg': f'⚠️ No video clips (.mp4, .mov, .mkv) found in "{folder_name}".',
            'type': 'error'
        }, to=sid)
        return

    total_mb = sum(f.stat().st_size for f in vids) / (1024 * 1024)
    _emit_step(sid, '🎬', f'Found {len(vids)} clip(s) (~{round(total_mb, 1)} MB)')

    # Build EditOptions object
    if not edit_opts:
        edit_opts = {
            "transition_type": "random",
            "bgm_track": "auto",
            "bgm_volume": 0.30,
            "aspect_ratio": "9:16",
            "platform": "youtube_shorts",
            "color_grade": True,
            "randomize_clips": False
        }

    options = EditOptions(
        transition_type=edit_opts.get("transition_type", "random"),
        transition_duration=float(edit_opts.get("transition_duration", 0.75)),
        bgm_track=edit_opts.get("bgm_track"),
        bgm_volume=float(edit_opts.get("bgm_volume", 0.30)),
        aspect_ratio=edit_opts.get("aspect_ratio", "16:9"),
        color_grade=bool(edit_opts.get("color_grade", True)),
        max_clips=edit_opts.get("max_clips"),
        max_clip_duration=edit_opts.get("max_clip_duration"),
        randomize_clips=bool(edit_opts.get("randomize_clips", False)),
        variation_seed=edit_opts.get("variation_seed"),
        sfx_preference=edit_opts.get("sfx_preference"),
        creative_mode=bool(edit_opts.get("creative_mode", False)),
        creative_request=str(edit_opts.get("creative_request", "")),
        game=edit_opts.get("game", "valorant"),
        platform=edit_opts.get("platform", "youtube_shorts"),
        target_duration=int(edit_opts.get("target_duration", 45))
    )

    # Initialize Job in Database with options saved in extra_metadata
    _emit_step(sid, '📝', 'Initializing job record with editor parameters...')
    session = SessionLocal()
    job = Job(
        status='queued',
        input_path=str(p_folder.resolve()),
        extra_metadata=json.dumps(edit_opts)
    )
    session.add(job)
    session.commit()
    job_id = job.id
    session.close()
    if options.randomize_clips and options.variation_seed is None:
        options.variation_seed = job_id

    _emit_step(sid, '⚡', f'Job #{job_id} scheduled - starting video rendering engine...')

    creative_model = None
    if options.creative_mode:
        try:
            from app.ai.remote_json_model import RemoteJSONModel
            from dashboard.puter_bridge import PuterBrowserTransport

            transport = PuterBrowserTransport(socketio, sid)
            creative_model = RemoteJSONModel(
                transport.call, model_name=f"puter:{transport.model_name}"
            )
            _emit_step(sid, '🧠', f'Creative AI via Puter.js in your browser ({transport.model_name}, free, no API keys).')
        except Exception:
            creative_model = None

    try:
        output_path = orchestrate_job(
            job_id=job_id,
            options=options,
            progress_callback=lambda icon, msg: _emit_step(sid, icon, msg),
            model=creative_model
        )
        out_p = pathlib.Path(output_path)
        out_size_mb = round(out_p.stat().st_size / (1024 * 1024), 2)
        out_filename = out_p.name

        _emit_step(sid, '✨', f'Rendering complete! File: {out_filename} ({out_size_mb} MB)')
        _emit_step(sid, '✅', 'Pipeline finished successfully.')

        # Build feature summary
        tr_info = f"✨ Transitions: {options.transition_type.capitalize()}" if options.transition_type != "none" else "✂️ Cuts: Direct Cuts"
        bgm_info = f"🎵 BGM: {options.bgm_track.replace('_', ' ').title()}" if options.bgm_track else "🔇 BGM: Original Audio"
        fmt_info = f"📱 Format: {options.aspect_ratio}"

        socketio.emit('chat_response', {
            'msg': f'🎉 Done! Your edited video is ready.\n\n{tr_info}\n{bgm_info}\n{fmt_info}\n📁 File: {out_filename} ({out_size_mb} MB)\n🆔 Job #{job_id}',
            'video_url': f'/output/{out_filename}',
            'filename': out_filename,
            'size_mb': out_size_mb,
            'job_id': job_id,
            'type': 'success'
        }, to=sid)

    except Exception as e:
        _emit_step(sid, '❌', f'Pipeline error: {str(e)}')
        socketio.emit('chat_response', {
            'msg': f'❌ Job #{job_id} failed during editing: {str(e)}',
            'type': 'error'
        }, to=sid)


def _run_publish(sid, platforms, requested_job_id=None):
    from app.publishing.publisher import publish_video
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    session = SessionLocal()
    try:
        job = session.get(Job, requested_job_id) if requested_job_id else (
            session.query(Job)
            .filter(Job.status == "completed", Job.output_path.isnot(None))
            .order_by(Job.id.desc())
            .first()
        )
        if not job or job.status != "completed" or not job.output_path:
            socketio.emit('chat_response', {
                'msg': 'No completed video was found to publish. Finish an edit first, or specify a completed job number.',
                'type': 'error'
            }, to=sid)
            return
        job_id = job.id
        video_path = pathlib.Path(job.output_path)
    finally:
        session.close()

    if not video_path.is_file():
        socketio.emit('chat_response', {
            'msg': f'Output for job #{job_id} is missing: {video_path}',
            'type': 'error'
        }, to=sid)
        return

    results = publish_video(video_path, platforms)
    lines = [f'Publish results for job #{job_id}:']
    for platform, result in results.items():
        state = 'Published' if result['success'] else 'Not published'
        lines.append(f"{platform.title()}: {state}. {result['message']}")
    socketio.emit('chat_response', {'msg': '\n'.join(lines), 'type': 'success' if all(r['success'] for r in results.values()) else 'error'}, to=sid)


# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/research')
def research_page():
    return render_template('youtube_research.html')


def _research_store() -> VideoObservationStore:
    store = VideoObservationStore(
        allow_derived_observations=_youtube_derived_observations_approved()
    )
    try:
        pool = json.loads(RESEARCH_CANDIDATES.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        pool = {}
    updated_at = pool.get('updated_at') if isinstance(pool, dict) else None
    candidates = pool.get('items', []) if isinstance(pool, dict) else []
    if isinstance(candidates, list):
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
        current_candidates = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            timestamp = item.get('discovered_at') or updated_at
            try:
                discovered_at = dt.datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            except (TypeError, ValueError):
                continue
            if discovered_at.tzinfo is None:
                discovered_at = discovered_at.replace(tzinfo=dt.timezone.utc)
            if discovered_at >= cutoff:
                current_candidates.append(item)
        if len(current_candidates) != len(candidates):
            pool['items'] = current_candidates
            pool['candidate_count'] = len(current_candidates)
            temporary = RESEARCH_CANDIDATES.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(pool, indent=2, ensure_ascii=True), encoding='utf-8')
            temporary.replace(RESEARCH_CANDIDATES)
        store.register_videos([
            {**item, 'discovered_at': item.get('discovered_at') or updated_at}
            for item in current_candidates
        ])
    return store


@app.route('/api/research/state')
def api_research_state():
    store = _research_store()
    example_path = pathlib.Path(PROJECT_ROOT) / 'training' / 'features' / 'examples.jsonl'
    example_count = 0
    if example_path.is_file():
        with example_path.open('r', encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    example_count += 1
    return jsonify({
        **store.stats(),
        'local_reference_examples': example_count,
        'active_model_present': (pathlib.Path(PROJECT_ROOT) / 'training' / 'models' / 'active.json').is_file(),
        'youtube_player_observations_trainable': False,
        'youtube_derived_observations_approved': _youtube_derived_observations_approved(),
    })


@app.route('/api/research/videos')
def api_research_videos():
    return jsonify(_research_store().list_videos())


@app.route('/api/research/discover', methods=['POST'])
def api_research_discover():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'error': 'Request body must be a JSON object.'}), 400
    topic = str(payload.get('topic', '')).strip()[:120]
    if len(topic) < 3:
        return jsonify({'error': 'Enter a topic of at least three characters.'}), 400
    try:
        limit = max(1, min(10, int(payload.get('limit', 5))))
    except (TypeError, ValueError):
        return jsonify({'error': 'limit must be an integer between 1 and 10.'}), 400
    try:
        discovered = YouTubeResearchAgent().discover(
            results_per_query=limit,
            strategies=[topic],
            license_filter='creativeCommon',
            order='viewCount',
            lookback_days=max(1, min(365, int(os.getenv('YOUTUBE_RESEARCH_LOOKBACK_DAYS', '90')))),
        )
        registered = _research_store().register_videos(discovered)
    except Exception as exc:
        app.logger.warning('YouTube reference discovery failed: %s', type(exc).__name__)
        return jsonify({'error': 'YouTube discovery failed. Check the server-side Data API configuration and quota.'}), 503
    unseen = [item for item in registered if item.get('research_status') == 'discovered']
    return jsonify({'videos': unseen, 'found': len(discovered), 'new_to_research': len(unseen)})


@app.route('/api/research/session', methods=['POST'])
def api_research_session():
    if not _youtube_derived_observations_approved():
        return jsonify({'error': 'Persistent YouTube observations require prior written YouTube approval. Preview-only mode is active.'}), 403
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'error': 'Request body must be a JSON object.'}), 400
    try:
        return jsonify(_research_store().create_session(str(payload.get('video_id', ''))))
    except ResearchValidationError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/research/session/<session_id>')
def api_research_session_detail(session_id: str):
    if not re.fullmatch(r'[a-f0-9]{32}', session_id):
        return jsonify({'error': 'Invalid research session id.'}), 400
    store = _research_store()
    session = store.get_session(session_id)
    if session is None:
        return jsonify({'error': 'Research session was not found.'}), 404
    return jsonify({'session': session, 'observations': store.observations_for(session_id)})


@app.route('/api/research/session/<session_id>/finish', methods=['POST'])
def api_research_finish_session(session_id: str):
    if not _youtube_derived_observations_approved():
        return jsonify({'error': 'Persistent YouTube observations require prior written YouTube approval. Preview-only mode is active.'}), 403
    if not re.fullmatch(r'[a-f0-9]{32}', session_id):
        return jsonify({'error': 'Invalid research session id.'}), 400
    try:
        return jsonify(_research_store().finish_session(session_id))
    except ResearchValidationError as exc:
        return jsonify({'error': str(exc)}), 404


@app.route('/api/research/observations', methods=['POST'])
def api_research_observation():
    if not _youtube_derived_observations_approved():
        return jsonify({'error': 'Persistent YouTube observations require prior written YouTube approval. Preview-only mode is active.'}), 403
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'error': 'Request body must be a JSON object.'}), 400
    try:
        observation = _research_store().add_observation(payload)
    except ResearchValidationError as exc:
        return jsonify({'error': str(exc)}), 400
    return jsonify({'observation': observation}), 201


@app.route('/api/research/train', methods=['POST'])
def api_research_train():
    try:
        pipeline = ReferenceTrainingPipeline()
        result = pipeline.train_candidate()
        youtube_observations_used = sum(
            len(item.get("provenance", {}).get("observation_ids", []))
            for item in pipeline._read_examples()
            if item.get("provenance", {}).get("platform") == "youtube"
        )
    except Exception as exc:
        app.logger.warning('Local reference training failed: %s', type(exc).__name__)
        return jsonify({'error': 'Training failed while processing the local rights-cleared reference dataset.'}), 500
    return jsonify({
        'result': result,
        'source': 'rights-cleared local references and explicitly approved observations',
        'youtube_player_observations_used': youtube_observations_used,
    })

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve rendered videos with range support for seamless browser playback."""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/jobs')
def api_jobs():
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    session = SessionLocal()
    jobs = session.query(Job).order_by(Job.id.desc()).limit(20).all()
    data = [
        {
            'id': j.id,
            'status': j.status,
            'input_path': j.input_path,
            'output_path': j.output_path,
            'output_url': f"/output/{pathlib.Path(j.output_path).name}" if j.output_path else None,
            'extra_metadata': json.loads(j.extra_metadata) if j.extra_metadata else {},
            'created_at': j.created_at.isoformat() if j.created_at else None
        }
        for j in jobs
    ]
    session.close()
    return jsonify(data)

@app.route('/api/inputs')
def api_inputs():
    return jsonify(_list_input_folders())

@app.route('/api/bgm')
def api_bgm():
    from app.audio.bgm_manager import list_bgm_tracks

    return jsonify(list_bgm_tracks())

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    from app.storage.db import SessionLocal
    from app.storage.models import Job

    payload = request.get_json(silent=True) or {}
    try:
        job_id = int(payload.get('edit_id'))
        rating = int(payload.get('rating'))
    except (TypeError, ValueError):
        return jsonify({'error': 'edit_id and rating are required.'}), 400
    if rating not in (-1, 1):
        return jsonify({'error': 'rating must be 1 or -1.'}), 400

    session = SessionLocal()
    job = session.get(Job, job_id)
    if not job or job.status != 'completed' or not job.output_path:
        session.close()
        return jsonify({'error': 'Feedback is only accepted for completed edits.'}), 404
    context = {'job_id': job_id, 'output': pathlib.Path(job.output_path).name}
    plan_path = pathlib.Path(job.output_path).with_suffix('.edit-plan.json')
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            context.update({'strategy': plan.get('strategy'), 'platform': plan.get('platform')})
        except (OSError, json.JSONDecodeError):
            pass
    session.close()

    allowed_tags = {'too_many_effects', 'too_slow', 'too_fast', 'bgm_mismatch', 'captions_good', 'transitions_bad', 'hook_good'}
    tags = payload.get('tags', [])
    if not isinstance(tags, list):
        return jsonify({'error': 'tags must be a list.'}), 400
    selected_tags = [tag for tag in tags if isinstance(tag, str) and tag in allowed_tags]
    ReferenceTrainingPipeline().record_feedback(
        edit_id=str(job_id),
        rating=rating,
        tags=selected_tags,
        notes=str(payload.get('notes', ''))[:1000],
        context=context,
    )
    return jsonify({'status': 'recorded', 'edit_id': job_id})


# ---- SocketIO Events ----

@socketio.on('connect')
def handle_connect():
    folders = _list_input_folders()
    total_clips = sum(f['videos'] for f in folders)
    emit('status', {
        'msg': f'Ready • {len(folders)} folders • {total_clips} clips'
    })
    emit('chat_response', {
        'msg': (
            '👋 Welcome to Gaming Video Agent — Pro Video Editor!\n\n'
            'I can autonomously assemble your clips with:\n'
            '• ✨ **Transitions** (Smooth Fade, Wipe, Slide, Dissolve, Circle, Random)\n'
            '• 🎵 **Background Music** (Cyberpunk Energy, Lo-Fi Chill, Epic Gaming)\n'
            '• 📱 **Aspect Ratios** (16:9 YouTube, 9:16 YouTube Shorts / Instagram Reels)\n'
            '• 🎨 **Gaming Vibrancy & Contrast Color Grading**\n\n'
            '• 🎲 **Fresh edits** with shuffled clips, varied trims, and music choices\n'
            '• 🎧 **Trending licensed audio** on request (Jamendo API key required)\n\n'
            '💬 Just tell me what you want, e.g.:\n'
            '  "Make transitions and add bgm"\n'
            '  "Create a shorts video with wipe transitions and lofi music"\n'
            '  "Make a random edit with trending audio"'
        ),
        'type': 'greeting'
    })

@socketio.on('chat_message')
def handle_chat_message(data):
    from flask import request
    msg = data.get('msg', '').strip()
    client_opts = data.get('options', {})
    if not msg:
        return
    sid = request.sid
    intent, params = _parse_intent(msg)
    edit_opts = _parse_editing_options(msg, client_opts)

    if intent == 'greeting':
        emit('chat_response', {
            'msg': '🎮 Hey there! Ready to edit. Tell me what you want to do (e.g. "make transitions and add bgm", "make shorts video", "list clips").',
            'type': 'greeting'
        })

    elif intent == 'help':
        emit('chat_response', {
            'msg': (
                '🤖 Editor Capabilities & Commands:\n\n'
                '• "make transitions and add bgm" — edits clips with smooth crossfade & music\n'
                '• "add wipe transition and epic music" — custom wipe transitions & epic track\n'
                '• "make a shorts video" — 9:16 vertical video for YouTube Shorts / Instagram Reels\n'
                '• "random transitions" — varied transitions per cut with synchronized whoosh SFX\n'
                '• "make a random edit with trending audio" — shuffled clips and a weekly-popular licensed music track\n'
                '• "list clips" — show available input gameplay folders\n'
                '• "status" — view recent rendering jobs'
                '\n• "publish latest to YouTube and Instagram" — publish the most recent completed edit'
            )
        })

    elif intent == 'publish':
        platforms = params['platforms']
        if not platforms:
            emit('chat_response', {
                'msg': 'Choose a destination, for example: "publish latest to YouTube", "publish job 12 to Instagram", or "publish latest to YouTube and Instagram".'
            })
        else:
            job_text = f"job #{params['job_id']}" if params['job_id'] else 'latest completed video'
            emit('chat_response', {'msg': f"Starting requested publish for {job_text} to {', '.join(platforms)}. Nothing is uploaded unless you request it."})
            threading.Thread(
                target=_run_publish,
                args=(sid, platforms, params['job_id']),
                daemon=True
            ).start()

    elif intent == 'list':
        folders = _list_input_folders()
        if not folders:
            emit('chat_response', {'msg': '📂 No input folders found in the input/ directory.'})
        else:
            lines = ['📂 Available gameplay folders:\n']
            for i, f in enumerate(folders, 1):
                badge = f"({f['videos']} clips, ~{f['size_mb']} MB)" if f['videos'] > 0 else "(empty)"
                lines.append(f'  {i}. {f["name"]} {badge}')
            lines.append('\n💡 Type: "make transitions and add bgm from 1"')
            emit('chat_response', {'msg': '\n'.join(lines)})

    elif intent == 'status':
        from app.storage.db import SessionLocal
        from app.storage.models import Job

        session = SessionLocal()
        jobs = session.query(Job).order_by(Job.id.desc()).limit(8).all()
        if not jobs:
            emit('chat_response', {'msg': '📊 No jobs recorded yet. Type "make transitions and add bgm" to start!'})
        else:
            lines = ['📊 Recent Jobs:\n']
            for j in jobs:
                badge = '✅ Completed' if j.status == 'completed' else ('⏳ Processing' if j.status == 'processing' else ('❌ Failed' if j.status == 'failed' else '🕒 Queued'))
                in_name = os.path.basename(j.input_path) if j.input_path else 'N/A'
                lines.append(f'• Job #{j.id} [{badge}] — {in_name}')
                if j.output_path:
                    out_name = os.path.basename(j.output_path)
                    lines.append(f'   Output: {out_name}')
            emit('chat_response', {'msg': '\n'.join(lines)})
        session.close()

    elif intent in ('process', 'process_default'):
        folder = params['folder']
        name = params['name']
        edit_opts["creative_mode"] = True
        edit_opts["creative_request"] = msg

        # Confirm edit parameters in chat
        tr_desc = f"{edit_opts['transition_type'].capitalize()} transitions" if edit_opts['transition_type'] != 'none' else "Direct cuts"
        bgm_desc = f"{edit_opts['bgm_track'].replace('_', ' ').title()} BGM" if edit_opts['bgm_track'] else "Original game audio only"
        fmt_desc = f"{edit_opts['aspect_ratio']}"

        emit('chat_response', {
            'msg': f'🚀 Starting full edit for **{name}**:\n• ✨ {tr_desc}\n• 🎵 {bgm_desc}\n• 📱 Format: {fmt_desc}\n• 🎨 Enhanced Gaming Color Grading'
        })

        threading.Thread(target=_run_pipeline, args=(sid, folder, name, edit_opts), daemon=True).start()

    elif intent == 'no_input':
        emit('chat_response', {
            'msg': '⚠️ No video folders found in the input/ directory. Please place your clips in input/ first!',
            'type': 'error'
        })

    else:
        emit('chat_response', {
            'msg': (
                f'🤔 I heard: "{msg}"\n'
                'To edit a video, say: "make transitions and add bgm" or "create a shorts montage".\n'
                'Type "help" for all editing commands!'
            )
        })


# ---- Startup ----

def run_dashboard(host='127.0.0.1', port=5000):
    from app.audio.bgm_manager import ensure_bgm_library
    try:
        from app.storage.db import init_db

        init_db()
    except Exception as exc:
        app.logger.warning(
            'Database unavailable; research routes remain usable, while job history and editing are disabled (%s).',
            type(exc).__name__,
        )
    ensure_bgm_library()
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if os.getenv('YOUTUBE_RESEARCH_ENABLED', '').lower() in {'1', 'true', 'yes'}:
        research_agent = YouTubeResearchAgent()
        if research_agent.api_key:
            interval = max(3600, int(os.getenv('YOUTUBE_RESEARCH_INTERVAL_SECONDS', '86400')))
            lookback_days = max(1, int(os.getenv('YOUTUBE_RESEARCH_LOOKBACK_DAYS', '30')))
            query_limit = max(1, min(len(SEARCH_STRATEGIES), int(os.getenv('YOUTUBE_RESEARCH_QUERY_LIMIT', '2'))))
            results_per_query = max(1, min(50, int(os.getenv('YOUTUBE_RESEARCH_RESULTS_PER_QUERY', '5'))))
            threading.Thread(
                target=research_agent.run_periodically,
                kwargs={
                    'interval_seconds': interval,
                    'strategies': SEARCH_STRATEGIES[:query_limit],
                    'results_per_query': results_per_query,
                    'order': os.getenv('YOUTUBE_RESEARCH_ORDER', 'viewCount'),
                    'lookback_days': lookback_days,
                },
                name='youtube-metadata-research',
                daemon=True,
            ).start()
        else:
            app.logger.warning('YouTube research enabled but YOUTUBE_DATA_API_KEY is not configured.')
    print('\n======================================================')
    print('  Gaming Video Agent — PRO EDITOR Dashboard RUNNING!')
    print(f'  Local URL:  http://localhost:{port}')
    print(f'  Network:    http://127.0.0.1:{port}')
    print('======================================================\n')
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    run_dashboard()
