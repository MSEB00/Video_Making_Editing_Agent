(() => {
  const byId = (id) => document.getElementById(id);
  const videoList = byId('videoList');
  const liveStatus = byId('liveStatus');
  let videos = [];
  let selectedVideo = null;
  let activeSession = null;
  let player = null;
  let playerReady = false;
  let playerTimer = null;
  let observations = [];
  let observationWritingApproved = false;

  function setStatus(message, error = false) {
    liveStatus.textContent = message;
    liveStatus.classList.toggle('error', error);
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {'Content-Type': 'application/json', ...(options.headers || {})}
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function secondsText(value) {
    const seconds = Math.max(0, Math.floor(Number(value) || 0));
    return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }

  function playerTime() {
    if (!playerReady || !player || typeof player.getCurrentTime !== 'function') return 0;
    return Math.max(0, Number(player.getCurrentTime()) || 0);
  }

  function updatePlayerReadout() {
    if (!playerReady || !player) return;
    const current = playerTime();
    const duration = Math.max(0, Number(player.getDuration()) || selectedVideo?.duration_seconds || 0);
    byId('timeReadout').textContent = `${secondsText(current)} / ${secondsText(duration)}`;
    byId('selectedDuration').textContent = secondsText(duration);
    const seekBar = byId('seekBar');
    seekBar.max = String(Math.max(1, duration));
    if (document.activeElement !== seekBar) seekBar.value = String(Math.min(current, duration || current));
  }

  function updatePlaybackState(event) {
    const states = new Map([[-1, 'Unstarted'], [0, 'Ended'], [1, 'Playing'], [2, 'Paused'], [3, 'Buffering'], [5, 'Cued']]);
    byId('playbackState').textContent = states.get(event.data) || 'Player ready';
    updatePlayerReadout();
  }

  function onPlayerReady() {
    playerReady = true;
    byId('playerStatus').textContent = 'Official YouTube player ready. Playback and timestamp are available; media is not captured.';
    byId('startSession').disabled = !selectedVideo || Boolean(activeSession) || !observationWritingApproved;
    updatePlayerReadout();
    const savedSession = selectedVideo && window.sessionStorage.getItem(`research-session-${selectedVideo.video_id}`);
    if (savedSession) loadObservationSession(savedSession).catch((error) => setStatus(error.message, true));
    if (playerTimer) window.clearInterval(playerTimer);
    playerTimer = window.setInterval(updatePlayerReadout, 500);
  }

  function loadIntoPlayer(video) {
    selectedVideo = video;
    byId('selectedTitle').textContent = video.title || 'YouTube reference';
    byId('selectedDuration').textContent = secondsText(video.duration_seconds);
    byId('startSession').disabled = Boolean(activeSession) || !window.YT?.Player || !observationWritingApproved;
    byId('playerStatus').textContent = 'Loading reference in the official embedded player...';
    if (player && playerReady) {
      player.cueVideoById(video.video_id);
      const savedSession = window.sessionStorage.getItem(`research-session-${video.video_id}`);
      if (savedSession) loadObservationSession(savedSession).catch((error) => setStatus(error.message, true));
      return;
    }
    if (!window.YT?.Player) {
      byId('playerStatus').textContent = 'The YouTube IFrame API has not loaded. Check network access to youtube.com.';
      return;
    }
    player = new window.YT.Player('player', {
      width: '100%',
      height: '100%',
      videoId: video.video_id,
      playerVars: {
        autoplay: 0,
        controls: 1,
        enablejsapi: 1,
        origin: window.location.origin,
        playsinline: 1,
        rel: 0
      },
      events: {
        onReady: onPlayerReady,
        onStateChange: updatePlaybackState,
        onError: (event) => {
          const code = Number(event.data);
          byId('playerStatus').textContent = `This video cannot be embedded here (player error ${code}). Open it on YouTube to review.`;
        }
      }
    });
  }

  function renderVideos() {
    videoList.replaceChildren();
    for (const video of videos) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'video-item';
      button.setAttribute('role', 'listitem');
      button.setAttribute('aria-current', selectedVideo?.video_id === video.video_id ? 'true' : 'false');
      const title = document.createElement('span');
      title.className = 'video-title';
      title.textContent = video.title || video.video_id;
      const meta = document.createElement('span');
      meta.className = 'video-meta';
      const topics = Array.isArray(video.discovery_topics) ? video.discovery_topics.join(' · ') : '';
      meta.textContent = `${video.channel || 'Unknown channel'} · ${secondsText(video.duration_seconds)} · ${video.observation_status || 'not observed'}${topics ? ` · ${topics}` : ''}`;
      button.append(title, meta);
      button.addEventListener('click', () => {
        if (activeSession) {
          setStatus('Finish the current observation session before switching references.', true);
          return;
        }
        loadIntoPlayer(video);
        renderVideos();
      });
      videoList.appendChild(button);
    }
    byId('videoCount').textContent = String(videos.length);
  }

  async function refreshVideos() {
    const result = await requestJson('/api/research/videos');
    videos = result;
    renderVideos();
  }

  async function refreshResearchState() {
    const state = await requestJson('/api/research/state');
    byId('localExamples').textContent = String(state.local_reference_examples ?? 0);
    byId('youtubeNotes').textContent = String(state.observation_count ?? 0);
    byId('eligibleObservations').textContent = String(state.training_eligible_youtube_observations ?? 0);
    byId('activeModel').textContent = state.active_model_present ? 'Yes' : 'No';
    observationWritingApproved = state.youtube_derived_observations_approved === true;
    const formControls = document.querySelectorAll('#observationForm input, #observationForm select, #observationForm textarea, #markStart, #markEnd');
    formControls.forEach((control) => { control.disabled = !observationWritingApproved; });
    byId('saveObservation').disabled = !observationWritingApproved || !activeSession;
    byId('startSession').disabled = !observationWritingApproved || !selectedVideo || Boolean(activeSession) || !playerReady;
    byId('approvalNotice').textContent = observationWritingApproved
      ? 'Annotation storage is enabled by local configuration. Confirm the required written YouTube approval applies to this use. Notes remain excluded from model training.'
      : 'Preview-only mode: persistent notes and session markers are disabled. YouTube\'s current API policies may restrict derived observations; enable annotation storage only after receiving the required written approval.';
  }

  function renderMarkers() {
    const list = byId('markerList');
    list.replaceChildren();
    byId('markerCount').textContent = String(observations.length);
    if (!observations.length) {
      const empty = document.createElement('span');
      empty.className = 'count';
      empty.textContent = 'No observations yet';
      list.appendChild(empty);
      return;
    }
    for (const observation of observations) {
      const row = document.createElement('div');
      row.className = 'marker';
      const seek = document.createElement('button');
      seek.type = 'button';
      seek.textContent = `${secondsText(observation.start_seconds)}–${secondsText(observation.end_seconds)}`;
      seek.addEventListener('click', () => {
        if (playerReady) player.seekTo(observation.start_seconds, true);
      });
      const description = document.createElement('div');
      const label = document.createElement('strong');
      label.textContent = observation.structure_label.replaceAll('_', ' ');
      const note = document.createElement('p');
      const decisions = Object.entries(observation.editing_decisions || {})
        .filter(([, value]) => value === 'observed')
        .map(([key]) => key)
        .join(', ');
      note.textContent = [observation.researcher_note, decisions ? `Observed: ${decisions}` : ''].filter(Boolean).join('\n');
      description.append(label, note);
      row.append(seek, description);
      list.appendChild(row);
    }
  }

  async function loadObservationSession(sessionId) {
    const result = await requestJson(`/api/research/session/${encodeURIComponent(sessionId)}`);
    activeSession = result.session;
    observations = result.observations || [];
    byId('sessionStatus').textContent = 'Observation session active';
    byId('startSession').disabled = true;
    byId('finishSession').disabled = false;
    byId('saveObservation').disabled = false;
    renderMarkers();
  }

  byId('searchForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = byId('searchButton');
    button.disabled = true;
    setStatus('Searching YouTube metadata...');
    try {
      const result = await requestJson('/api/research/discover', {
        method: 'POST',
        body: JSON.stringify({topic: byId('topicInput').value, limit: byId('resultLimit').value})
      });
      videos = [...result.videos, ...videos.filter((oldVideo) => !result.videos.some((newVideo) => newVideo.video_id === oldVideo.video_id))];
      renderVideos();
      setStatus(`Found ${result.found} candidates; ${result.new_to_research} are new to this research pool.`);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      button.disabled = false;
    }
  });

  byId('startSession').addEventListener('click', async () => {
    if (!selectedVideo) return;
    try {
      activeSession = await requestJson('/api/research/session', {
        method: 'POST',
        body: JSON.stringify({video_id: selectedVideo.video_id})
      });
      window.sessionStorage.setItem(`research-session-${selectedVideo.video_id}`, activeSession.session_id);
      observations = [];
      byId('sessionStatus').textContent = `Session ${activeSession.session_id.slice(0, 8)}`;
      byId('startSession').disabled = true;
      byId('finishSession').disabled = false;
      byId('saveObservation').disabled = false;
      renderMarkers();
      setStatus('Session started. Watch manually and add timestamped observations.');
    } catch (error) {
      setStatus(error.message, true);
    }
  });

  byId('finishSession').addEventListener('click', async () => {
    if (!activeSession) return;
    try {
      await requestJson(`/api/research/session/${encodeURIComponent(activeSession.session_id)}/finish`, {method: 'POST', body: '{}'});
      activeSession = null;
      byId('sessionStatus').textContent = 'No active session';
      byId('startSession').disabled = !selectedVideo || !playerReady;
      byId('finishSession').disabled = true;
      byId('saveObservation').disabled = true;
      if (selectedVideo) window.sessionStorage.removeItem(`research-session-${selectedVideo.video_id}`);
      await refreshVideos();
      await refreshResearchState();
      setStatus('Session finished. Notes remain excluded from model training.');
    } catch (error) {
      setStatus(error.message, true);
    }
  });

  byId('markStart').addEventListener('click', () => { byId('startTime').value = playerTime().toFixed(1); });
  byId('markEnd').addEventListener('click', () => { byId('endTime').value = playerTime().toFixed(1); });
  byId('seekBack').addEventListener('click', () => { if (playerReady) player.seekTo(Math.max(0, playerTime() - 5), true); });
  byId('seekBar').addEventListener('change', (event) => { if (playerReady) player.seekTo(Number(event.target.value), true); });

  byId('observationForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!activeSession) return;
    const optionalNumber = (id) => byId(id).value === '' ? null : Number(byId(id).value);
    const speechValue = byId('speechPresent').value;
    const editingDecisions = {};
    document.querySelectorAll('[data-decision]').forEach((select) => {
      editingDecisions[select.dataset.decision] = select.value;
    });
    const payload = {
      session_id: activeSession.session_id,
      start_seconds: Number(byId('startTime').value),
      end_seconds: Number(byId('endTime').value),
      structure_label: byId('structureLabel').value,
      editing_decisions: editingDecisions,
      context: {
        visual_intensity: optionalNumber('visualIntensity'),
        audio_intensity: optionalNumber('audioIntensity'),
        speech_present: speechValue === 'unknown' ? null : speechValue === 'yes',
        gameplay_context: byId('gameplayContext').value,
        mood: ''
      },
      rights_confirmed: byId('rightsConfirmed').checked,
      confidence: optionalNumber('confidence'),
      note: byId('researchNote').value
    };
    try {
      const result = await requestJson('/api/research/observations', {method: 'POST', body: JSON.stringify(payload)});
      observations.push(result.observation);
      renderMarkers();
      await refreshResearchState();
      setStatus('Observation saved as a human-authored note; it is not training data.');
      byId('researchNote').value = '';
    } catch (error) {
      setStatus(error.message, true);
    }
  });

  byId('trainLocal').addEventListener('click', async () => {
    const button = byId('trainLocal');
    button.disabled = true;
    setStatus('Training from rights-cleared local references only...');
    try {
      const result = await requestJson('/api/research/train', {method: 'POST', body: '{}'});
      const training = result.result;
      setStatus(training.status === 'promoted' ? `Model ${training.version} promoted from local references.` : `Training status: ${training.status}; examples: ${training.example_count ?? training.training_count ?? 0}.`);
      await refreshResearchState();
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      button.disabled = false;
    }
  });

  byId('seekBack').disabled = true;
  window.addEventListener('youtube-api-ready', () => {
    if (selectedVideo && !player) loadIntoPlayer(selectedVideo);
  });
  if (window.YT?.Player) window.dispatchEvent(new Event('youtube-api-ready'));

  Promise.all([refreshVideos(), refreshResearchState()]).catch((error) => setStatus(error.message, true));
})();