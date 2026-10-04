"""
app/audio/bgm_manager.py
-------------------------
Manages background music assets and generation for the Gaming Video Agent.
"""
from __future__ import annotations

import os
import math
import json
import random
import wave
import struct
import pathlib
import requests
from typing import List, Dict, Optional

BGM_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "bgm"

def ensure_bgm_library():
    """Ensure the user-managed background music directory exists."""
    BGM_DIR.mkdir(parents=True, exist_ok=True)

def list_bgm_tracks() -> List[Dict[str, str]]:
    """List user-managed or Jamendo-downloaded BGM assets."""
    BGM_DIR.mkdir(parents=True, exist_ok=True)
    tracks = []
    for f in sorted(BGM_DIR.glob("*.wav")) + sorted(BGM_DIR.glob("*.mp3")):
        tracks.append({
            "name": f.stem.replace("_", " ").title(),
            "id": f.stem,
            "filename": f.name,
            "path": str(f.resolve())
        })
    return tracks

def get_track_by_name_or_genre(query: Optional[str]) -> Optional[pathlib.Path]:
    """Resolve an explicit local audio asset; creative selection belongs to the AI planner."""
    if not query:
        return None

    q = query.lower()
    if q in {"random", "shuffle", "surprise"}:
        tracks = list(BGM_DIR.glob("*.wav")) + list(BGM_DIR.glob("*.mp3"))
        return random.choice(tracks) if tracks else None
    requested_path = pathlib.Path(query)
    if requested_path.is_file():
        return requested_path

    for track in list_bgm_tracks():
        if q in {track["id"].lower(), track["filename"].lower(), track["name"].lower()}:
            return pathlib.Path(track["path"])
    return None


def _generate_cyberpunk_track(dst_path: pathlib.Path, duration=60):
    sample_rate = 44100
    num_samples = sample_rate * duration
    bpm = 128
    beat_len = 60.0 / bpm
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        beat_t = t % beat_len
        bar_num = int(t / (beat_len * 4)) % 4
        beat_index = int(t / beat_len) % 4
        sub_beat = t % (beat_len / 4)

        # Kick
        kick_env = math.exp(-25 * beat_t)
        kick_freq = 150 * math.exp(-30 * beat_t) + 45
        kick = math.sin(2 * math.pi * kick_freq * beat_t) * kick_env if beat_t < 0.25 else 0.0

        # Snare
        snare = 0.0
        if beat_index in (1, 3):
            snare_env = math.exp(-15 * beat_t)
            noise = (math.sin(t * 15321) * 0.5 + math.sin(t * 27819) * 0.5) * snare_env
            snare_tone = math.sin(2 * math.pi * 180 * beat_t) * snare_env
            snare = (noise * 0.7 + snare_tone * 0.3) * 0.6 if beat_t < 0.3 else 0.0

        # Hi-hat
        hat_t = t % (beat_len / 2)
        hat = (math.sin(t * 43219) + math.sin(t * 89123)) * 0.5 * math.exp(-50 * hat_t) * 0.15

        # Bass
        bass_roots = [110.0, 87.31, 130.81, 98.0]
        root = bass_roots[bar_num]
        bass = (math.sin(2 * math.pi * (root / 2) * t) * 0.6 + math.sin(2 * math.pi * root * t) * 0.4) * math.exp(-8 * sub_beat) * 0.45

        # Arp
        chord_notes = [
            [220.0, 261.63, 329.63, 440.0],
            [174.61, 220.0, 261.63, 349.23],
            [261.63, 329.63, 392.0, 523.25],
            [196.0, 246.94, 293.66, 392.0]
        ]
        arp_idx = int((t % beat_len) / (beat_len / 4)) % 4
        synth = math.sin(2 * math.pi * chord_notes[bar_num][arp_idx] * t) * math.exp(-12 * sub_beat) * 0.2

        val = max(-1.0, min(1.0, kick * 0.5 + snare * 0.4 + hat * 0.2 + bass * 0.45 + synth * 0.3))
        samples.append(int(val * 32767))

    with wave.open(str(dst_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))


def _generate_lofi_track(dst_path: pathlib.Path, duration=60):
    sample_rate = 44100
    num_samples = sample_rate * duration
    bpm = 84
    beat_len = 60.0 / bpm
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        beat_t = t % beat_len
        bar_num = int(t / (beat_len * 4)) % 4
        beat_index = int(t / beat_len) % 4

        # Vinyl warmth/crackle
        crackle = (math.sin(t * 54321) * math.sin(t * 12345)) * 0.03

        # Soft deep kick on beats 0 and 2.5
        kick = 0.0
        if beat_index == 0 or (beat_index == 2 and beat_t > beat_len * 0.45):
            k_env = math.exp(-18 * beat_t)
            kick = math.sin(2 * math.pi * (70 * math.exp(-20 * beat_t) + 38) * beat_t) * k_env * 0.55

        # Soft snare / rimshot on beats 1 and 3
        snare = 0.0
        if beat_index in (1, 3):
            s_env = math.exp(-20 * beat_t)
            snare = (math.sin(t * 31415) * 0.4 + math.sin(2 * math.pi * 210 * beat_t) * 0.6) * s_env * 0.35

        # Rhodes style jazz chords (Cmaj7 - Am7 - Dm7 - G7)
        chords = [
            [261.63, 329.63, 392.00, 493.88], # Cmaj7
            [220.00, 261.63, 329.63, 392.00], # Am7
            [293.66, 349.23, 440.00, 523.25], # Dm7
            [196.00, 246.94, 293.66, 349.23]  # G7
        ]
        rhodes = 0.0
        c_notes = chords[bar_num]
        chord_env = math.exp(-1.5 * (t % (beat_len * 4)))
        for note in c_notes:
            rhodes += math.sin(2 * math.pi * note * t) * 0.12 * chord_env

        # Sub bass
        bass_roots = [65.41, 55.0, 73.42, 49.0]
        sub = math.sin(2 * math.pi * bass_roots[bar_num] * t) * 0.4 * chord_env

        val = max(-1.0, min(1.0, kick + snare + rhodes + sub + crackle))
        samples.append(int(val * 32767))

    with wave.open(str(dst_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))


def _generate_epic_track(dst_path: pathlib.Path, duration=60):
    sample_rate = 44100
    num_samples = sample_rate * duration
    bpm = 140
    beat_len = 60.0 / bpm
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        beat_t = t % beat_len
        bar_num = int(t / (beat_len * 4)) % 4
        sub_beat = t % (beat_len / 4)

        # Heavy punchy kick on every quarter note
        kick = math.sin(2 * math.pi * (180 * math.exp(-35 * beat_t) + 50) * beat_t) * math.exp(-22 * beat_t) * 0.6 if beat_t < 0.28 else 0.0

        # Clap
        clap = 0.0
        if int(t / beat_len) % 2 == 1:
            clap = (math.sin(t * 42819) * 0.7 + math.sin(2 * math.pi * 320 * beat_t) * 0.3) * math.exp(-22 * beat_t) * 0.5 if beat_t < 0.25 else 0.0

        # Cinematic rolling bass pulse
        pulse_roots = [65.41, 77.78, 58.27, 49.0] # C, Eb, Bb, G
        bass_freq = pulse_roots[bar_num]
        bass = (math.sin(2 * math.pi * bass_freq * t) + 0.3 * math.sin(2 * math.pi * bass_freq * 2 * t)) * math.exp(-6 * sub_beat) * 0.5

        # Brass hit on bar downbeat
        brass = 0.0
        if (t % (beat_len * 4)) < 0.6:
            br_env = math.exp(-3 * (t % (beat_len * 4)))
            brass = (math.sin(2 * math.pi * bass_freq * 2 * t) + math.sin(2 * math.pi * bass_freq * 3 * t)) * br_env * 0.35

        val = max(-1.0, min(1.0, kick + clap + bass + brass))
        samples.append(int(val * 32767))

    with wave.open(str(dst_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))

if __name__ == '__main__':
    ensure_bgm_library()
    print("BGM Library ready:", [t['name'] for t in list_bgm_tracks()])
