"""
app/utilities/hardware.py
-------------------------
Detects available hardware (GPU, CUDA, NVENC) and exposes a singleton
HardwareInfo dataclass so all modules can make consistent decisions about
which encoder / inference backend to use.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from app.utilities.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class HardwareInfo:
    has_nvidia_gpu: bool
    gpu_name: str
    gpu_vram_mb: int
    driver_version: str
    nvenc_available: bool   # FFmpeg compiled with --enable-nvenc
    cuda_available: bool    # FFmpeg compiled with --enable-cuda-llvm
    preferred_encoder: str  # "h264_nvenc" | "libx264"
    preferred_decoder: str  # "h264_cuvid" | "" (empty = software)
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareInfo:
    """
    Query nvidia-smi and the local FFmpeg build for hardware capabilities.
    Returns a cached HardwareInfo object — safe to call many times.
    """
    gpu_name = ""
    gpu_vram_mb = 0
    driver_version = ""
    has_nvidia = False

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                gpu_vram_mb = int(parts[1])
                driver_version = parts[2]
                has_nvidia = True
                log.info(
                    "NVIDIA GPU detected",
                    extra={"gpu": gpu_name, "vram_mb": gpu_vram_mb},
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        log.info("No NVIDIA GPU detected — CPU mode.")

    # Check FFmpeg for NVENC / CUDA support
    nvenc_available = False
    cuda_available = False
    if has_nvidia:
        try:
            enc_result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True, text=True, timeout=15,
            )
            nvenc_available = "h264_nvenc" in enc_result.stdout
            dec_result = subprocess.run(
                ["ffmpeg", "-decoders"],
                capture_output=True, text=True, timeout=15,
            )
            cuda_available = "h264_cuvid" in dec_result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    preferred_encoder = "h264_nvenc" if nvenc_available else "libx264"
    preferred_decoder = "h264_cuvid" if cuda_available else ""

    info = HardwareInfo(
        has_nvidia_gpu=has_nvidia,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        driver_version=driver_version,
        nvenc_available=nvenc_available,
        cuda_available=cuda_available,
        preferred_encoder=preferred_encoder,
        preferred_decoder=preferred_decoder,
    )
    log.info(
        "Hardware profile",
        extra={
            "encoder": preferred_encoder,
            "decoder": preferred_decoder or "software",
        },
    )
    return info
