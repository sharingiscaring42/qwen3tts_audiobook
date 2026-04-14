from __future__ import annotations

from pathlib import Path

import modal

FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

SCALEDOWN_WINDOW = 300
TIMEOUT = 1200
MEMORY = 16384
GPU_TYPE = "A10"
MODEL_DIR = "/models"

IMAGE_BASE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
IMAGE_ENV = {
    "CUDA_HOME": "/usr/local/cuda",
    "HF_HOME": "/models/hf_cache",
    "TRANSFORMERS_CACHE": "/models/transformers_cache",
    "HF_HUB_CACHE": "/models/hf_hub_cache",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "ENABLE_DEBUG_TIMINGS": "0",
}

SERVER_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.11")
    .entrypoint([])
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "sox",
        "libsox-fmt-all",
        "build-essential",
        "ninja-build",
        "cmake",
    )
    .pip_install("packaging", "setuptools", "wheel")
    .pip_install("qwen-tts>=0.0.5", extra_options="--no-deps")
    .pip_install("qwen-asr==0.0.6")
    .pip_install(
        "transformers==4.57.6",
        "accelerate>=0.25.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.2",
        "numpy>=1.24.0",
        "onnxruntime>=1.18.0",
        "gradio>=4.0.0",
        "sox>=1.4.1",
        "fastapi[standard]>=0.110.0",
        "pydantic>=2.0.0",
    )
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .run_commands(f"python -m pip install --no-cache-dir '{FLASH_ATTN_WHEEL_URL}'")
    .env(IMAGE_ENV)
    .add_local_file(str(SERVER_DIR / "voice_cloner_core.py"), remote_path="/root/voice_cloner_core.py")
    .add_local_file(str(SERVER_DIR / "asr_core.py"), remote_path="/root/asr_core.py")
    .add_local_file(str(SERVER_DIR / "api_contracts.py"), remote_path="/root/api_contracts.py")
)

app = modal.App(f"qwen3-full-{GPU_TYPE}", image=image)
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
