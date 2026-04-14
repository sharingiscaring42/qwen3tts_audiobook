"""Deploy entrypoint for single-container Qwen3 TTS + ASR service.

Deploy with:
    modal deploy server/modal_server.py
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

import modal
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    .pip_install("transformers==4.57.6")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .run_commands(f"python -m pip install --no-cache-dir '{FLASH_ATTN_WHEEL_URL}'")
    .env(IMAGE_ENV)
    .add_local_file(str(PROJECT_ROOT / "voice_cloner_core.py"), remote_path="/root/voice_cloner_core.py")
    .add_local_file(str(PROJECT_ROOT / "asr_core.py"), remote_path="/root/asr_core.py")
    .add_local_file(str(PROJECT_ROOT / "api_contracts.py"), remote_path="/root/api_contracts.py")
)

app = modal.App(f"qwen3-full-{GPU_TYPE}", image=image)
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)

from api_contracts import validate_asr_request, validate_tts_request
from asr_core import AsrCore
from voice_cloner_core import ATTN_IMPLEMENTATION, VoiceClonerCore


@app.cls(
    gpu=GPU_TYPE,
    max_containers=1,
    min_containers=0,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={MODEL_DIR: volume},
    enable_memory_snapshot=True,
    timeout=TIMEOUT,
    memory=MEMORY,
)
class Qwen3SpeechService:
    @modal.enter()
    def load_default(self):
        self.tts_model_cache: dict[str, VoiceClonerCore] = {}
        self.asr_model_cache: dict[str, AsrCore] = {}
        self._get_tts_core("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def _get_tts_core(self, model_id: str) -> VoiceClonerCore:
        core = self.tts_model_cache.get(model_id)
        if core is not None:
            return core
        model_path = f"{MODEL_DIR}/{model_id}"
        core = VoiceClonerCore(
            model_path=model_path,
            model_id=model_id,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        core.load_model()
        self.tts_model_cache[model_id] = core
        return core

    def _get_asr_core(self, model_id: str) -> AsrCore:
        core = self.asr_model_cache.get(model_id)
        if core is not None:
            return core
        model_path = f"{MODEL_DIR}/{model_id}"
        core = AsrCore(model_id=model_id, model_path=model_path)
        core.load_model()
        self.asr_model_cache[model_id] = core
        return core

    def _evict_tts_models(self) -> None:
        self.tts_model_cache = {}
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _evict_asr_models(self) -> None:
        self.asr_model_cache = {}
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @modal.method()
    def generate_tts(self, request: dict) -> dict:
        validated = validate_tts_request(request)
        # Single-container mode: free ASR models before TTS generation to avoid GPU OOM.
        self._evict_asr_models()
        core = self._get_tts_core(validated["tts_model"])
        result = core.synthesize(
            text=validated["text"],
            tts_mode=validated["tts_mode"],
            language=validated["language"],
            max_new_tokens=validated["max_new_tokens"],
            ref_audio_base64=validated["ref_audio_base64"],
            ref_text=validated["ref_text"],
            speaker=validated["speaker"],
            prompt_instruct_text=validated["prompt_instruct_text"],
            prompt=validated["prompt"],
        )
        result["tts_mode"] = validated["tts_mode"]
        result["tts_model"] = validated["tts_model"]
        return result

    @modal.method()
    def transcribe_audio(self, validated: dict) -> dict:
        # Single-container mode: free TTS models before ASR transcription to avoid GPU OOM.
        self._evict_tts_models()
        core = self._get_asr_core(validated["asr_model"])
        result = core.transcribe(
            audio_base64=validated["audio_base64"],
            language=validated["language"],
            return_timestamps=validated["return_timestamps"],
            align_ref_text=validated["align_ref_text"],
            aligner_model=validated["aligner_model"],
        )
        result["asr_model"] = validated["asr_model"]
        return result

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        try:
            return self.generate_tts.local(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: dict) -> dict:
        try:
            validated = validate_asr_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            return self.transcribe_audio.local(validated)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @modal.fastapi_endpoint(method="GET")
    def settings(self) -> dict:
        def package_version(name: str) -> str:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return "not-installed"

        return {
            "service": "tts_asr_unified_single_container",
            "gpu_type": GPU_TYPE,
            "scaledown_window": SCALEDOWN_WINDOW,
            "timeout": TIMEOUT,
            "memory": MEMORY,
            "model_dir": MODEL_DIR,
            "default_tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "default_asr_model": "Qwen/Qwen3-ASR-1.7B",
            "loaded_tts_models": sorted(getattr(self, "tts_model_cache", {}).keys()),
            "loaded_asr_models": sorted(getattr(self, "asr_model_cache", {}).keys()),
            "attn_implementation": ATTN_IMPLEMENTATION,
            "versions": {
                "python": platform.python_version(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
                "torchaudio": package_version("torchaudio"),
                "qwen-tts": package_version("qwen-tts"),
            },
        }


@app.local_entrypoint()
def main():
    print("Single-container Qwen3 speech app loaded. Deploy with: modal deploy server/modal_server.py")
