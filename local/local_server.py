"""
Local FastAPI server for Qwen3 TTS voice cloning (WSL2 + pip).

Uses the same core logic as the Modal deployment, but runs fully locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.voice_cloner_core import (
    VoiceClonerCore,
    MODEL_ID as DEFAULT_MODEL_ID,
    ATTN_IMPLEMENTATION as DEFAULT_ATTN_IMPLEMENTATION,
)


MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
_raw_attn = os.getenv("ATTN_IMPLEMENTATION", DEFAULT_ATTN_IMPLEMENTATION)
ATTN_IMPLEMENTATION = _raw_attn or None

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_ID) if MODEL_DIR else MODEL_ID

app = FastAPI(title="Qwen3 TTS Voice Cloner (Local)", version="1.0.0")

core: VoiceClonerCore | None = None


class GenerateRequest(BaseModel):
    text: Union[str, List[str]]
    ref_audio_base64: Optional[str] = None
    reference_audio_base64: Optional[str] = None
    ref_text: Optional[str] = None
    reference_text: Optional[str] = None
    language: str = Field(default="Auto")
    max_new_tokens: int = Field(default=2048, ge=1)


@app.on_event("startup")
def _startup() -> None:
    global core
    instance = VoiceClonerCore(
        model_path=MODEL_PATH,
        model_id=MODEL_ID,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    instance.load_model()
    core = instance


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Qwen3 TTS local server"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    ref_audio_base64 = request.ref_audio_base64 or request.reference_audio_base64 or ""
    ref_text = request.ref_text or request.reference_text

    if not ref_audio_base64:
        raise HTTPException(status_code=400, detail="Missing ref_audio_base64")
    if ref_text is None:
        raise HTTPException(status_code=400, detail="Missing ref_text")

    if core is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return core.clone_voice(
        text=request.text,
        ref_audio_base64=ref_audio_base64,
        ref_text=ref_text,
        language=request.language,
        max_new_tokens=request.max_new_tokens,
    )


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "healthy",
        "model": MODEL_ID,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_load_seconds": getattr(core, "model_load_seconds", None),
        "model_loaded_at": getattr(core, "model_loaded_at", None),
    }


@app.get("/settings")
def settings() -> dict:
    import platform
    from importlib import metadata

    def package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "model_path": getattr(core, "model_path", None),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "model_id": MODEL_ID,
        "model_dir": MODEL_DIR,
        "versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "torchaudio": package_version("torchaudio"),
            "qwen-tts": package_version("qwen-tts"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
