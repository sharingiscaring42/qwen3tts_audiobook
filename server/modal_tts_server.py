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

try:
    from api_contracts import validate_tts_request
    from modal_app import (
        MEMORY,
        MODEL_DIR,
        SCALEDOWN_WINDOW,
        TIMEOUT,
        GPU_TYPE,
        app,
        volume,
    )
    from voice_cloner_core import ATTN_IMPLEMENTATION, VoiceClonerCore
except ModuleNotFoundError:
    from server.api_contracts import validate_tts_request
    from server.modal_app import (
        MEMORY,
        MODEL_DIR,
        SCALEDOWN_WINDOW,
        TIMEOUT,
        GPU_TYPE,
        app,
        volume,
    )
    from server.voice_cloner_core import ATTN_IMPLEMENTATION, VoiceClonerCore


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
class Qwen3TTSService:
    @modal.enter()
    def load_default(self):
        self.model_cache: dict[str, VoiceClonerCore] = {}
        self._get_core("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def _get_core(self, model_id: str) -> VoiceClonerCore:
        core = self.model_cache.get(model_id)
        if core is not None:
            return core
        model_path = f"{MODEL_DIR}/{model_id}"
        core = VoiceClonerCore(
            model_path=model_path,
            model_id=model_id,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        core.load_model()
        self.model_cache[model_id] = core
        return core

    @modal.method()
    def generate_tts(self, request: dict) -> dict:
        validated = validate_tts_request(request)
        core = self._get_core(validated["tts_model"])
        result = core.synthesize(**validated)
        result["tts_mode"] = validated["tts_mode"]
        result["tts_model"] = validated["tts_model"]
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

    @modal.fastapi_endpoint(method="GET")
    def settings(self) -> dict:
        def package_version(name: str) -> str:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return "not-installed"

        model_cache = getattr(self, "model_cache", {})
        loaded_models = sorted(model_cache.keys()) if isinstance(model_cache, dict) else []

        return {
            "service": "tts",
            "gpu_type": GPU_TYPE,
            "scaledown_window": SCALEDOWN_WINDOW,
            "timeout": TIMEOUT,
            "memory": MEMORY,
            "model_dir": MODEL_DIR,
            "default_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "loaded_models": loaded_models,
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
