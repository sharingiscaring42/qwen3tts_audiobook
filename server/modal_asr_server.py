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
    from api_contracts import validate_asr_request
    from asr_core import AsrCore
    from modal_app import (
        MEMORY,
        MODEL_DIR,
        SCALEDOWN_WINDOW,
        TIMEOUT,
        GPU_TYPE,
        app,
        volume,
    )
except ModuleNotFoundError:
    from server.api_contracts import validate_asr_request
    from server.asr_core import AsrCore
    from server.modal_app import (
        MEMORY,
        MODEL_DIR,
        SCALEDOWN_WINDOW,
        TIMEOUT,
        GPU_TYPE,
        app,
        volume,
    )


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
class Qwen3ASRService:
    @modal.enter()
    def load_default(self):
        self.model_cache: dict[str, AsrCore] = {}
        self._get_core("Qwen/Qwen3-ASR-1.7B")

    def _get_core(self, model_id: str) -> AsrCore:
        core = self.model_cache.get(model_id)
        if core is not None:
            return core
        model_path = f"{MODEL_DIR}/{model_id}"
        core = AsrCore(model_id=model_id, model_path=model_path)
        core.load_model()
        self.model_cache[model_id] = core
        return core

    @modal.method()
    def transcribe_audio(self, request: dict) -> dict:
        validated = validate_asr_request(request)
        core = self._get_core(validated["asr_model"])
        result = core.transcribe(**validated)
        result["asr_model"] = validated["asr_model"]
        return result

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: dict) -> dict:
        try:
            return self.transcribe_audio.local(request)
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
            "service": "asr",
            "gpu_type": GPU_TYPE,
            "scaledown_window": SCALEDOWN_WINDOW,
            "timeout": TIMEOUT,
            "memory": MEMORY,
            "model_dir": MODEL_DIR,
            "default_model": "Qwen/Qwen3-ASR-1.7B",
            "loaded_models": loaded_models,
            "versions": {
                "python": platform.python_version(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
                "torchaudio": package_version("torchaudio"),
            },
        }
