from __future__ import annotations

import base64
import io
import time
from typing import Any


class AsrCore:
    def __init__(self, model_id: str, model_path: str | None = None) -> None:
        self.model_id = model_id
        self.model_path = model_path or model_id
        self.model: Any = None
        self.model_with_aligner: Any = None
        self.model_with_aligner_id: str | None = None
        self.model_load_seconds: float | None = None
        self.model_loaded_at: float | None = None

    def load_model(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        start = time.monotonic()
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=256,
        )
        self.model_load_seconds = time.monotonic() - start
        self.model_loaded_at = time.time()

    def _decode_audio(self, audio_base64: str):
        import numpy as np
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_base64)
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float32)
        return data, int(sr)

    def _ensure_model_with_aligner(self, aligner_model_id: str) -> dict[str, Any]:
        if self.model_with_aligner is not None and self.model_with_aligner_id == aligner_model_id:
            return {"loaded": True, "model_id": aligner_model_id, "cached": True}

        import torch
        from qwen_asr import Qwen3ASRModel

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            self.model_with_aligner = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=256,
                forced_aligner=aligner_model_id,
                forced_aligner_kwargs={
                    "dtype": dtype,
                    "device_map": device_map,
                },
            )
            self.model_with_aligner_id = aligner_model_id
            return {"loaded": True, "model_id": aligner_model_id, "cached": False}
        except Exception as exc:
            self.model_with_aligner = None
            self.model_with_aligner_id = None
            return {"loaded": False, "model_id": aligner_model_id, "cached": False, "warning": str(exc)}

    def transcribe(
        self,
        *,
        audio_base64: str,
        language: str = "auto",
        return_timestamps: bool = False,
        align_ref_text: str | None = None,
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ASR model not loaded")

        request_start = time.monotonic()
        audio, sr = self._decode_audio(audio_base64)

        language_arg = None if not language or language.lower() == "auto" else language
        active_model = self.model
        aligner_info = None
        if return_timestamps:
            aligner_info = self._ensure_model_with_aligner(aligner_model)
            if aligner_info.get("loaded"):
                active_model = self.model_with_aligner

        result = active_model.transcribe(
            audio=(audio, sr),
            language=language_arg,
            return_time_stamps=return_timestamps,
        )
        processing_seconds = time.monotonic() - request_start

        item = result[0] if isinstance(result, list) and result else result
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            detected_language = item.get("language", language)
            time_stamps = item.get("time_stamps")
        else:
            text = str(getattr(item, "text", ""))
            detected_language = getattr(item, "language", language)
            time_stamps = getattr(item, "time_stamps", None)

        response: dict[str, Any] = {
            "success": True,
            "text": text,
            "language": detected_language or language,
            "model_id": self.model_id,
            "processing_seconds": processing_seconds,
            "model_load_seconds": self.model_load_seconds,
            "model_loaded_at": self.model_loaded_at,
        }

        if return_timestamps:
            if time_stamps is not None:
                response["segments"] = time_stamps
            response["aligner"] = aligner_info or {"loaded": False, "model_id": aligner_model, "cached": False}
            if align_ref_text:
                response["align_ref_text"] = align_ref_text

        return response
