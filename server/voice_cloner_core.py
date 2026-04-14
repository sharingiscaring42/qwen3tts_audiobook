"""
Shared core logic for Qwen3 TTS inference.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"


def _normalize_audio(wav, eps: float = 1e-12, clip: bool = True):
    import numpy as np

    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


class VoiceClonerCore:
    def __init__(
        self,
        model_path: str | None = None,
        model_id: str = MODEL_ID,
        attn_implementation: str | None = ATTN_IMPLEMENTATION,
    ) -> None:
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        self.model_path = model_path or model_id
        self.model: Any = None
        self.model_load_seconds = None
        self.model_loaded_at = None
        self.voice_clone_prompt_cache: dict[str, Any] = {}

    def load_model(self) -> None:
        import importlib

        import torch
        from qwen_tts import Qwen3TTSModel
        from transformers.utils.import_utils import is_flash_attn_2_available

        print("=" * 60)
        print(f"Initializing Qwen3 TTS model: {self.model_id}")
        print("=" * 60)

        model_path = self.model_path

        if model_path != self.model_id and not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}")
            print("Falling back to HuggingFace Hub download...")
            model_path = self.model_id
        elif model_path == self.model_id:
            print(f"Loading model from HuggingFace Hub: {model_path}")
        else:
            print(f"Loading model from local path: {model_path}")

        self.model_path = model_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs = {
            "device_map": "cuda" if device == "cuda" else None,
            "dtype": torch.bfloat16 if device == "cuda" else None,
        }
        if device == "cuda" and self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        load_start = time.monotonic()
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                **{k: v for k, v in load_kwargs.items() if v is not None},
            )
        except Exception:
            if load_kwargs.get("attn_implementation"):
                print("WARNING: custom attention load failed, retrying with defaults")
                load_kwargs.pop("attn_implementation", None)
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    **{k: v for k, v in load_kwargs.items() if v is not None},
                )
            else:
                raise

        self.model_load_seconds = time.monotonic() - load_start
        self.model_loaded_at = time.time()
        self.voice_clone_prompt_cache = {}

        print("torch:", torch.__version__)
        print("transformers:", __import__("transformers").__version__)
        print("transformers.flash_attn_2_available:", is_flash_attn_2_available())
        try:
            fa = importlib.import_module("flash_attn")
            print("flash_attn:", getattr(fa, "__version__", "unknown"))
        except Exception as exc:
            print("flash_attn: NOT AVAILABLE", repr(exc))

        print("Model loaded successfully")

    def _build_result(
        self,
        *,
        wavs,
        sample_rate: int,
        text,
        language,
        processing_seconds: float,
        request_start: float,
        timing_breakdown: dict[str, float] | None = None,
    ) -> dict:
        import soundfile as sf

        if isinstance(text, list):
            audio_base64s = []
            durations = []
            for wav in wavs:
                output_buffer = io.BytesIO()
                sf.write(output_buffer, wav, sample_rate, format="WAV")
                output_buffer.seek(0)
                audio_base64s.append(base64.b64encode(output_buffer.read()).decode("utf-8"))
                durations.append(len(wav) / sample_rate)
            result = {
                "audio_base64s": audio_base64s,
                "sample_rate": sample_rate,
                "duration_seconds": durations,
                "text": text,
                "language": language,
                "success": True,
                "processing_seconds": processing_seconds,
                "model_load_seconds": self.model_load_seconds,
                "model_loaded_at": self.model_loaded_at,
            }
        else:
            generated_audio = wavs[0]
            output_buffer = io.BytesIO()
            sf.write(output_buffer, generated_audio, sample_rate, format="WAV")
            output_buffer.seek(0)
            output_base64 = base64.b64encode(output_buffer.read()).decode("utf-8")
            result = {
                "audio_base64": output_base64,
                "sample_rate": sample_rate,
                "duration_seconds": len(generated_audio) / sample_rate,
                "text": text,
                "language": language,
                "success": True,
                "processing_seconds": processing_seconds,
                "model_load_seconds": self.model_load_seconds,
                "model_loaded_at": self.model_loaded_at,
            }

        if timing_breakdown is not None:
            result["timing_breakdown"] = timing_breakdown

        result["total_seconds"] = time.monotonic() - request_start
        return result

    def _run_base_clone(
        self,
        text,
        ref_audio_base64: str,
        ref_text: str,
        language: str,
        max_new_tokens: int,
    ) -> dict:
        import hashlib

        import soundfile as sf
        import torch

        audio_bytes = base64.b64decode(ref_audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        ref_wav, ref_sr = sf.read(audio_buffer)
        ref_wav = _normalize_audio(ref_wav)

        cache_key = hashlib.sha256(f"{ref_audio_base64}||{ref_text}".encode("utf-8")).hexdigest()
        voice_clone_prompt = self.voice_clone_prompt_cache.get(cache_key)

        prompt_start = time.monotonic()
        if voice_clone_prompt is None:
            voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=(ref_wav, int(ref_sr)),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
            self.voice_clone_prompt_cache[cache_key] = voice_clone_prompt
        prompt_seconds = time.monotonic() - prompt_start

        processing_start = time.monotonic()
        with torch.no_grad():
            if isinstance(text, list):
                lang_batch = language if isinstance(language, list) else [language] * len(text)
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=lang_batch,
                    voice_clone_prompt=voice_clone_prompt,
                    max_new_tokens=max_new_tokens,
                )
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    max_new_tokens=max_new_tokens,
                )
        processing_seconds = time.monotonic() - processing_start

        return {
            "wavs": wavs,
            "sample_rate": sr,
            "processing_seconds": processing_seconds,
            "timing_breakdown": {
                "prompt_seconds": prompt_seconds,
                "generate_seconds": processing_seconds,
            },
        }

    def _run_generic_generate(self, *, text, language: str, max_new_tokens: int, **kwargs) -> dict:
        import torch

        processing_start = time.monotonic()
        with torch.no_grad():
            # API compatibility fallback across qwen-tts versions.
            attempts = [
                {"text": text, "language": language, "max_new_tokens": max_new_tokens, **kwargs},
                {"text": text, "max_new_tokens": max_new_tokens, **kwargs},
                {"text": text, **kwargs},
            ]

            last_exc: Exception | None = None
            for params in attempts:
                try:
                    wavs, sr = self.model.generate(**params)
                    return {
                        "wavs": wavs,
                        "sample_rate": sr,
                        "processing_seconds": time.monotonic() - processing_start,
                        "timing_breakdown": {
                            "prompt_seconds": 0.0,
                            "generate_seconds": time.monotonic() - processing_start,
                        },
                    }
                except TypeError as exc:
                    last_exc = exc
                    continue

            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Failed to invoke generate()")

    def synthesize(
        self,
        *,
        text,
        tts_mode: str,
        language: str = "Auto",
        max_new_tokens: int = 2048,
        ref_audio_base64: str = "",
        ref_text: str = "",
        speaker: str = "",
        prompt_instruct_text: str = "",
        prompt: str = "",
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        request_start = time.monotonic()
        if tts_mode == "base_clone":
            run = self._run_base_clone(
                text=text,
                ref_audio_base64=ref_audio_base64,
                ref_text=ref_text,
                language=language,
                max_new_tokens=max_new_tokens,
            )
        elif tts_mode == "custom_voice":
            run = self._run_generic_generate(
                text=text,
                language=language,
                max_new_tokens=max_new_tokens,
                speaker=speaker,
                prompt_instruct_text=prompt_instruct_text,
            )
        elif tts_mode == "voice_design":
            run = self._run_generic_generate(
                text=text,
                language=language,
                max_new_tokens=max_new_tokens,
                prompt=prompt,
            )
        else:
            raise ValueError(f"Unsupported tts_mode: {tts_mode}")

        return self._build_result(
            wavs=run["wavs"],
            sample_rate=run["sample_rate"],
            text=text,
            language=language,
            processing_seconds=run["processing_seconds"],
            request_start=request_start,
            timing_breakdown=run.get("timing_breakdown"),
        )

    # Backward compatibility for local server usage.
    def clone_voice(
        self,
        text,
        ref_audio_base64: str,
        ref_text: str,
        language: str = "Auto",
        max_new_tokens: int = 2048,
    ) -> dict:
        return self.synthesize(
            text=text,
            tts_mode="base_clone",
            ref_audio_base64=ref_audio_base64,
            ref_text=ref_text,
            language=language,
            max_new_tokens=max_new_tokens,
        )
