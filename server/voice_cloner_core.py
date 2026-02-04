"""
Shared core logic for Qwen3 TTS voice cloning.

This module is used by both the Modal server and local WSL2 server so they
stay in sync.
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
        import torch
        import importlib
        from qwen_tts import Qwen3TTSModel
        from transformers.utils.import_utils import is_flash_attn_2_available

        print("=" * 60)
        print("Initializing Qwen3 TTS Voice Cloner...")
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

        try:
            print("Loading model weights...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
                print(
                    f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                )

            load_start = time.monotonic()
            load_kwargs = {
                "device_map": "cuda" if device == "cuda" else None,
                "dtype": torch.bfloat16 if device == "cuda" else None,
            }

            if device == "cuda" and self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation

            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    **{k: v for k, v in load_kwargs.items() if v is not None},
                )
            except Exception:
                if device == "cuda" and load_kwargs.get("attn_implementation"):
                    print(
                        "WARNING: flash-attn3 not available, retrying without custom attention impl."
                    )
                    load_kwargs.pop("attn_implementation", None)
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_path,
                        **{k: v for k, v in load_kwargs.items() if v is not None},
                    )
                else:
                    raise

            print("torch:", torch.__version__)
            print("transformers:", __import__("transformers").__version__)
            print("transformers.flash_attn_2_available:", is_flash_attn_2_available())
            try:
                fa = importlib.import_module("flash_attn")
                print("flash_attn:", getattr(fa, "__version__", "unknown"))
            except Exception as exc:
                print("flash_attn: NOT AVAILABLE", repr(exc))

            print("requested_attn_implementation:", load_kwargs.get("attn_implementation"))

            for name in ["attn_implementation", "_attn_implementation"]:
                if hasattr(self.model, name):
                    print(f"model.{name} =", getattr(self.model, name))

            config = getattr(self.model, "config", None)
            if config is not None and hasattr(config, "attn_implementation"):
                print("model.config.attn_implementation =", config.attn_implementation)

            self.model_load_seconds = time.monotonic() - load_start
            self.model_loaded_at = time.time()
            self.voice_clone_prompt_cache = {}

            print("Model loaded successfully!")
            print("=" * 60)

        except Exception as exc:
            print(f"ERROR loading model: {exc}")
            raise

    def clone_voice(
        self,
        text,
        ref_audio_base64: str,
        ref_text: str,
        language: str = "Auto",
        max_new_tokens: int = 2048,
    ) -> dict:
        import torch
        import soundfile as sf
        import numpy as np
        import time

        if self.model is None:
            raise RuntimeError("Model not loaded")

        model = self.model

        print(f"\n{'='*60}")
        print("Processing voice cloning request...")
        if isinstance(text, list):
            text_lengths = [len(item) for item in text]
            total_chars = sum(text_lengths)
            print(f"Text list items: {len(text)}")
            print(f"Text lengths: {text_lengths}")
            print(f"Text total chars: {total_chars}")
        else:
            print(f"Text chars: {len(text)}")
            print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"Language: {language}")
        print(f"{'='*60}\n")
        request_start = time.monotonic()
        debug_timings = os.getenv("ENABLE_DEBUG_TIMINGS", "0") == "1"

        try:
            print(f"Reference audio provided: {len(ref_audio_base64)} chars (base64)")

            print("Decoding reference audio from base64...")
            audio_bytes = base64.b64decode(ref_audio_base64)
            audio_buffer = io.BytesIO(audio_bytes)
            ref_wav, ref_sr = sf.read(audio_buffer)

            ref_wav = _normalize_audio(ref_wav)

            cache_key = None
            if ref_audio_base64 and ref_text is not None:
                import hashlib

                key_material = f"{ref_audio_base64}||{ref_text}"
                cache_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()

            voice_clone_prompt = None
            if cache_key:
                voice_clone_prompt = self.voice_clone_prompt_cache.get(cache_key)

            if voice_clone_prompt is None:
                print("Creating voice clone prompt...")
                prompt_start = time.monotonic()
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=(ref_wav, int(ref_sr)),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                prompt_seconds = time.monotonic() - prompt_start
                if cache_key:
                    self.voice_clone_prompt_cache[cache_key] = voice_clone_prompt
            else:
                prompt_seconds = 0.0

            print("Generating speech with voice clone...")

            processing_start = time.monotonic()
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if isinstance(text, list):
                    language_batch = language
                    if isinstance(language, str):
                        language_batch = [language] * len(text)

                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language_batch,
                        voice_clone_prompt=voice_clone_prompt,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=voice_clone_prompt,
                        max_new_tokens=max_new_tokens,
                    )

                sample_rate_out = sr
                if sample_rate_out is None:
                    raise RuntimeError("Failed to determine sample rate for batch output")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            processing_seconds = time.monotonic() - processing_start

            if isinstance(text, list):
                total_duration = sum(len(wav) for wav in wavs) / sample_rate_out
                print(f"Speech generated: {total_duration:.2f}s (batch)")
            else:
                generated_audio = wavs[0]
                print(f"Speech generated: {len(generated_audio)/sample_rate_out:.2f}s")

            print("Encoding output audio...")
            encode_start = time.monotonic()
            if isinstance(text, list):
                audio_base64s = []
                durations = []
                for wav in wavs:
                    output_buffer = io.BytesIO()
                    sf.write(output_buffer, wav, sample_rate_out, format="WAV")
                    output_buffer.seek(0)
                    audio_base64s.append(base64.b64encode(output_buffer.read()).decode("utf-8"))
                    durations.append(len(wav) / sample_rate_out)

                result = {
                    "audio_base64s": audio_base64s,
                    "sample_rate": sample_rate_out,
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
                sf.write(output_buffer, generated_audio, sample_rate_out, format="WAV")
                output_buffer.seek(0)
                output_base64 = base64.b64encode(output_buffer.read()).decode("utf-8")

                result = {
                    "audio_base64": output_base64,
                    "sample_rate": sample_rate_out,
                    "duration_seconds": len(generated_audio) / sample_rate_out,
                    "text": text,
                    "language": language,
                    "success": True,
                    "processing_seconds": processing_seconds,
                    "model_load_seconds": self.model_load_seconds,
                    "model_loaded_at": self.model_loaded_at,
                }

            encode_seconds = time.monotonic() - encode_start

            if debug_timings:
                gpu_mem = None
                if torch.cuda.is_available():
                    gpu_mem = {
                        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    }
                result["timing_breakdown"] = {
                    "prompt_seconds": prompt_seconds,
                    "generate_seconds": processing_seconds,
                    "encode_seconds": encode_seconds,
                }
                result["gpu_memory"] = gpu_mem

            print(f"\n{'='*60}")
            print("Request completed successfully!")
            if isinstance(result.get("duration_seconds"), list):
                total_audio = sum(result["duration_seconds"])
                print(f"Generated: {total_audio:.2f}s of audio (batch)")
            else:
                total_audio = result["duration_seconds"]
                print(f"Generated: {total_audio:.2f}s of audio")
            if processing_seconds > 0:
                inv_rtf = total_audio / processing_seconds
                print(f"1/RTF (audio/processing): {inv_rtf:.3f}x")
            print(f"{'='*60}\n")

            return result

        except Exception as exc:
            print(f"\nERROR during voice cloning: {exc}")
            import traceback

            traceback.print_exc()
            return {
                "audio_base64": "",
                "sample_rate": 0,
                "duration_seconds": 0.0,
                "text": text,
                "success": False,
                "error": str(exc),
                "processing_seconds": time.monotonic() - request_start,
                "model_load_seconds": self.model_load_seconds,
                "model_loaded_at": self.model_loaded_at,
            }
