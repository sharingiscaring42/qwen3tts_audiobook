"""
Qwen3 TTS Voice Cloning Server on Modal

This module provides a voice cloning service using Qwen3-TTS 1.7B Base model.
The Base model supports voice cloning from reference audio + text.

It's designed to:
- Run on Modal cloud with GPU acceleration
- Use Modal Volume for persistent model storage
- Limit to max 1 instance for cost control
- Scale to zero when idle
"""

import modal
from pathlib import Path

FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"
# ATTN_IMPLEMENTATION = "flash_attention_2"  # Alternative attention implementation
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SCALEDOWN_WINDOW = 300
TIMEOUT = 1200
MEMORY = 16384

# Image with all dependencies pre-installed
IMAGE_BASE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
GPU_TYPE = "A10"
IMAGE_ENV = {
    "CUDA_HOME": "/usr/local/cuda",
    "HF_HOME": "/models/hf_cache",
    "TRANSFORMERS_CACHE": "/models/transformers_cache",
    "HF_HUB_CACHE": "/models/hf_hub_cache",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "ENABLE_DEBUG_TIMINGS": "0",
    # "ENABLE_CUDA_CLEANUP": "1",
}

image = (
    modal.Image.from_registry(
        IMAGE_BASE,
        add_python="3.11",
    )
    .entrypoint([])  # optional: silence base image entrypoint noise
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
    # .pip_install("numpy>=1.24.0")
    .pip_install(
        "qwen-tts>=0.0.5",
        extra_options="--no-deps",
    )
    .pip_install(
        "transformers==4.57.3",
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
    .pip_install("torch==2.8.0")
    .pip_install("torchaudio==2.8.0")
    .run_commands(f"python -m pip install --no-cache-dir '{FLASH_ATTN_WHEEL_URL}'")
    # .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .env(IMAGE_ENV)
)

# App definition
app = modal.App("qwen3-tts-voice-cloner", image=image)

# Volume for model persistence
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    # gpu="T4",  # Cost-effective GPU for 1.7B model (16GB VRAM)
    gpu=GPU_TYPE,  # Cost-effective GPU for 1.7B model (24GB VRAM)
    # gpu="A100",  # Cost-effective GPU for 1.7B model (40GB VRAM)


    max_containers=1,  # HARD LIMIT: Never exceed 1 instance
    min_containers=0,  # Scale to zero when idle (no charges)
    scaledown_window=SCALEDOWN_WINDOW,  # Keep warm for 60s after last request
    volumes={MODEL_DIR: volume},
    enable_memory_snapshot=True,  # Fast cold starts by restoring memory state
    timeout=TIMEOUT,  # 20 minutes timeout per request
    memory=MEMORY,  # 16GB RAM
)
class Qwen3VoiceCloner:
    """
    Voice cloning service using Qwen3 TTS 1.7B Base model
    
    Usage:
        1. Deploy: modal deploy modal_server.py
        2. Get endpoint URL from deployment output
        3. Send POST request with text, ref_audio_base64, and ref_text
    """
    
    @modal.enter()
    def load_model(self):
        """
        Load model once when container starts.
        This runs only on cold start, not on every request.
        """
        import torch
        import os
        import time
        
        print("=" * 60)
        print("Initializing Qwen3 TTS Voice Cloner...")
        print("=" * 60)
        
        model_path = f"{MODEL_DIR}/{MODEL_ID}"
        
        # Check if model exists in volume
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}")
            print("Falling back to HuggingFace Hub download...")
            model_path = MODEL_ID
        else:
            print(f"Loading model from Volume: {model_path}")
        self.model_path = model_path
        
        try:
            # Load model using qwen-tts package
            print("Loading model weights...")
            
            from qwen_tts import Qwen3TTSModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if device == "cuda":
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Load using qwen-tts API
            load_start = time.monotonic()
            load_kwargs = {
                "device_map": "cuda" if device == "cuda" else None,
                "dtype": torch.bfloat16 if device == "cuda" else None,
            }

            if device == "cuda" and ATTN_IMPLEMENTATION:
                load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    **{k: v for k, v in load_kwargs.items() if v is not None},
                )
            except Exception as e:
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

            # >>> ADD THESE LINES RIGHT HERE <<<
            import importlib
            from transformers.utils.import_utils import is_flash_attn_2_available
            print("torch:", torch.__version__)
            print("transformers:", __import__("transformers").__version__)
            print("transformers.flash_attn_2_available:", is_flash_attn_2_available())
            try:
                fa = importlib.import_module("flash_attn")
                print("flash_attn:", getattr(fa, "__version__", "unknown"))
            except Exception as e:
                print("flash_attn: NOT AVAILABLE", repr(e))

            print("requested_attn_implementation:", load_kwargs.get("attn_implementation"))

            # Try to print what attention impl is actually set to (best-effort)
            for name in ["attn_implementation", "_attn_implementation"]:
                if hasattr(self.model, name):
                    print(f"model.{name} =", getattr(self.model, name))
            # Some models store it under config
            config = getattr(self.model, "config", None)
            if config is not None and hasattr(config, "attn_implementation"):
                print("model.config.attn_implementation =", config.attn_implementation)
            # <<< END ADD >>>


            self.model_load_seconds = time.monotonic() - load_start
            self.model_loaded_at = time.time()
            self.voice_clone_prompt_cache = {}
            
            print("Model loaded successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise
    
    @modal.method()
    def clone_voice(
        self,
        text,
        ref_audio_base64: str,
        ref_text: str,
        language: str = "Auto",
        max_new_tokens: int = 2048,
    ) -> dict:
        """
        Clone voice and generate speech
        
        Args:
            text: Text to synthesize into speech (string or list of strings)
            ref_audio_base64: Base64-encoded reference audio (WAV format)
            ref_text: Transcript of the reference audio
            language: Language (default: "Auto" for automatic detection)
            
        Returns:
            Dictionary containing:
            - audio_base64: Generated audio as base64 string
            - sample_rate: Audio sample rate in Hz
            - duration_seconds: Duration of generated audio
            - text: Original text input
        """
        import torch
        import soundfile as sf
        import io
        import base64
        import numpy as np
        import time
        import os
        
        print(f"\n{'='*60}")
        print(f"Processing voice cloning request...")
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

            # Decode base64 to audio array and normalize like the official demo
            print("Decoding reference audio from base64...")
            audio_bytes = base64.b64decode(ref_audio_base64)
            audio_buffer = io.BytesIO(audio_bytes)
            ref_wav, ref_sr = sf.read(audio_buffer)

            def _normalize_audio(wav, eps=1e-12, clip=True):
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

            ref_wav = _normalize_audio(ref_wav)

            # Build or reuse voice clone prompt
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
                voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=(ref_wav, int(ref_sr)),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                prompt_seconds = time.monotonic() - prompt_start
                if cache_key:
                    self.voice_clone_prompt_cache[cache_key] = voice_clone_prompt
            else:
                prompt_seconds = 0.0

            # Generate speech
            print("Generating speech with voice clone...")

            processing_start = time.monotonic()
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if isinstance(text, list):
                    language_batch = language
                    if isinstance(language, str):
                        language_batch = [language] * len(text)

                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language_batch,
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
            
            # Encode output audio to base64
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
                    "model_load_seconds": getattr(self, "model_load_seconds", None),
                    "model_loaded_at": getattr(self, "model_loaded_at", None),
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
                    "model_load_seconds": getattr(self, "model_load_seconds", None),
                    "model_loaded_at": getattr(self, "model_loaded_at", None),
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
            
        except Exception as e:
            print(f"\nERROR during voice cloning: {e}")
            import traceback
            traceback.print_exc()
            return {
                "audio_base64": "",
                "sample_rate": 0,
                "duration_seconds": 0.0,
                "text": text,
                "success": False,
                "error": str(e),
                "processing_seconds": time.monotonic() - request_start,
                "model_load_seconds": getattr(self, "model_load_seconds", None),
                "model_loaded_at": getattr(self, "model_loaded_at", None),
            }
        # finally:
        #     try:
        #         import gc
        #         if os.getenv("ENABLE_CUDA_CLEANUP", "1") == "1":
        #             if torch.cuda.is_available():
        #                 torch.cuda.empty_cache()
        #             gc.collect()
        #     except Exception:
        #         pass
    
    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """
        FastAPI endpoint for HTTP access
        
        Example request:
        {
            "text": "Hello, this is my cloned voice speaking.",
            "ref_audio_base64": "<base64-encoded-wav-audio>",
            "ref_text": "Transcript of the reference audio.",
            "language": "English"  // optional, default: "Auto"
        }
        """
        return self.clone_voice.local(
            text=request.get("text", ""),
            ref_audio_base64=request.get("ref_audio_base64", ""),
            ref_text=request.get("ref_text", ""),
            language=request.get("language", "Auto"),
            max_new_tokens=request.get("max_new_tokens", 2048),
        )
    
    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint"""
        import torch
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_load_seconds": getattr(self, "model_load_seconds", None),
            "model_loaded_at": getattr(self, "model_loaded_at", None),
        }

    @modal.fastapi_endpoint(method="GET")
    def settings(self) -> dict:
        """Runtime settings snapshot"""
        import platform
        import torch
        from importlib import metadata

        def package_version(name: str) -> str:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return "not-installed"

        return {
            "scaledown_window": SCALEDOWN_WINDOW,
            "timeout": TIMEOUT,
            "memory": MEMORY,
            "model_path": getattr(self, "model_path", None),
            "attn_implementation": ATTN_IMPLEMENTATION,
            "gpu_type": GPU_TYPE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "image_base": IMAGE_BASE,
            "image_env": IMAGE_ENV,
            "versions": {
                "python": platform.python_version(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
                "torchaudio": package_version("torchaudio"),
                "qwen-tts": package_version("qwen-tts"),
            },
        }


# Local testing entrypoint
@app.local_entrypoint()
def main(
    test_text: str = "Hello, this is my cloned voice speaking.",
    ref_audio_path: str = "ref_audio.wav",
    ref_text_path: str = "ref_text.txt",
    language: str = "Auto",
):
    """
    Test the voice cloning service locally
    
    Usage:
        modal run modal_server.py --test-text "Your text here" --ref-audio-path "your_audio.wav"
    """
    import base64
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print("LOCAL TEST: Qwen3 TTS Voice Cloner")
    print("=" * 60 + "\n")
    
    # Read reference audio
    audio_path = Path(ref_audio_path)
    if not audio_path.exists():
        print(f"ERROR: Reference audio not found: {audio_path}")
        print("Please provide a reference audio file.")
        return
    
    with open(audio_path, "rb") as f:
        ref_audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"Loaded reference audio: {audio_path}")
    
    # Read reference text
    text_path = Path(ref_text_path)
    if text_path.exists():
        with open(text_path, "r") as f:
            ref_text = f.read().strip()
    else:
        print(f"ERROR: Reference text file not found: {text_path}")
        print("Please create a text file with the transcript of your reference audio.")
        return
    
    print(f"Reference text: {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}")
    print(f"Text to synthesize: {test_text}\n")
    
    # Call remote function
    print("Calling Modal function...")
    cloner = Qwen3VoiceCloner()
    result = cloner.clone_voice.remote(
        text=test_text,
        ref_audio_base64=ref_audio_base64,
        ref_text=ref_text,
        language=language,
    )
    
    if result.get("success"):
        # Save output
        output_path = Path("output.wav")
        audio_data = base64.b64decode(result["audio_base64"])
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        print("\n" + "=" * 60)
        print(f"SUCCESS!")
        print(f"Generated audio: {result['duration_seconds']:.2f}s")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Saved to: {output_path.absolute()}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"FAILED: {result.get('error', 'Unknown error')}")
        print("=" * 60)
