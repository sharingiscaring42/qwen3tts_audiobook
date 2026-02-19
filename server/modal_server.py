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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_cloner_core import VoiceClonerCore, MODEL_ID, ATTN_IMPLEMENTATION

FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
# ATTN_IMPLEMENTATION = "flash_attention_2"  # Alternative attention implementation
SCALEDOWN_WINDOW = 300 # 5 minutes scaledown window to keep container warm after last request
TIMEOUT = 1200
MEMORY = 16384

# Image with all dependencies pre-installed
IMAGE_BASE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
GPU_TYPE = "A10"
# ATTN_MODE = "sdpa_flash"  # options: "sdpa_flash", "auto", "flash_attn3"
IMAGE_ENV = {
    "CUDA_HOME": "/usr/local/cuda",
    "HF_HOME": "/models/hf_cache",
    "TRANSFORMERS_CACHE": "/models/transformers_cache",
    "HF_HUB_CACHE": "/models/hf_hub_cache",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "ENABLE_DEBUG_TIMINGS": "0",
    # "ATTN_IMPLEMENTATION": "flash_attention_2" if ATTN_MODE == "sdpa_flash" else (
    #     "kernels-community/flash-attn3" if ATTN_MODE == "flash_attn3" else "default"
    # ),
    # "ENABLE_CUDA_CLEANUP": "1",
}

CORE_FILE = PROJECT_ROOT / "voice_cloner_core.py"

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
    # .pip_install("kernels")
    # .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .env(IMAGE_ENV)
    .add_local_file(str(CORE_FILE), remote_path="/root/voice_cloner_core.py")
)

# App definition
app = modal.App(f"qwen3-tts-voice-cloner-{GPU_TYPE}", image=image)

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
        print("=" * 60)
        print("Initializing Qwen3 TTS Voice Cloner...")
        print("=" * 60)
        model_path = f"{MODEL_DIR}/{MODEL_ID}"
        self.core = VoiceClonerCore(
            model_path=model_path,
            model_id=MODEL_ID,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        self.core.load_model()
    
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
        return self.core.clone_voice(
            text=text,
            ref_audio_base64=ref_audio_base64,
            ref_text=ref_text,
            language=language,
            max_new_tokens=max_new_tokens,
        )
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
    
    # @modal.fastapi_endpoint(method="GET")
    # def health(self) -> dict:
    #     """Health check endpoint"""
    #     import torch
    #     core = getattr(self, "core", None)
    #     return {
    #         "status": "healthy",
    #         "model": MODEL_ID,
    #         "gpu_available": torch.cuda.is_available(),
    #         "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    #         "model_load_seconds": getattr(core, "model_load_seconds", None),
    #         "model_loaded_at": getattr(core, "model_loaded_at", None),
    #     }

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
            "model_path": getattr(getattr(self, "core", None), "model_path", None),
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
