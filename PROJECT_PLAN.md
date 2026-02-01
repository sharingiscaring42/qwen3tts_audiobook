# Qwen3 TTS 1.7B Voice Cloning on Modal - Project Plan

## Project Overview

Build a voice cloning service using Qwen3-TTS-1.7B deployed on Modal with:
- **Server**: Modal Function/Class with web endpoint for TTS generation
- **Client**: Python script to interact with the deployed service
- **Model Storage**: Modal Volume for persisting model weights (avoid re-downloading)
- **Cost Control**: Max 1 instance, min_containers=0, scaledown_window=60s

## Key Modal Concepts to Understand

### 1. **Apps, Functions, and Classes**
- `modal.App`: Groups Functions for deployment
- `@app.function()`: Decorator for serverless functions
- `@app.cls()`: Class-based approach for stateful containers (needed for model loading)
- `@modal.enter()`: Lifecycle hook - runs once when container starts (load model here)
- `@modal.method()`: Exposes class method as callable function

### 2. **Volumes (Critical for Model Storage)**
- Modal Volumes = distributed file system (like shared disk)
- Model weights stored here persist between container restarts
- Use `modal.Volume.from_name()` to attach to functions
- Must call `volume.commit()` to persist changes
- Up to 2.5 GB/s bandwidth, <50k files recommended per volume

### 3. **Scaling Controls (Cost Management)**
- `max_containers=1`: Hard limit to 1 instance
- `min_containers=0`: Scale to zero when idle (no charges)
- `scaledown_window=60`: Container stays warm 60s after last request
- `enable_memory_snapshot=True`: Fast cold starts by restoring memory state

### 4. **Web Endpoints**
- `@modal.fastapi_endpoint()`: Simple HTTP endpoint
- `@modal.asgi_app()`: Full FastAPI/ASGI app
- `modal serve`: Development with live reload
- `modal deploy`: Production deployment

## Step-by-Step Setup Guide

### Step 1: Install Modal CLI and Authenticate

```bash
# Install Modal Python SDK
pip install modal

# Authenticate with Modal
modal setup
# Or if that fails:
python -m modal setup

# Verify setup
modal --version
```

### Step 2: Create Project Structure

```
qwen3-tts-modal/
├── server/modal_server.py      # Modal app with TTS inference
├── server/download_model.py    # One-time script to download model to Volume
├── client/client.py            # Local client to call the service
├── test_reference.wav          # Sample reference audio for testing
├── test_reference.txt          # Transcript for reference audio
└── requirements.txt            # Local dependencies
```

### Step 3: Create Modal Volume for Model Storage

```bash
# Create a Volume to store Qwen3 TTS model (~3-4GB for 1.7B model)
modal volume create qwen3-tts-models

# Verify creation
modal volume list
```

### Step 4: Download Model to Volume (One-time Setup)

Create `server/download_model.py` to download model weights from Hugging Face to the Volume:

```python
import modal

# Create a simple image with just the download dependencies
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Fast transfers
)

app = modal.App("qwen3-tts-download", image=download_image)

# Create or get the volume
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
MODEL_DIR = "/models"

@app.function(
    volumes={MODEL_DIR: volume},
    timeout=600,  # 10 minutes for download
)
def download_qwen3_tts():
    from huggingface_hub import snapshot_download
    import os
    
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    local_path = f"{MODEL_DIR}/{model_id}"
    
    print(f"Downloading {model_id} to {local_path}...")
    
    # Download model to volume
    snapshot_download(
        repo_id=model_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
    )
    
    print(f"Download complete! Model saved to {local_path}")
    print(f"Size: {os.path.getsize(local_path) / 1024 / 1024:.2f} MB")
    
    # IMPORTANT: Commit changes to persist to volume
    volume.commit()
    print("Volume committed successfully!")

@app.local_entrypoint()
def main():
    download_qwen3_tts.remote()
```

Run the download:
```bash
modal run server/download_model.py
```

### Step 5: Create the TTS Server

Create `server/modal_server.py`:

```python
import modal
from pathlib import Path
import io
import base64

# Create image with all TTS dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")  # Audio processing deps
    .pip_install(
        "qwen-tts",           # Qwen3 TTS package
        "torch>=2.0",         # PyTorch
        "transformers>=4.40", # HuggingFace transformers
        "accelerate",         # For model loading optimization
        "soundfile",          # Audio I/O
        "numpy",
        "fastapi[standard]",  # For web endpoint
    )
    .env({
        "HF_HOME": "/models/hf_cache",  # Cache HF models in volume
        "TRANSFORMERS_CACHE": "/models/transformers_cache",
    })
)

app = modal.App("qwen3-tts-server", image=image)

# Attach the volume with pre-downloaded models
volume = modal.Volume.from_name("qwen3-tts-models")
MODEL_DIR = "/models"

# TTS Service Class with max 1 container limit
@app.cls(
    gpu="T4",  # T4 is cost-effective for 1.7B model; use A10 for faster inference
    max_containers=1,  # HARD LIMIT: Never more than 1 instance
    min_containers=0,  # Scale to zero when idle (no charges)
    scaledown_window=60,  # Keep warm for 60s after last request
    volumes={MODEL_DIR: volume},
    enable_memory_snapshot=True,  # Fast cold starts
    timeout=300,  # 5 min timeout per request
)
class Qwen3TTSService:
    """Voice cloning service using Qwen3 TTS 1.7B"""
    
    @modal.enter()
    def setup(self):
        """Load model once when container starts"""
        import torch
        from qwen_tts import Qwen3TTSModel
        
        print("Loading Qwen3 TTS 1.7B model...")
        model_path = f"{MODEL_DIR}/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        
        # Load model to CPU first (for memory snapshot compatibility)
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cpu",  # Load to CPU first
            torch_dtype=torch.float16,
        )
        
        # Move to GPU after snapshot (in separate method if using GPU snapshot)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        
        print("Model loaded successfully!")
    
    @modal.method()
    def clone_voice(
        self,
        text: str,
        reference_audio_base64: str,
        reference_text: str,
    ) -> dict:
        """
        Generate speech using voice cloning
        
        Args:
            text: Text to synthesize
            reference_audio_base64: Reference audio as base64 string
            reference_text: Transcript of reference audio
        
        Returns:
            dict with generated audio as base64 and metadata
        """
        import torch
        import soundfile as sf
        import io
        import base64
        import numpy as np
        
        # Decode reference audio
        audio_bytes = base64.b64decode(reference_audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Read audio data
        reference_audio, sample_rate = sf.read(audio_buffer)
        
        # Generate speech
        print(f"Generating speech for: {text[:50]}...")
        
        with torch.no_grad():
            result = self.model.generate(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                sample_rate=sample_rate,
            )
        
        # Convert output to base64
        output_buffer = io.BytesIO()
        sf.write(output_buffer, result.audio, result.sample_rate, format="WAV")
        output_buffer.seek(0)
        audio_base64 = base64.b64encode(output_buffer.read()).decode("utf-8")
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": result.sample_rate,
            "duration_seconds": len(result.audio) / result.sample_rate,
            "text": text,
        }
    
    @modal.fastapi_endpoint(method="POST")
    def generate_endpoint(self, request: dict) -> dict:
        """FastAPI endpoint for web access"""
        return self.clone_voice(
            text=request["text"],
            reference_audio_base64=request["reference_audio_base64"],
            reference_text=request["reference_text"],
        )

# Local entrypoint for testing
@app.local_entrypoint()
def test():
    """Test the TTS service locally"""
    import base64
    
    # Read test reference audio
    with open("test_reference.wav", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Read reference text
    with open("test_reference.txt", "r") as f:
        reference_text = f.read().strip()
    
    # Test text to synthesize
    test_text = "Hello, this is a test of the voice cloning system."
    
    # Call the service
    service = Qwen3TTSService()
    result = service.clone_voice.remote(
        text=test_text,
        reference_audio_base64=audio_base64,
        reference_text=reference_text,
    )
    
    print(f"Generated audio: {result['duration_seconds']:.2f} seconds")
    
    # Save output locally
    audio_data = base64.b64decode(result["audio_base64"])
    with open("output.wav", "wb") as f:
        f.write(audio_data)
    
    print("Output saved to output.wav")
```

### Step 6: Create the Client

Create `client/client.py`:

```python
#!/usr/bin/env python3
"""
Client for Qwen3 TTS Voice Cloning Service on Modal
"""

import base64
import requests
import argparse
from pathlib import Path

# Modal web endpoint URL (get this after deployment)
DEFAULT_ENDPOINT = "https://your-username--qwen3-tts-server-Qwen3TTSService-generate-endpoint.modal.run"


def clone_voice(
    text: str,
    reference_audio_path: str,
    reference_text: str,
    endpoint_url: str = DEFAULT_ENDPOINT,
    output_path: str = "output.wav",
) -> None:
    """
    Clone a voice and generate speech
    
    Args:
        text: Text to synthesize
        reference_audio_path: Path to reference audio file (.wav)
        reference_text: Transcript of reference audio
        endpoint_url: Modal web endpoint URL
        output_path: Where to save the output audio
    """
    # Read and encode reference audio
    audio_path = Path(reference_audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    
    print(f"Reading reference audio: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Prepare request
    payload = {
        "text": text,
        "reference_audio_base64": audio_base64,
        "reference_text": reference_text,
    }
    
    print(f"Sending request to Modal endpoint...")
    print(f"Text to synthesize: {text[:50]}...")
    
    # Call Modal endpoint
    response = requests.post(endpoint_url, json=payload, timeout=300)
    response.raise_for_status()
    
    result = response.json()
    
    # Decode and save audio
    audio_data = base64.b64decode(result["audio_base64"])
    
    output_file = Path(output_path)
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"Success! Generated {result['duration_seconds']:.2f} seconds of audio")
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Saved to: {output_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3 TTS Voice Cloning Client")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--reference-audio", "-a", required=True, help="Path to reference audio .wav file")
    parser.add_argument("--reference-text", "-t", required=True, help="Transcript of reference audio")
    parser.add_argument("--endpoint", "-e", default=DEFAULT_ENDPOINT, help="Modal endpoint URL")
    parser.add_argument("--output", "-o", default="output.wav", help="Output audio file path")
    
    args = parser.parse_args()
    
    clone_voice(
        text=args.text,
        reference_audio_path=args.reference_audio,
        reference_text=args.reference_text,
        endpoint_url=args.endpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
```

### Step 7: Development and Testing

```bash
# Test locally (spins up ephemeral container, tears down after)
modal run server/modal_server.py

# Development mode with live reload (for web endpoints)
modal serve server/modal_server.py
# Press Ctrl+C to stop
```

### Step 8: Deploy to Production

```bash
# Deploy the app (persists until you stop it)
modal deploy server/modal_server.py

# Get the endpoint URL from the output
# Or check: modal app list

# View logs
modal app logs qwen3-tts-server
```

### Step 9: Manage Deployment

```bash
# List deployed apps
modal app list

# Stop the app (stops charging)
modal app stop qwen3-tts-server

# Delete the volume (if needed)
modal volume delete qwen3-tts-models
```

## Cost Control Strategy

### Budget-Friendly Configuration

```python
@app.cls(
    gpu="T4",              # $0.000164/second (~$0.59/hour)
    # Alternative: "A10" for faster inference (~$1.10/hour)
    max_containers=1,      # NEVER more than 1 instance
    min_containers=0,      # Scale to zero (pay nothing when idle)
    scaledown_window=60,   # 60s grace period
    volumes={MODEL_DIR: volume},
    enable_memory_snapshot=True,  # Faster cold starts (sub-second)
)
```

### Cost Estimates (T4 GPU)

| Scenario | Duration | Cost |
|----------|----------|------|
| Cold start + 1 request | ~30s | ~$0.005 |
| Warm container + 1 request | ~5s | ~$0.001 |
| 100 requests (warm) | ~8 min | ~$0.08 |
| 24h always-on | 24h | ~$14 |

### Cost Saving Tips

1. **Keep min_containers=0**: Only pay when actively processing
2. **Use T4 GPU**: Cheapest option, sufficient for 1.7B model
3. **Batch requests**: Process multiple texts in one call if possible
4. **Use memory snapshots**: Reduces cold start time (and cost)
5. **Stop when not needed**: `modal app stop` when done testing

## Monitoring and Debugging

```bash
# View app status and containers
modal app list

# View real-time logs
modal app logs qwen3-tts-server --follow

# Access container shell for debugging
modal shell --app qwen3-tts-server

# Check volume contents
modal volume ls qwen3-tts-models

# Download file from volume
modal volume get qwen3-tts-models /path/on/volume local_file.txt
```

## Advanced: Using Modal Secrets (Optional)

If you need HuggingFace token for gated models:

```bash
# Create secret
modal secret create huggingface-token HF_TOKEN=your_token_here
```

```python
@app.cls(
    gpu="T4",
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-token")],  # Inject secret
    # ... other config
)
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Local Machine                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ client/client.py    │──│ Reference    │──│ Text Input   │       │
│  │ (calls API)  │  │ Audio + Text │  │ to Synthesize│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼ HTTP POST
┌─────────────────────────────────────────────────────────────┐
│                    Modal Cloud Platform                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Web Endpoint (@modal.fastapi_endpoint)                │ │
│  │  └─► @modal.cls (Qwen3TTSService)                     │ │
│  │      ├─► @modal.enter()                               │ │
│  │      │   └─► Load Qwen3 1.7B model from Volume        │ │
│  │      │                                                 │ │
│  │      └─► clone_voice()                                │ │
│  │          ├─► Decode reference audio                   │ │
│  │          ├─► Run TTS inference on GPU                 │ │
│  │          └─► Return generated audio                   │ │
│  │                                                        │ │
│  │  Configuration:                                        │ │
│  │  • max_containers=1 (hard limit)                      │ │
│  │  • min_containers=0 (scale to zero)                   │ │
│  │  • GPU: T4 (cost-effective)                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  modal.Volume("qwen3-tts-models")                      │ │
│  │  └─► /models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice      │ │
│  │      (persisted model weights, ~3-4GB)                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Download the model** to Volume (one-time ~10 min)
2. **Deploy the server** with `modal deploy`
3. **Test with client** script
4. **Monitor costs** in Modal dashboard
5. **Optimize** based on usage patterns (consider memory snapshots)

## Troubleshooting

### Model Not Found
```bash
# Check volume contents
modal volume ls qwen3-tts-models

# If empty, re-run download script
modal run server/download_model.py
```

### Out of Memory
```python
# Try different GPU with more VRAM
@app.cls(gpu="A10")  # 24GB VRAM vs T4's 16GB
```

### Slow Cold Starts
```python
# Enable memory snapshots
@app.cls(
    enable_memory_snapshot=True,
    # ... rest of config
)
```

### Container Keeps Dying
Check logs: `modal app logs qwen3-tts-server --follow`

## Resources

- [Modal Docs](https://modal.com/docs/)
- [Qwen3 TTS HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Modal Pricing](https://modal.com/pricing)
