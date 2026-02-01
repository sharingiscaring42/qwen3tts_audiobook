# Qwen3 TTS 1.7B Voice Cloning on Modal

A production-ready voice cloning service using Qwen3-TTS 1.7B deployed on Modal's serverless cloud infrastructure.

## Install

```bash
pip install -r requirements.txt
```

Notes:
- The GPU/server dependencies are installed inside the Modal container on deploy.
- `ffmpeg` is required locally to run audio concatenation/transcode.

## Features

- **Voice Cloning**: Clone any voice with 3-10 seconds of reference audio
- **Cost-Effective**: Max 1 instance, scales to zero when idle
- **Fast Cold Starts**: Uses Modal Memory Snapshots for sub-second startup
- **Persistent Storage**: Model weights stored in Modal Volume (no re-downloads)
- **Simple API**: RESTful HTTP endpoint for easy integration

## Architecture

```
Local Client → HTTP POST → Modal Web Endpoint → GPU Container → Voice Clone
                                    ↓
                              Modal Volume (Model Storage)
```

## Quick Start

### 1. Authenticate with Modal

```bash
modal setup
# Or: python -m modal setup
```

### 2. Download Model (One-time)

```bash
# Download Qwen3 TTS 1.7B model (~3-4GB) to Modal Volume
modal run server/download_model.py
```

### 3. Deploy Server

```bash
# Deploy persistent app
modal deploy server/modal_server.py

# Note the endpoint URL from the output
```

### 3.5 Configure .env (required for clients)

After deploy, Modal prints two URLs:
- `Qwen3VoiceCloner.generate` → use this for `ENDPOINT_URL`
- `Qwen3VoiceCloner.settings` → use this for `SETTING_URL`

Create `.env` in repo root:
```bash
ENDPOINT_URL=https://your-endpoint.modal.run
SETTING_URL=https://your-endpoint.modal.run
```


### 4. Clone a Voice

**Option A: Command-line client (single request)**
```bash
python client/client.py "Hello, this is my cloned voice!" \
  -a reference_audio.wav \
  -t "This is the transcript of my reference audio"
```

**Option B: Editable client (edit in code)**
```bash
# Edit the CONFIG section in client/client_editable.py, then just run:
python client/client_editable.py
```

## Project Structure

```
.
├── server/modal_server.py          # Main Modal app with TTS service
├── server/download_model.py        # One-time model download script
├── client/client.py                # CLI client (command-line arguments)
├── client/client_editable.py       # Editable client (edit in code)
├── client/client_batch_from_text.py# Batch client from text file
├── client/client_batch_from_book.py# Batch client from EPUB/PDF
├── client/book_extract.py          # EPUB/PDF extractor + summary
├── client/book_audio_concat.py     # Concatenate/transcode audio chunks
├── input/                          # Text inputs
├── ref/                            # Reference audio + transcript
├── output/                         # Generated audio outputs
├── result/                         # Run logs/settings snapshots
├── book_extracts/                  # Optional extract output folder
└── PROJECT_PLAN.md                 # Detailed architecture and planning
```

## Configuration

### Cost Control (Max 1 Instance)

The server is configured with strict limits to control costs (current defaults in `server/modal_server.py`):

```python
@app.cls(
    gpu="A10",             # 24GB VRAM
    max_containers=1,      # NEVER more than 1 instance
    min_containers=0,      # Scale to zero when idle
    scaledown_window=300,  # 5m grace period
    enable_memory_snapshot=True,  # Fast cold starts
)
```

### GPU Options

- **A10**: 24GB VRAM, ~$1.10/hour - current config
- **T4**: 16GB VRAM, ~$0.59/hour - lower cost
- **A100-40GB**: 40GB VRAM, ~$2.20/hour - more headroom

Edit `server/modal_server.py` to change GPU type.

## Usage Examples

### Python Client

```python
from client import clone_voice

result = clone_voice(
    text="Hello world!",
    reference_audio_path="my_voice.wav",
    reference_text="This is my voice speaking.",
    endpoint_url="https://your-endpoint.modal.run",
    output_path="output.wav"
)
```

### cURL / HTTP

```bash
curl -X POST https://your-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "reference_audio_base64": "<base64-encoded-audio>",
    "reference_text": "Reference audio transcript."
  }'
```

### Health Check

```bash
python client/client.py --health -e https://your-endpoint.modal.run

# Or with curl:
curl https://your-endpoint.modal.run/health
```

### Book Extraction + Summary (EPUB/PDF)

```bash
# Summary only (quick overview)
python client/book_extract.py --input "path/to/book.epub" --summary-only

# Full extract + summary
python client/book_extract.py --input "path/to/book.pdf"

# Custom output directory
python client/book_extract.py --input "book.epub" --output-dir "output/book"
```

### Batch Voice Clone from EPUB/PDF

```bash
python client/client_batch_from_book.py --input "path/to/book.epub" --start-chapter 1 --end-chapter 5
```

### Concatenate + Compress Audio Output

```bash
# Opus (recommended for size/quality)
python client/book_audio_concat.py --input-dir output/book/<book_name>/intermediary_audio --format ogg --bitrate 48k

# AAC (good for native players)
python client/book_audio_concat.py --input-dir output/book/<book_name>/intermediary_audio --format m4a --bitrate 64k
```

## Development

### Local Testing

```bash
# Run locally (creates ephemeral container)
modal run server/modal_server.py

# Development mode with live reload
modal serve server/modal_server.py
```

### Monitoring

```bash
# View logs
modal app logs qwen3-tts-voice-cloner --follow

# View app status
modal app list

# Access container shell
modal shell --app qwen3-tts-voice-cloner
```

## Cost Estimates (A10)

Assumptions from current runs:
- A10 GPU: ~$1.10/hour
- CPU + memory overhead: ~ $0.15/hour
- Total: ~$1.25/hour
- Observed 1/RTF: ~8x to ~11x (varies by settings and batch stability)

Estimated cost per 1 hour of generated audio:
- 1/RTF 8x: 1/8 hour compute -> ~$0.16
- 1/RTF 11x: 1/11 hour compute -> ~$0.11

**Tips to save money:**
- Keep `min_containers=0` (scales to zero)
- Stop app when not needed: `modal app stop qwen3-tts-voice-cloner`

## Modal Concepts Used

1. **Apps**: `modal.App` groups Functions for deployment
2. **Classes**: `@app.cls()` for stateful containers with model loading
3. **Lifecycle Hooks**: `@modal.enter()` runs once on container startup
4. **Volumes**: Persistent storage for model weights
5. **Web Endpoints**: HTTP API via `@modal.fastapi_endpoint()`
6. **Memory Snapshots**: Fast cold starts by restoring container state
7. **Scaling Controls**: `max_containers`, `min_containers` for cost control

## Troubleshooting

### Model Not Found

```bash
# Re-download model to volume
modal run server/download_model.py
```

### Cold Start Taking Too Long

First request may take 30-60s. Subsequent requests are faster. Enable memory snapshots for sub-second cold starts.

### Out of Memory

```python
# In server/modal_server.py, upgrade GPU:
@app.cls(gpu="A10")  # 24GB VRAM
```

### Connection Issues

```bash
# Check app is deployed
modal app list

# Verify endpoint URL
curl https://your-endpoint.modal.run/health
```

## Current A10 Setup Notes

- Stable batch setting: `BATCH_SIZE=20` with chunks just under 60s (about 900-1000 chars).
- Total batch text size over ~19k-20k chars can trigger OOM.
- Long or "runaway" audio can slow a batch; the retry logic handles these chunks but often halves 1/RTF for that batch.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` helps memory fragmentation; cleanup block is currently disabled and needs testing for perf impact.

## Resources

- [Modal Documentation](https://modal.com/docs/)
- [Qwen3 TTS HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Modal Pricing](https://modal.com/pricing)

## License

MIT - See LICENSE file for details.
