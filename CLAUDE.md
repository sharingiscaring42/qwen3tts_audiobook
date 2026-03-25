# CLAUDE.md — Qwen3 TTS Audiobook

## Project Overview

This project is a **voice cloning audiobook pipeline** using the `Qwen3-TTS-1.7B-Base` model. It supports two deployment modes:

1. **Modal (cloud)** — primary path: GPU container on Modal serverless infrastructure
2. **Local (WSL2 + pip)** — secondary path: FastAPI server running on a local GPU

The pipeline takes text or a book (EPUB/PDF) + a reference voice audio clip and generates narrated audio via batched TTS inference, then concatenates the chunks into a final audio file.

---

## Repository Structure

```
.
├── server/
│   ├── modal_server.py         # Modal app: Qwen3VoiceCloner class + endpoints
│   ├── voice_cloner_core.py    # Shared inference logic (used by both server modes)
│   └── download_model.py       # One-time Modal Volume model download script
│
├── local/
│   ├── local_server.py         # FastAPI local server (WSL2)
│   ├── download_model.py       # Download model weights to local disk
│   └── __init__.py
│
├── client/
│   ├── client.py               # Simple single-request CLI client
│   ├── client_editable.py      # Editable config-at-top client
│   ├── client_batch_from_text.py  # Batch TTS from .txt file
│   ├── client_batch_from_book.py  # Batch TTS from EPUB/PDF
│   ├── book_extract.py         # EPUB/PDF → chapter JSON/summary
│   ├── book_audio_concat.py    # ffmpeg concat + transcode WAV chunks
│   └── book_interactive.py     # Interactive book workflow helper
│
├── web/
│   └── app.py                  # Gradio web UI (wraps text + book workflows)
│
├── ref/                        # Reference voice profiles (gitignored)
│   └── <profile_name>/
│       ├── ref_audio.wav
│       └── ref_text.txt
│
├── input/                      # Text inputs (gitignored)
├── output/                     # Generated audio (gitignored)
│   ├── text/<name>/
│   │   ├── intermediary_audio/ # Per-chunk WAV files
│   │   └── final_audio/        # Concatenated output
│   └── book/<book_name>/
│       ├── intermediary_audio/
│       ├── final_audio/
│       ├── extract.json
│       └── summary.txt
├── result/                     # Run logs / settings snapshots (gitignored)
├── models/                     # Local model weights (gitignored)
│
├── requirements.txt            # Client-only deps (modal, requests, gradio, etc.)
├── requirements_local.txt      # Local server deps (torch 2.5 + cu121)
├── requirements_local_flash3.txt  # Local server deps (torch 2.8 + cu128, flash-attn3)
├── .env                        # Endpoint URLs (gitignored)
├── README.md
└── PROJECT_PLAN.md
```

---

## Key Files and Their Roles

### `server/voice_cloner_core.py`
The single source of truth for inference. Both the Modal server and the local FastAPI server import `VoiceClonerCore` from here.

- `MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"` — the model used
- `ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"` — default attention backend
- `VoiceClonerCore.load_model()` — loads model at startup; falls back gracefully if flash-attn3 is unavailable
- `VoiceClonerCore.clone_voice(text, ref_audio_base64, ref_text, language, max_new_tokens)` — handles both single string and list-of-strings (batch) input
- **Voice clone prompt caching**: SHA-256 hash of `(ref_audio_base64, ref_text)` is used as cache key so the expensive `create_voice_clone_prompt` call is skipped on repeated requests with the same voice
- Returns dict with `audio_base64` (single) or `audio_base64s` (batch), `duration_seconds`, `processing_seconds`, `success`, etc.

### `server/modal_server.py`
Modal deployment. Reads `voice_cloner_core.py` and copies it into the container image with `.add_local_file()`.

Key constants to edit when changing deployment:
- `GPU_TYPE = "A10"` — change to `"T4"`, `"A100"`, `"H100"` as needed
- `SCALEDOWN_WINDOW = 300` — seconds to keep container warm after last request
- `TIMEOUT = 1200`, `MEMORY = 16384`
- `max_containers=1` — **never change this**; cost control hard limit
- `min_containers=0` — scales to zero when idle
- `enable_memory_snapshot=True` — fast cold starts

Endpoints exposed:
- `POST /generate` — main TTS generation (calls `clone_voice.local()`)
- `GET /settings` — runtime settings snapshot (library versions, GPU, attn backend)

The `health` endpoint is commented out; use `settings` instead.

### `local/local_server.py`
FastAPI server for local GPU (WSL2). Imports `VoiceClonerCore` from `server/voice_cloner_core.py`.

- Reads `MODEL_DIR`, `MODEL_ID`, `ATTN_IMPLEMENTATION` from environment variables
- Endpoints: `POST /generate`, `GET /health`, `GET /settings`, `GET /`
- Run: `python local/local_server.py` → listens on `0.0.0.0:8000`

### `client/client_batch_from_text.py` and `client/client_batch_from_book.py`
The primary batch clients. Both share the same CONFIG block at the top:

```python
CARD = "A10"           # determines batch size and target chunk sizes
USE_LOCAL = False      # flip to True to use local server
LANGUAGE = "English"   # or "French"
```

GPU card settings (target seconds per chunk, batch size) are hardcoded in the `SETTINGS` dict within each file:
- A10: batch 20 (EN) / 17 (FR), target 60s chunks
- A100: batch 56 (EN) / 28 (FR), target 30s chunks
- H100: batch 64 (EN) / 56 (FR), target 60s chunks

**Retry logic**: Chunks producing audio longer than `MAX_AUDIO_SECONDS=90` are deferred and re-run in the final batch with `RETRY_MAX_NEW_TOKENS=1500` instead of 2048. This handles "runaway" generations.

**Stdout tee**: All output is simultaneously written to a timestamped log in `result/`.

### `client/book_extract.py`
Extracts EPUB/PDF into chapter JSON. Uses `ebooklib` for EPUB (follows TOC), `pypdf` for PDF (follows outline). All text is whitespace-normalized to a single line.

### `client/book_audio_concat.py`
Wraps `ffmpeg` to:
1. Sort WAV chunks by numeric suffix (`basename_1.wav`, `basename_2.wav`, ...)
2. Build an ffmpeg concat list, trimming the last `TRIM_TAIL_SECONDS=0.1` from each chunk to remove TTS silence artifacts
3. Transcode to OGG/Opus or M4A/AAC

### `web/app.py`
Gradio web UI at `http://localhost:7860`. Two tabs:
- **client_batch_from_text**: paste/upload text, pick voice profile, generate
- **client_batch_from_book**: upload EPUB/PDF, extract chapters, select range, generate

Card defaults auto-populate when GPU card or language changes. Reference profiles are auto-discovered from `ref/<name>/ref_audio.wav` + `ref_text.txt`.

Endpoint resolution order (highest priority first):
1. Manual override in the UI
2. `LOCAL_ENDPOINT_URL` from `.env` (if "Use local" checked)
3. `ENDPOINT_URL_<CARD>` from `.env` (e.g. `ENDPOINT_URL_A10`)
4. `ENDPOINT_URL` from `.env`

---

## Environment Configuration

Create `.env` in the repo root:

```bash
# Modal endpoints (one per GPU card you deploy)
ENDPOINT_URL_A10=https://your-username--qwen3-tts-voice-cloner-a10-...generate.modal.run
SETTING_URL_A10=https://your-username--qwen3-tts-voice-cloner-a10-...settings.modal.run

ENDPOINT_URL_A100=https://...
SETTING_URL_A100=https://...

# Fallback
ENDPOINT_URL=https://your-default-endpoint.modal.run

# Local server
LOCAL_ENDPOINT_URL=http://localhost:8000/generate
LOCAL_SETTING_URL=http://localhost:8000/settings
```

`.env` is gitignored. Never commit it.

---

## Reference Voice Profiles

Store reference voices in `ref/<profile_name>/`:
```
ref/
└── my_voice/
    ├── ref_audio.wav   # 3–10 seconds of clean speech, WAV format
    └── ref_text.txt    # exact transcript of ref_audio.wav
```

`ref/` is gitignored. The web UI and batch clients auto-discover profiles from this directory.

---

## Development Workflows

### Modal Deployment

```bash
# Install local deps
pip install -r requirements.txt

# Authenticate
modal setup

# Download model to Modal Volume (one-time)
modal run server/download_model.py

# Deploy
modal deploy server/modal_server.py

# Dev mode (live reload)
modal serve server/modal_server.py

# View logs
modal app logs qwen3-tts-voice-cloner-A10 --follow

# Stop (saves money)
modal app stop qwen3-tts-voice-cloner-A10
```

### Local Server (WSL2)

```bash
# Standard stack (torch 2.5 + cuda 12.1)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements_local.txt
python local/download_model.py --model-dir models  # optional
python local/local_server.py

# Flash-attn3 stack (torch 2.8 + cuda 12.8)
python3.11 -m venv .venv_flash3 && source .venv_flash3/bin/activate
pip install -r requirements_local_flash3.txt
python local/local_server.py
```

Attention backend can be overridden via environment variable:
```bash
export ATTN_IMPLEMENTATION=flash_attention_2   # SDPA flash
export ATTN_IMPLEMENTATION=default             # PyTorch auto
export ATTN_IMPLEMENTATION=kernels-community/flash-attn3
```

### Web UI

```bash
python web/app.py
# Open http://localhost:7860
```

### Batch Generation (text file)

Edit the CONFIG block at the top of `client/client_batch_from_text.py`, then:
```bash
python client/client_batch_from_text.py
```

### Batch Generation (book)

```bash
python client/client_batch_from_book.py --input path/to/book.epub --start-chapter 1 --end-chapter 5
```

### Audio Concatenation (standalone)

```bash
python client/book_audio_concat.py \
  --input-dir output/book/<name>/intermediary_audio \
  --format ogg --bitrate 48k
```

---

## API Contract

### Request (POST `/generate`)

```json
{
  "text": "Hello world" | ["chunk1", "chunk2"],
  "ref_audio_base64": "<base64-encoded WAV>",
  "ref_text": "Transcript of reference audio",
  "language": "Auto" | "English" | "French",
  "max_new_tokens": 2048
}
```

### Response (single string input)

```json
{
  "audio_base64": "<base64-encoded WAV>",
  "sample_rate": 24000,
  "duration_seconds": 5.4,
  "text": "...",
  "language": "English",
  "success": true,
  "processing_seconds": 1.2,
  "model_load_seconds": 18.3,
  "model_loaded_at": 1712345678.0
}
```

### Response (list input / batch)

```json
{
  "audio_base64s": ["<b64>", "<b64>", ...],
  "sample_rate": 24000,
  "duration_seconds": [4.1, 5.2, ...],
  "text": ["chunk1", "chunk2"],
  "language": "English",
  "success": true,
  "processing_seconds": 8.7,
  "timing_breakdown": {...},  // only when ENABLE_DEBUG_TIMINGS=1
  "gpu_memory": {...}         // only when ENABLE_DEBUG_TIMINGS=1
}
```

---

## Conventions and Important Notes

### Cost Control (Modal)
- `max_containers=1` is a hard limit — **never increase this** without understanding billing implications
- `min_containers=0` ensures the container scales to zero when idle
- Always run `modal app stop` when done for the day

### Chunk Sizing
- Text is split at sentence boundaries (last `.` within the window), targeting `TARGET_SECONDS * CHARS_PER_SECOND` characters
- `MAX_CHUNK_MULTIPLIER=1.05` adds a 5% buffer
- Total batch text over ~19k–20k characters can trigger OOM on A10

### Audio Artifacts
- Each WAV chunk has a 0.1s tail trim during concatenation to remove trailing silence/artifacts
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set in the Modal image to reduce fragmentation

### Shared Core
- `server/voice_cloner_core.py` is the single source of truth — both Modal and local servers import from it
- When modifying inference logic, change `voice_cloner_core.py` only; Modal picks it up via `.add_local_file()`

### Gitignored Directories
The following are **never committed**: `.env`, `ref/`, `output/`, `result/`, `input/`, `models/`, `*.wav`, `*.ogg`, `*.mp3`

### Python Version
The project targets **Python 3.11**. The Modal image uses `add_python="3.11"` and WSL2 setup uses `python3.11`.

### No Tests
There is no automated test suite. Manual testing is done via `modal run server/modal_server.py` (Modal) or by running `python local/local_server.py` and hitting the endpoints directly.

### Branch for AI Development
Work on feature branch `claude/add-claude-documentation-M9jl6`. Never push directly to `main`.
