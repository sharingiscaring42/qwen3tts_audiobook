# Plan: `qwen3-full` Multi-Model TTS + ASR (Single Modal App, 2 Endpoints)

## Summary
Implement support for all requested Qwen3 TTS/ASR models with one Modal app exposing exactly two public endpoints:
1. `POST /generate` for all TTS modes/models
2. `POST /transcribe` for ASR (+ optional aligner path)

Keep current Base voice-clone behavior as default and preserve existing `.env` endpoint compatibility.

## Branch and Delivery
1. Create branch from current `main`:
   - `git checkout -b qwen3-full`
2. Implement in small commits:
   - Server architecture + APIs
   - Client updates (`client_batch_from_book.py` + new ASR script)
   - Web integration (new ASR tab + TTS model controls)
   - Docs and examples

## Server Architecture (One Modal App, Readable Split)
1. Add `server/modal_app.py`
   - Owns shared `modal.Image`, `modal.Volume`, constants, and `app = modal.App(...)`.
2. Add `server/modal_tts_server.py`
   - Defines TTS class/endpoints on shared `app`.
3. Add `server/modal_asr_server.py`
   - Defines ASR class/endpoints on shared `app`.
4. Keep `server/modal_server.py` as deploy entrypoint
   - Imports both modules so both classes register on the same app.
   - `modal deploy server/modal_server.py` yields two endpoints.
5. Keep using current volume name (`qwen3-tts-models`) for backward compatibility; store ASR/aligner under `/models/Qwen/...` too.

## Public API / Interface Changes

### 1) TTS Endpoint (existing path, expanded contract)
`POST /generate` request:
- `text: str | list[str]` (required)
- `tts_mode: "base_clone" | "custom_voice" | "voice_design"` (default `base_clone`)
- `tts_model:`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (default)
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `language: str = "Auto"`
- `max_new_tokens: int = 2048`
- `ref_audio_base64`, `ref_text` required only for `base_clone`
- `speaker`, `prompt_instruct_text` used for `custom_voice`
- `prompt` used for `voice_design`

Validation:
- strict mode-based validation, clear `400` errors for missing required fields by mode.

Response:
- keep current batch/single output structure (`audio_base64` or `audio_base64s`, durations, timing, success/error).

### 2) ASR Endpoint (new)
`POST /transcribe` request:
- `audio_base64: str` (required)
- `asr_model:`
  - `Qwen/Qwen3-ASR-1.7B` (default)
  - `Qwen/Qwen3-ASR-0.6B`
- `language: str = "auto"`
- `return_timestamps: bool = false`
- `align_ref_text: str | None` (optional; used when align requested)
- `aligner_model: "Qwen/Qwen3-ForcedAligner-0.6B"` (default)

Response:
- `text`, `language`, `model_id`, `processing_seconds`
- optional `segments` / alignment metadata when `return_timestamps=true`.

## Core Logic Changes
1. Add TTS core model manager (cache by `tts_model`) to avoid reload on every request.
2. Extend generation paths:
   - base clone path (existing)
   - custom voice path (`speaker`, `prompt_instruct_text`)
   - voice design path (`prompt`)
3. Add ASR core model manager (cache by `asr_model`) and optional aligner lazy-load.
4. Shared error mapping for predictable client/web behavior.

## Model Download Workflow
1. Extend `server/download_model.py` to support:
   - `--task tts|asr|aligner|all`
   - `--model-id` override
2. Add presets for all requested models:
   - TTS: all 5 listed by you
   - ASR: 1.7B, 0.6B
   - Aligner: 0.6B
3. Keep one script and one volume path.

## Client Changes

### `client/client_batch_from_book.py` (required)
1. Keep default behavior identical (`base_clone` + 1.7B Base).
2. Add CLI flags:
- `--tts-mode`
- `--tts-model`
- `--speaker`
- `--prompt-instruct-text`
- `--voice-design-prompt`
- optional `--endpoint-url` override
3. Payload builder includes mode-specific fields.
4. Keep current chunking/batching/retry behavior.

### New ASR script
Add `client/client_asr_from_audio.py`:
- Inputs: audio path, model, endpoint, timestamps toggle, optional align text.
- Outputs: `<name>.txt` + `<name>.json` (metadata + optional segments).
- Same `.env` loading style as existing clients.

## Web App Integration (`web/app.py`)
1. TTS controls:
- Add `tts_mode` + `tts_model` dropdowns in current workflow.
- Show/hide mode-specific fields (`speaker`, `prompt_instruct_text`, `voice_prompt`) while preserving current reference-audio UX for clone mode.
2. New ASR tab:
- Upload audio
- ASR model selector
- `return_timestamps` checkbox
- optional align reference text box
- endpoint override input
- outputs:
  - transcript textbox
  - downloadable `.txt`
  - downloadable `.json`
  - run log/status panel
3. Keep existing tabs and behavior intact for current users.

## Environment and Compatibility
1. Backward-compatible endpoint lookup:
- Existing `ENDPOINT_URL_A10/A100/H100` continues to work for TTS.
2. Add new optional keys:
- `TTS_ENDPOINT_URL` (global override)
- `ASR_ENDPOINT_URL`
- `TTS_SETTINGS_URL` / `ASR_SETTINGS_URL` (if needed)
3. If new keys are absent, fallback to existing behavior.

## Tests and Validation Scenarios

### Unit/contract tests
1. TTS request validation:
- required fields per mode
- invalid mode/model rejection
2. ASR request validation:
- missing audio
- timestamps flag with/without align text
3. Payload builder tests in `client_batch_from_book.py`.

### Integration/smoke tests
1. TTS Base clone end-to-end unchanged.
2. TTS CustomVoice and VoiceDesign produce audio with correct model routing.
3. ASR transcript-only returns text for both ASR models.
4. ASR with timestamps path triggers aligner lazy-load and returns structured metadata.
5. Web:
- TTS default path still works
- ASR tab transcribes and exports txt/json.

## Assumptions and Defaults
1. Default TTS mode/model remains `base_clone` + `Qwen3-TTS-12Hz-1.7B-Base`.
2. Default ASR model is `Qwen3-ASR-1.7B`.
3. One Modal app with two endpoints is the target deployment.
4. Forced aligner is available now through optional ASR flow (not mandatory for normal transcription).
5. Transformers backend is used for ASR in this iteration (no vLLM in scope).
