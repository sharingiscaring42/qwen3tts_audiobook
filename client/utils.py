"""Shared utilities for TTS clients."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# GPU card settings
# ---------------------------------------------------------------------------

CARD_SETTINGS: dict = {
    "A10": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {"BATCH_SIZE": 20},
            "French":  {"BATCH_SIZE": 17},
        },
    },
    "A100": {
        "TARGET_SECONDS": 30,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {"BATCH_SIZE": 56},
            "French":  {"BATCH_SIZE": 28},
        },
    },
    "H100": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {"BATCH_SIZE": 64},
            "French":  {"BATCH_SIZE": 56},
        },
    },
}


def card_defaults(card: str, language: str) -> tuple[int, int, float, int]:
    """Return (target_seconds, chars_per_second, max_chunk_multiplier, batch_size)."""
    cfg = CARD_SETTINGS[card]
    batch_size = cfg["LANG"].get(language, cfg["LANG"]["English"])["BATCH_SIZE"]
    return cfg["TARGET_SECONDS"], cfg["CHARS_PER_SECOND"], cfg["MAX_CHUNK_MULTIPLIER"], batch_size


# ---------------------------------------------------------------------------
# Environment / config helpers
# ---------------------------------------------------------------------------

def load_env(path: str = ".env") -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Ignores blank lines and # comments."""
    if not os.path.exists(path):
        return {}
    data: dict[str, str] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    target_seconds: int,
    chars_per_second: int,
    max_chunk_multiplier: float = 1.05,
) -> list[str]:
    """Split text into chunks targeting ~target_seconds of audio each.

    Cuts at the last period within the character window so chunks end
    at sentence boundaries rather than mid-sentence.
    """
    max_chars = max(1, int(target_seconds * chars_per_second * max_chunk_multiplier))
    chunks: list[str] = []
    idx = 0
    length = len(text)

    while idx < length:
        window_end = min(idx + max_chars, length)
        if window_end >= length:
            chunk = text[idx:].strip()
            if chunk:
                chunks.append(chunk)
            break

        window = text[idx:window_end]
        reverse_period = window[::-1].find(".")
        cut_end = window_end if reverse_period == -1 else window_end - reverse_period
        chunk = text[idx:cut_end].strip()
        if chunk:
            chunks.append(chunk)
        idx = cut_end

    return chunks


# ---------------------------------------------------------------------------
# Audio / text I/O
# ---------------------------------------------------------------------------

def read_audio_b64(path: str | Path) -> str:
    """Read an audio file and return it base64-encoded."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: str | Path) -> str:
    """Read a text file and return its stripped contents."""
    with open(path, "r") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------

def fetch_server_settings(url: str) -> dict:
    """GET the server settings endpoint and return parsed JSON."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def clone_voice_chunk(endpoint_url: str, payload: dict) -> dict:
    """POST a generation payload and return the result dict with roundtrip time."""
    request_start = time.monotonic()
    response = requests.post(
        endpoint_url,
        json=payload,
        timeout=900,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    result = response.json()
    result["client_roundtrip_seconds"] = time.monotonic() - request_start
    return result


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class Tee:
    """Write to both a file and another stream simultaneously."""

    def __init__(self, file_obj, stream):
        self.file_obj = file_obj
        self.stream = stream

    def write(self, data):
        self.file_obj.write(data)
        self.stream.write(data)

    def flush(self):
        self.file_obj.flush()
        self.stream.flush()


# ---------------------------------------------------------------------------
# Batch generation loop
# ---------------------------------------------------------------------------

def run_generation(
    *,
    endpoint_url: str,
    pending: list[tuple[int, str]],
    total_chunks: int,
    ref_audio_base64: str,
    ref_text: str,
    language: str,
    batch_size: int,
    output_dir: Path,
    output_basename: str,
    max_new_tokens: int = 2048,
    retry_on_long_audio: bool = True,
    max_audio_seconds: float = 90.0,
    retry_max_new_tokens: int = 1500,
) -> tuple[float, float]:
    """Run batched TTS generation for pending (index, text) pairs.

    Chunks whose audio exceeds *max_audio_seconds* are deferred to the final
    batch and re-run with *retry_max_new_tokens*.  Output WAV files are saved
    as ``<output_dir>/<output_basename>_<index>.wav``.

    Returns:
        (total_audio_seconds, total_processing_seconds)

    Raises:
        SystemExit(1) on unrecoverable request or server errors.
    """
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0
    retry_queue: list[tuple[int, str]] = []

    def _batches(items: list, size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _process_batch(batch: list[tuple[int, str]], is_final: bool) -> float:
        nonlocal total_processing_seconds

        total_chars = sum(len(t) for _, t in batch)
        max_chars = max((len(t) for _, t in batch), default=0)
        print(f"Batch size: {len(batch)}")
        print(f"Batch total chars: {total_chars}")
        print(f"Batch max chars: {max_chars}")

        payload = {
            "text": [t for _, t in batch],
            "ref_audio_base64": ref_audio_base64,
            "ref_text": ref_text,
            "language": language,
            "max_new_tokens": retry_max_new_tokens if is_final else max_new_tokens,
        }

        try:
            result = clone_voice_chunk(endpoint_url, payload)
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out.")
            raise SystemExit(1)
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Cannot connect to endpoint: {endpoint_url}")
            raise SystemExit(1)
        except requests.exceptions.HTTPError as e:
            print(f"ERROR: Server error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise SystemExit(1)
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed: {e}")
            raise SystemExit(1)

        if not result.get("success", True):
            print(f"ERROR: Server processing failed: {result.get('error', 'Unknown error')}")
            raise SystemExit(1)

        audio_base64s = result.get("audio_base64s")
        durations = result.get("duration_seconds")
        processing_seconds = float(result.get("processing_seconds", 0.0))
        timing_breakdown = result.get("timing_breakdown")
        gpu_memory = result.get("gpu_memory")

        if not isinstance(audio_base64s, list) or not isinstance(durations, list):
            print("ERROR: Expected batch response with audio_base64s and duration_seconds list")
            raise SystemExit(1)

        batch_audio_used = 0.0
        total_processing_seconds += processing_seconds

        for (chunk_idx, chunk_text), audio_b64, audio_seconds in zip(batch, audio_base64s, durations):
            audio_seconds = float(audio_seconds)
            if not is_final and retry_on_long_audio and audio_seconds > max_audio_seconds:
                print(
                    f"Chunk {chunk_idx}/{total_chunks} | chars={len(chunk_text)} | "
                    f"audio={audio_seconds:.2f}s | deferred retry"
                )
                retry_queue.append((chunk_idx, chunk_text))
            else:
                output_file = output_dir / f"{output_basename}_{chunk_idx}.wav"
                audio_data = base64.b64decode(audio_b64)
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                print(
                    f"Chunk {chunk_idx}/{total_chunks} | chars={len(chunk_text)} | "
                    f"audio={audio_seconds:.2f}s | saved={output_file}"
                )
                batch_audio_used += audio_seconds

        rtf = processing_seconds / batch_audio_used if batch_audio_used > 0 else 0.0
        inv_rtf = batch_audio_used / processing_seconds if processing_seconds > 0 else 0.0
        print(f"Batch processing time: {processing_seconds:.2f} seconds")
        print(f"Batch RTF (processing/audio): {rtf:.3f}")
        print(f"Batch 1/RTF (audio/processing): {inv_rtf:.3f}x")
        if timing_breakdown:
            print(
                "Timing breakdown: "
                f"prompt={timing_breakdown.get('prompt_seconds', 0.0):.2f}s, "
                f"generate={timing_breakdown.get('generate_seconds', 0.0):.2f}s, "
                f"encode={timing_breakdown.get('encode_seconds', 0.0):.2f}s"
            )
        if gpu_memory:
            print(
                "GPU memory: "
                f"allocated={gpu_memory.get('allocated_gb', 0.0):.2f} GB, "
                f"reserved={gpu_memory.get('reserved_gb', 0.0):.2f} GB"
            )
        return batch_audio_used

    batches = list(_batches(pending, batch_size))
    last_batch: list[tuple[int, str]] = []
    if batches:
        last_batch = batches.pop()

    for batch in batches:
        print("-" * 60)
        total_audio_seconds += _process_batch(batch, is_final=False)

    if last_batch or retry_queue:
        combined = last_batch + retry_queue
        print("-" * 60)
        if retry_queue:
            print("Last batch includes deferred retries")
        total_audio_seconds += _process_batch(combined, is_final=True)

    return total_audio_seconds, total_processing_seconds
