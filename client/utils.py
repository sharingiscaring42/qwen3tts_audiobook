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
