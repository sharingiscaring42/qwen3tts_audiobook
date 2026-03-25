"""Shared utilities for TTS clients."""
from __future__ import annotations

import base64
import os
from pathlib import Path


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


def read_audio_b64(path: str | Path) -> str:
    """Read an audio file and return it base64-encoded."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: str | Path) -> str:
    """Read a text file and return its stripped contents."""
    with open(path, "r") as f:
        return f.read().strip()
