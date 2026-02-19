#!/usr/bin/env python3
"""
Batch voice cloning client with text file input.

Splits input text into ~60s chunks by character count,
then extends to the next period (.) to cut cleanly.
"""

import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ============================================
# CONFIG - EDIT THESE VALUES
# ============================================

def load_env(path: str = ".env") -> dict:
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


_env = load_env()

CARD = "A10"
USE_LOCAL = False

# Path to your reference audio file (WAV format recommended, 3-10 seconds)
# This is the voice you want to clone
NARRATOR="anne"

REFERENCE_AUDIO_PATH = f"ref/{NARRATOR}/ref_audio.wav"

# Path to a text file containing the transcript of your reference audio
# (what is being said in the reference audio file)
REFERENCE_TEXT_PATH = f"ref/{NARRATOR}/ref_text.txt"
# Input text file for generation
NAME_FILE = "texte_fr_famille.txt"
TEXT_PATH = f"input/txt/{NAME_FILE}"
# Language ("Auto" or explicit language like "English")
LANGUAGE = "English"

SETTINGS = {
    "A10": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {
                "BATCH_SIZE": 20,
            },
            "French": {
                "BATCH_SIZE": 17,
            },
        },
    },
    "A100": {
        "TARGET_SECONDS": 30,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {
                "BATCH_SIZE": 56,
            },
            "French": {
                "BATCH_SIZE": 28,
            },
        },
    },
    "H100": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {
            "English": {
                "BATCH_SIZE": 64,
            },
            "French": {
                "BATCH_SIZE": 56,
            },
        },
    },
}

TARGET_SECONDS = SETTINGS[CARD]["TARGET_SECONDS"]
CHARS_PER_SECOND = SETTINGS[CARD]["CHARS_PER_SECOND"]
MAX_CHUNK_MULTIPLIER = SETTINGS[CARD]["MAX_CHUNK_MULTIPLIER"]
BATCH_SIZE = SETTINGS[CARD]["LANG"][LANGUAGE]["BATCH_SIZE"]

# Modal endpoint URL
ENDPOINT_URL = _env.get(f"ENDPOINT_URL_{CARD}", "https://your-endpoint.modal.run")
SETTING_URL = _env.get(f"SETTING_URL_{CARD}", ENDPOINT_URL)

LOCAL_ENDPOINT_URL = _env.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
LOCAL_SETTING_URL = _env.get("LOCAL_SETTING_URL", "http://localhost:8000/settings")

if USE_LOCAL:
    ENDPOINT_URL = LOCAL_ENDPOINT_URL
    SETTING_URL = LOCAL_SETTING_URL

# Cap generation length per request (hard cap on output length)
MAX_NEW_TOKENS = 2048



# Defer long generations for retry in later batches
RETRY_ON_LONG_AUDIO = True
MAX_AUDIO_SECONDS = 90
RETRY_MAX_NEW_TOKENS = 1500
RETRY_BATCH_SIZE = 8


# Output directory and base filename
OUTPUT_DIR = "output/texte_famille"
OUTPUT_BASENAME = f"{NAME_FILE[:-4]}"

# ============================================
# END CONFIG
# ============================================


def read_audio_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


# def normalize_text(text: str) -> str:
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     text = text.replace("…", "...")
#     while "...." in text:
#         text = text.replace("....", "...")
#     while ".." in text:
#         text = text.replace("..", ".")
#     while "!!!" in text:
#         text = text.replace("!!!", "!!")
#     while "!!" in text:
#         text = text.replace("!!", "!")
#     while "???" in text:
#         text = text.replace("???", "??")
#     while "??" in text:
#         text = text.replace("??", "?")
#     text = text.replace("!.", "!").replace("?.", "?")
#     text = text.replace("!?", "!").replace("?!", "?")
#     text = text.replace("—", "-").replace("–", "-")
#     while "--" in text:
#         text = text.replace("--", "-")
#     text = text.replace(" ,", ",").replace(" .", ".")
#     text = " ".join(text.split())
#     return text.strip()


def split_text(text: str, target_seconds: int, chars_per_second: int) -> list[str]:
    max_chars = max(1, int(target_seconds * chars_per_second * MAX_CHUNK_MULTIPLIER))
    chunks = []
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
        if reverse_period == -1:
            cut_end = window_end
        else:
            cut_end = window_end - reverse_period

        chunk = text[idx:cut_end].strip()
        if chunk:
            chunks.append(chunk)
        idx = cut_end

    return chunks


def clone_voice_chunk(payload: dict) -> dict:
    request_start = time.monotonic()
    response = requests.post(
        ENDPOINT_URL,
        json=payload,
        timeout=900,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    result = response.json()
    result["client_roundtrip_seconds"] = time.monotonic() - request_start
    return result


def clone_voice_single(text: str, ref_audio_base64: str, ref_text: str, language: str, max_new_tokens: int) -> dict:
    payload = {
        "text": text,
        "ref_audio_base64": ref_audio_base64,
        "ref_text": ref_text,
        "language": language,
        "max_new_tokens": max_new_tokens,
    }
    return clone_voice_chunk(payload)


def settings_url(endpoint_url: str) -> str:
    cleaned = endpoint_url.rstrip("/")
    if "/" in cleaned and cleaned.rsplit("/", 1)[-1] in {"generate", "clone_voice"}:
        return f"{cleaned.rsplit('/', 1)[0]}/settings"
    return f"{cleaned}/settings"


def fetch_server_settings(endpoint_url: str) -> dict:
    # url = settings_url(endpoint_url)
    url = endpoint_url
    response = requests.get(url, timeout=50)
    response.raise_for_status()
    return response.json()




def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("result").mkdir(exist_ok=True)
    log_path = Path("result") / f"run_settings_{timestamp}.txt"
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0

    class Tee:
        def __init__(self, file_obj, stream):
            self.file_obj = file_obj
            self.stream = stream

        def write(self, data):
            self.file_obj.write(data)
            self.stream.write(data)

        def flush(self):
            self.file_obj.flush()
            self.stream.flush()

    with open(log_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(log_file, original_stdout)
        timing_breakdown = None
        gpu_memory = None
        try:
            print(f"Run timestamp: {timestamp}")
            print(f"Settings log: {log_path}")
            print()

            print("Server settings:")
            try:
                server_settings = fetch_server_settings(SETTING_URL)
                for key, value in server_settings.items():
                    print(f"{key}: {value}")
            except requests.RequestException as exc:
                print(f"ERROR: Failed to fetch server settings: {exc}")
            print()

            print("Client settings:")
            print(f"LANGUAGE = {LANGUAGE}")
            print(f"TARGET_SECONDS = {TARGET_SECONDS}")
            print(f"CHARS_PER_SECOND = {CHARS_PER_SECOND}")
            print(f"MAX_CHUNK_MULTIPLIER = {MAX_CHUNK_MULTIPLIER}")
            print(f"BATCH_SIZE = {BATCH_SIZE}")
            print(f"MAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
            print(f"RETRY_ON_LONG_AUDIO = {RETRY_ON_LONG_AUDIO}")
            print(f"MAX_AUDIO_SECONDS = {MAX_AUDIO_SECONDS}")
            print(f"RETRY_MAX_NEW_TOKENS = {RETRY_MAX_NEW_TOKENS}")
            print(f"RETRY_BATCH_SIZE = {RETRY_BATCH_SIZE}")
            print()

            print("=" * 60)
            print("Qwen3 TTS Batch Voice Cloning Client")
            print("=" * 60)
            print(f"Endpoint: {ENDPOINT_URL}")
            print(f"Reference audio: {REFERENCE_AUDIO_PATH}")
            print(f"Reference text: {REFERENCE_TEXT_PATH}")
            print(f"Input text: {TEXT_PATH}")
            print(f"Target seconds: {TARGET_SECONDS}")
            print(f"Chars/sec: {CHARS_PER_SECOND}")
            print(f"Output dir: {OUTPUT_DIR}")
            print()
            audio_path = Path(REFERENCE_AUDIO_PATH)
            if not audio_path.exists():
                print(f"ERROR: Reference audio not found: {audio_path}")
                return 1

            text_path = Path(REFERENCE_TEXT_PATH)
            if not text_path.exists():
                print(f"ERROR: Reference text not found: {text_path}")
                return 1

            input_text_path = Path(TEXT_PATH)
            if not input_text_path.exists():
                print(f"ERROR: Input text file not found: {input_text_path}")
                return 1

            print("Reading reference audio...")
            ref_audio_base64 = read_audio_file(str(audio_path))
            print(f"Audio encoded: {len(ref_audio_base64)} characters")

            print("Reading reference text...")
            ref_text = read_text_file(str(text_path))
            print(f"Reference text length: {len(ref_text)} characters")

            print("Reading input text...")
            full_text = read_text_file(str(input_text_path))
            print(f"Input text length: {len(full_text)} characters")

            chunks = split_text(full_text, TARGET_SECONDS, CHARS_PER_SECOND)
            print(f"Split into {len(chunks)} chunk(s)")
            print()

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            def chunk_batches(items, size):
                for i in range(0, len(items), size):
                    yield items[i:i + size]

            chunk_index = 1
            retry_queue = []

            batches = list(chunk_batches(chunks, BATCH_SIZE))
            last_batch = []
            if batches:
                last_batch = batches.pop()

            for batch in batches:
                print("-" * 60)
                total_chars = sum(len(item) for item in batch)
                max_chars = max(len(item) for item in batch)
                print(f"Batch size: {len(batch)}")
                print(f"Batch total chars: {total_chars}")
                print(f"Batch max chars: {max_chars}")

                payload = {
                    "text": batch,
                    "ref_audio_base64": ref_audio_base64,
                    "ref_text": ref_text,
                    "language": LANGUAGE,
                    "max_new_tokens": MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(payload)
                except requests.exceptions.Timeout:
                    print("ERROR: Request timed out.")
                    return 1
                except requests.exceptions.ConnectionError:
                    print(f"ERROR: Cannot connect to endpoint: {ENDPOINT_URL}")
                    return 1
                except requests.exceptions.HTTPError as e:
                    print(f"ERROR: Server error: {e.response.status_code}")
                    print(f"Response: {e.response.text}")
                    return 1

                if not result.get("success", True):
                    print(f"ERROR: Server processing failed: {result.get('error', 'Unknown error')}")
                    return 1

                audio_base64s = result.get("audio_base64s")
                durations = result.get("duration_seconds")
                processing_seconds = float(result.get("processing_seconds", 0.0))
                timing_breakdown = result.get("timing_breakdown")
                gpu_memory = result.get("gpu_memory")

                if not isinstance(audio_base64s, list) or not isinstance(durations, list):
                    print("ERROR: Expected batch response with audio_base64s and duration_seconds list")
                    return 1

                batch_audio_used = 0.0
                total_processing_seconds += processing_seconds

                for chunk_text, audio_b64, audio_seconds in zip(batch, audio_base64s, durations):
                    audio_seconds = float(audio_seconds)
                    if RETRY_ON_LONG_AUDIO and audio_seconds > MAX_AUDIO_SECONDS:
                        print(
                            f"Chunk {chunk_index}/{len(chunks)} | chars={len(chunk_text)} | "
                            f"audio={audio_seconds:.2f}s | deferred retry"
                        )
                        retry_queue.append((chunk_index, chunk_text))
                    else:
                        output_file = Path(OUTPUT_DIR) / f"{OUTPUT_BASENAME}_{chunk_index}.wav"
                        audio_data = base64.b64decode(audio_b64)
                        with open(output_file, "wb") as f:
                            f.write(audio_data)
                        print(
                            f"Chunk {chunk_index}/{len(chunks)} | chars={len(chunk_text)} | "
                            f"audio={audio_seconds:.2f}s | saved={output_file}"
                        )
                        batch_audio_used += audio_seconds
                    chunk_index += 1

                total_audio_seconds += batch_audio_used
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

            # Process last batch with deferred retries appended
            if last_batch or retry_queue:
                combined = []
                if last_batch:
                    combined.extend([(i + 1, txt) for i, txt in enumerate(last_batch, start=chunk_index - len(last_batch))])
                combined.extend(retry_queue)

                combined_texts = [item[1] for item in combined]
                combined_indices = [item[0] for item in combined]

                print("-" * 60)
                total_chars = sum(len(item[1]) for item in combined)
                max_chars = max(len(item[1]) for item in combined)
                print(f"Batch size: {len(combined)}")
                print(f"Batch total chars: {total_chars}")
                print(f"Batch max chars: {max_chars}")
                if retry_queue:
                    print("Last batch includes deferred retries")

                payload = {
                    "text": combined_texts,
                    "ref_audio_base64": ref_audio_base64,
                    "ref_text": ref_text,
                    "language": LANGUAGE,
                    "max_new_tokens": RETRY_MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(payload)
                except requests.exceptions.RequestException as e:
                    print(f"Final batch failed: {e}")
                    return 1

                if not result.get("success", True):
                    print(f"Final batch failed: {result.get('error', 'Unknown error')}")
                    return 1

                audio_base64s = result.get("audio_base64s")
                durations = result.get("duration_seconds")
                processing_seconds = float(result.get("processing_seconds", 0.0))
                timing_breakdown = result.get("timing_breakdown")
                gpu_memory = result.get("gpu_memory")

                if not isinstance(audio_base64s, list) or not isinstance(durations, list):
                    print("ERROR: Expected batch response with audio_base64s and duration_seconds list")
                    return 1

                batch_audio_used = 0.0
                total_processing_seconds += processing_seconds

                for idx, audio_b64, audio_seconds, text_item in zip(combined_indices, audio_base64s, durations, combined_texts):
                    audio_seconds = float(audio_seconds)
                    output_file = Path(OUTPUT_DIR) / f"{OUTPUT_BASENAME}_{idx}.wav"
                    audio_data = base64.b64decode(audio_b64)
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    print(
                        f"Chunk {idx}/{len(chunks)} | chars={len(text_item)} | "
                        f"audio={audio_seconds:.2f}s | saved={output_file}"
                    )
                    batch_audio_used += audio_seconds

                total_audio_seconds += batch_audio_used
                rtf = processing_seconds / batch_audio_used if batch_audio_used > 0 else 0.0
                inv_rtf = batch_audio_used / processing_seconds if processing_seconds > 0 else 0.0
                print(f"Batch processing time: {processing_seconds:.2f} seconds")
                print(f"Batch RTF (processing/audio): {rtf:.3f}")
                print(f"Batch 1/RTF (audio/processing): {inv_rtf:.3f}x")
        finally:
            print("=" * 60)
            print("TOTAL SUMMARY")
            print(f"Total audio generated: {total_audio_seconds:.2f} seconds")
            print(f"Total processing time: {total_processing_seconds:.2f} seconds")

            avg_rtf = total_processing_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0
            avg_inv_rtf = total_audio_seconds / total_processing_seconds if total_processing_seconds > 0 else 0.0
            print(f"Average RTF (processing/audio): {avg_rtf:.3f}")
            print(f"Average 1/RTF (audio/processing): {avg_inv_rtf:.3f}x")
            print("=" * 60)
            sys.stdout = original_stdout

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
