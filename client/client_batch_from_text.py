#!/usr/bin/env python3
"""
Batch voice cloning client with text file input.

Splits input text into ~60s chunks by character count,
then extends to the next period (.) to cut cleanly.
Supports resume: chunks whose output files already exist are skipped.
"""

import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path

import requests

from utils import (
    CARD_SETTINGS,
    Tee,
    card_defaults,
    clone_voice_chunk,
    fetch_server_settings,
    load_env,
    read_audio_b64,
    read_text_file,
    split_text,
)

# ============================================
# CONFIG - EDIT THESE VALUES
# (all can be overridden via CLI flags)
# ============================================

_env = load_env()

CARD = "A10"
USE_LOCAL = False

NARRATOR = "anne"
REFERENCE_AUDIO_PATH = f"ref/{NARRATOR}/ref_audio.wav"
REFERENCE_TEXT_PATH = f"ref/{NARRATOR}/ref_text.txt"
NAME_FILE = "texte_fr_famille.txt"
TEXT_PATH = f"input/txt/{NAME_FILE}"
LANGUAGE = "English"

TARGET_SECONDS, CHARS_PER_SECOND, MAX_CHUNK_MULTIPLIER, BATCH_SIZE = card_defaults(CARD, LANGUAGE)

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


def clone_voice_single(text: str, ref_audio_base64: str, ref_text: str, language: str, max_new_tokens: int) -> dict:
    payload = {
        "text": text,
        "ref_audio_base64": ref_audio_base64,
        "ref_text": ref_text,
        "language": language,
        "max_new_tokens": max_new_tokens,
    }
    return clone_voice_chunk(ENDPOINT_URL, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch voice cloning from a text file")
    parser.add_argument("--input", dest="text_path", help="Path to input .txt file (overrides TEXT_PATH)")
    parser.add_argument("--narrator", help="Narrator name under ref/ (overrides NARRATOR)")
    parser.add_argument("--language", choices=list(CARD_SETTINGS["A10"]["LANG"].keys()) + ["Auto"], help="Language (overrides LANGUAGE)")
    parser.add_argument("--card", choices=list(CARD_SETTINGS.keys()), help="GPU card key (overrides CARD)")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory (overrides OUTPUT_DIR)")
    args = parser.parse_args()

    # Apply CLI overrides
    card = args.card or CARD
    language = args.language or LANGUAGE
    narrator = args.narrator or NARRATOR
    text_path_str = args.text_path or TEXT_PATH
    output_dir_str = args.output_dir or OUTPUT_DIR

    target_seconds, chars_per_second, max_chunk_multiplier, batch_size = card_defaults(card, language)

    ref_audio_path = Path(f"ref/{narrator}/ref_audio.wav") if args.narrator else Path(REFERENCE_AUDIO_PATH)
    ref_text_path = Path(f"ref/{narrator}/ref_text.txt") if args.narrator else Path(REFERENCE_TEXT_PATH)
    text_path = Path(text_path_str)
    output_dir = Path(output_dir_str)
    output_basename = text_path.stem

    endpoint_url = ENDPOINT_URL
    setting_url = SETTING_URL

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("result").mkdir(exist_ok=True)
    log_path = Path("result") / f"run_settings_{timestamp}.txt"
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0

    with open(log_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(log_file, original_stdout)
        try:
            print(f"Run timestamp: {timestamp}")
            print(f"Settings log: {log_path}")
            print()

            print("Server settings:")
            try:
                server_settings = fetch_server_settings(setting_url)
                for key, value in server_settings.items():
                    print(f"{key}: {value}")
            except requests.RequestException as exc:
                print(f"ERROR: Failed to fetch server settings: {exc}")
            print()

            print("Client settings:")
            print(f"CARD = {card}")
            print(f"LANGUAGE = {language}")
            print(f"TARGET_SECONDS = {target_seconds}")
            print(f"CHARS_PER_SECOND = {chars_per_second}")
            print(f"MAX_CHUNK_MULTIPLIER = {max_chunk_multiplier}")
            print(f"BATCH_SIZE = {batch_size}")
            print(f"MAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
            print(f"RETRY_ON_LONG_AUDIO = {RETRY_ON_LONG_AUDIO}")
            print(f"MAX_AUDIO_SECONDS = {MAX_AUDIO_SECONDS}")
            print(f"RETRY_MAX_NEW_TOKENS = {RETRY_MAX_NEW_TOKENS}")
            print()

            print("=" * 60)
            print("Qwen3 TTS Batch Voice Cloning Client")
            print("=" * 60)
            print(f"Endpoint: {endpoint_url}")
            print(f"Reference audio: {ref_audio_path}")
            print(f"Reference text: {ref_text_path}")
            print(f"Input text: {text_path}")
            print(f"Output dir: {output_dir}")
            print()

            if not ref_audio_path.exists():
                print(f"ERROR: Reference audio not found: {ref_audio_path}")
                return 1
            if not ref_text_path.exists():
                print(f"ERROR: Reference text not found: {ref_text_path}")
                return 1
            if not text_path.exists():
                print(f"ERROR: Input text file not found: {text_path}")
                return 1

            print("Reading reference audio...")
            ref_audio_base64 = read_audio_b64(ref_audio_path)
            print(f"Audio encoded: {len(ref_audio_base64)} characters")

            print("Reading reference text...")
            ref_text = read_text_file(ref_text_path)
            print(f"Reference text length: {len(ref_text)} characters")

            print("Reading input text...")
            full_text = read_text_file(text_path)
            print(f"Input text length: {len(full_text)} characters")

            chunks = split_text(full_text, target_seconds, chars_per_second, max_chunk_multiplier)
            print(f"Split into {len(chunks)} chunk(s)")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Checkpoint: find already-completed chunks
            done_set: set[int] = set()
            for p in output_dir.glob(f"{output_basename}_*.wav"):
                suffix = p.stem.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    done_set.add(int(suffix))
            if done_set:
                print(f"Checkpoint: {len(done_set)} chunk(s) already done, skipping.")

            # Build (1-based index, text) for pending chunks only
            pending = [(i + 1, t) for i, t in enumerate(chunks) if (i + 1) not in done_set]
            if not pending:
                print("All chunks already complete. Nothing to do.")
                return 0
            print(f"Generating {len(pending)} of {len(chunks)} chunk(s).")
            print()

            def chunk_batches(items, size):
                for i in range(0, len(items), size):
                    yield items[i:i + size]

            retry_queue: list[tuple[int, str]] = []

            batches = list(chunk_batches(pending, batch_size))
            last_batch: list[tuple[int, str]] = []
            if batches:
                last_batch = batches.pop()

            for batch in batches:
                print("-" * 60)
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
                    "max_new_tokens": MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(endpoint_url, payload)
                except requests.exceptions.Timeout:
                    print("ERROR: Request timed out.")
                    return 1
                except requests.exceptions.ConnectionError:
                    print(f"ERROR: Cannot connect to endpoint: {endpoint_url}")
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

                for (chunk_idx, chunk_text), audio_b64, audio_seconds in zip(batch, audio_base64s, durations):
                    audio_seconds = float(audio_seconds)
                    if RETRY_ON_LONG_AUDIO and audio_seconds > MAX_AUDIO_SECONDS:
                        print(
                            f"Chunk {chunk_idx}/{len(chunks)} | chars={len(chunk_text)} | "
                            f"audio={audio_seconds:.2f}s | deferred retry"
                        )
                        retry_queue.append((chunk_idx, chunk_text))
                    else:
                        output_file = output_dir / f"{output_basename}_{chunk_idx}.wav"
                        audio_data = base64.b64decode(audio_b64)
                        with open(output_file, "wb") as f:
                            f.write(audio_data)
                        print(
                            f"Chunk {chunk_idx}/{len(chunks)} | chars={len(chunk_text)} | "
                            f"audio={audio_seconds:.2f}s | saved={output_file}"
                        )
                        batch_audio_used += audio_seconds

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
                combined = last_batch + retry_queue
                combined_texts = [t for _, t in combined]
                combined_indices = [i for i, _ in combined]

                print("-" * 60)
                total_chars = sum(len(t) for _, t in combined)
                max_chars = max((len(t) for _, t in combined), default=0)
                print(f"Batch size: {len(combined)}")
                print(f"Batch total chars: {total_chars}")
                print(f"Batch max chars: {max_chars}")
                if retry_queue:
                    print("Last batch includes deferred retries")

                payload = {
                    "text": combined_texts,
                    "ref_audio_base64": ref_audio_base64,
                    "ref_text": ref_text,
                    "language": language,
                    "max_new_tokens": RETRY_MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(endpoint_url, payload)
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
                    output_file = output_dir / f"{output_basename}_{idx}.wav"
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
