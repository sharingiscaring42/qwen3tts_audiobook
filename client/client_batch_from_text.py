#!/usr/bin/env python3
"""
Batch voice cloning client with text file input.

Splits input text into ~60s chunks by character count,
then extends to the next period (.) to cut cleanly.
Supports resume: chunks whose output files already exist are skipped.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import requests

from utils import (
    CARD_SETTINGS,
    Tee,
    card_defaults,
    fetch_server_settings,
    load_env,
    read_audio_b64,
    read_text_file,
    run_generation,
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

            total_audio_seconds, total_processing_seconds = run_generation(
                endpoint_url=endpoint_url,
                pending=pending,
                total_chunks=len(chunks),
                ref_audio_base64=ref_audio_base64,
                ref_text=ref_text,
                language=language,
                batch_size=batch_size,
                output_dir=output_dir,
                output_basename=output_basename,
                max_new_tokens=MAX_NEW_TOKENS,
                retry_on_long_audio=RETRY_ON_LONG_AUDIO,
                max_audio_seconds=MAX_AUDIO_SECONDS,
                retry_max_new_tokens=RETRY_MAX_NEW_TOKENS,
            )
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
