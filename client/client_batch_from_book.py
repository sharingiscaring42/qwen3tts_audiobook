import argparse
import sys
from datetime import datetime
from pathlib import Path

import requests

from book_extract import extract_book, write_extract_json, write_summary_txt
from utils import (
    Tee,
    card_defaults,
    fetch_server_settings,
    load_env,
    read_audio_b64,
    read_text_file,
    run_generation,
    split_text,
)

_env = load_env()

# A10 24GB   1/RTF 9x       Modal: 1.10$ -> 0.13 $/h        fal.ai:  N/A
# A100 40GB  1/RTF 18x      Modal: 2.10$ -> 0.12 $/h        fal.ai:  0.99$ -> 0.055 $/h
# H100 80GB  1/RTF 36x      Modal: 3.95$ -> 0.11 $/h        fal.ai:  1.89$ -> 0.0525 $/h
# ============================================
# CONFIG - EDIT THESE VALUES
# ============================================

CARD = "A10"
USE_LOCAL = False

REFERENCE_AUDIO_PATH = "ref/jeff_hays_0/ref_audio.wav"
REFERENCE_TEXT_PATH = "ref/jeff_hays_0/ref_text.txt"
LANGUAGE = "English"

# REFERENCE_AUDIO_PATH = "ref/herve_lacroix/ref_audio.wav"
# REFERENCE_TEXT_PATH = "ref/herve_lacroix/ref_text.txt"
# LANGUAGE = "French"

TARGET_SECONDS, CHARS_PER_SECOND, MAX_CHUNK_MULTIPLIER, BATCH_SIZE = card_defaults(CARD, LANGUAGE)

# Cap generation length per request (hard cap on output length)
MAX_NEW_TOKENS = 2048
# Defer long generations for retry in later batches
RETRY_ON_LONG_AUDIO = True
MAX_AUDIO_SECONDS = 90
RETRY_MAX_NEW_TOKENS = 1500
RETRY_BATCH_SIZE = 8

# Output directory and base filename
OUTPUT_DIR = "output/book"
OUTPUT_BASENAME = "book"

# Modal endpoint URL
ENDPOINT_URL = _env.get(f"ENDPOINT_URL_{CARD}", "https://your-endpoint.modal.run")
SETTING_URL = _env.get(f"SETTING_URL_{CARD}", ENDPOINT_URL)

LOCAL_ENDPOINT_URL = _env.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
LOCAL_SETTING_URL = _env.get("LOCAL_SETTING_URL", "http://localhost:8000/settings")

if USE_LOCAL:
    ENDPOINT_URL = LOCAL_ENDPOINT_URL
    SETTING_URL = LOCAL_SETTING_URL

# ============================================
# END CONFIG
# ============================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch voice cloning from EPUB/PDF")
    parser.add_argument("--input", required=True)
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--end-chapter", type=int)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    book_name = Path(args.input).stem
    base_output_dir = Path(OUTPUT_DIR) / book_name
    intermediary_dir = base_output_dir / "intermediary_audio"
    final_dir = base_output_dir / "final_audio"
    output_basename = book_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    intermediary_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(exist_ok=True)
    log_path = Path("result") / f"run_settings_book_{timestamp}.txt"
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0

    with open(log_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(log_file, original_stdout)
        try:
            print(f"Run timestamp: {timestamp}")
            print(f"Settings log: {log_path}")
            print(f"Input file: {args.input}")
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
            print(f"CARD = {CARD}")
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

            audio_path = Path(REFERENCE_AUDIO_PATH)
            if not audio_path.exists():
                print(f"ERROR: Reference audio not found: {audio_path}")
                return 1

            text_path = Path(REFERENCE_TEXT_PATH)
            if not text_path.exists():
                print(f"ERROR: Reference text not found: {text_path}")
                return 1

            print("Reading reference audio...")
            ref_audio_base64 = read_audio_b64(audio_path)
            print(f"Audio encoded: {len(ref_audio_base64)} characters")

            print("Reading reference text...")
            ref_text = read_text_file(text_path)
            print(f"Reference text length: {len(ref_text)} characters")

            print("Extracting book...")
            extract = extract_book(args.input)
            extract_path = base_output_dir / "extract.json"
            summary_path = base_output_dir / "summary.txt"
            write_extract_json(extract, str(extract_path))
            write_summary_txt(extract["chapters"], str(summary_path))
            print(f"Extract saved: {extract_path}")
            print(f"Summary saved: {summary_path}")

            chapters = extract["chapters"]
            start_index = max(1, args.start_chapter)
            end_index = args.end_chapter or len(chapters)
            end_index = min(end_index, len(chapters))
            selected = [c for c in chapters if start_index <= c["index"] <= end_index]

            if not selected:
                print("ERROR: No chapters selected")
                return 1

            print(f"Selected chapters: {start_index}..{end_index} (total {len(selected)})")

            chunks = []
            for chapter in selected:
                chapter_chunks = split_text(chapter["text"], TARGET_SECONDS, CHARS_PER_SECOND, MAX_CHUNK_MULTIPLIER)
                chunks.extend(chapter_chunks)

            print(f"Split into {len(chunks)} chunk(s)")

            # Checkpoint: find already-completed chunks
            done_set: set[int] = set()
            for p in intermediary_dir.glob(f"{output_basename}_*.wav"):
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

            total_audio_seconds, total_processing_seconds = run_generation(
                endpoint_url=ENDPOINT_URL,
                pending=pending,
                total_chunks=len(chunks),
                ref_audio_base64=ref_audio_base64,
                ref_text=ref_text,
                language=LANGUAGE,
                batch_size=BATCH_SIZE,
                output_dir=intermediary_dir,
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
