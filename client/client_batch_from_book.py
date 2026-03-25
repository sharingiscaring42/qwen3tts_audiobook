import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path

import requests

from book_extract import extract_book, write_extract_json, write_summary_txt
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

            def chunk_batches(items, size):
                for i in range(0, len(items), size):
                    yield items[i:i + size]

            retry_queue: list[tuple[int, str]] = []

            batches = list(chunk_batches(pending, BATCH_SIZE))
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
                    "language": LANGUAGE,
                    "max_new_tokens": MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(ENDPOINT_URL, payload)
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

                for (chunk_idx, chunk_text), audio_b64, audio_seconds in zip(batch, audio_base64s, durations):
                    audio_seconds = float(audio_seconds)
                    if RETRY_ON_LONG_AUDIO and audio_seconds > MAX_AUDIO_SECONDS:
                        print(
                            f"Chunk {chunk_idx}/{len(chunks)} | chars={len(chunk_text)} | "
                            f"audio={audio_seconds:.2f}s | deferred retry"
                        )
                        retry_queue.append((chunk_idx, chunk_text))
                    else:
                        output_file = intermediary_dir / f"{output_basename}_{chunk_idx}.wav"
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
                    "language": LANGUAGE,
                    "max_new_tokens": RETRY_MAX_NEW_TOKENS,
                }

                try:
                    result = clone_voice_chunk(ENDPOINT_URL, payload)
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
                    output_file = intermediary_dir / f"{output_basename}_{idx}.wav"
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
