import argparse
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from book_extract import extract_book, write_extract_json, write_summary_txt
from client_batch_from_text import split_text


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

REFERENCE_AUDIO_PATH = "ref/jeff_hays_0/ref_audio.wav"
REFERENCE_TEXT_PATH = "ref/jeff_hays_0/ref_text.txt"
LANGUAGE = "English"

SETTINGS = {
    "A10": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {"English": {"BATCH_SIZE": 20}, "French": {"BATCH_SIZE": 17}},
    },
    "A100": {
        "TARGET_SECONDS": 30,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {"English": {"BATCH_SIZE": 56}, "French": {"BATCH_SIZE": 28}},
    },
    "H100": {
        "TARGET_SECONDS": 60,
        "CHARS_PER_SECOND": 15,
        "MAX_CHUNK_MULTIPLIER": 1.05,
        "LANG": {"English": {"BATCH_SIZE": 64}, "French": {"BATCH_SIZE": 56}},
    },
}

TARGET_SECONDS = SETTINGS[CARD]["TARGET_SECONDS"]
CHARS_PER_SECOND = SETTINGS[CARD]["CHARS_PER_SECOND"]
MAX_CHUNK_MULTIPLIER = SETTINGS[CARD]["MAX_CHUNK_MULTIPLIER"]
BATCH_SIZE = SETTINGS[CARD]["LANG"][LANGUAGE]["BATCH_SIZE"]

MAX_NEW_TOKENS = 2048
RETRY_ON_LONG_AUDIO = True
MAX_AUDIO_SECONDS = 90
RETRY_MAX_NEW_TOKENS = 1500
RETRY_BATCH_SIZE = 8

DEFAULT_TTS_MODE = "base_clone"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

OUTPUT_DIR = "output/book"


def resolve_endpoint(endpoint_override: str | None = None) -> str:
    if endpoint_override:
        return endpoint_override
    if USE_LOCAL:
        return _env.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
    key = f"TTS_ENDPOINT_URL_{CARD}"
    value = _env.get(key, "").strip()
    if not value:
        raise RuntimeError(f"Missing {key} in .env")
    return value


def resolve_settings_url(endpoint_url: str) -> str:
    key = f"TTS_SETTINGS_URL_{CARD}"
    return _env.get(key, endpoint_url).strip() or endpoint_url


def read_audio_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def fetch_server_settings(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def build_tts_payload(
    *,
    text,
    language: str,
    max_new_tokens: int,
    tts_mode: str,
    tts_model: str,
    ref_audio_base64: str = "",
    ref_text: str = "",
    speaker: str = "",
    prompt_instruct_text: str = "",
    voice_design_prompt: str = "",
) -> dict:
    payload = {
        "text": text,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "tts_mode": tts_mode,
        "tts_model": tts_model,
    }

    if tts_mode == "base_clone":
        payload["ref_audio_base64"] = ref_audio_base64
        payload["ref_text"] = ref_text
    elif tts_mode == "custom_voice":
        payload["speaker"] = speaker
        if prompt_instruct_text:
            payload["prompt_instruct_text"] = prompt_instruct_text
    elif tts_mode == "voice_design":
        payload["prompt"] = voice_design_prompt

    return payload


def clone_voice_chunk(endpoint_url: str, payload: dict) -> dict:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch voice cloning from EPUB/PDF")
    parser.add_argument("--input", required=True)
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--end-chapter", type=int)
    parser.add_argument("--tts-mode", default=DEFAULT_TTS_MODE, choices=["base_clone", "custom_voice", "voice_design"])
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--speaker", default="")
    parser.add_argument("--prompt-instruct-text", default="")
    parser.add_argument("--voice-design-prompt", default="")
    parser.add_argument("--endpoint-url", default="")
    args = parser.parse_args()

    endpoint_url = resolve_endpoint(args.endpoint_url.strip() or None)
    settings_url = resolve_settings_url(endpoint_url)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    book_name = Path(args.input).stem
    base_output_dir = Path(OUTPUT_DIR) / book_name
    intermediary_dir = base_output_dir / "intermediary_audio"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    intermediary_dir.mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(exist_ok=True)
    log_path = Path("result") / f"run_settings_book_{timestamp}.txt"
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
        try:
            print(f"Run timestamp: {timestamp}")
            print(f"Settings log: {log_path}")
            print(f"Endpoint URL: {endpoint_url}")
            print(f"Input file: {args.input}\n")

            print("Server settings:")
            try:
                server_settings = fetch_server_settings(settings_url)
                for key, value in server_settings.items():
                    print(f"{key}: {value}")
            except requests.RequestException as exc:
                print(f"WARN: Failed to fetch server settings: {exc}")
            print()

            ref_audio_base64 = ""
            ref_text = ""
            if args.tts_mode == "base_clone":
                audio_path = Path(REFERENCE_AUDIO_PATH)
                text_path = Path(REFERENCE_TEXT_PATH)
                if not audio_path.exists():
                    print(f"ERROR: Reference audio not found: {audio_path}")
                    return 1
                if not text_path.exists():
                    print(f"ERROR: Reference text not found: {text_path}")
                    return 1
                ref_audio_base64 = read_audio_file(str(audio_path))
                ref_text = read_text_file(str(text_path))
            elif args.tts_mode == "custom_voice" and not args.speaker.strip():
                print("ERROR: --speaker is required for --tts-mode custom_voice")
                return 1
            elif args.tts_mode == "voice_design" and not args.voice_design_prompt.strip():
                print("ERROR: --voice-design-prompt is required for --tts-mode voice_design")
                return 1

            print("Extracting book...")
            extract = extract_book(args.input)
            extract_path = base_output_dir / "extract.json"
            summary_path = base_output_dir / "summary.txt"
            write_extract_json(extract, str(extract_path))
            write_summary_txt(extract["chapters"], str(summary_path))

            chapters = extract["chapters"]
            start_index = max(1, args.start_chapter)
            end_index = args.end_chapter or len(chapters)
            end_index = min(end_index, len(chapters))
            selected = [c for c in chapters if start_index <= c["index"] <= end_index]
            if not selected:
                print("ERROR: No chapters selected")
                return 1

            chunks = []
            for chapter in selected:
                chunks.extend(split_text(chapter["text"], TARGET_SECONDS, CHARS_PER_SECOND))
            print(f"Split into {len(chunks)} chunk(s)")

            def chunk_batches(items, size):
                for i in range(0, len(items), size):
                    yield items[i : i + size]

            chunk_index = 1
            retry_queue = []
            batches = list(chunk_batches(chunks, BATCH_SIZE))
            last_batch = batches.pop() if batches else []

            def submit_batch(batch_texts: list[str], token_cap: int, indices: list[int] | None = None):
                nonlocal chunk_index, total_audio_seconds, total_processing_seconds
                payload = build_tts_payload(
                    text=batch_texts,
                    language=LANGUAGE,
                    max_new_tokens=token_cap,
                    tts_mode=args.tts_mode,
                    tts_model=args.tts_model,
                    ref_audio_base64=ref_audio_base64,
                    ref_text=ref_text,
                    speaker=args.speaker,
                    prompt_instruct_text=args.prompt_instruct_text,
                    voice_design_prompt=args.voice_design_prompt,
                )
                result = clone_voice_chunk(endpoint_url, payload)
                if not result.get("success", True):
                    raise RuntimeError(result.get("error", "Unknown error"))

                audio_base64s = result.get("audio_base64s")
                durations = result.get("duration_seconds")
                if not isinstance(audio_base64s, list) or not isinstance(durations, list):
                    raise RuntimeError("Expected batch response with audio_base64s/duration_seconds")

                processing_seconds = float(result.get("processing_seconds", 0.0))
                total_processing_seconds += processing_seconds

                batch_audio_used = 0.0
                for i, (chunk_text, audio_b64, audio_seconds) in enumerate(zip(batch_texts, audio_base64s, durations)):
                    idx = indices[i] if indices else chunk_index
                    audio_seconds = float(audio_seconds)
                    if (
                        RETRY_ON_LONG_AUDIO
                        and indices is None
                        and audio_seconds > MAX_AUDIO_SECONDS
                    ):
                        retry_queue.append((idx, chunk_text))
                    else:
                        output_file = intermediary_dir / f"{book_name}_{idx}.wav"
                        with open(output_file, "wb") as f:
                            f.write(base64.b64decode(audio_b64))
                        batch_audio_used += audio_seconds
                    if indices is None:
                        chunk_index += 1

                total_audio_seconds += batch_audio_used

            for batch in batches:
                submit_batch(batch, MAX_NEW_TOKENS)

            combined = []
            if last_batch:
                start_idx = chunk_index
                combined.extend([(start_idx + i, txt) for i, txt in enumerate(last_batch)])
            combined.extend(retry_queue)
            if combined:
                submit_batch(
                    [txt for _, txt in combined],
                    RETRY_MAX_NEW_TOKENS,
                    indices=[idx for idx, _ in combined],
                )
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out")
            return 1
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Cannot connect to endpoint: {endpoint_url}")
            return 1
        except requests.exceptions.HTTPError as exc:
            print(f"ERROR: Server error {exc.response.status_code}: {exc.response.text}")
            return 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1
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
