#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import requests


def load_env(path: str = ".env") -> dict[str, str]:
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


def encode_file(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_asr_endpoint(env: dict[str, str], card: str, override: str | None) -> str:
    if override:
        return override
    key = f"ASR_ENDPOINT_URL_{card}"
    value = env.get(key, "").strip()
    if not value:
        raise RuntimeError(f"Missing {key} in .env")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio via /transcribe endpoint")
    parser.add_argument("--audio", required=True, help="Path to input audio")
    parser.add_argument("--card", default="A10", choices=["A10", "A100", "H100"])
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--endpoint-url", default="")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--return-timestamps", action="store_true")
    parser.add_argument("--align-ref-text", default="")
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--output-dir", default="output/asr")
    args = parser.parse_args()

    env = load_env()
    endpoint = resolve_asr_endpoint(env, args.card, args.endpoint_url.strip() or None)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio not found: {audio_path}")
        return 1

    payload = {
        "audio_base64": encode_file(audio_path),
        "asr_model": args.asr_model,
        "language": args.language,
        "return_timestamps": args.return_timestamps,
        "aligner_model": args.aligner_model,
    }
    if args.align_ref_text.strip():
        payload["align_ref_text"] = args.align_ref_text.strip()

    try:
        response = requests.post(endpoint, json=payload, timeout=900, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        print(f"ERROR: request failed: {exc}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    with open(txt_path, "w") as f:
        f.write(result.get("text", ""))
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved transcript: {txt_path}")
    print(f"Saved metadata:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
