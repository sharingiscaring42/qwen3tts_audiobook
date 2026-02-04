"""
Download Qwen3 TTS model weights to a local directory.

Usage:
  python local/download_model.py --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --model-dir models
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen3 TTS model to local disk")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Local model directory (root where model ID folder will be created)",
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    target_dir = os.path.join(args.model_dir, args.model_id)

    print(f"Downloading {args.model_id} to {target_dir}...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
