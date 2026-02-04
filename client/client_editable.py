#!/usr/bin/env python3
"""
Qwen3 TTS Voice Cloning Client - Editable Version

Edit the CONFIG section below, then run:
    python client/client_editable.py

No command line arguments needed - everything is configured in the code!
"""

import base64
import os
from pathlib import Path
import requests
import time

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

USE_LOCAL = False

# Your Modal endpoint URL (get this from `modal deploy` output)
ENDPOINT_URL = _env.get("ENDPOINT_URL", "https://your-endpoint.modal.run")
LOCAL_ENDPOINT_URL = _env.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
ACTIVE_ENDPOINT_URL = LOCAL_ENDPOINT_URL if USE_LOCAL else ENDPOINT_URL

# Path to your reference audio file (WAV format recommended, 3-10 seconds)
# This is the voice you want to clone
REFERENCE_AUDIO_PATH = "ref/jeff_hays_0/ref_audio.wav"

# Path to a text file containing the transcript of your reference audio
# (what is being said in the reference audio file)
REFERENCE_TEXT_PATH = "ref/jeff_hays_0/ref_text.txt"

# Text you want to synthesize in the cloned voice
TEXT_TO_SYNTHESIZE = """The western side of each ridge was steep and difficult, but the eastward slopes were gentler, furrowed with many gullies and narrow ravines. All night the three companions scrambled in this bony land, climbing to the crest of the first and tallest ridge, and down again into the darkness of a deep winding valley on the other side."""

# Language (use "Auto" for automatic detection, or specify like "English", "Chinese")
LANGUAGE = "Auto"

# Where to save the output audio
OUTPUT_PATH = "output/output.wav"

# ============================================
# END CONFIG
# ============================================


def read_audio_file(path: str) -> str:
    """Read audio file and return as base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: str) -> str:
    """Read text file"""
    with open(path, "r") as f:
        return f.read().strip()


def clone_voice(endpoint_url: str):
    """
    Clone voice using the configuration above.
    Run this function after editing the CONFIG section.
    """
    print("=" * 60)
    print("Qwen3 TTS Voice Cloning Client (Editable)")
    print("=" * 60)
    print(f"\nEndpoint: {endpoint_url}")
    print(f"Reference audio: {REFERENCE_AUDIO_PATH}")
    print(f"Reference text file: {REFERENCE_TEXT_PATH}")
    print(f"Language: {LANGUAGE}")
    print(f"Text to synthesize: {TEXT_TO_SYNTHESIZE[:60]}{'...' if len(TEXT_TO_SYNTHESIZE) > 60 else ''}")
    print()
    
    # Validate inputs
    audio_path = Path(REFERENCE_AUDIO_PATH)
    if not audio_path.exists():
        print(f"ERROR: Reference audio not found: {audio_path}")
        print(f"\nPlease check REFERENCE_AUDIO_PATH in the CONFIG section.")
        return 1
    
    text_path = Path(REFERENCE_TEXT_PATH)
    if not text_path.exists():
        print(f"ERROR: Reference text file not found: {text_path}")
        print(f"\nPlease check REFERENCE_TEXT_PATH in the CONFIG section.")
        print(f"Create a text file with the transcript of your reference audio.")
        return 1
    
    # Read and encode audio
    print("Reading and encoding reference audio...")
    try:
        ref_audio_base64 = read_audio_file(str(audio_path))
        print(f"Audio encoded: {len(ref_audio_base64)} characters")
    except Exception as e:
        print(f"ERROR reading audio file: {e}")
        return 1
    
    # Read reference text
    print("Reading reference text...")
    try:
        ref_text = read_text_file(str(text_path))
        print(f"Reference text: {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}")
    except Exception as e:
        print(f"ERROR reading text file: {e}")
        return 1
    
    # Prepare request
    payload = {
        "text": TEXT_TO_SYNTHESIZE,
        "ref_audio_base64": ref_audio_base64,
        "ref_text": ref_text,
        "language": LANGUAGE,
    }
    
    # Send request
    print(f"\nSending request to endpoint...")
    print(f"This may take 30-60 seconds on cold start, or 5-10 seconds if warm.")
    print()
    
    try:
        request_start = time.monotonic()
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=300,  # 5 minute timeout
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        request_end = time.monotonic()
    except requests.exceptions.Timeout:
        print(f"\nERROR: Request timed out.")
        print("The server may be starting up (cold start). Try again in a moment.")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to endpoint: {endpoint_url}")
        print("Tip: Check that ENDPOINT_URL is correct and the service is deployed.")
        return 1
    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: Server error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return 1
    
    result = response.json()
    total_roundtrip = request_end - request_start
    
    # Check for server-side errors
    if not result.get("success", True):
        print(f"\nERROR: Server processing failed: {result.get('error', 'Unknown error')}")
        return 1
    
    # Save output
    output_file = Path(OUTPUT_PATH)
    try:
        audio_data = base64.b64decode(result["audio_base64"])
        with open(output_file, "wb") as f:
            f.write(audio_data)
    except Exception as e:
        print(f"\nERROR saving output file: {e}")
        return 1
    
    print("=" * 60)
    print(f"SUCCESS!")
    print(f"Generated audio: {result['duration_seconds']:.2f} seconds")
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Language: {result.get('language', 'Auto')}")
    if result.get("model_load_seconds") is not None:
        print(f"Model load time (server): {result['model_load_seconds']:.2f} seconds")
    if result.get("processing_seconds") is not None:
        print(f"Processing time (server): {result['processing_seconds']:.2f} seconds")
    print(f"Total round-trip (client): {total_roundtrip:.2f} seconds")

    duration_seconds = result.get("duration_seconds", 0.0)
    processing_seconds = result.get("processing_seconds")
    if duration_seconds > 0 and processing_seconds:
        rtf = processing_seconds / duration_seconds
        inv_rtf = duration_seconds / processing_seconds if processing_seconds > 0 else 0.0
        print(f"RTF (processing/audio): {rtf:.3f}")
        print(f"1/RTF (audio/processing): {inv_rtf:.3f}x")
    print(f"Saved to: {output_file.absolute()}")
    print("=" * 60)
    
    return 0


def health_check(endpoint_url: str):
    """Check if the service is healthy"""
    health_url = endpoint_url.replace("/generate", "/health")
    
    print(f"Checking health at: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        print(f"\nStatus: {result.get('status', 'unknown')}")
        if result.get('gpu_available'):
            print(f"GPU: {result.get('gpu_name', 'Unknown')}")
        print(f"Model: {result.get('model', 'Unknown')}")
        return 0
        
    except Exception as e:
        print(f"\nHealth check failed: {e}")
        return 1


def main():
    """
    Main entry point.
    Edit the CONFIG section at the top of this file, then run:
        python client_editable.py
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--health":
        return health_check(ACTIVE_ENDPOINT_URL)

    return clone_voice(ACTIVE_ENDPOINT_URL)


if __name__ == "__main__":
    exit(main())
