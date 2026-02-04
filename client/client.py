#!/usr/bin/env python3
"""
Qwen3 TTS Voice Cloning Client

Simple client to interact with the Modal-deployed voice cloning service.

Usage:
    python client/client.py "Hello world" -a reference.wav -t "This is the reference text"
    
    # With custom endpoint
    python client/client.py "Hello world" -a ref.wav -t "ref text" -e https://your-endpoint.modal.run
"""

import base64
import argparse
import os
from pathlib import Path
import requests
from typing import Optional


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

DEFAULT_ENDPOINT = _env.get("ENDPOINT_URL", "https://your-endpoint.modal.run")
LOCAL_ENDPOINT = _env.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
ACTIVE_ENDPOINT = LOCAL_ENDPOINT if USE_LOCAL else DEFAULT_ENDPOINT


def read_audio_file(path: str) -> str:
    """Read audio file and return as base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clone_voice(
    text: str,
    reference_audio_path: str,
    reference_text: str,
    endpoint_url: str = DEFAULT_ENDPOINT,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Clone a voice and generate speech using Modal endpoint
    
    Args:
        text: Text to synthesize
        reference_audio_path: Path to reference audio file (WAV recommended)
        reference_text: Transcript of the reference audio
        endpoint_url: Modal web endpoint URL
        output_path: Where to save output (defaults to "output_<timestamp>.wav")
        verbose: Print progress messages
        
    Returns:
        Response dictionary from the server
    """
    if verbose:
        print("=" * 60)
        print("Qwen3 TTS Voice Cloning Client")
        print("=" * 60)
        print(f"\nEndpoint: {endpoint_url}")
        print(f"Reference audio: {reference_audio_path}")
        print(f"Text to synthesize: {text[:60]}{'...' if len(text) > 60 else ''}")
        print()
    
    # Validate inputs
    audio_path = Path(reference_audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    
    # Read and encode audio
    if verbose:
        print("Reading and encoding reference audio...")
    audio_base64 = read_audio_file(str(audio_path))
    if verbose:
        print(f"Audio encoded: {len(audio_base64)} characters")
    
    # Prepare request
    payload = {
        "text": text,
        "reference_audio_base64": audio_base64,
        "reference_text": reference_text,
    }
    
    # Send request
    if verbose:
        print(f"\nSending request to Modal endpoint...")
        print(f"This may take 30-60 seconds on cold start, or 5-10 seconds if warm.")
        print()
    
    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=300,  # 5 minute timeout
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise TimeoutError("Request timed out. The server may be starting up (cold start). Try again.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to endpoint: {endpoint_url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Server error: {e.response.status_code} - {e.response.text}")
    
    result = response.json()
    
    # Check for server-side errors
    if not result.get("success", True):
        raise RuntimeError(f"Server processing error: {result.get('error', 'Unknown error')}")
    
    # Save output
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.wav"
    
    output_file = Path(output_path)
    audio_data = base64.b64decode(result["audio_base64"])
    
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    if verbose:
        print("=" * 60)
        print(f"SUCCESS!")
        print(f"Generated audio: {result['duration_seconds']:.2f} seconds")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Saved to: {output_file.absolute()}")
        print("=" * 60)
    
    return result


def health_check(endpoint_url: str = DEFAULT_ENDPOINT) -> dict:
    """Check if the service is healthy"""
    health_url = endpoint_url.replace("/generate", "/health")
    
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice using Qwen3 TTS on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python client.py "Hello world" -a ref.wav -t "reference text here"
  
  # With custom output file
  python client.py "Hello world" -a ref.wav -t "reference text" -o my_voice.wav
  
  # With custom endpoint
  python client.py "Hello world" -a ref.wav -t "text" -e https://my-endpoint.modal.run
  
  # Health check
  python client.py --health -e https://my-endpoint.modal.run
        """
    )
    
    parser.add_argument("text", nargs="?", help="Text to synthesize into speech")
    parser.add_argument("-a", "--audio", dest="reference_audio", help="Path to reference audio file (WAV)")
    parser.add_argument("-t", "--text", dest="reference_text", help="Transcript of reference audio")
    parser.add_argument("-e", "--endpoint", default=ACTIVE_ENDPOINT, help="Endpoint URL")
    parser.add_argument("-o", "--output", help="Output audio file path (default: output_TIMESTAMP.wav)")
    parser.add_argument("--health", action="store_true", help="Check service health")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")
    
    args = parser.parse_args()
    
    endpoint_url = args.endpoint

    # Health check mode
    if args.health:
        result = health_check(endpoint_url)
        print(f"Health status: {result.get('status', 'unknown')}")
        if result.get('gpu_available'):
            print(f"GPU: {result.get('gpu_name', 'Unknown')}")
        print(f"Model: {result.get('model', 'Unknown')}")
        return
    
    # Validate required arguments
    if not args.text:
        parser.error("Text is required (unless using --health)")
    if not args.reference_audio:
        parser.error("Reference audio path is required (-a/--audio)")
    if not args.reference_text:
        parser.error("Reference text is required (-t/--text)")
    
    try:
        result = clone_voice(
            text=args.text,
            reference_audio_path=args.reference_audio,
            reference_text=args.reference_text,
            endpoint_url=endpoint_url,
            output_path=args.output,
            verbose=not args.quiet,
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except TimeoutError as e:
        print(f"\nError: {e}")
        print("Tip: First request may take longer due to cold start. Try again.")
        return 1
    except ConnectionError as e:
        print(f"\nError: {e}")
        print("Tip: Check that the endpoint URL is correct and the service is deployed.")
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
