"""
Script to download Qwen3 TTS model to Modal Volume

Run this once to persist model weights to a Modal Volume.
This avoids downloading the model from HuggingFace on every cold start.
"""

import modal

# Simple image with just download dependencies
download_image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface_hub>=0.25.0",
        "hf-transfer>=0.1.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable fast transfers
        "HF_HUB_CACHE": "/models/hf_hub_cache",
    })
)

app = modal.App("qwen3-tts-model-downloader", image=download_image)

# Create or get volume
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
MODEL_DIR = "/models"


@app.function(
    volumes={MODEL_DIR: volume},
    timeout=600,  # 10 minutes for large model download
)
def download_model(
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # Base model for voice cloning
    revision: str = "main",
):
    """
    Download Qwen3 TTS model from HuggingFace Hub to Modal Volume
    
    Args:
        model_id: HuggingFace model identifier
        revision: Model revision/tag to download
    """
    from huggingface_hub import snapshot_download
    import os
    
    local_path = f"{MODEL_DIR}/{model_id}"
    
    print("=" * 60)
    print(f"Downloading model: {model_id}")
    print(f"Revision: {revision}")
    print(f"Target: {local_path}")
    print("=" * 60)
    
    # Check if already exists
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"\nModel already exists at {local_path}")
        print("Checking for updates...")
    
    try:
        # Download model
        print("\nStarting download...")
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        # Calculate size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(local_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        print("\n" + "=" * 60)
        print(f"Download complete!")
        print(f"Location: {local_path}")
        print(f"Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
        print("=" * 60)
        
        # Commit to persist
        print("\nCommitting to volume...")
        volume.commit()
        print("Volume committed successfully!")
        
        return {
            "success": True,
            "model_id": model_id,
            "local_path": local_path,
            "size_mb": size_mb,
            "size_gb": size_gb,
        }
        
    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # Base model for voice cloning
    revision: str = "main",
):
    """
    Run model download
    
    Usage:
        modal run download_model.py
        
    Or with specific model:
        modal run download_model.py --model-id "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    """
    print("\nStarting model download to Modal Volume...\n")
    
    result = download_model.remote(model_id=model_id, revision=revision)
    
    if result["success"]:
        print(f"\nModel downloaded successfully!")
        print(f"Size: {result['size_gb']:.2f} GB")
        print(f"\nYou can now deploy the server with: modal deploy modal_server.py")
    else:
        print(f"\nDownload failed: {result.get('error', 'Unknown error')}")
        exit(1)
