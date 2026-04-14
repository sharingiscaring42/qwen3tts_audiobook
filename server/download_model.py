"""Download Qwen3 TTS/ASR/Aligner models to the shared Modal volume."""

from __future__ import annotations

import modal

MODEL_PRESETS = {
    "tts": [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ],
    "asr": [
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ASR-0.6B",
    ],
    "aligner": [
        "Qwen/Qwen3-ForcedAligner-0.6B",
    ],
}


def resolve_models(task: str, model_id: str | None) -> list[str]:
    if model_id:
        return [model_id]
    if task == "all":
        return MODEL_PRESETS["tts"] + MODEL_PRESETS["asr"] + MODEL_PRESETS["aligner"]
    return MODEL_PRESETS[task]


image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub>=0.25.0", "hf-transfer>=0.1.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/models/hf_hub_cache",
        }
    )
)

app = modal.App("qwen3-model-downloader", image=image)
volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
MODEL_DIR = "/models"


@app.function(volumes={MODEL_DIR: volume}, timeout=60 * 60)
def download_models(task: str = "tts", model_id: str | None = None, revision: str = "main"):
    import os

    from huggingface_hub import snapshot_download

    models = resolve_models(task, model_id)
    summaries = []

    for repo_id in models:
        local_path = f"{MODEL_DIR}/{repo_id}"
        print("=" * 60)
        print(f"Downloading model: {repo_id}")
        print(f"Target: {local_path}")
        if os.path.exists(local_path) and os.listdir(local_path):
            print("Model directory exists, checking/updating files...")

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(local_path):
            for filename in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, filename))

        size_gb = total_size / (1024**3)
        summaries.append({"model_id": repo_id, "local_path": local_path, "size_gb": size_gb})
        print(f"Completed {repo_id} ({size_gb:.2f} GB)")

    print("Committing volume...")
    volume.commit()

    return {"success": True, "task": task, "models": summaries}


@app.local_entrypoint()
def main(task: str = "tts", model_id: str = "", revision: str = "main"):
    model_id_arg = model_id.strip() or None
    if task not in {"tts", "asr", "aligner", "all"}:
        raise SystemExit("task must be one of: tts, asr, aligner, all")

    result = download_models.remote(task=task, model_id=model_id_arg, revision=revision)
    if not result.get("success"):
        raise SystemExit(1)

    print("Downloaded models:")
    for item in result["models"]:
        print(f"- {item['model_id']} ({item['size_gb']:.2f} GB) -> {item['local_path']}")
