from __future__ import annotations

from typing import Any

TTS_MODES = {"base_clone", "custom_voice", "voice_design"}

TTS_MODELS_BY_MODE = {
    "base_clone": {
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
    "custom_voice": {
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
    "voice_design": {
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
}

DEFAULT_TTS_MODE = "base_clone"
DEFAULT_TTS_MODEL_BY_MODE = {
    "base_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
}
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"

ALIGNER_MODELS = {
    "Qwen/Qwen3-ForcedAligner-0.6B",
}
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _validate_text(value: Any) -> str | list[str]:
    if isinstance(value, str):
        if value.strip():
            return value
        raise ValueError("Field 'text' must be a non-empty string")

    if isinstance(value, list):
        if not value:
            raise ValueError("Field 'text' list cannot be empty")
        normalized: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Field 'text[{idx}]' must be a non-empty string")
            normalized.append(item)
        return normalized

    raise ValueError("Field 'text' must be a string or list of strings")


def _validate_choice(field: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid '{field}': '{value}'. Allowed values: {options}")
    return value


def validate_tts_request(request: dict[str, Any]) -> dict[str, Any]:
    text = _validate_text(request.get("text"))
    tts_mode = request.get("tts_mode", DEFAULT_TTS_MODE)
    if not isinstance(tts_mode, str):
        raise ValueError("Field 'tts_mode' must be a string")
    tts_mode = _validate_choice("tts_mode", tts_mode, TTS_MODES)

    default_model = DEFAULT_TTS_MODEL_BY_MODE[tts_mode]
    raw_model = request.get("tts_model", default_model)
    if not isinstance(raw_model, str):
        raise ValueError("Field 'tts_model' must be a string")
    tts_model = _validate_choice("tts_model", raw_model, TTS_MODELS_BY_MODE[tts_mode])

    language = request.get("language", "Auto")
    if not isinstance(language, str):
        raise ValueError("Field 'language' must be a string")

    max_new_tokens = request.get("max_new_tokens", 2048)
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("Field 'max_new_tokens' must be an integer >= 1")

    data = {
        "text": text,
        "tts_mode": tts_mode,
        "tts_model": tts_model,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "ref_audio_base64": "",
        "ref_text": "",
        "speaker": "",
        "prompt_instruct_text": "",
        "prompt": "",
    }

    if tts_mode == "base_clone":
        ref_audio = _as_non_empty_str(
            request.get("ref_audio_base64") or request.get("reference_audio_base64")
        )
        ref_text = _as_non_empty_str(request.get("ref_text") or request.get("reference_text"))
        if not ref_audio:
            raise ValueError("Field 'ref_audio_base64' is required for tts_mode='base_clone'")
        if not ref_text:
            raise ValueError("Field 'ref_text' is required for tts_mode='base_clone'")
        data["ref_audio_base64"] = ref_audio
        data["ref_text"] = ref_text

    elif tts_mode == "custom_voice":
        speaker = _as_non_empty_str(request.get("speaker"))
        if not speaker:
            raise ValueError("Field 'speaker' is required for tts_mode='custom_voice'")
        data["speaker"] = speaker
        data["prompt_instruct_text"] = _as_non_empty_str(request.get("prompt_instruct_text")) or ""

    else:
        prompt = _as_non_empty_str(request.get("prompt"))
        if not prompt:
            raise ValueError("Field 'prompt' is required for tts_mode='voice_design'")
        data["prompt"] = prompt

    return data


def validate_asr_request(request: dict[str, Any]) -> dict[str, Any]:
    audio_base64 = _as_non_empty_str(request.get("audio_base64"))
    if not audio_base64:
        raise ValueError("Field 'audio_base64' is required")

    asr_model = request.get("asr_model", DEFAULT_ASR_MODEL)
    if not isinstance(asr_model, str):
        raise ValueError("Field 'asr_model' must be a string")
    asr_model = _validate_choice("asr_model", asr_model, ASR_MODELS)

    language = request.get("language", "auto")
    if not isinstance(language, str):
        raise ValueError("Field 'language' must be a string")

    return_timestamps = request.get("return_timestamps", False)
    if not isinstance(return_timestamps, bool):
        raise ValueError("Field 'return_timestamps' must be a boolean")

    align_ref_text = _as_non_empty_str(request.get("align_ref_text"))

    aligner_model = request.get("aligner_model", DEFAULT_ALIGNER_MODEL)
    if not isinstance(aligner_model, str):
        raise ValueError("Field 'aligner_model' must be a string")
    aligner_model = _validate_choice("aligner_model", aligner_model, ALIGNER_MODELS)

    if align_ref_text and not return_timestamps:
        raise ValueError("Field 'align_ref_text' requires return_timestamps=true")

    return {
        "audio_base64": audio_base64,
        "asr_model": asr_model,
        "language": language,
        "return_timestamps": return_timestamps,
        "align_ref_text": align_ref_text,
        "aligner_model": aligner_model,
    }
