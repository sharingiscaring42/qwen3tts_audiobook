from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from client_batch_from_book import build_tts_payload


def test_build_payload_base_clone():
    payload = build_tts_payload(
        text=["a"],
        language="English",
        max_new_tokens=100,
        tts_mode="base_clone",
        tts_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ref_audio_base64="audio",
        ref_text="text",
        speaker="",
        prompt_instruct_text="",
        voice_design_prompt="",
    )
    assert payload["ref_audio_base64"] == "audio"
    assert payload["ref_text"] == "text"
    assert "speaker" not in payload


def test_build_payload_custom_voice():
    payload = build_tts_payload(
        text="a",
        language="English",
        max_new_tokens=100,
        tts_mode="custom_voice",
        tts_model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker="Chelsie",
        prompt_instruct_text="Calm tone",
        ref_audio_base64="",
        ref_text="",
        voice_design_prompt="",
    )
    assert payload["speaker"] == "Chelsie"
    assert payload["prompt_instruct_text"] == "Calm tone"
    assert "ref_audio_base64" not in payload


def test_build_payload_voice_design():
    payload = build_tts_payload(
        text="a",
        language="English",
        max_new_tokens=100,
        tts_mode="voice_design",
        tts_model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        speaker="",
        prompt_instruct_text="",
        ref_audio_base64="",
        ref_text="",
        voice_design_prompt="deep and cinematic",
    )
    assert payload["prompt"] == "deep and cinematic"
    assert "speaker" not in payload
