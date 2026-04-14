from server.api_contracts import validate_asr_request, validate_tts_request


def test_tts_base_clone_requires_reference_fields():
    try:
        validate_tts_request({"text": "hello", "tts_mode": "base_clone"})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "ref_audio_base64" in str(exc)


def test_tts_custom_voice_requires_speaker():
    try:
        validate_tts_request(
            {
                "text": "hello",
                "tts_mode": "custom_voice",
                "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            }
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "speaker" in str(exc)


def test_tts_voice_design_requires_prompt():
    try:
        validate_tts_request(
            {
                "text": "hello",
                "tts_mode": "voice_design",
                "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            }
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "prompt" in str(exc)


def test_asr_audio_required():
    try:
        validate_asr_request({})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "audio_base64" in str(exc)


def test_asr_align_requires_timestamps():
    try:
        validate_asr_request({"audio_base64": "abc", "align_ref_text": "x", "return_timestamps": False})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "return_timestamps=true" in str(exc)
