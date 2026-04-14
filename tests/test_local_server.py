"""Smoke tests for local/local_server.py using a mocked VoiceClonerCore."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if fastapi (or the local server) cannot be imported
pytest.importorskip("fastapi", reason="fastapi not installed")

import local.local_server as ls
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a TestClient backed by a mocked VoiceClonerCore instance."""
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = None
    mock_instance.model_path = "/fake/model"
    mock_instance.model_load_seconds = 0.1
    mock_instance.model_loaded_at = 0.0
    mock_instance.clone_voice.return_value = {
        "audio_base64": "dGVzdA==",
        "sample_rate": 24000,
        "duration_seconds": 1.0,
        "success": True,
        "processing_seconds": 0.05,
    }

    with patch.object(ls, "VoiceClonerCore", return_value=mock_instance):
        with TestClient(ls.app, raise_server_exceptions=True) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root_returns_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_settings_returns_dict(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_id" in data
    assert "versions" in data


def test_generate_missing_ref_audio_returns_400(client):
    resp = client.post("/generate", json={
        "text": "hello",
        "ref_text": "some transcript",
    })
    assert resp.status_code == 400


def test_generate_missing_ref_text_returns_400(client):
    resp = client.post("/generate", json={
        "text": "hello",
        "ref_audio_base64": "dGVzdA==",
    })
    assert resp.status_code == 400


def test_generate_success(client):
    resp = client.post("/generate", json={
        "text": "hello",
        "ref_audio_base64": "dGVzdA==",
        "ref_text": "test transcript",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "audio_base64" in data


def test_generate_calls_core_clone_voice(client):
    # Verify the mock was called with the right arguments
    ls.core.clone_voice.reset_mock()
    client.post("/generate", json={
        "text": "hello world",
        "ref_audio_base64": "dGVzdA==",
        "ref_text": "transcript",
        "language": "English",
    })
    ls.core.clone_voice.assert_called_once()
    call_kwargs = ls.core.clone_voice.call_args.kwargs
    assert call_kwargs["text"] == "hello world"
    assert call_kwargs["language"] == "English"
