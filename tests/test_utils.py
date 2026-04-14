"""Smoke tests for client/utils.py."""
import io
import os
import tempfile

import pytest

from utils import (
    CARD_SETTINGS,
    Tee,
    card_defaults,
    load_env,
    read_audio_b64,
    read_text_file,
    split_text,
)


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------

def test_split_text_returns_list():
    chunks = split_text("Hello world.", 60, 15)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_split_text_cuts_at_period():
    # With a window large enough for each sentence but not both, should cut at the period
    text = "First sentence. Second sentence."
    # Window = int(2 * 17 * 1.05) = 35, larger than full text → single chunk
    # Use target that forces a split: window = int(1 * 17 * 1.05) = 17
    chunks = split_text(text, 1, 17)
    # "First sentence." is 15 chars, fits in window of 17; period found → cuts there
    assert chunks[0] == "First sentence."
    # All chunks should be non-empty
    assert all(c for c in chunks)


def test_split_text_empty_string():
    assert split_text("", 60, 15) == []


def test_split_text_short_text_single_chunk():
    text = "Short."
    chunks = split_text(text, 60, 15)
    assert chunks == ["Short."]


def test_split_text_no_period_falls_back_to_window():
    # No period in the text; should still split without crashing
    text = "a" * 2000
    chunks = split_text(text, 1, 15)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# load_env
# ---------------------------------------------------------------------------

def test_load_env_missing_file_returns_empty():
    result = load_env("/nonexistent/path/.env")
    assert result == {}


def test_load_env_parses_key_value():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("FOO=bar\n")
        f.write("# comment line\n")
        f.write("BAZ=qux\n")
        f.write("\n")
        name = f.name
    try:
        result = load_env(name)
        assert result["FOO"] == "bar"
        assert result["BAZ"] == "qux"
        assert "# comment line" not in result
    finally:
        os.unlink(name)


def test_load_env_ignores_lines_without_equals():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NOEQUALS\n")
        f.write("KEY=value\n")
        name = f.name
    try:
        result = load_env(name)
        assert "NOEQUALS" not in result
        assert result["KEY"] == "value"
    finally:
        os.unlink(name)


# ---------------------------------------------------------------------------
# card_defaults
# ---------------------------------------------------------------------------

def test_card_defaults_a10_english():
    target, cps, mult, batch = card_defaults("A10", "English")
    assert target == CARD_SETTINGS["A10"]["TARGET_SECONDS"]
    assert cps == CARD_SETTINGS["A10"]["CHARS_PER_SECOND"]
    assert batch == CARD_SETTINGS["A10"]["LANG"]["English"]["BATCH_SIZE"]


def test_card_defaults_all_cards_english():
    for card in CARD_SETTINGS:
        target, cps, mult, batch = card_defaults(card, "English")
        assert target > 0
        assert cps > 0
        assert 1.0 <= mult <= 2.0
        assert batch > 0


def test_card_defaults_unknown_language_falls_back_to_english():
    _, _, _, batch_en = card_defaults("A10", "English")
    _, _, _, batch_unknown = card_defaults("A10", "Klingon")
    assert batch_unknown == batch_en


# ---------------------------------------------------------------------------
# read_audio_b64 / read_text_file
# ---------------------------------------------------------------------------

def test_read_audio_b64_roundtrip():
    import base64
    data = b"\x00\x01\x02\x03\xff"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(data)
        name = f.name
    try:
        encoded = read_audio_b64(name)
        assert base64.b64decode(encoded) == data
    finally:
        os.unlink(name)


def test_read_text_file_strips_whitespace():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("  hello world  \n")
        name = f.name
    try:
        result = read_text_file(name)
        assert result == "hello world"
    finally:
        os.unlink(name)


# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------

def test_tee_writes_to_both_streams():
    buf1 = io.StringIO()
    buf2 = io.StringIO()
    tee = Tee(buf1, buf2)
    tee.write("hello")
    tee.flush()
    assert buf1.getvalue() == "hello"
    assert buf2.getvalue() == "hello"


def test_tee_flush_propagates():
    buf1 = io.StringIO()
    buf2 = io.StringIO()
    tee = Tee(buf1, buf2)
    tee.write("x")
    tee.flush()
    assert buf1.getvalue() == "x"
