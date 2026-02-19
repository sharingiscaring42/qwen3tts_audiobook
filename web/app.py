#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import requests


CARD_DEFAULTS = {
    "A10": {
        "target_seconds": 60,
        "chars_per_second": 15,
        "max_chunk_multiplier": 1.05,
        "batch_size": {"English": 20, "French": 17},
    },
    "A100": {
        "target_seconds": 30,
        "chars_per_second": 15,
        "max_chunk_multiplier": 1.05,
        "batch_size": {"English": 56, "French": 28},
    },
    "H100": {
        "target_seconds": 60,
        "chars_per_second": 15,
        "max_chunk_multiplier": 1.05,
        "batch_size": {"English": 64, "French": 56},
    },
}

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.book_audio_concat import build_concat_list
from client.book_extract import extract_book, write_extract_json, write_summary_txt

OUTPUT_BOOK = ROOT / "output" / "book"
OUTPUT_TEXT = ROOT / "output" / "text"

CUSTOM_CSS = """
:root {
  --bg-0: #0b1018;
  --bg-1: #111827;
  --bg-2: #182235;
  --panel: #131d2d;
  --panel-2: #1a2436;
  --text-0: #edf3ff;
  --text-1: #c2cee2;
  --muted: #96a6c1;
  --accent: #3db4c0;
  --accent-2: #f5a84b;
}

.gradio-container {
  background:
    radial-gradient(950px 380px at 8% -8%, rgba(61, 180, 192, 0.22) 0%, rgba(61, 180, 192, 0) 62%),
    radial-gradient(900px 360px at 94% -10%, rgba(245, 168, 75, 0.2) 0%, rgba(245, 168, 75, 0) 60%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
  color: var(--text-0);
}

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container [class*='panel'] {
  background: linear-gradient(180deg, rgba(23, 33, 51, 0.92) 0%, rgba(20, 30, 47, 0.92) 100%);
  border: 1px solid rgba(163, 183, 220, 0.18);
  border-radius: 12px;
}

.gradio-container label,
.gradio-container .label,
.gradio-container .prose,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container span {
  color: var(--text-0);
}

.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong {
  color: var(--text-1);
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--panel-2) !important;
  color: var(--text-0) !important;
  border: 1px solid rgba(160, 182, 222, 0.24) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--muted) !important;
}

.gradio-container .tab-nav button {
  color: var(--text-1) !important;
}

.gradio-container .tab-nav button.selected {
  color: var(--text-0) !important;
  border-bottom-color: var(--accent) !important;
}

.gradio-container button.primary,
.gradio-container .primary {
  background: linear-gradient(90deg, #2c9eab 0%, #297fbc 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  color: #f7fcff !important;
}

.gradio-container button.secondary {
  background: rgba(255, 255, 255, 0.06) !important;
  color: var(--text-0) !important;
  border: 1px solid rgba(165, 186, 222, 0.28) !important;
}

.hero {
  background: linear-gradient(120deg, #182b49 0%, #1c5c6b 58%, #2a7b74 100%);
  color: var(--text-0);
  border-radius: 16px;
  padding: 18px 20px;
  border: 1px solid rgba(190, 219, 255, 0.24);
  box-shadow: 0 12px 30px rgba(8, 14, 26, 0.45);
  margin-bottom: 12px;
}

.hero h1 {
  margin: 0;
  font-size: 1.45rem;
}

.hero p {
  margin: 6px 0 0;
  color: #dbe7f8;
}

.section-title {
  color: var(--accent-2);
  font-weight: 700;
  margin-top: 2px;
}
"""


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def sanitize_name(name: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def split_text(text: str, target_seconds: int, chars_per_second: int, max_chunk_multiplier: float) -> list[str]:
    max_chars = max(1, int(target_seconds * chars_per_second * max_chunk_multiplier))
    chunks: list[str] = []
    idx = 0
    length = len(text)

    while idx < length:
        window_end = min(idx + max_chars, length)
        if window_end >= length:
            chunk = text[idx:].strip()
            if chunk:
                chunks.append(chunk)
            break

        window = text[idx:window_end]
        reverse_period = window[::-1].find(".")
        cut_end = window_end if reverse_period == -1 else window_end - reverse_period
        chunk = text[idx:cut_end].strip()
        if chunk:
            chunks.append(chunk)
        idx = cut_end

    return chunks


def encode_audio(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text(path: Path) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def list_reference_profiles() -> list[str]:
    ref_root = ROOT / "ref"
    if not ref_root.exists():
        return []
    profiles = []
    for path in sorted(ref_root.iterdir()):
        if not path.is_dir():
            continue
        audio = path / "ref_audio.wav"
        text = path / "ref_text.txt"
        if audio.exists() and text.exists():
            profiles.append(path.name)
    return profiles


def profile_paths(profile: str | None) -> tuple[Path | None, Path | None]:
    if not profile:
        return None, None
    base = ROOT / "ref" / profile
    audio = base / "ref_audio.wav"
    text = base / "ref_text.txt"
    return (audio if audio.exists() else None, text if text.exists() else None)


def endpoint_for_card(card: str, use_local: bool, manual_endpoint: str | None, env_data: dict[str, str]) -> str:
    if manual_endpoint and manual_endpoint.strip():
        return manual_endpoint.strip()
    if use_local:
        return env_data.get("LOCAL_ENDPOINT_URL", "http://localhost:8000/generate")
    return env_data.get(f"ENDPOINT_URL_{card}", env_data.get("ENDPOINT_URL", "https://your-endpoint.modal.run"))


def defaults_for_card_language(card: str, language: str) -> tuple[int, int, float, int]:
    cfg = CARD_DEFAULTS[card]
    batch = cfg["batch_size"].get(language, cfg["batch_size"]["English"])
    return cfg["target_seconds"], cfg["chars_per_second"], cfg["max_chunk_multiplier"], batch


def resolve_reference(
    profile: str | None,
    uploaded_audio: str | None,
    uploaded_ref_text_file: str | None,
    ref_text_input: str,
) -> tuple[str, str, str]:
    audio_path: Path | None = Path(uploaded_audio) if uploaded_audio else None
    profile_audio, profile_text = profile_paths(profile)

    if audio_path is None:
        audio_path = profile_audio
    if audio_path is None or not audio_path.exists():
        raise ValueError("Missing reference audio. Upload one or select a reference profile.")

    if ref_text_input and ref_text_input.strip():
        ref_text = ref_text_input.strip()
    elif uploaded_ref_text_file:
        ref_text = read_text(Path(uploaded_ref_text_file))
    elif profile_text is not None and profile_text.exists():
        ref_text = read_text(profile_text)
    else:
        raise ValueError("Missing reference text. Type text, upload .txt, or select a reference profile.")

    ref_audio_b64 = encode_audio(audio_path)
    source_desc = f"audio={audio_path}"
    return ref_audio_b64, ref_text, source_desc


def clone_batch(
    endpoint: str,
    chunks: list[str],
    ref_audio_base64: str,
    ref_text: str,
    language: str,
    batch_size: int,
    max_new_tokens: int,
    output_dir: Path,
    basename: str,
) -> tuple[list[Path], str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = 1
    written: list[Path] = []
    logs: list[str] = []
    total_audio = 0.0
    total_processing = 0.0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        payload = {
            "text": batch,
            "ref_audio_base64": ref_audio_base64,
            "ref_text": ref_text,
            "language": language,
            "max_new_tokens": max_new_tokens,
        }
        response = requests.post(endpoint, json=payload, timeout=900, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        if not result.get("success", True):
            raise RuntimeError(result.get("error", "Unknown generation error"))

        audio_base64s = result.get("audio_base64s")
        durations = result.get("duration_seconds")
        if not isinstance(audio_base64s, list) or not isinstance(durations, list):
            raise RuntimeError("Server response missing batch outputs")

        processing_seconds = float(result.get("processing_seconds", 0.0))
        total_processing += processing_seconds

        for text_chunk, b64, duration in zip(batch, audio_base64s, durations):
            out = output_dir / f"{basename}_{chunk_idx}.wav"
            with open(out, "wb") as f:
                f.write(base64.b64decode(b64))
            d = float(duration)
            total_audio += d
            logs.append(f"chunk={chunk_idx} chars={len(text_chunk)} audio={d:.2f}s saved={out}")
            written.append(out)
            chunk_idx += 1

    inv_rtf = total_audio / total_processing if total_processing > 0 else 0.0
    logs.append(f"total_audio={total_audio:.2f}s total_processing={total_processing:.2f}s 1/RTF={inv_rtf:.3f}x")
    return written, "\n".join(logs)


def concat_chunks(intermediary_dir: Path, basename: str, output_format: str, bitrate: str) -> Path:
    output_dir = intermediary_dir.parent / "final_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "ogg" if output_format == "ogg" else "m4a"
    output_path = output_dir / f"{basename}_full.{ext}"

    concat_list = intermediary_dir / "concat_list.txt"
    build_concat_list(str(intermediary_dir), basename, str(concat_list))
    codec = "libopus" if output_format == "ogg" else "aac"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg concat failed")
    return output_path


def on_card_or_lang_change(card: str, language: str):
    target, cps, mult, batch = defaults_for_card_language(card, language)
    return target, cps, mult, batch


def get_ref_profile_text(profile: str | None):
    _, text_path = profile_paths(profile)
    if text_path is None:
        return ""
    return read_text(text_path)


def run_text_workflow(
    card: str,
    language: str,
    use_local: bool,
    endpoint_override: str,
    target_seconds: int,
    chars_per_second: int,
    max_chunk_multiplier: float,
    batch_size: int,
    max_new_tokens: int,
    profile: str,
    uploaded_audio: str | None,
    uploaded_ref_text_file: str | None,
    ref_text_input: str,
    text_input: str,
    txt_upload: str | None,
    output_name: str,
    output_format: str,
    bitrate: str,
):
    env_data = load_env(ROOT / ".env")
    endpoint = endpoint_for_card(card, use_local, endpoint_override, env_data)
    basename = sanitize_name(output_name, "text")

    full_text = (text_input or "").strip()
    if txt_upload:
        full_text = read_text(Path(txt_upload))
    if not full_text:
        raise gr.Error("Provide text in the textbox or upload a .txt file.")

    try:
        ref_audio_b64, ref_text, ref_source = resolve_reference(
            profile=profile,
            uploaded_audio=uploaded_audio,
            uploaded_ref_text_file=uploaded_ref_text_file,
            ref_text_input=ref_text_input,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    chunks = split_text(full_text, target_seconds, chars_per_second, max_chunk_multiplier)
    if not chunks:
        raise gr.Error("Text splitting produced no chunks.")

    run_root = OUTPUT_TEXT / basename
    intermediary = run_root / "intermediary_audio"
    try:
        written, log = clone_batch(
            endpoint=endpoint,
            chunks=chunks,
            ref_audio_base64=ref_audio_b64,
            ref_text=ref_text,
            language=language,
            batch_size=int(batch_size),
            max_new_tokens=int(max_new_tokens),
            output_dir=intermediary,
            basename=basename,
        )
        final_audio = concat_chunks(intermediary, basename, output_format, bitrate)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    details = [
        f"endpoint={endpoint}",
        f"card={card} language={language}",
        f"ref={ref_source}",
        f"chunks={len(written)}",
        f"saved={run_root}",
        log,
    ]
    return "\n".join(details), str(final_audio), str(final_audio)


def extract_book_action(book_file: str | None):
    if not book_file:
        raise gr.Error("Upload an EPUB or PDF first.")

    book_path = Path(book_file)
    book_name = sanitize_name(book_path.stem, "book")
    out_root = OUTPUT_BOOK / book_name
    out_root.mkdir(parents=True, exist_ok=True)

    data = extract_book(str(book_path))
    extract_path = out_root / "extract.json"
    summary_path = out_root / "summary.txt"
    intermediary_dir = out_root / "intermediary_audio"
    intermediary_dir.mkdir(parents=True, exist_ok=True)
    write_extract_json(data, str(extract_path))
    write_summary_txt(data["chapters"], str(summary_path))

    chapter_rows = [[c["index"], c["title"], c["word_count"], c["preview"]] for c in data["chapters"]]
    chapter_count = len(chapter_rows)
    max_ch = max(1, chapter_count)
    details = (
        f"book={book_name}\n"
        f"chapters={chapter_count}\n"
        f"extract={extract_path}\n"
        f"summary={summary_path}"
    )

    return (
        details,
        chapter_rows,
        gr.update(minimum=1, maximum=max_ch, value=1),
        gr.update(minimum=1, maximum=max_ch, value=max_ch),
        str(intermediary_dir),
        book_name,
        json.dumps(data),
    )


def run_book_generation(
    extract_json: str,
    card: str,
    language: str,
    use_local: bool,
    endpoint_override: str,
    target_seconds: int,
    chars_per_second: int,
    max_chunk_multiplier: float,
    batch_size: int,
    max_new_tokens: int,
    profile: str,
    uploaded_audio: str | None,
    uploaded_ref_text_file: str | None,
    ref_text_input: str,
    start_chapter: int,
    end_chapter: int,
    output_format: str,
    bitrate: str,
):
    if not extract_json:
        raise gr.Error("Run extract first.")

    data = json.loads(extract_json)
    chapters = data.get("chapters", [])
    if not chapters:
        raise gr.Error("No chapters found in extract.")

    start = max(1, int(start_chapter))
    end = min(int(end_chapter), len(chapters))
    selected = [c for c in chapters if start <= int(c["index"]) <= end]
    if not selected:
        raise gr.Error("Selected chapter range is empty.")

    env_data = load_env(ROOT / ".env")
    endpoint = endpoint_for_card(card, use_local, endpoint_override, env_data)

    try:
        ref_audio_b64, ref_text, ref_source = resolve_reference(
            profile=profile,
            uploaded_audio=uploaded_audio,
            uploaded_ref_text_file=uploaded_ref_text_file,
            ref_text_input=ref_text_input,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    chunks: list[str] = []
    for chapter in selected:
        chunks.extend(
            split_text(
                chapter["text"],
                target_seconds,
                chars_per_second,
                max_chunk_multiplier,
            )
        )
    if not chunks:
        raise gr.Error("No text chunks to generate.")

    book_name = sanitize_name(Path(data.get("source_path", "book")).stem, "book")
    run_root = OUTPUT_BOOK / book_name
    intermediary = run_root / "intermediary_audio"

    try:
        written, log = clone_batch(
            endpoint=endpoint,
            chunks=chunks,
            ref_audio_base64=ref_audio_b64,
            ref_text=ref_text,
            language=language,
            batch_size=int(batch_size),
            max_new_tokens=int(max_new_tokens),
            output_dir=intermediary,
            basename=book_name,
        )
        final_audio = concat_chunks(intermediary, book_name, output_format, bitrate)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    details = [
        f"book={book_name}",
        f"endpoint={endpoint}",
        f"chapters={start}..{end}",
        f"ref={ref_source}",
        f"chunks={len(written)}",
        f"saved={run_root}",
        log,
    ]
    return "\n".join(details), str(final_audio), str(final_audio)


def run_concat_only(intermediary_dir: str, basename: str, output_format: str, bitrate: str):
    if not intermediary_dir:
        raise gr.Error("Provide intermediary audio directory.")
    try:
        final = concat_chunks(Path(intermediary_dir), sanitize_name(basename, "book"), output_format, bitrate)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return str(final), str(final), f"concat_done={final}"


def build_ui() -> gr.Blocks:
    profiles = list_reference_profiles()
    default_profile = profiles[0] if profiles else None

    with gr.Blocks(title="Qwen3 TTS Workflow", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class='hero'>
              <h1>Qwen3 TTS Workflow Studio</h1>
              <p>Run text and book generation workflows from one clean dashboard, with card-aware defaults and quick overrides.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("<div class='section-title'>Run Configuration</div>")
                with gr.Group():
                    with gr.Row():
                        card = gr.Dropdown(choices=list(CARD_DEFAULTS.keys()), value="A10", label="Card")
                        language = gr.Dropdown(choices=["English", "French", "Auto"], value="English", label="Language")
                        use_local = gr.Checkbox(value=False, label="Use local endpoint")
                    endpoint_override = gr.Textbox(
                        label="Endpoint override",
                        placeholder="Optional full /generate URL (leave blank to use .env)",
                    )
                    with gr.Row():
                        target_seconds = gr.Number(value=60, precision=0, label="Target seconds")
                        chars_per_second = gr.Number(value=15, precision=0, label="Chars per second")
                        max_chunk_multiplier = gr.Number(value=1.05, precision=2, label="Max chunk multiplier")
                        batch_size = gr.Number(value=20, precision=0, label="Batch size")
                        max_new_tokens = gr.Number(value=2048, precision=0, label="Max new tokens")

            with gr.Column(scale=5):
                gr.Markdown("<div class='section-title'>Voice Reference</div>")
                with gr.Group():
                    profile = gr.Dropdown(
                        choices=profiles,
                        value=default_profile,
                        label="Reference profile",
                        info="Uses ref/<profile>/ref_audio.wav and ref_text.txt",
                        allow_custom_value=False,
                    )
                    with gr.Row():
                        uploaded_audio = gr.Audio(type="filepath", label="Upload reference audio")
                        uploaded_ref_text_file = gr.File(file_types=[".txt"], label="Upload reference text (.txt)")
                    ref_text_input = gr.Textbox(
                        label="Reference text",
                        lines=4,
                        value=get_ref_profile_text(default_profile),
                        placeholder="Type reference transcript here (this overrides uploaded/profile text)",
                    )

        with gr.Tabs():
            with gr.Tab("Workflow: client_batch_from_text"):
                gr.Markdown("### Text Source")
                with gr.Row():
                    with gr.Column(scale=7):
                        text_input = gr.Textbox(label="Write text directly", lines=14, placeholder="Paste or type your text...")
                    with gr.Column(scale=3):
                        txt_upload = gr.File(file_types=[".txt"], label="Upload .txt instead")
                        text_output_name = gr.Textbox(label="Output name", value="text_run")
                        text_format = gr.Dropdown(choices=["ogg", "m4a"], value="ogg", label="Final format")
                        text_bitrate = gr.Textbox(label="Bitrate", value="48k")
                        text_run_btn = gr.Button("Generate + Concat Text Audio", variant="primary")

                gr.Markdown("### Result")
                with gr.Row():
                    text_player = gr.Audio(label="Play result", type="filepath")
                    text_download = gr.File(label="Download result")
                text_log = gr.Textbox(label="Run log", lines=11, interactive=False)

            with gr.Tab("Workflow: client_batch_from_book"):
                gr.Markdown("### 1) Book Extract")
                with gr.Row():
                    book_file = gr.File(file_types=[".epub", ".pdf"], label="Book file (EPUB/PDF)")
                    extract_btn = gr.Button("Extract Book + Summary", variant="primary")
                extract_details = gr.Textbox(label="Extract details", lines=4, interactive=False)
                chapter_table = gr.Dataframe(
                    headers=["Index", "Title", "Words", "Preview"],
                    datatype=["number", "str", "number", "str"],
                    label="Book interactive chapter view",
                    wrap=True,
                )

                gr.Markdown("### 2) Chapter Selection + Generation")
                with gr.Row():
                    start_chapter = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Start chapter")
                    end_chapter = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="End chapter")
                    book_format = gr.Dropdown(choices=["ogg", "m4a"], value="ogg", label="Final format")
                    book_bitrate = gr.Textbox(label="Bitrate", value="48k")
                book_run_btn = gr.Button("Generate + Concat Book Audio", variant="primary")

                gr.Markdown("### 3) Result")
                with gr.Row():
                    book_player = gr.Audio(label="Play result", type="filepath")
                    book_download = gr.File(label="Download result")
                book_log = gr.Textbox(label="Run log", lines=10, interactive=False)

                gr.Markdown("### Audio Concat Only")
                with gr.Row():
                    concat_dir = gr.Textbox(
                        label="Intermediary directory",
                        placeholder="output/book/<name>/intermediary_audio",
                    )
                    concat_base = gr.Textbox(label="Basename", placeholder="book name")
                    concat_format = gr.Dropdown(choices=["ogg", "m4a"], value="ogg", label="Format")
                    concat_bitrate = gr.Textbox(label="Bitrate", value="48k")
                concat_btn = gr.Button("Concat Existing Chunks")
                with gr.Row():
                    concat_player = gr.Audio(label="Play concat result", type="filepath")
                    concat_download = gr.File(label="Download concat result")
                concat_log = gr.Textbox(label="Concat log", interactive=False)

                extract_json_state = gr.State("")

        card.change(
            on_card_or_lang_change,
            inputs=[card, language],
            outputs=[target_seconds, chars_per_second, max_chunk_multiplier, batch_size],
        )
        language.change(
            on_card_or_lang_change,
            inputs=[card, language],
            outputs=[target_seconds, chars_per_second, max_chunk_multiplier, batch_size],
        )
        profile.change(get_ref_profile_text, inputs=[profile], outputs=[ref_text_input])

        text_run_btn.click(
            run_text_workflow,
            inputs=[
                card,
                language,
                use_local,
                endpoint_override,
                target_seconds,
                chars_per_second,
                max_chunk_multiplier,
                batch_size,
                max_new_tokens,
                profile,
                uploaded_audio,
                uploaded_ref_text_file,
                ref_text_input,
                text_input,
                txt_upload,
                text_output_name,
                text_format,
                text_bitrate,
            ],
            outputs=[text_log, text_player, text_download],
        )

        extract_btn.click(
            extract_book_action,
            inputs=[book_file],
            outputs=[
                extract_details,
                chapter_table,
                start_chapter,
                end_chapter,
                concat_dir,
                concat_base,
                extract_json_state,
            ],
        )

        book_run_btn.click(
            run_book_generation,
            inputs=[
                extract_json_state,
                card,
                language,
                use_local,
                endpoint_override,
                target_seconds,
                chars_per_second,
                max_chunk_multiplier,
                batch_size,
                max_new_tokens,
                profile,
                uploaded_audio,
                uploaded_ref_text_file,
                ref_text_input,
                start_chapter,
                end_chapter,
                book_format,
                book_bitrate,
            ],
            outputs=[book_log, book_player, book_download],
        )

        concat_btn.click(
            run_concat_only,
            inputs=[concat_dir, concat_base, concat_format, concat_bitrate],
            outputs=[concat_player, concat_download, concat_log],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
