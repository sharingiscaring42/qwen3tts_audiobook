import os
import subprocess
import sys
from pathlib import Path

from book_extract import extract_book, write_summary_txt


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


def prompt(text: str) -> str:
    return input(text).strip()


def print_summary_preview(summary_path: Path, lines: int = 50) -> None:
    if not summary_path.exists():
        return
    with open(summary_path, "r") as f:
        for _ in range(lines):
            line = f.readline()
            if not line:
                break
            print(line.rstrip("\n"))


def main() -> int:
    print("Interactive Book Prep")
    print("=" * 60)

    env = load_env()
    if not env.get("ENDPOINT_URL"):
        print("WARNING: ENDPOINT_URL not set in .env")
    if not env.get("SETTING_URL"):
        print("WARNING: SETTING_URL not set in .env")

    book_path_input = prompt("Path to EPUB/PDF: ")
    if not book_path_input:
        print("ERROR: No path provided")
        return 1
    book_path = str(Path(book_path_input).expanduser().resolve())
    if not os.path.exists(book_path):
        print(f"ERROR: File not found: {book_path}")
        return 1

    book_name = Path(book_path).stem
    output_root = Path("output/book") / book_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("\nExtracting and summarizing...")
    data = extract_book(book_path)
    summary_path = output_root / "summary.txt"
    write_summary_txt(data["chapters"], str(summary_path))
    print(f"Summary saved: {summary_path}")

    print("\nSummary preview (first 50 lines):")
    print("-" * 60)
    print_summary_preview(summary_path, lines=50)
    print("-" * 60)

    total_chapters = len(data["chapters"])
    print(f"Chapters detected: {total_chapters}")

    start_raw = prompt("Start chapter (default: 1): ")
    end_raw = prompt(f"End chapter (default: {total_chapters}): ")

    start_chapter = 1
    end_chapter = total_chapters
    if start_raw:
        start_chapter = max(1, int(start_raw))
    if end_raw:
        end_chapter = min(total_chapters, int(end_raw))

    print(f"Selected chapters: {start_chapter}..{end_chapter}")

    print("\nAudio format options: ogg (default), m4a")
    fmt = prompt("Output format (press Enter for ogg): ")
    if not fmt:
        fmt = "ogg"
    fmt = fmt.lower()
    if fmt not in {"ogg", "m4a"}:
        print("ERROR: Unsupported format. Choose ogg or m4a.")
        return 1

    bitrate_default = "48k" if fmt == "ogg" else "64k"
    bitrate = prompt(f"Bitrate (default: {bitrate_default}): ")
    if not bitrate:
        bitrate = bitrate_default

    batch_cmd = [
        sys.executable,
        "client/client_batch_from_book.py",
        "--input",
        book_path,
        "--start-chapter",
        str(start_chapter),
        "--end-chapter",
        str(end_chapter),
    ]
    print("\nRunning batch client:")
    print(" ".join(batch_cmd))
    batch_result = subprocess.run(batch_cmd)
    if batch_result.returncode != 0:
        print("ERROR: Batch client failed")
        return batch_result.returncode

    concat_cmd = [
        sys.executable,
        "client/book_audio_concat.py",
        "--input-dir",
        f"output/book/{book_name}/intermediary_audio",
        "--format",
        fmt,
        "--bitrate",
        bitrate,
    ]
    print("\nRunning audio concat:")
    print(" ".join(concat_cmd))
    concat_result = subprocess.run(concat_cmd)
    if concat_result.returncode != 0:
        print("ERROR: Audio concat failed")
        return concat_result.returncode

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
