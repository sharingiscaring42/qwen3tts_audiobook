import argparse
import argparse
import subprocess
from pathlib import Path

TRIM_TAIL_SECONDS = 0.1


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_concat_list(input_dir: str, basename: str, output_path: str) -> int:
    files = []
    for path in sorted(Path(input_dir).glob(f"{basename}_*.wav"), key=lambda p: int(p.stem.split("_")[-1])):
        files.append(path)
    if not files:
        raise FileNotFoundError("No WAV files found to concatenate")

    with open(output_path, "w") as f:
        for path in files:
            duration = probe_duration(path)
            outpoint = max(0.0, duration - TRIM_TAIL_SECONDS)
            f.write(f"file '{path.resolve()}'\n")
            f.write(f"outpoint {outpoint:.6f}\n")

    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate WAV chunks and transcode")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--basename")
    parser.add_argument("--output")
    parser.add_argument("--bitrate", default="48k")
    parser.add_argument("--format", choices=["ogg", "m4a"], default="ogg")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    book_name = input_dir.parent.name if input_dir.name == "intermediary_audio" else input_dir.name
    basename = args.basename or book_name

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_dir.parent / "final_audio" if input_dir.name == "intermediary_audio" else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ext = "ogg" if args.format == "ogg" else "m4a"
        output_path = output_dir / f"{basename}_full.{output_ext}"

    concat_list = Path(args.input_dir) / "concat_list.txt"
    count = build_concat_list(args.input_dir, basename, str(concat_list))
    print(f"Found {count} chunks")
    print(f"Trimming last {TRIM_TAIL_SECONDS:.3f}s of each chunk")

    if args.format == "ogg":
        codec = "libopus"
    else:
        codec = "aac"

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
        args.bitrate,
        str(output_path),
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
