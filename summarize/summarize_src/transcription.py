"""Transcription utilities for the summarize tool."""

import re
import subprocess
from pathlib import Path

from .config import Config


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def find_transcription_file(
    file_path: str | Path, transcribe_output_dir: Path
) -> Path | None:
    """Find an existing transcription file for a given input path/URL."""
    if isinstance(file_path, Path):
        stem = file_path.stem
        txt_path = transcribe_output_dir / f"{stem}.txt"
        if txt_path.exists():
            return txt_path
        matching = list(transcribe_output_dir.glob(f"{stem}*.txt"))
        if matching:
            return max(matching, key=lambda p: p.stat().st_mtime)
    else:
        video_id = extract_video_id(file_path)
        if video_id:
            matching = list(transcribe_output_dir.glob(f"{video_id}*.txt"))
            if matching:
                return max(matching, key=lambda p: p.stat().st_mtime)

        stem = file_path.split("/")[-1].split("\\")[-1]
        if "?" in stem:
            stem = stem.split("?")[0]

        if not stem.endswith(".txt"):
            txt_path = transcribe_output_dir / f"{stem}.txt"
            if txt_path.exists():
                return txt_path

        matching = list(transcribe_output_dir.glob(f"{stem}*.txt"))
        if matching:
            return max(matching, key=lambda p: p.stat().st_mtime)

        if "." in stem:
            base_stem = stem.rsplit(".", 1)[0]
            txt_path = transcribe_output_dir / f"{base_stem}.txt"
            if txt_path.exists():
                return txt_path
            matching = list(transcribe_output_dir.glob(f"{base_stem}*.txt"))
            if matching:
                return max(matching, key=lambda p: p.stat().st_mtime)

    return None


def transcribe_file(
    file_path: str | Path, config: Config, verbose: bool = False
) -> Path:
    """Transcribe a video/audio file using the transcribe tool."""
    config.transcribe_output_dir.mkdir(parents=True, exist_ok=True)

    input_str = str(file_path)
    transcribe_cmd = ["transcribe", input_str]
    transcribe_cmd.extend(config.build_transcribe_args())

    if verbose:
        print(f"Transcribing: {input_str}")
        print(f"Transcribe output dir: {config.transcribe_output_dir}")

    try:
        result = subprocess.run(  # noqa: S603
            transcribe_cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if verbose:
            print(result.stdout)

        txt_file = find_transcription_file(input_str, config.transcribe_output_dir)
        if txt_file:
            return txt_file

        raise FileNotFoundError(f"Could not find transcription output for {file_path}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Transcription failed: {e.stderr}") from e


def read_transcription(file_path: Path | str) -> str:
    """Read transcription from a text file."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    content = file_path.read_text(encoding="utf-8")

    lines = content.split("\n")
    transcript_lines = []
    in_transcript = False

    metadata_prefixes = (
        "source:",
        "video_id:",
        "video_title:",
        "url:",
        "model:",
        "language:",
        "duration:",
    )

    for line in lines:
        line_lower = line.lower().strip()
        stripped = line.strip()

        if stripped.startswith("---") and stripped.endswith("---"):
            if in_transcript:
                break
            in_transcript = True
            continue
        if in_transcript:
            transcript_lines.append(line)
        elif stripped.startswith(metadata_prefixes):
            continue
        elif stripped and not any(
            k in line_lower
            for k in ["video", "title", "language", "source", "duration"]
        ):
            in_transcript = True
            transcript_lines.append(line)

    return "\n".join(transcript_lines).strip() or content
