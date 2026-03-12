"""File output handling for transcripts."""

import re
from pathlib import Path


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text for use in filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", text)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized.strip(". ")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(" ", 1)[0]
    if not sanitized:
        sanitized = "untitled"
    return sanitized


def format_srt_timestamp(seconds: float) -> str:
    """Format timestamp for SRT (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms %= 3600000
    mm = ms // 60000
    ms %= 60000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_transcript_txt(
    segments: list[dict],
    output_path: Path,
    metadata: dict | None = None,
) -> Path:
    """
    Write transcript as plain text.

    Args:
        segments: List of segment dictionaries
        output_path: Output file path
        metadata: Optional metadata dict (title, source, etc.)

    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if metadata:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 50 + "\n\n")

        for segment in segments:
            f.write(segment["text"].strip() + "\n")

    return output_path


def write_transcript_srt(
    segments: list[dict],
    output_path: Path,
) -> Path:
    """
    Write transcript as SRT subtitles.

    Args:
        segments: List of segment dictionaries with text, start, end
        output_path: Output file path

    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_timestamp(segment['start'])} --> "
                f"{format_srt_timestamp(segment['end'])}\n"
            )
            f.write(segment["text"].strip() + "\n\n")

    return output_path


def write_transcripts(
    segments: list[dict],
    output_base: Path,
    metadata: dict | None = None,
    write_srt: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """
    Write transcript to files.

    Args:
        segments: List of segment dictionaries
        output_base: Base output path (without extension)
        metadata: Optional metadata dict
        write_srt: Whether to write SRT file
        verbose: Enable verbose output

    Returns:
        List of written file paths
    """
    written = []

    txt_path = output_base.with_suffix(".txt")
    write_transcript_txt(segments, txt_path, metadata)
    written.append(txt_path)
    if verbose:
        print(f"Written TXT: {txt_path}")

    if write_srt:
        srt_path = output_base.with_suffix(".srt")
        write_transcript_srt(segments, srt_path)
        written.append(srt_path)
        if verbose:
            print(f"Written SRT: {srt_path}")

    return written
