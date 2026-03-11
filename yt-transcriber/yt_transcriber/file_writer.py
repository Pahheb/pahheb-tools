"""File output handling for transcripts."""

import re
from pathlib import Path
from typing import Optional


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for use in filenames.

    Args:
        text: Text to sanitize
        max_length: Maximum filename length (excluding extension)

    Returns:
        Sanitized filename-safe text
    """
    # Replace special characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", text)
    # Replace multiple spaces with single space
    sanitized = re.sub(r"\s+", " ", sanitized)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(" ", 1)[0]
    # Fallback if empty after sanitization
    if not sanitized:
        sanitized = "untitled"
    return sanitized


def write_transcript(
    video_id: str,
    title: str,
    transcript: str,
    output_dir: Path,
    source: str,
    language: Optional[str] = None,
) -> Path:
    """
    Write transcript to a text file.

    Args:
        video_id: YouTube video ID
        title: Video title
        transcript: Transcript text
        output_dir: Directory to save file
        source: Source of transcript ('official' or 'whisper')
        language: Optional language code

    Returns:
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize title for filename
    safe_title = sanitize_filename(title, max_length=80)

    # Create filename: {video_id}_{title}.txt
    filename = f"{video_id}_{safe_title}.txt"
    file_path = output_dir / filename

    # Write transcript with metadata header
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Title: {title}\n")
        if language:
            f.write(f"Language: {language}\n")
        f.write(f"Source: {source}\n")
        f.write("-" * 50 + "\n\n")
        f.write(transcript)

    return file_path


def combine_transcripts(
    file_paths: list[Path],
    output_dir: Path,
    output_filename: str = "combined.txt",
) -> Path:
    """
    Combine multiple transcript files into one.

    Args:
        file_paths: List of paths to transcript files
        output_dir: Directory to save combined file
        output_filename: Name of combined file

    Returns:
        Path to combined file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / output_filename

    with open(combined_path, "w", encoding="utf-8") as f:
        for file_path in file_paths:
            if not file_path.exists():
                continue

            # Extract video info from filename or read header
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Try to parse header
            video_id = "unknown"
            title = "unknown"
            for line in lines[:5]:  # Check first 5 lines
                if line.startswith("Video ID:"):
                    video_id = line.split(":", 1)[1].strip()
                elif line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip()

            # Write separator
            f.write(f"\n{'=' * 70}\n")
            f.write(f"=== {title} ({video_id}) ===\n")
            f.write(f"{'=' * 70}\n\n")

            # Write transcript content (skip header)
            transcript_start = False
            for line in lines:
                if line.startswith("-" * 50):
                    transcript_start = True
                    continue
                if transcript_start:
                    f.write(line + "\n")

    return combined_path


def get_output_filename(video_id: str, title: str) -> str:
    """
    Generate output filename for a video.

    Args:
        video_id: YouTube video ID
        title: Video title

    Returns:
        Sanitized filename
    """
    safe_title = sanitize_filename(title, max_length=80)
    return f"{video_id}_{safe_title}.txt"
