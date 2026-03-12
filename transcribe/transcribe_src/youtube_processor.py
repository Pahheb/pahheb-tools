"""YouTube transcription processor."""

import shutil
from pathlib import Path

from .config import Config
from .file_writer import write_transcripts
from .whisper import transcribe_audio
from .youtube_downloader import download_youtube_audio, get_youtube_video_info


def process_youtube_video(
    config: Config,
    url: str,
    verbose: bool = False,
) -> list[Path]:
    """
    Process a YouTube video.

    Args:
        config: Configuration object
        url: YouTube URL or video ID
        verbose: Enable verbose output

    Returns:
        List of written output files

    Raises:
        ValueError: If URL is invalid
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing YouTube URL: {url}")
        print(f"{'=' * 60}")

    if verbose:
        print("\n[1/4] Getting video info...")

    video_info = get_youtube_video_info(url, verbose=verbose)
    video_id = video_info["id"]
    video_title = video_info["title"]

    if verbose:
        print(f"  Title: {video_title}")
        print(f"  Video ID: {video_id}")

    output_base = output_dir / f"{video_id}_{sanitize_filename(video_title, 60)}"

    if verbose:
        print("\n[2/4] Downloading audio...")

    audio_path = download_youtube_audio(url, work_dir, verbose=verbose)

    if verbose:
        print(f"  Downloaded: {audio_path}")

    if verbose:
        print(f"\n[3/4] Transcribing with Whisper ({config.model_size})...")

    segments = transcribe_audio(
        audio_path=audio_path,
        model_size=config.model_size,
        language=config.language,
        device=config.device,
        compute_type=config.compute_type,
        vad_filter=config.vad,
        verbose=verbose,
    )

    if verbose:
        print(f"  Got {len(segments)} segments")

    if verbose:
        print("\n[4/4] Writing output files...")

    metadata = {
        "source": "youtube",
        "video_id": video_id,
        "video_title": video_title,
        "url": video_info["url"],
        "model": config.model_size,
    }
    if config.language:
        metadata["language"] = config.language

    output_files = write_transcripts(
        segments=segments,
        output_base=output_base,
        metadata=metadata,
        write_srt=config.srt,
        verbose=verbose,
    )

    if config.cleanup:
        if verbose:
            print(f"\nCleaning up work directory: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_files


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text for use in filenames."""
    import re

    sanitized = re.sub(r'[<>:"/\\|?*]', "_", text)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized.strip(". ")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(" ", 1)[0]
    if not sanitized:
        sanitized = "untitled"
    return sanitized
