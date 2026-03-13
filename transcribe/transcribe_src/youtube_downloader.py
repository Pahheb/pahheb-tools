"""YouTube audio download using yt-dlp."""

import json
import subprocess
from pathlib import Path


class YouTubeDownloadError(Exception):
    """Raised when YouTube download fails."""

    pass


def download_youtube_audio(
    url: str,
    output_dir: Path,
    verbose: bool = False,
) -> Path:
    """
    Download audio from YouTube video/playlist.

    Args:
        url: YouTube URL or video ID
        output_dir: Directory to save audio
        verbose: Enable verbose output

    Returns:
        Path to downloaded audio file

    Raises:
        YouTubeDownloadError: If download fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = set(output_dir.glob("*.wav"))

    output_template = str(output_dir / "%(title)s_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--output",
        output_template,
    ]

    if not verbose:
        cmd.append("--quiet")

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise YouTubeDownloadError(f"yt-dlp failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise YouTubeDownloadError("Download timed out")
    except FileNotFoundError:
        raise YouTubeDownloadError("yt-dlp not found. Install with: pip install yt-dlp")

    audio_files = list(output_dir.glob("*.wav"))
    new_files = [f for f in audio_files if f not in existing_files]

    if not new_files:
        if audio_files:
            return max(audio_files, key=lambda p: p.stat().st_mtime)
        raise YouTubeDownloadError("No audio file was downloaded")

    if verbose:
        print(f"Downloaded: {new_files[0]}")

    return max(new_files, key=lambda p: p.stat().st_mtime)


def get_youtube_video_info(
    url: str,
    verbose: bool = False,
) -> dict:
    """
    Get video information from YouTube.

    Args:
        url: YouTube URL or video ID
        verbose: Enable verbose output

    Returns:
        Dictionary with video info

    Raises:
        YouTubeDownloadError: If info retrieval fails
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
    ]

    if not verbose:
        cmd.append("--quiet")

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise YouTubeDownloadError(f"yt-dlp failed: {result.stderr}")

        info = json.loads(result.stdout)

        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "url": info.get("webpage_url"),
            "duration": info.get("duration"),
        }

    except subprocess.TimeoutExpired:
        raise YouTubeDownloadError("Info retrieval timed out")
    except json.JSONDecodeError as e:
        raise YouTubeDownloadError(f"Failed to parse video info: {e}")
