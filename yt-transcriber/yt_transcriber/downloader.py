"""Audio download from YouTube using yt-dlp."""

import subprocess
from pathlib import Path


class DownloadError(Exception):
    """Raised when audio download fails."""

    pass


def download_audio(video_id: str, output_dir: Path) -> Path:
    """
    Download audio stream from YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save downloaded audio

    Returns:
        Path to downloaded audio file

    Raises:
        DownloadError: If download fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = output_dir / f"{video_id}_audio.wav"

    try:
        # Use yt-dlp to download audio as WAV
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",  # Best quality
            "--output",
            str(output_path.with_suffix(".%(ext)s")),
            url,
        ]

        # Run yt-dlp
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise DownloadError(f"yt-dlp failed: {result.stderr}")

        # yt-dlp adds the extension, so we check if the .wav file exists
        if not output_path.exists():
            # Try finding the actual file (yt-dlp might have changed the name)
            possible_files = list(output_dir.glob(f"{video_id}_audio.*"))
            if possible_files:
                output_path = possible_files[0]
            else:
                raise DownloadError(f"Downloaded file not found: {output_path}")

        return output_path

    except subprocess.TimeoutExpired:
        raise DownloadError(f"Download timed out for video {video_id}")
    except FileNotFoundError:
        raise DownloadError("yt-dlp not found. Please install it with: pip install yt-dlp")
    except Exception as e:
        raise DownloadError(f"Failed to download audio for {video_id}: {e}") from e


def cleanup_audio_file(audio_path: Path) -> None:
    """
    Remove downloaded audio file.

    Args:
        audio_path: Path to audio file to remove
    """
    try:
        if audio_path.exists():
            audio_path.unlink()
    except Exception:
        # Ignore cleanup errors
        pass
