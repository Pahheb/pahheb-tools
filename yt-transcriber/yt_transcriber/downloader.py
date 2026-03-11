"""Audio download from YouTube using pytube."""

from pathlib import Path
from typing import Optional

from pytube import Stream, YouTube


class DownloadError(Exception):
    """Raised when audio download fails."""

    pass


def download_audio(video_id: str, output_dir: Path) -> Path:
    """
    Download audio stream from YouTube video.

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

    try:
        yt = YouTube(url)

        # Get best audio stream
        audio_stream: Optional[Stream] = None

        # Prefer adaptive streams (audio-only)
        adaptive_streams = yt.streams.filter(adaptive=True, only_audio=True)
        if adaptive_streams:
            audio_stream = adaptive_streams.order_by("abr").last()
        else:
            # Fallback to progressive streams with audio
            progressive_streams = yt.streams.filter(progressive=True, only_audio=False)
            if progressive_streams:
                audio_stream = progressive_streams.order_by("abr").last()

        if not audio_stream:
            raise DownloadError(f"No audio stream available for video {video_id}")

        # Create temporary file path
        # Use .m4a for adaptive streams, .mp4 for progressive
        ext = "m4a" if audio_stream.adaptive else "mp4"
        output_path = output_dir / f"{video_id}_audio.{ext}"

        # Download the audio
        audio_stream.download(output_path=str(output_dir), filename=f"{video_id}_audio.{ext}")

        return output_path

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
