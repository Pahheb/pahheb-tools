"""Official YouTube transcript extraction using youtube-transcript-api."""

import re
from typing import Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


class NoTranscriptError(Exception):
    """Raised when no transcript is available for a video."""

    pass


def sanitize_text(text: str) -> str:
    """
    Clean up transcript text.

    Args:
        text: Raw transcript text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def get_official_transcript(video_id: str, language: Optional[str] = None) -> tuple[str, str]:
    """
    Get official transcript for a YouTube video.

    Args:
        video_id: YouTube video ID
        language: Optional language code (e.g., 'en', 'es')

    Returns:
        Tuple of (transcript_text, language_code)

    Raises:
        NoTranscriptError: If no transcript is available
    """
    try:
        # Try to get transcript in specified language or auto-detect
        languages = [language] if language else None
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=languages, preserve_formatting=False
        )

        # Combine segments into full text
        full_text = " ".join([entry["text"] for entry in transcript])

        # Get the actual language used
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        actual_language = "unknown"
        for t in transcript_list:
            if t.language_code.startswith(language if language else "en"):
                actual_language = t.language_code
                break

        return sanitize_text(full_text), actual_language

    except TranscriptsDisabled:
        raise NoTranscriptError(f"Transcripts are disabled for video {video_id}")

    except NoTranscriptFound:
        raise NoTranscriptError(f"No transcript found for video {video_id}")

    except VideoUnavailable:
        raise NoTranscriptError(f"Video {video_id} is unavailable")

    except Exception as e:
        raise NoTranscriptError(f"Error fetching transcript for {video_id}: {e}")


def get_available_transcripts(video_id: str) -> list[dict]:
    """
    Get list of available transcripts for a video.

    Args:
        video_id: YouTube video ID

    Returns:
        List of transcript info dicts with 'language', 'language_code', 'is_generated'
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return [
            {
                "language": t.language,
                "language_code": t.language_code,
                "is_generated": t.is_generated,
            }
            for t in transcript_list
        ]
    except Exception:
        return []
