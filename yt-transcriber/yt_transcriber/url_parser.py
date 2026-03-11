"""URL parsing and validation for YouTube videos and playlists."""

import re
from pathlib import Path
from typing import Optional


# YouTube URL patterns
VIDEO_PATTERNS = [
    # https://www.youtube.com/watch?v=VIDEO_ID
    r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    # https://youtube.com/watch?v=VIDEO_ID
    r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    # https://youtu.be/VIDEO_ID
    r"youtu\.be/([a-zA-Z0-9_-]{11})",
    # https://m.youtube.com/watch?v=VIDEO_ID
    r"m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
]

PLAYLIST_PATTERNS = [
    # https://www.youtube.com/playlist?list=PLAYLIST_ID
    r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)",
    # https://youtube.com/playlist?list=PLAYLIST_ID
    r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)",
]

VIDEO_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{11}")
PLAYLIST_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{34}")


def is_youtube_url(url: str) -> bool:
    """Check if the string is a YouTube URL."""
    url_lower = url.lower().strip()
    return "youtube.com" in url_lower or "youtu.be" in url_lower


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.

    Args:
        url: YouTube URL string

    Returns:
        Video ID if found, None otherwise
    """
    url_stripped = url.strip()

    # Direct video ID (11 characters)
    if VIDEO_ID_PATTERN.fullmatch(url_stripped):
        return url_stripped

    # Check URL patterns (case-insensitive search but preserve original case)
    for pattern in VIDEO_PATTERNS:
        match = re.search(pattern, url_stripped, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def is_playlist_url(url: str) -> bool:
    """Check if the URL is a YouTube playlist URL."""
    url_lower = url.lower().strip()

    # Check playlist patterns
    for pattern in PLAYLIST_PATTERNS:
        if re.search(pattern, url_lower):
            return True

    # Also check for list parameter in watch URLs
    if "youtube.com" in url_lower and "list=" in url_lower:
        return True

    return False


def extract_playlist_id(url: str) -> Optional[str]:
    """
    Extract playlist ID from YouTube playlist URL.

    Args:
        url: YouTube playlist URL string

    Returns:
        Playlist ID if found, None otherwise
    """
    url_stripped = url.strip()

    # Check playlist patterns (case-insensitive search but preserve original case)
    for pattern in PLAYLIST_PATTERNS:
        match = re.search(pattern, url_stripped, re.IGNORECASE)
        if match:
            return match.group(1)

    # Check for list parameter in watch URLs
    list_match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url_stripped, re.IGNORECASE)
    if list_match:
        return list_match.group(1)

    return None


def is_file_with_urls(path: str) -> bool:
    """
    Check if the path points to a file containing YouTube URLs.

    Args:
        path: File path string

    Returns:
        True if file exists and contains YouTube URLs
    """
    file_path = Path(path)

    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    # Read first few lines to check for URLs
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in range(10):  # Check first 10 lines
                line = f.readline()
                if not line:
                    break
                if is_youtube_url(line.strip()):
                    return True
    except (IOError, UnicodeDecodeError):
        return False

    return False


def parse_input(input_str: str) -> tuple[str, str, Optional[str]]:
    """
    Parse input string to determine type and extract identifiers.

    Args:
        input_str: URL, video ID, playlist ID, or file path

    Returns:
        Tuple of (type, identifier, playlist_id_or_none)
        Type can be: 'video', 'playlist', 'file'
    """
    input_str = input_str.strip()

    # Check if it's a file path
    if is_file_with_urls(input_str):
        return "file", input_str, None

    # Check if it's a playlist URL
    if is_playlist_url(input_str):
        playlist_id = extract_playlist_id(input_str)
        if playlist_id:
            return "playlist", playlist_id, None

    # Check if it's a YouTube URL
    if is_youtube_url(input_str):
        video_id = extract_video_id(input_str)
        if video_id:
            return "video", video_id, None

    # Check if it's a direct video ID
    if VIDEO_ID_PATTERN.fullmatch(input_str):
        return "video", input_str, None

    # Check if it's a direct playlist ID
    if PLAYLIST_ID_PATTERN.fullmatch(input_str):
        return "playlist", input_str, None

    raise ValueError(f"Invalid input: {input_str}")


def read_urls_from_file(file_path: str) -> list[str]:
    """
    Read YouTube URLs from a file.

    Args:
        file_path: Path to file containing URLs

    Returns:
        List of YouTube URLs
    """
    urls = []
    path = Path(file_path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and is_youtube_url(line):
                    urls.append(line)
    except (IOError, UnicodeDecodeError) as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

    return urls
