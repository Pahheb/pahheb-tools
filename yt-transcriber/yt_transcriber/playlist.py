"""Playlist processing with rate limiting."""

import asyncio
import time
from pathlib import Path
from typing import Optional

from pytube import Playlist


class PlaylistError(Exception):
    """Raised when playlist processing fails."""

    pass


def get_playlist_videos(playlist_url: str) -> list[dict]:
    """
    Extract all video IDs and titles from a playlist.

    Args:
        playlist_url: YouTube playlist URL

    Returns:
        List of dicts with 'video_id' and 'title'

    Raises:
        PlaylistError: If playlist cannot be accessed
    """
    try:
        playlist = Playlist(playlist_url)
        videos = []

        # Get all video URLs from playlist
        for url in playlist.video_urls:
            # Extract video ID from URL
            video_id = url.split("v=")[1].split("&")[0] if "v=" in url else None
            if video_id:
                videos.append(
                    {
                        "video_id": video_id,
                        "url": url,
                        "title": "",  # Will be fetched later
                    }
                )

        return videos

    except Exception as e:
        raise PlaylistError(f"Failed to access playlist: {e}")


async def process_playlist_with_retry(
    playlist_url: str,
    process_video_func,
    sleep_delay: int = 2,
    max_retries: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Process all videos in playlist with rate limiting and retry logic.

    Args:
        playlist_url: YouTube playlist URL
        process_video_func: Function to process each video
        sleep_delay: Delay between videos in seconds
        max_retries: Maximum retry attempts for failed videos

    Returns:
        Tuple of (successful_videos, failed_videos)
    """
    try:
        playlist = Playlist(playlist_url)
    except Exception as e:
        raise PlaylistError(f"Failed to access playlist: {e}")

    # Get all video URLs
    video_urls = list(playlist.video_urls)
    total_videos = len(video_urls)

    if total_videos == 0:
        return [], []

    successful = []
    failed = []

    for index, url in enumerate(video_urls, 1):
        # Extract video ID
        video_id = url.split("v=")[1].split("&")[0] if "v=" in url else None
        if not video_id:
            failed.append({"video_id": "unknown", "url": url, "reason": "Invalid URL"})
            continue

        # Retry logic
        for attempt in range(max_retries):
            try:
                # Process the video
                result = await process_video_func(video_id, index, total_videos)

                if result:
                    successful.append({"video_id": video_id, "url": url, "result": result})
                else:
                    failed.append(
                        {"video_id": video_id, "url": url, "reason": "Processing returned None"}
                    )

                break  # Success, exit retry loop

            except Exception as e:
                if attempt == max_retries - 1:
                    failed.append({"video_id": video_id, "url": url, "reason": str(e)})
                else:
                    # Exponential backoff for rate limiting
                    delay = sleep_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

        # Sleep between videos (unless last video)
        if index < total_videos:
            await asyncio.sleep(sleep_delay)

    return successful, failed
