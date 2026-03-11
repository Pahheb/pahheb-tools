"""Playlist processing with rate limiting using yt-dlp."""

import asyncio
import subprocess
import json


class PlaylistError(Exception):
    """Raised when playlist processing fails."""

    pass


def get_playlist_videos(playlist_url: str) -> list[dict]:
    """
    Extract all video IDs and titles from a playlist using yt-dlp.

    Args:
        playlist_url: YouTube playlist URL

    Returns:
        List of dicts with 'video_id' and 'title'

    Raises:
        PlaylistError: If playlist cannot be accessed
    """
    try:
        # Use yt-dlp to get playlist info in JSON format
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print-json",
            "--no-warnings",
            playlist_url,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise PlaylistError(f"yt-dlp failed: {result.stderr}")

        # Parse JSON output - yt-dlp outputs one JSON object per line
        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    info = json.loads(line)
                    if info.get("_type") == "url":
                        video_id = info.get("id") or info.get("url")
                        if video_id:
                            videos.append(
                                {
                                    "video_id": video_id,
                                    "url": info.get("url", ""),
                                    "title": info.get("title", ""),
                                }
                            )
                except json.JSONDecodeError:
                    continue

        return videos

    except subprocess.TimeoutExpired:
        raise PlaylistError(f"Playlist extraction timed out for {playlist_url}")
    except FileNotFoundError:
        raise PlaylistError("yt-dlp not found. Please install it with: pip install yt-dlp")
    except Exception as e:
        raise PlaylistError(f"Failed to access playlist: {e}") from e


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
        videos = get_playlist_videos(playlist_url)
    except PlaylistError as e:
        raise e

    total_videos = len(videos)

    if total_videos == 0:
        return [], []

    successful = []
    failed = []

    for index, video_info in enumerate(videos, 1):
        video_id = video_info.get("video_id")
        url = video_info.get("url", "")
        title = video_info.get("title", "")

        if not video_id:
            failed.append({"video_id": "unknown", "url": url, "reason": "Invalid video ID"})
            continue

        # Retry logic
        for attempt in range(max_retries):
            try:
                # Process the video
                result = await process_video_func(video_id, index, total_videos)

                if result:
                    successful.append(
                        {
                            "video_id": video_id,
                            "url": url,
                            "title": title,
                            "result": result,
                        }
                    )
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
