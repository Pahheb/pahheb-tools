"""CLI argument parsing for yt-transcriber."""

import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos using official transcripts or Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yt-transcribe https://www.youtube.com/watch?v=dQw4w9WgXcQ
  yt-transcribe https://youtu.be/VIDEO_ID --model-size medium
  yt-transcribe https://www.youtube.com/playlist?list=PLAYLIST_ID
  yt-transcribe playlist_url --output-dir ./my_transcriptions --combine
  yt-transcribe input.txt --language en --sleep 3 --verbose
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="YouTube URL, playlist URL, video ID, or file with URLs",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./transcriptions",
        help="Output directory for transcription files (default: ./transcriptions)",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Create combined.txt with all transcriptions",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr'). Default: auto-detect",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small - balanced)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for transcription (default: auto)",
    )

    parser.add_argument(
        "--sleep",
        type=int,
        default=2,
        help="Seconds to wait between videos (default: 2)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information",
    )

    return parser.parse_args()
