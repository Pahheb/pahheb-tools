"""CLI argument parsing for transcribe tool."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe local audio/video files or YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local file transcription
  transcribe /path/to/audio.mp3
  transcribe video.mov --model medium --srt

  # YouTube transcription
  transcribe https://www.youtube.com/watch?v=VIDEO_ID --source youtube
  transcribe https://youtu.be/VIDEO_ID -s youtube --srt

  # With audio enhancement
  transcribe audio.mp3 --audio-enhance --srt

  # With RNNoise denoising
  transcribe audio.mp3 --denoise --denoise-model ./models/std.rnnn
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input file path or YouTube URL",
    )

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        parser.exit()

    args.output_dir = Path(args.output_dir).expanduser().resolve()

    return args
