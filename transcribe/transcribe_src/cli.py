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

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["local", "youtube"],
        default="local",
        help="Input source type (default: local)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./transcriptions",
        help="Output directory (default: ./transcriptions)",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es'). Default: auto-detect",
    )

    parser.add_argument(
        "--model",
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
        help="Device for transcription (default: auto)",
    )

    parser.add_argument(
        "--compute",
        type=str,
        default=None,
        help="Compute type (int8, int8_float16, float16, float32)",
    )

    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply RNNoise denoising",
    )

    parser.add_argument(
        "--denoise-model",
        type=str,
        default=None,
        help="Path to RNNoise model file (default: auto-search in ./models)",
    )

    parser.add_argument(
        "--vad",
        action="store_true",
        help="Apply voice activity detection filtering",
    )

    parser.add_argument(
        "--audio-enhance",
        action="store_true",
        help="Apply audio enhancement filters (highpass/lowpass/compand/loudnorm)",
    )

    parser.add_argument(
        "--srt",
        action="store_true",
        help="Generate SRT subtitle file",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate files after transcription",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        parser.exit()

    args.output_dir = Path(args.output_dir).expanduser().resolve()

    return args
