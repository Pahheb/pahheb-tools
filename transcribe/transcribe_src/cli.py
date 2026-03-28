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
  # Single file
  transcribe /path/to/audio.mp3
  transcribe video.mov --model medium --srt

  # YouTube video
  transcribe https://www.youtube.com/watch?v=VIDEO_ID --source youtube

  # Multiple files (auto-detects local vs YouTube)
  transcribe audio.mp3 video.mov https://youtu.be/VIDEO_ID

  # Multiple files with audio enhancement
  transcribe audio1.mp3 audio2.mp3 --audio-enhance --srt
        """,
    )

    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input file paths or YouTube URLs (one or more)",
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["local", "youtube"],
        default=None,
        help="Force input source type for all inputs (default: auto-detect per input)",
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

    args.output_dir = Path(args.output_dir).expanduser().resolve()

    return args
