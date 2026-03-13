"""CLI argument parsing for summarize tool."""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="summarize",
        description="Summarize video/audio transcriptions using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize pre-transcribed files (per-video summaries)
  summarize transcription1.txt transcription2.txt

  # Combine summaries into one file (post-processing merge)
  summarize ./transcriptions/*.txt --combine

  # Unified summary: merge transcriptions first, then summarize once
  summarize ./transcriptions/*.txt --unified

  # Skip combined files (useful when transcribe used --combine)
  summarize ./transcriptions/*.txt --unified --skip-combined

  # Transcribe then summarize (with transcribe options)
  summarize video.mp4 --transcribe
  summarize audio.wav --transcribe --combine
  summarize video.mp4 --transcribe --transcribe-model medium --transcribe-srt
  summarize https://youtube.com/watch?v=ID --transcribe --transcribe-source youtube

  # Use different AI provider
  summarize file.txt --provider ollama --model llama3.2
  summarize file.txt --provider huggingface

  # Control output
  summarize file.txt --output-dir ./summaries --output-format md
  summarize file.txt --summary-length brief
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="*",
        help="Input files (transcribed .txt files) or video/audio files (with --transcribe)",
    )

    parser.add_argument(
        "--transcribe",
        "-t",
        action="store_true",
        help="First transcribe video/audio files before summarizing",
    )

    transcribe_group = parser.add_argument_group(
        "transcribe options",
        "Options for transcribing video/audio (only used with --transcribe)",
    )

    transcribe_group.add_argument(
        "--transcribe-source",
        "-ts",
        type=str,
        choices=["local", "youtube"],
        default="local",
        help="Input source type (default: local)",
    )

    transcribe_group.add_argument(
        "--transcribe-language",
        "-tl",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'en', 'es'). Default: auto-detect",
    )

    transcribe_group.add_argument(
        "--transcribe-model",
        "-tm",
        type=str,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="small",
        help="Whisper model size (default: small - balanced)",
    )

    transcribe_group.add_argument(
        "--transcribe-device",
        "-td",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for transcription (default: auto)",
    )

    transcribe_group.add_argument(
        "--transcribe-compute",
        type=str,
        default=None,
        help="Compute type (int8, int8_float16, float16, float32)",
    )

    transcribe_group.add_argument(
        "--transcribe-denoise",
        action="store_true",
        help="Apply RNNoise denoising",
    )

    transcribe_group.add_argument(
        "--transcribe-vad",
        action="store_true",
        help="Apply voice activity detection filtering",
    )

    transcribe_group.add_argument(
        "--transcribe-audio-enhance",
        action="store_true",
        help="Apply audio enhancement filters",
    )

    transcribe_group.add_argument(
        "--transcribe-srt",
        action="store_true",
        help="Generate SRT subtitle file",
    )

    transcribe_group.add_argument(
        "--transcribe-cleanup",
        action="store_true",
        help="Delete intermediate files after transcription",
    )

    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["ollama", "huggingface", "watsonx"],
        default="ollama",
        help="AI provider for summarization (default: ollama)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model to use for summarization (default: provider-specific)",
    )

    parser.add_argument(
        "--combine",
        "-c",
        action="store_true",
        help="Combine all per-video summaries into one output file (post-processing merge)",
    )

    parser.add_argument(
        "--unified",
        "-u",
        action="store_true",
        help="Merge all transcriptions into one and create a single unified summary. "
        "Useful when you want one summary for multiple videos. "
        "Note: If combined*.txt file is detected in inputs, you may want to use --skip-combined",
    )

    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip any combined*.txt files in the input. "
        "Useful when the transcribe tool was used with --combine, "
        "to avoid duplicating content in your summary",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./summaries",
        help="Output directory for summaries (default: ./summaries)",
    )

    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["txt", "md", "json"],
        default="txt",
        help="Output format (default: txt)",
    )

    parser.add_argument(
        "--summary-length",
        "-l",
        type=str,
        choices=["brief", "standard", "detailed"],
        default="standard",
        help="Length of summary (default: standard)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(0)

    args.output_dir = Path(args.output_dir).expanduser().resolve()

    args.input_files = []
    for input_path in args.input:
        path = Path(input_path)
        is_url = input_path.startswith(
            ("http://", "https://", "youtube.com", "youtu.be")
        )

        if is_url and args.transcribe and args.transcribe_source == "youtube":
            args.input_files.append(path)
        elif path.exists():
            args.input_files.append(path)
        elif args.transcribe and args.transcribe_source == "youtube":
            args.input_files.append(path)
        else:
            print(
                f"Warning: Input file not found: {input_path}",
                file=sys.stderr,
            )

    if not args.input_files:
        print("Error: No valid input files provided", file=sys.stderr)
        parser.exit(1)

    return args
