"""CLI entry point for summarize tool."""

import re
import subprocess
import sys
from pathlib import Path

from .cli import parse_args
from .config import Config
from .file_writer import (
    write_summary_json,
    write_summary_md,
    write_summary_txt,
)
from .summarizer import (
    ProviderNotAvailableError,
    get_provider,
    SummarizerError,
)


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def find_transcription_file(
    file_path: str | Path, transcribe_output_dir: Path
) -> Path | None:
    """Find an existing transcription file for a given input path/URL."""
    if isinstance(file_path, Path):
        stem = file_path.stem
        txt_path = transcribe_output_dir / f"{stem}.txt"
        if txt_path.exists():
            return txt_path
        matching = list(transcribe_output_dir.glob(f"{stem}*.txt"))
        if matching:
            return max(matching, key=lambda p: p.stat().st_mtime)
    else:
        video_id = extract_video_id(file_path)
        if video_id:
            matching = list(transcribe_output_dir.glob(f"{video_id}*.txt"))
            if matching:
                return max(matching, key=lambda p: p.stat().st_mtime)

        stem = file_path.split("/")[-1].split("\\")[-1]
        if "?" in stem:
            stem = stem.split("?")[0]

        if not stem.endswith(".txt"):
            txt_path = transcribe_output_dir / f"{stem}.txt"
            if txt_path.exists():
                return txt_path

        matching = list(transcribe_output_dir.glob(f"{stem}*.txt"))
        if matching:
            return max(matching, key=lambda p: p.stat().st_mtime)

        if "." in stem:
            base_stem = stem.rsplit(".", 1)[0]
            txt_path = transcribe_output_dir / f"{base_stem}.txt"
            if txt_path.exists():
                return txt_path
            matching = list(transcribe_output_dir.glob(f"{base_stem}*.txt"))
            if matching:
                return max(matching, key=lambda p: p.stat().st_mtime)

    return None


def transcribe_file(
    file_path: str | Path, config: Config, verbose: bool = False
) -> Path:
    """Transcribe a video/audio file using the transcribe tool."""
    config.transcribe_output_dir.mkdir(parents=True, exist_ok=True)

    input_str = str(file_path)
    transcribe_cmd = ["transcribe", input_str]
    transcribe_cmd.extend(config.build_transcribe_args())

    if verbose:
        print(f"Transcribing: {input_str}")
        print(f"Transcribe output dir: {config.transcribe_output_dir}")

    try:
        result = subprocess.run(
            transcribe_cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if verbose:
            print(result.stdout)

        if isinstance(file_path, Path):
            stem = file_path.stem
        else:
            stem = file_path.split("/")[-1].split("\\")[-1]
            if "?" in stem:
                stem = stem.split("?")[0]

        txt_file = config.transcribe_output_dir / f"{stem}.txt"
        if txt_file.exists():
            return txt_file

        matching_files = list(config.transcribe_output_dir.glob(f"{stem}*.txt"))
        if matching_files:
            return max(matching_files, key=lambda p: p.stat().st_mtime)

        raise FileNotFoundError(f"Could not find transcription output for {file_path}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Transcription failed: {e.stderr}")


def read_transcription(file_path: Path) -> str:
    """Read transcription from a text file."""
    content = file_path.read_text(encoding="utf-8")

    lines = content.split("\n")
    transcript_lines = []
    in_transcript = False

    for line in lines:
        if "---" in line and in_transcript:
            break
        if in_transcript:
            transcript_lines.append(line)
        elif line.strip() and not any(
            k in line.lower()
            for k in ["video", "title", "language", "source", "duration"]
        ):
            in_transcript = True
            transcript_lines.append(line)

    return "\n".join(transcript_lines).strip() or content


def summarize_file(
    file_path: Path,
    config: Config,
    provider,
    verbose: bool = False,
) -> Path:
    """Summarize a single transcription file."""
    if verbose:
        print(f"Reading: {file_path}")

    text = read_transcription(file_path)

    if not text.strip():
        raise ValueError(f"No transcription content found in {file_path}")

    if verbose:
        print(f"Summarizing {len(text)} characters...")

    result = provider.summarize(text, summary_type=config.summary_length)

    output_filename = file_path.stem
    if config.output_format == "txt":
        output_filename += ".txt"
    elif config.output_format == "md":
        output_filename += ".md"
    else:
        output_filename += ".json"

    output_path = config.output_dir / output_filename

    metadata = {
        "source": str(file_path),
        "summary_type": config.summary_length,
        **result.metadata,
    }

    if config.output_format == "txt":
        write_summary_txt(result.summary, output_path, metadata)
    elif config.output_format == "md":
        write_summary_md(
            result.summary,
            output_path,
            result.key_points,
            metadata,
        )
    else:
        write_summary_json(
            result.summary,
            output_path,
            result.key_points,
            metadata,
        )

    return output_path


def summarize_unified(
    file_paths: list[Path],
    config: Config,
    provider,
    verbose: bool = False,
) -> Path:
    """Merge all transcriptions into one and create a single unified summary."""
    if verbose:
        print(f"Merging {len(file_paths)} transcriptions for unified summary...")

    combined_text = []
    for file_path in file_paths:
        text = read_transcription(file_path)
        combined_text.append(f"=== {file_path.stem} ===\n{text}")

    full_text = "\n\n".join(combined_text)

    if verbose:
        print(f"Summarizing unified text ({len(full_text)} characters)...")

    result = provider.summarize(full_text, summary_type=config.summary_length)

    output_filename = "unified_summary"
    if config.output_format == "txt":
        output_filename += ".txt"
    elif config.output_format == "md":
        output_filename += ".md"
    else:
        output_filename += ".json"

    output_path = config.output_dir / output_filename

    metadata = {
        "source": "unified",
        "files": [str(p) for p in file_paths],
        "summary_type": config.summary_length,
        **result.metadata,
    }

    if config.output_format == "txt":
        write_summary_txt(result.summary, output_path, metadata)
    elif config.output_format == "md":
        write_summary_md(
            result.summary,
            output_path,
            result.key_points,
            metadata,
        )
    else:
        write_summary_json(
            result.summary,
            output_path,
            result.key_points,
            metadata,
        )

    return output_path


def summarize_combined(
    file_paths: list[Path],
    config: Config,
    provider,
    verbose: bool = False,
) -> Path:
    """Combine multiple summaries into one file (post-processing merge)."""
    if verbose:
        print(f"Creating combined summary from {len(file_paths)} files...")

    summaries = []
    key_points = []
    for file_path in file_paths:
        text = read_transcription(file_path)
        if verbose:
            print(f"Summarizing {file_path.name}...")

        result = provider.summarize(text, summary_type=config.summary_length)
        summaries.append(f"=== {file_path.stem} ===\n{result.summary}")
        key_points.extend(result.key_points)

    combined_text = "\n\n".join(summaries)

    output_filename = "combined_summary"
    if config.output_format == "txt":
        output_filename += ".txt"
    elif config.output_format == "md":
        output_filename += ".md"
    else:
        output_filename += ".json"

    output_path = config.output_dir / output_filename

    metadata = {
        "source": "combined",
        "files": [str(p) for p in file_paths],
        "summary_type": config.summary_length,
    }

    if config.output_format == "txt":
        write_summary_txt(combined_text, output_path, metadata)
    elif config.output_format == "md":
        write_summary_md(
            combined_text,
            output_path,
            key_points,
            metadata,
        )
    else:
        write_summary_json(
            combined_text,
            output_path,
            key_points,
            metadata,
        )

    return output_path


def main():
    """Main entry point."""
    args = parse_args()

    config = Config(
        provider=args.provider,
        model=args.model,
        input_files=args.input_files,
        output_dir=args.output_dir,
        output_format=args.output_format,
        summary_length=args.summary_length,
        combine=args.combine,
        unified=args.unified,
        skip_combined=args.skip_combined,
        transcribe_first=args.transcribe,
        verbose=args.verbose,
        transcribe_source=args.transcribe_source,
        transcribe_language=args.transcribe_language,
        transcribe_model=args.transcribe_model,
        transcribe_device=args.transcribe_device,
        transcribe_compute=args.transcribe_compute,
        transcribe_denoise=args.transcribe_denoise,
        transcribe_vad=args.transcribe_vad,
        transcribe_audio_enhance=args.transcribe_audio_enhance,
        transcribe_srt=args.transcribe_srt,
        transcribe_cleanup=args.transcribe_cleanup,
    )

    if config.verbose:
        print(f"Provider: {config.provider}")
        print(f"Model: {config.model or 'default'}")
        print(f"Output dir: {config.output_dir}")
        print(f"Transcribe output dir: {config.transcribe_output_dir}")
        print(f"Combine: {config.combine}")
        print(f"Unified: {config.unified}")
        print(f"Skip combined: {config.skip_combined}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    combined_files = config.find_combined_files()
    if combined_files:
        print(
            f"\nWarning: Detected {len(combined_files)} combined file(s) in input: "
            f"{', '.join(f.name for f in combined_files)}",
            file=sys.stderr,
        )
        if config.unified:
            print(
                "Warning: Using --unified with combined files may result in duplicate content. "
                "Consider using --skip-combined to exclude them.",
                file=sys.stderr,
            )
        elif not config.skip_combined:
            print(
                "Note: Both individual files and combined file will be summarized. "
                "Use --skip-combined to exclude combined files.",
                file=sys.stderr,
            )

    input_files = config.filter_input_files()

    if len(input_files) < len(config.input_files):
        print(
            f"\nSkipped {len(config.input_files) - len(input_files)} combined file(s) as requested.",
            file=sys.stderr,
        )

    try:
        provider = get_provider(config.provider, config.model)

        if not provider.is_available():
            if config.provider == "ollama":
                print(
                    "Error: Ollama is not running. Please start it with 'ollama serve'",
                    file=sys.stderr,
                )
                print(
                    "Or install Ollama from: https://github.com/ollama/ollama",
                    file=sys.stderr,
                )
            elif config.provider == "huggingface":
                print(
                    "Error: HuggingFace transformers not available or model cannot be loaded.",
                    file=sys.stderr,
                )
                print(
                    "Install with: pip install transformers torch",
                    file=sys.stderr,
                )
            elif config.provider == "watsonx":
                print(
                    "Error: watsonx.ai credentials not configured.",
                    file=sys.stderr,
                )
                print(
                    "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables.",
                    file=sys.stderr,
                )
            sys.exit(1)

        processed_files = []

        if config.transcribe_first:
            for file_path in config.input_files:
                txt_path = find_transcription_file(
                    file_path, config.transcribe_output_dir
                )
                if txt_path:
                    if config.verbose:
                        print(f"Using existing transcription: {txt_path}")
                    processed_files.append(txt_path)
                else:
                    txt_path = transcribe_file(file_path, config, config.verbose)
                    processed_files.append(txt_path)
            processed_files = config.filter_input_files()
        else:
            processed_files = config.filter_input_files()

        if not processed_files:
            print("Error: No valid input files to process", file=sys.stderr)
            sys.exit(1)

        if config.unified:
            output_path = summarize_unified(
                processed_files, config, provider, config.verbose
            )
            print(f"\n✓ Unified summary saved to: {output_path}")
        elif config.combine:
            output_path = summarize_combined(
                processed_files, config, provider, config.verbose
            )
            print(f"\n✓ Combined summary saved to: {output_path}")
        else:
            for file_path in processed_files:
                output_path = summarize_file(
                    file_path, config, provider, config.verbose
                )
                print(f"✓ Summary saved to: {output_path}")

    except ProviderNotAvailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SummarizerError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if config.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
