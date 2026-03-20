"""CLI entry point for summarize tool."""

import sys
from pathlib import Path

from .cli import parse_args
from .config import Config
from .file_writer import write_summary
from .pipeline import _transcribe_then_combine, _transcribe_then_summarize_thread
from .summarizer import (
    ProviderNotAvailableError,
    SummarizerError,
    get_provider,
)
from .transcription import find_transcription_file, read_transcription, transcribe_file


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

    output_path = config.output_dir / f"{file_path.stem}.{config.output_format}"

    metadata = {
        "source": str(file_path),
        "summary_type": config.summary_length,
        **result.metadata,
    }

    write_summary(
        result.summary, output_path, config.output_format, result.key_points, metadata
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

    output_path = config.output_dir / f"unified_summary.{config.output_format}"

    metadata = {
        "source": "unified",
        "files": [str(p) for p in file_paths],
        "summary_type": config.summary_length,
        **result.metadata,
    }

    write_summary(
        result.summary, output_path, config.output_format, result.key_points, metadata
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

    output_path = config.output_dir / f"combined_summary.{config.output_format}"

    metadata = {
        "source": "combined",
        "files": [str(p) for p in file_paths],
        "summary_type": config.summary_length,
    }

    write_summary(
        combined_text, output_path, config.output_format, key_points, metadata
    )

    return output_path


def _summarize_with_provider(file_path, config, provider, verbose):
    """Summarize a single file (used as callback for threaded pipeline)."""
    return summarize_file(file_path, config, provider, verbose)


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
        single_threaded=args.single_threaded,
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
            f"{', '.join(f.name if isinstance(f, Path) else f for f in combined_files)}",
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
            sys.exit(1)

        processed_files: list[Path] = []
        transcribed_count = 0
        transcribed_errors = 0
        summarized_errors = 0

        if config.transcribe_first:
            if config.unified or config.single_threaded:
                for file_path in config.input_files:
                    txt_path = find_transcription_file(
                        file_path, config.transcribe_output_dir
                    )
                    if txt_path:
                        if config.verbose:
                            print(f"Using existing transcription: {txt_path}")
                        processed_files.append(txt_path)
                    else:
                        try:
                            txt_path = transcribe_file(
                                file_path, config, config.verbose
                            )
                            processed_files.append(txt_path)
                            transcribed_count += 1
                        except Exception as e:
                            print(
                                f"Warning: Transcription failed for {file_path}: {e}",
                                file=sys.stderr,
                            )
                            transcribed_errors += 1
            elif config.combine:
                processed_files, tc, te = _transcribe_then_combine(
                    config.input_files, config, config.verbose
                )
                transcribed_count = tc
                transcribed_errors = te
            else:
                summarize_fn = lambda fp, c, v: _summarize_with_provider(  # noqa: E731
                    fp, c, provider, v
                )
                (
                    outputs,
                    transcribed_count,
                    transcribed_errors,
                    summarized_errors,
                ) = _transcribe_then_summarize_thread(
                    config.input_files, config, summarize_fn, config.verbose
                )
                if transcribed_errors or summarized_errors:
                    total = len(outputs) + transcribed_errors + summarized_errors
                    errs = transcribed_errors + summarized_errors
                    print(
                        f"\nCompleted with errors: {total - errs} succeeded, "
                        f"{errs} failed.",
                        file=sys.stderr,
                    )
                    if not outputs:
                        sys.exit(1)
                elif outputs:
                    print(f"\n✓ All {len(outputs)} files summarized successfully.")
                else:
                    print("Error: No valid input files to process", file=sys.stderr)
                    sys.exit(1)
                return
        else:
            processed_files = config.filter_input_files()  # type: ignore[assignment]

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
            summarized_count = 0
            for file_path in processed_files:
                try:
                    output_path = summarize_file(
                        file_path, config, provider, config.verbose
                    )
                    print(f"✓ Summary saved to: {output_path}")
                    summarized_count += 1
                except Exception as e:
                    print(
                        f"Warning: Summarization failed for {file_path}: {e}",
                        file=sys.stderr,
                    )
                    summarized_errors += 1
            if transcribed_errors or summarized_errors:
                total = transcribed_count + summarized_count
                errs = transcribed_errors + summarized_errors
                print(
                    f"\nCompleted with errors: {total} succeeded, {errs} failed.",
                    file=sys.stderr,
                )
                sys.exit(1)

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
