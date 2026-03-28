"""CLI entry point for transcribe tool."""

import re
import sys
from pathlib import Path

from .cli import parse_args
from .config import Config
from .local_processor import process_local_file
from .youtube_processor import process_youtube_video


def _is_youtube_url(text: str) -> bool:
    """Check if a string looks like a YouTube URL."""
    return bool(
        re.search(r"(?:youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)", text)
    )


def _process_input(
    config: Config,
    input_arg: str,
    force_source: str | None,
    verbose: bool = False,
) -> list[Path]:
    """Process a single input (file or URL). Returns list of output files."""
    source = force_source or ("youtube" if _is_youtube_url(input_arg) else "local")

    if source == "youtube":
        return process_youtube_video(config, input_arg, verbose=verbose)

    input_path = Path(input_arg)
    return process_local_file(config, input_path, verbose=verbose)


def main():
    """Main entry point."""
    args = parse_args()

    config = Config(
        source=args.source or "local",
        output_dir=args.output_dir,
        language=args.language,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute,
        denoise=args.denoise,
        denoise_model=args.denoise_model,
        vad=args.vad,
        audio_enhance=args.audio_enhance,
        srt=args.srt,
        cleanup=args.cleanup,
        verbose=args.verbose,
    )

    total = len(args.inputs)
    succeeded = 0
    failed = 0

    try:
        for i, input_arg in enumerate(args.inputs, 1):
            if total > 1:
                print(f"\n[{i}/{total}] Processing: {input_arg}")

            try:
                output_files = _process_input(
                    config, input_arg, args.source, config.verbose
                )

                if total > 1:
                    print("  ✓ Done")
                else:
                    source_label = args.source or (
                        "YouTube" if _is_youtube_url(input_arg) else "local"
                    )
                    if source_label == "youtube":
                        print("\n✓ Successfully transcribed YouTube video")
                    else:
                        print(f"\n✓ Successfully transcribed: {Path(input_arg).name}")

                for f in output_files:
                    print(f"  → {f}")

                succeeded += 1
            except Exception as e:
                print(f"  ✗ Error: {e}", file=sys.stderr)
                failed += 1

        if total > 1:
            if failed:
                print(
                    f"\nCompleted: {succeeded}/{total} succeeded, {failed} failed.",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                print(f"\n✓ All {total} files transcribed successfully.")
        elif failed:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
