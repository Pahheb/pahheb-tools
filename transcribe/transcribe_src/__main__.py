"""CLI entry point for transcribe tool."""

import sys
from pathlib import Path

from .cli import parse_args
from .config import Config
from .local_processor import process_local_file
from .youtube_processor import process_youtube_video


def main():
    """Main entry point."""
    args = parse_args()

    config = Config(
        source=args.source,
        input_path=args.input,
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

    input_path = Path(args.input)

    if config.source == "youtube":
        try:
            output_files = process_youtube_video(config, args.input, verbose=args.verbose)
            print("\n✓ Successfully transcribed YouTube video")
            for f in output_files:
                print(f"  → {f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            output_files = process_local_file(config, input_path, verbose=args.verbose)
            print(f"\n✓ Successfully transcribed: {input_path.name}")
            for f in output_files:
                print(f"  → {f}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
