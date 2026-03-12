"""Local file transcription processor."""

import shutil
from pathlib import Path

from .audio_processor import process_audio
from .config import Config
from .file_writer import write_transcripts
from .whisper import transcribe_audio


def process_local_file(
    config: Config,
    input_path: Path,
    verbose: bool = False,
) -> list[Path]:
    """
    Process a local audio/video file.

    Args:
        config: Configuration object
        input_path: Path to input file
        verbose: Enable verbose output

    Returns:
        List of written output files

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file type is not supported
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    supported_extensions = {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".webm",
    }
    if input_path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file type: {input_path.suffix}. "
            f"Supported: {', '.join(supported_extensions)}"
        )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "work" / input_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    output_base = output_dir / input_path.stem

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {input_path.name}")
        print(f"{'=' * 60}")

    if verbose:
        print("\n[1/3] Processing audio...")

    processed_audio = process_audio(
        input_path=input_path,
        work_dir=work_dir,
        denoise=config.denoise,
        denoise_model=config.denoise_model,
        audio_enhance=config.audio_enhance,
        verbose=verbose,
    )

    if verbose:
        print(f"  Processed audio: {processed_audio}")

    if verbose:
        print(f"\n[2/3] Transcribing with Whisper ({config.model_size})...")

    segments = transcribe_audio(
        audio_path=processed_audio,
        model_size=config.model_size,
        language=config.language,
        device=config.device,
        compute_type=config.compute_type,
        vad_filter=config.vad,
        verbose=verbose,
    )

    if verbose:
        print(f"  Got {len(segments)} segments")

    if verbose:
        print("\n[3/3] Writing output files...")

    metadata = {
        "source": "local",
        "input_file": str(input_path),
        "model": config.model_size,
    }
    if config.language:
        metadata["language"] = config.language

    output_files = write_transcripts(
        segments=segments,
        output_base=output_base,
        metadata=metadata,
        write_srt=config.srt,
        verbose=verbose,
    )

    if config.cleanup:
        if verbose:
            print(f"\nCleaning up work directory: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_files
