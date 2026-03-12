"""Audio processing using FFmpeg and optionally RNNoise."""

import subprocess
from pathlib import Path
from typing import Optional


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""

    pass


def check_ffmpeg_arnndn_support() -> bool:
    """Check if ffmpeg has the arnndn filter (RNNoise) compiled in."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        return " arnndn " in result.stdout
    except Exception:
        return False


def find_denoise_model(search_paths: list[Path]) -> Optional[Path]:
    """Search for RNNoise model file in given paths."""
    model_names = ["std.rnnn", "model.rnnn"]

    for search_path in search_paths:
        if not search_path.exists():
            continue
        if search_path.is_file():
            if search_path.name in model_names:
                return search_path
        elif search_path.is_dir():
            for model_name in model_names:
                model_path = search_path / model_name
                if model_path.exists():
                    return model_path

    return None


def get_default_model_search_paths() -> list[Path]:
    """Get default search paths for RNNoise model."""
    return [
        Path("./models"),
        Path.home() / ".local" / "share" / "transcribe" / "models",
    ]


def process_audio(
    input_path: Path,
    work_dir: Path,
    denoise: bool = False,
    denoise_model: Optional[str] = None,
    audio_enhance: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Process audio file to 16kHz mono WAV suitable for Whisper.

    Args:
        input_path: Path to input audio/video file
        work_dir: Working directory for intermediate files
        denoise: Whether to apply RNNoise denoising
        denoise_model: Path to RNNoise model (optional)
        audio_enhance: Whether to apply audio enhancement filters
        verbose: Whether to show verbose output

    Returns:
        Path to processed audio file
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "audio_16k_mono.wav"

    filters = []

    if audio_enhance:
        filters.extend(
            [
                "highpass=f=80",
                "lowpass=f=8000",
                "compand=0.3|0.8:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2",
                "loudnorm",
            ]
        )

    if denoise:
        has_arnndn = check_ffmpeg_arnndn_support()
        if has_arnndn:
            model_path = None

            if denoise_model:
                model_path = Path(denoise_model)
                if not model_path.exists():
                    raise AudioProcessingError(f"Denoise model not found: {model_path}")
            else:
                search_paths = get_default_model_search_paths()
                model_path = find_denoise_model(search_paths)

            if model_path:
                filters.insert(0, f"arnndn=m={model_path}")
            else:
                if verbose:
                    print("Warning: RNNoise model not found, skipping denoising")
        else:
            if verbose:
                print(
                    "Warning: ffmpeg does not support RNNoise (arnndn filter), skipping denoising"
                )

    af = ",".join(filters) if filters else "anull"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        af,
        str(output_path),
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise AudioProcessingError(f"ffmpeg failed: {result.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        raise AudioProcessingError("Audio processing timed out")

    if not output_path.exists():
        raise AudioProcessingError(f"Output file not created: {output_path}")

    return output_path
