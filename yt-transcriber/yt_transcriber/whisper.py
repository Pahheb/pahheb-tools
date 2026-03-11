"""Local transcription using faster-whisper with AMD GPU support."""

from pathlib import Path
from typing import Optional

import torch


def check_amd_gpu() -> bool:
    """
    Check if AMD GPU with ROCm is available.

    Returns:
        True if AMD GPU with ROCm is available
    """
    try:
        # Check if PyTorch detects ROCm
        if hasattr(torch, "version") and hasattr(torch.version, "hip"):
            if torch.version.hip is not None:
                return torch.cuda.is_available()
        return False
    except Exception:
        return False


def get_device() -> str:
    """
    Get the best available device for Whisper inference.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if check_amd_gpu():
        return "cuda"
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def transcribe_with_whisper(
    audio_path: Path,
    model_size: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
) -> str:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        language: Optional language code
        device: Device to use (auto, cuda, cpu)

    Returns:
        Plain text transcription
    """
    from faster_whisper import WhisperModel

    # Determine device
    if device == "auto":
        device_str = get_device()
    else:
        device_str = device

    # Ensure device string is valid for faster-whisper
    # faster-whisper uses 'cuda' for GPU, 'cpu' for CPU
    if device_str not in ("cuda", "cpu", "mps"):
        device_str = "cpu"

    # Create model
    model = WhisperModel(
        model_size, device=device_str, compute_type="int8" if device_str == "cpu" else "float16"
    )

    # Transcribe
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,  # Use VAD to filter non-speech segments
    )

    # Combine segments into full text
    full_text = " ".join([segment.text for segment in segments])

    return full_text.strip()
