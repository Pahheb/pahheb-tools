"""Whisper transcription with AMD GPU / ROCm support."""

from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from faster_whisper import WhisperModel


def check_amd_gpu() -> bool:
    """Check if AMD GPU with ROCm is available."""
    try:
        if hasattr(torch, "version") and hasattr(torch.version, "hip"):
            if torch.version.hip is not None:
                return torch.cuda.is_available()
        return False
    except Exception:
        return False


def get_device() -> str:
    """Get the best available device for Whisper inference."""
    if check_amd_gpu():
        return "cuda"
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_compute_type(device: str, compute_type: Optional[str] = None) -> str:
    """Get compute type based on device."""
    if compute_type:
        return compute_type

    if device == "cpu":
        return "int8"
    elif device in ("cuda", "mps"):
        return "float16"
    return "int8"


class WhisperTranscriber:
    """Whisper transcription handler."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use (auto, cuda, cpu, mps)
            compute_type: Compute type (int8, float16, etc.)
            language: Language code for transcription
            verbose: Enable verbose output
        """
        self.model_size = model_size
        self.language = language
        self.verbose = verbose

        if device == "auto":
            device = get_device()

        self.device = device
        self.compute_type = get_compute_type(device, compute_type)

        if self.verbose:
            print(f"Loading Whisper model: {model_size}")
            print(f"  Device: {device}")
            print(f"  Compute type: {self.compute_type}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        vad_filter: bool = False,
    ) -> tuple[Iterator[Any], Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            vad_filter: Whether to use VAD filtering

        Returns:
            Tuple of (segments iterator, info)
        """
        if self.verbose:
            print(f"Transcribing: {audio_path}")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=vad_filter,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        return segments, info


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: Optional[str] = None,
    vad_filter: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """
    Transcribe audio file and return segments.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Language code
        device: Device to use
        compute_type: Compute type
        vad_filter: Use VAD filtering
        verbose: Verbose output

    Returns:
        List of segment dictionaries with text, start, end
    """
    transcriber = WhisperTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        verbose=verbose,
    )

    segments, info = transcriber.transcribe(audio_path, vad_filter=vad_filter)

    if verbose:
        print(f"Detected language: {info.language} ({info.language_probability:.2f})")

    results = []
    for segment in segments:
        results.append(
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
            }
        )

    return results
