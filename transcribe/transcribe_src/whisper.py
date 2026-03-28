"""Whisper transcription."""

from pathlib import Path

from faster_whisper import WhisperModel


def get_compute_type(device: str, compute_type: str | None = None) -> str:
    """Get compute type based on device."""
    if compute_type:
        return compute_type
    if device == "cpu":
        return "int8"
    elif device in ("cuda", "mps"):
        return "float16"
    return "int8"


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
    compute_type: str | None = None,
    vad_filter: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Transcribe audio file and return segments."""
    if device == "auto":
        device = "cpu"

    resolved_compute = get_compute_type(device, compute_type)

    if verbose:
        print(f"Loading Whisper model: {model_size}")
        print(f"  Device: {device}")
        print(f"  Compute type: {resolved_compute}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=resolved_compute,
    )

    if verbose:
        print(f"Transcribing: {audio_path}")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
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
