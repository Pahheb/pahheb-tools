"""Threading pipeline for transcribe-then-summarize workflows."""

import queue
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from .config import Config
from .transcription import find_transcription_file, transcribe_file

_SENTINEL = object()


def _transcribe_then_summarize_thread(
    input_files: Sequence[str | Path],
    config: Config,
    summarize_fn,
    verbose: bool = False,
) -> tuple[list[Path], int, int, int]:
    """Two-thread pipeline: transcriber feeds queue, summarizer consumes.

    Only one Whisper instance runs at a time (no GPU/CPU contention).
    Summarization starts immediately after the first transcription finishes.
    """
    summary_queue: queue.Queue[Path | object] = queue.Queue()
    output_paths: list[Path] = []
    errors: list[tuple[str, Exception]] = []
    lock = threading.Lock()

    def summarize_worker() -> None:
        """Consume transcribed files from queue and summarize."""
        while True:
            item = summary_queue.get(timeout=600)
            if item is _SENTINEL:
                summary_queue.task_done()
                break
            txt_path = item
            assert isinstance(txt_path, Path)
            try:
                out = summarize_fn(txt_path, config, verbose)
                print(f"✓ Summary saved to: {out}")
                with lock:
                    output_paths.append(out)
            except Exception as e:
                print(
                    f"Warning: Summarization failed for {txt_path}: {e}",
                    file=sys.stderr,
                )
                with lock:
                    errors.append(("summarize", e))
            finally:
                summary_queue.task_done()

    sum_thread = threading.Thread(target=summarize_worker, daemon=True)
    sum_thread.start()

    transcribed_count = 0
    transcribed_errors = 0

    # Transcriber runs on main thread (sequential within itself)
    for file_path in input_files:
        txt_path = find_transcription_file(file_path, config.transcribe_output_dir)
        if txt_path:
            if verbose:
                print(f"Using existing transcription: {txt_path}")
            summary_queue.put(txt_path)
        else:
            try:
                txt_path = transcribe_file(file_path, config, verbose)
                summary_queue.put(txt_path)
                transcribed_count += 1
            except Exception as e:
                print(
                    f"Warning: Transcription failed for {file_path}: {e}",
                    file=sys.stderr,
                )
                transcribed_errors += 1
                with lock:
                    errors.append(("transcribe", e))

    # Signal summarizer to finish
    summary_queue.put(_SENTINEL)
    sum_thread.join()

    summarized_errors = len([e for t, e in errors if t == "summarize"])

    return output_paths, transcribed_count, transcribed_errors, summarized_errors


def _transcribe_then_combine(
    input_files: Sequence[str | Path],
    config: Config,
    verbose: bool = False,
) -> tuple[list[Path], int, int]:
    """Sequential transcription followed by combine."""
    processed_files: list[Path] = []
    transcribed_count = 0
    transcribed_errors = 0

    for file_path in input_files:
        txt_path = find_transcription_file(file_path, config.transcribe_output_dir)
        if txt_path:
            if verbose:
                print(f"Using existing transcription: {txt_path}")
            processed_files.append(txt_path)
        else:
            try:
                txt_path = transcribe_file(file_path, config, verbose)
                processed_files.append(txt_path)
                transcribed_count += 1
            except Exception as e:
                print(
                    f"Warning: Transcription failed for {file_path}: {e}",
                    file=sys.stderr,
                )
                transcribed_errors += 1

    return processed_files, transcribed_count, transcribed_errors
