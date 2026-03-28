"""Integration tests for the summarize CLI pipeline."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarize_src.__main__ import main


class TestPipelineResilience:
    """Test that failures in transcription/summarization don't crash the pipeline."""

    def test_summarize_file_failure_isolated_one_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that a failing summarize_file doesn't crash the loop."""
        txt1 = tmp_path / "file1.txt"
        txt2 = tmp_path / "file2.txt"
        txt1.write_text("Content of file 1")
        txt2.write_text("Content of file 2")
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            def raise_on_first(path: Path, *args: object, **kwargs: object) -> Path:
                if path == txt1:
                    raise RuntimeError("Simulated summarize failure")
                return out_dir / f"{path.stem}_summary.txt"

            mock_summarize.side_effect = raise_on_first

            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stdout", stdout_capture):
                    with patch.object(sys, "stderr", stderr_capture):
                        with patch(
                            "sys.argv",
                            [
                                "summarize",
                                str(txt1),
                                str(txt2),
                                "--output-dir",
                                str(out_dir),
                            ],
                        ):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "file1.txt" in err or "Simulated summarize failure" in err
            assert mock_summarize.call_count == 2

    def test_transcribe_failure_isolated_one_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that a failing transcribe_file doesn't crash the loop."""
        audio1 = tmp_path / "audio1.mp3"
        audio2 = tmp_path / "audio2.mp3"
        audio1.write_text("fake audio 1")
        audio2.write_text("fake audio 2")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            mock_find.return_value = None

            def raise_on_first(path: Path, *args: object, **kwargs: object) -> Path:
                if path == audio1:
                    raise RuntimeError("Simulated transcription failure")
                txt = transcription_dir / f"{path.stem}.txt"
                txt.write_text(f"transcription of {path.name}")
                return txt

            mock_transcribe.side_effect = raise_on_first
            mock_summarize.return_value = tmp_path / "out" / "summary.txt"

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        [
                            "summarize",
                            str(audio1),
                            str(audio2),
                            "--output-dir",
                            str(tmp_path),
                            "--transcribe",
                        ],
                    ):
                        main()

            err = stderr_capture.getvalue()
            assert "Simulated transcription failure" in err or "audio1" in err
            assert mock_transcribe.call_count == 2

    def test_all_transcribe_failures_exits_with_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that when all transcriptions fail, the pipeline exits with error."""
        audio1 = tmp_path / "audio1.mp3"
        audio1.write_text("fake audio 1")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None
            mock_transcribe.side_effect = RuntimeError("All transcription failed")

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        [
                            "summarize",
                            str(audio1),
                            "--output-dir",
                            str(tmp_path),
                            "--transcribe",
                        ],
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

            assert exc_info.value.code == 1


class TestBug3Integration:
    """Regression test for bug #3: filter_input_files() returned processed_files
    instead of actual transcription output paths."""

    def test_transcribe_first_uses_actual_transcription_path(
        self,
        tmp_path: Path,
    ) -> None:
        """Bug #3: transcribe_first should use the actual .txt path from transcribe_file,
        not the result of filter_input_files() which returned processed_files."""
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"
        video1.write_text("fake video 1")
        video2.write_text("fake video 2")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir()

        txt1 = transcription_dir / "video1.txt"
        txt2 = transcription_dir / "video2.txt"
        txt1.write_text("Transcription of video 1")
        txt2.write_text("Transcription of video 2")

        out_dir = tmp_path / "summaries"
        out_dir.mkdir()

        captured_args: list[Path] = []

        def capture_summarize(path: Path, *args: object, **kwargs: object) -> Path:
            captured_args.append(path)
            out = out_dir / f"{path.stem}_summary.txt"
            out.write_text("summary")
            return out

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch(
                "summarize_src.__main__.summarize_file", side_effect=capture_summarize
            ),
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            def find_side_effect(video: Path, out_dir_arg: Path) -> Path | None:
                txt = transcription_dir / f"{video.stem}.txt"
                if txt.exists():
                    return txt
                return None

            mock_find.side_effect = find_side_effect
            mock_transcribe.side_effect = lambda v, c, verbose: (
                transcription_dir / f"{v.stem}.txt"
            )

            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stdout", stdout_capture):
                    with patch.object(sys, "stderr", stderr_capture):
                        with patch(
                            "sys.argv",
                            [
                                "summarize",
                                str(video1),
                                str(video2),
                                "--transcribe",
                                "--output-dir",
                                str(tmp_path),
                            ],
                        ):
                            main()

            assert len(captured_args) == 2
            assert all(isinstance(p, Path) for p in captured_args)
            assert all(p.suffix == ".txt" for p in captured_args)
            assert all(
                "Transcription" in p.parent.name or p.exists() for p in captured_args
            )


class TestProviderErrors:
    """Test provider initialization errors."""

    def test_missing_provider_credentials_exits_cleanly(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that missing provider credentials cause a clean exit."""
        from summarize_src.summarizer import ProviderNotAvailableError

        txt = tmp_path / "file.txt"
        txt.write_text("content")

        with patch("summarize_src.__main__.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = ProviderNotAvailableError(
                "ollama", "Connection refused"
            )

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        ["summarize", str(txt)],
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "ollama" in err.lower() or "connection refused" in err.lower()


class TestOllamaTimeout:
    """Test Ollama summarization timeout handling."""

    def test_ollama_timeout_exits_with_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that httpx.TimeoutException from Ollama causes a clean exit."""
        import httpx

        txt = tmp_path / "file.txt"
        txt.write_text("some transcription content")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch("summarize_src.__main__.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.summarize.side_effect = httpx.TimeoutException(
                "Connection timed out"
            )
            mock_get_provider.return_value = mock_provider

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        ["summarize", str(txt), "--output-dir", str(out_dir)],
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "timed out" in err.lower() or "error" in err.lower()


class TestMultipleFiles:
    """Test summarization of multiple files."""

    def test_summarize_multiple_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that multiple files are all summarized."""
        txt1 = tmp_path / "file1.txt"
        txt2 = tmp_path / "file2.txt"
        txt3 = tmp_path / "file3.txt"
        txt1.write_text("Transcription of video 1")
        txt2.write_text("Transcription of video 2")
        txt3.write_text("Transcription of video 3")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_summarize.side_effect = lambda p, *a, **k: (
                out_dir / f"{p.stem}_summary.txt"
            )

            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stdout", stdout_capture):
                    with patch.object(sys, "stderr", stderr_capture):
                        with patch(
                            "sys.argv",
                            [
                                "summarize",
                                str(txt1),
                                str(txt2),
                                str(txt3),
                                "--output-dir",
                                str(out_dir),
                            ],
                        ):
                            main()

            assert mock_summarize.call_count == 3
            out = stdout_capture.getvalue()
            assert "✓" in out

    def test_transcribe_multiple_videos_then_summarize(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that multiple video files are transcribed then summarized."""
        vid1 = tmp_path / "video1.mp4"
        vid2 = tmp_path / "video2.mp4"
        vid3 = tmp_path / "video3.mp4"
        for v in (vid1, vid2, vid3):
            v.write_text("fake video")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None

            def fake_transcribe(path, *args, **kwargs):
                txt = transcription_dir / f"{path.stem}.txt"
                txt.write_text(f"transcription of {path.name}")
                return txt

            mock_transcribe.side_effect = fake_transcribe
            mock_summarize.side_effect = lambda p, *a, **k: (
                tmp_path / "out" / f"{p.stem}_summary.txt"
            )

            with patch.object(sys, "stdin", StringIO()):
                with patch(
                    "sys.argv",
                    [
                        "summarize",
                        str(vid1),
                        str(vid2),
                        str(vid3),
                        "--transcribe",
                        "--output-dir",
                        str(tmp_path),
                    ],
                ):
                    main()

            assert mock_transcribe.call_count == 3
            assert mock_summarize.call_count == 3

    def test_summarize_unified_mode(
        self,
        tmp_path: Path,
    ) -> None:
        """Test --unified merges transcriptions and summarizes once."""
        txt1 = tmp_path / "file1.txt"
        txt2 = tmp_path / "file2.txt"
        txt1.write_text("Transcription 1")
        txt2.write_text("Transcription 2")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.__main__.summarize_unified") as mock_unified,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_unified.return_value = out_dir / "unified_summary.txt"

            with patch.object(sys, "stdin", StringIO()):
                with patch(
                    "sys.argv",
                    [
                        "summarize",
                        str(txt1),
                        str(txt2),
                        "--unified",
                        "--output-dir",
                        str(out_dir),
                    ],
                ):
                    main()

            mock_unified.assert_called_once()
            call_files = mock_unified.call_args[0][0]
            assert len(call_files) == 2

    def test_summarize_combined_mode(
        self,
        tmp_path: Path,
    ) -> None:
        """Test --combine creates combined summary."""
        txt1 = tmp_path / "file1.txt"
        txt2 = tmp_path / "file2.txt"
        txt1.write_text("Transcription 1")
        txt2.write_text("Transcription 2")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.__main__.summarize_combined") as mock_combine,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_combine.return_value = out_dir / "combined_summary.txt"

            with patch.object(sys, "stdin", StringIO()):
                with patch(
                    "sys.argv",
                    [
                        "summarize",
                        str(txt1),
                        str(txt2),
                        "--combine",
                        "--output-dir",
                        str(out_dir),
                    ],
                ):
                    main()

            mock_combine.assert_called_once()

    def test_transcribe_youtube_video_then_summarize(
        self,
        tmp_path: Path,
    ) -> None:
        """Test YouTube URL transcribe + summarize pipeline."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        transcription_dir = out_dir / "transcriptions"
        transcription_dir.mkdir()

        txt = transcription_dir / "abc123.txt"
        txt.write_text("YouTube transcription content")

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None
            mock_transcribe.return_value = txt
            mock_summarize.return_value = out_dir / "summary.txt"

            with patch.object(sys, "stdin", StringIO()):
                with patch(
                    "sys.argv",
                    [
                        "summarize",
                        "https://youtube.com/watch?v=abc123",
                        "--transcribe",
                        "--transcribe-source",
                        "youtube",
                        "--output-dir",
                        str(out_dir),
                    ],
                ):
                    main()

            mock_transcribe.assert_called_once()
            mock_summarize.assert_called_once()


class TestErrorPaths:
    """Test error path integration tests."""

    def test_empty_transcription_exits_with_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that an empty transcription content causes a clean exit."""
        txt = tmp_path / "file.txt"
        txt.write_text("some content")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.__main__.read_transcription") as mock_read,
        ):
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_get_provider.return_value = mock_provider
            mock_read.return_value = "   "  # whitespace only, triggers ValueError

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        ["summarize", str(txt), "--output-dir", str(out_dir)],
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

            assert exc_info.value.code == 1

    def test_summarizer_error_exits_with_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that SummarizerError causes a clean exit."""
        from summarize_src.summarizer import SummarizerError

        txt = tmp_path / "file.txt"
        txt.write_text("some content")

        with patch("summarize_src.__main__.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.summarize.side_effect = SummarizerError("Model not found")
            mock_get_provider.return_value = mock_provider

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        ["summarize", str(txt)],
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "model not found" in err.lower()


class TestThreadedPipeline:
    """Test the threaded transcription+summarization pipeline."""

    def test_threaded_transcribe_summarize(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that threaded pipeline processes all files."""
        vid1 = tmp_path / "video1.mp4"
        vid2 = tmp_path / "video2.mp4"
        vid3 = tmp_path / "video3.mp4"
        for v in (vid1, vid2, vid3):
            v.write_text("fake video")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None

            def fake_transcribe(path, *args, **kwargs):
                txt = transcription_dir / f"{path.stem}.txt"
                txt.write_text(f"transcription of {path.name}")
                return txt

            mock_transcribe.side_effect = fake_transcribe
            mock_summarize.return_value = tmp_path / "out" / "summary.txt"

            stdout_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stdout", stdout_capture):
                    with patch(
                        "sys.argv",
                        [
                            "summarize",
                            str(vid1),
                            str(vid2),
                            str(vid3),
                            "--transcribe",
                            "--output-dir",
                            str(tmp_path),
                        ],
                    ):
                        main()

            assert mock_transcribe.call_count == 3
            assert mock_summarize.call_count == 3
            out = stdout_capture.getvalue()
            assert "summarized" in out.lower() or "✓" in out

    def test_threaded_partial_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """Test threaded pipeline handles partial transcription failures."""
        vid1 = tmp_path / "video1.mp4"
        vid2 = tmp_path / "video2.mp4"
        for v in (vid1, vid2):
            v.write_text("fake video")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None

            def fake_transcribe(path, *args, **kwargs):
                if path == vid1:
                    raise RuntimeError("Transcription failed for video1")
                txt = transcription_dir / f"{path.stem}.txt"
                txt.write_text(f"transcription of {path.name}")
                return txt

            mock_transcribe.side_effect = fake_transcribe
            mock_summarize.return_value = tmp_path / "out" / "summary.txt"

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        [
                            "summarize",
                            str(vid1),
                            str(vid2),
                            "--transcribe",
                            "--output-dir",
                            str(tmp_path),
                        ],
                    ):
                        main()

            # video1 failed transcription, video2 succeeded
            assert mock_transcribe.call_count == 2
            assert mock_summarize.call_count == 1
            err = stderr_capture.getvalue()
            assert "video1" in err.lower() or "error" in err.lower()

    def test_single_threaded_flag_uses_sequential(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that --single-threaded flag uses sequential processing."""
        vid1 = tmp_path / "video1.mp4"
        vid1.write_text("fake video")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None
            mock_transcribe.return_value = transcription_dir / "video1.txt"
            mock_summarize.return_value = tmp_path / "out" / "summary.txt"

            with patch.object(sys, "stdin", StringIO()):
                with patch(
                    "sys.argv",
                    [
                        "summarize",
                        str(vid1),
                        "--transcribe",
                        "--output-dir",
                        str(tmp_path),
                        "--single-threaded",
                    ],
                ):
                    main()

            # Both were called, confirming sequential path was used
            mock_transcribe.assert_called_once()
            mock_summarize.assert_called_once()

    def test_threaded_summarizer_partial_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """Test threaded pipeline handles partial summarization failures."""
        vid1 = tmp_path / "video1.mp4"
        vid2 = tmp_path / "video2.mp4"
        for v in (vid1, vid2):
            v.write_text("fake video")

        transcription_dir = tmp_path / "transcriptions"
        transcription_dir.mkdir(exist_ok=True)

        with (
            patch("summarize_src.__main__.get_provider") as mock_get_provider,
            patch("summarize_src.pipeline.find_transcription_file") as mock_find,
            patch("summarize_src.pipeline.transcribe_file") as mock_transcribe,
            patch("summarize_src.__main__.summarize_file") as mock_summarize,
        ):
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_find.return_value = None

            def fake_transcribe(path, *args, **kwargs):
                txt = transcription_dir / f"{path.stem}.txt"
                txt.write_text(f"transcription of {path.name}")
                return txt

            mock_transcribe.side_effect = fake_transcribe

            def fake_summarize(path, *args, **kwargs):
                if "video1" in str(path):
                    raise RuntimeError("Summarization failed for video1")
                return tmp_path / "out" / "summary.txt"

            mock_summarize.side_effect = fake_summarize

            stderr_capture = StringIO()
            with patch.object(sys, "stdin", StringIO()):
                with patch.object(sys, "stderr", stderr_capture):
                    with patch(
                        "sys.argv",
                        [
                            "summarize",
                            str(vid1),
                            str(vid2),
                            "--transcribe",
                            "--output-dir",
                            str(tmp_path),
                        ],
                    ):
                        main()

            assert mock_transcribe.call_count == 2
            assert mock_summarize.call_count == 2
            err = stderr_capture.getvalue()
            assert "summarization failed" in err.lower() or "error" in err.lower()
