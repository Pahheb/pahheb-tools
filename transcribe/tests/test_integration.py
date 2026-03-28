"""Integration tests for the transcribe CLI pipeline."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from transcribe_src.__main__ import main


class TestLocalFilePipelineResilience:
    """Test that failures in the local file pipeline don't crash the CLI."""

    def test_missing_file_exits_with_error(self, tmp_path: Path) -> None:
        """Test that a missing input file causes a clean exit."""
        missing = tmp_path / "nonexistent.mp3"

        stderr_capture = StringIO()
        with patch.object(sys, "stderr", stderr_capture):
            with patch(
                "sys.argv",
                ["transcribe", str(missing)],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        err = stderr_capture.getvalue()
        assert "not found" in err.lower() or "nonexistent" in err.lower()

    def test_unsupported_file_type_exits_with_error(self, tmp_path: Path) -> None:
        """Test that an unsupported file extension causes a clean exit."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("not audio")

        stderr_capture = StringIO()
        with patch.object(sys, "stderr", stderr_capture):
            with patch(
                "sys.argv",
                ["transcribe", str(txt_file)],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        err = stderr_capture.getvalue()
        assert "unsupported" in err.lower() or ".txt" in err.lower()

    def test_audio_processor_failure_exits_with_error(self, tmp_path: Path) -> None:
        """Test that a failing audio processor causes a clean exit."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")

        with patch(
            "transcribe_src.local_processor.process_audio"
        ) as mock_process_audio:
            mock_process_audio.side_effect = RuntimeError("ffmpeg not found")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    ["transcribe", str(audio)],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "ffmpeg" in err.lower()

    def test_transcription_failure_exits_with_error(self, tmp_path: Path) -> None:
        """Test that a failing Whisper transcription causes a clean exit."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.side_effect = RuntimeError("Whisper model load failed")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    ["transcribe", str(audio)],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "whisper" in err.lower() or "load" in err.lower()


class TestYouTubePipelineResilience:
    """Test that failures in the YouTube pipeline don't crash the CLI."""

    def test_invalid_youtube_url_exits_with_error(self, tmp_path: Path) -> None:
        """Test that an invalid YouTube URL causes a clean exit."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch(
            "transcribe_src.youtube_processor.get_youtube_video_info"
        ) as mock_info:
            mock_info.side_effect = RuntimeError("Invalid YouTube URL")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    [
                        "transcribe",
                        "not-a-valid-url",
                        "--source",
                        "youtube",
                        "-o",
                        str(out_dir),
                    ],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

        assert exc_info.value.code == 1
        err = stderr_capture.getvalue()
        assert "invalid" in err.lower() or "url" in err.lower()

    def test_youtube_no_yt_dlp_exits_with_error(self, tmp_path: Path) -> None:
        """Test that missing yt-dlp causes a clean exit."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch(
            "transcribe_src.youtube_processor.get_youtube_video_info"
        ) as mock_info:
            mock_info.side_effect = RuntimeError("yt-dlp not found")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    [
                        "transcribe",
                        "https://youtube.com/watch?v=test",
                        "--source",
                        "youtube",
                        "-o",
                        str(out_dir),
                    ],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

        assert exc_info.value.code == 1
        err = stderr_capture.getvalue()
        assert "yt-dlp" in err.lower()

    def test_youtube_download_failure_exits_with_error(self, tmp_path: Path) -> None:
        """Test that a failed YouTube download causes a clean exit."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.side_effect = RuntimeError("Download failed")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    [
                        "transcribe",
                        "https://www.youtube.com/watch?v=abc123",
                        "--source",
                        "youtube",
                        "-o",
                        str(out_dir),
                    ],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "download" in err.lower()

    def test_youtube_info_failure_exits_with_error(self, tmp_path: Path) -> None:
        """Test that a failed video info fetch causes a clean exit."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch(
            "transcribe_src.youtube_processor.get_youtube_video_info"
        ) as mock_info:
            mock_info.side_effect = RuntimeError("yt-dlp not found")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    [
                        "transcribe",
                        "https://www.youtube.com/watch?v=abc123",
                        "--source",
                        "youtube",
                        "-o",
                        str(out_dir),
                    ],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

            assert exc_info.value.code == 1
            err = stderr_capture.getvalue()
            assert "yt-dlp" in err.lower() or "not found" in err.lower()

    def test_youtube_transcription_failure_exits_with_error(
        self, tmp_path: Path
    ) -> None:
        """Test that a failed Whisper transcription of YouTube audio causes exit."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.side_effect = RuntimeError("Whisper error")

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                with patch(
                    "sys.argv",
                    [
                        "transcribe",
                        "https://www.youtube.com/watch?v=abc123",
                        "--source",
                        "youtube",
                        "-o",
                        str(out_dir),
                    ],
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

            assert exc_info.value.code == 1


class TestSRTGeneration:
    """Test SRT generation paths in the pipeline."""

    def test_local_file_srt_generation(self, tmp_path: Path) -> None:
        """Test that --srt flag causes SRT file to be generated."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        fake_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
        ]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                ["transcribe", str(audio), "-o", str(out_dir), "--srt"],
            ):
                main()

            # Verify transcribe_audio was called (pipeline completed)
            mock_transcribe.assert_called_once()

    def test_youtube_srt_generation(self, tmp_path: Path) -> None:
        """Test that --srt flag causes SRT generation for YouTube videos."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        fake_segments = [
            {"start": 0.0, "end": 2.0, "text": "YouTube content"},
        ]

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                [
                    "transcribe",
                    "https://www.youtube.com/watch?v=abc123",
                    "--source",
                    "youtube",
                    "-o",
                    str(out_dir),
                    "--srt",
                ],
            ):
                main()

            mock_transcribe.assert_called_once()


class TestSuccessfulPipeline:
    """Test successful pipeline with output file verification."""

    def test_local_pipeline_writes_txt(self, tmp_path: Path) -> None:
        """Test that successful local pipeline writes correct .txt file."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"

        fake_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
            {"start": 1.0, "end": 3.0, "text": "This is a test"},
        ]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch("sys.argv", ["transcribe", str(audio), "-o", str(out_dir)]):
                main()

            # Verify output .txt file was written
            txt_path = out_dir / "audio.txt"
            assert txt_path.exists()
            content = txt_path.read_text()
            assert "Hello world" in content
            assert "This is a test" in content
            assert "source: local" in content

    def test_local_pipeline_with_srt_writes_srt(self, tmp_path: Path) -> None:
        """Test that --srt flag creates correct .srt file."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"

        fake_segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello world"},
            {"start": 1.5, "end": 3.0, "text": "Second line"},
        ]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv", ["transcribe", str(audio), "-o", str(out_dir), "--srt"]
            ):
                main()

            # Verify both .txt and .srt exist
            txt_path = out_dir / "audio.txt"
            srt_path = out_dir / "audio.srt"
            assert txt_path.exists()
            assert srt_path.exists()

            # Verify SRT content
            srt_content = srt_path.read_text()
            assert "1\n" in srt_content
            assert "00:00:00,000 --> 00:00:01,500" in srt_content
            assert "Hello world" in srt_content
            assert "2\n" in srt_content
            assert "00:00:01,500 --> 00:00:03,000" in srt_content
            assert "Second line" in srt_content

    def test_local_pipeline_with_language_metadata(self, tmp_path: Path) -> None:
        """Test that --language flag injects language into metadata."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"

        fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                ["transcribe", str(audio), "-o", str(out_dir), "--language", "en"],
            ):
                main()

            txt_path = out_dir / "audio.txt"
            content = txt_path.read_text()
            assert "language: en" in content

    def test_youtube_pipeline_writes_txt(self, tmp_path: Path) -> None:
        """Test that YouTube pipeline writes correct .txt file."""
        out_dir = tmp_path / "out"

        fake_segments = [
            {"start": 0.0, "end": 2.0, "text": "YouTube content"},
        ]

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                [
                    "transcribe",
                    "https://www.youtube.com/watch?v=abc123",
                    "--source",
                    "youtube",
                    "-o",
                    str(out_dir),
                ],
            ):
                main()

            # Verify output .txt file was written
            txt_files = list(out_dir.glob("*.txt"))
            assert len(txt_files) == 1
            content = txt_files[0].read_text()
            assert "YouTube content" in content
            assert "source: youtube" in content
            assert "video_id: abc123" in content

    def test_youtube_pipeline_with_srt_writes_srt(self, tmp_path: Path) -> None:
        """Test that YouTube pipeline with --srt creates .srt file."""
        out_dir = tmp_path / "out"

        fake_segments = [
            {"start": 0.0, "end": 2.0, "text": "YouTube content"},
        ]

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                [
                    "transcribe",
                    "https://www.youtube.com/watch?v=abc123",
                    "--source",
                    "youtube",
                    "-o",
                    str(out_dir),
                    "--srt",
                ],
            ):
                main()

            # Verify both .txt and .srt exist
            txt_files = list(out_dir.glob("*.txt"))
            srt_files = list(out_dir.glob("*.srt"))
            assert len(txt_files) == 1
            assert len(srt_files) == 1

            srt_content = srt_files[0].read_text()
            assert "00:00:00,000 --> 00:00:02,000" in srt_content
            assert "YouTube content" in srt_content


class TestVerboseAndCleanup:
    """Test verbose mode and cleanup flag behavior."""

    def test_local_verbose_mode(self, tmp_path: Path, capsys) -> None:
        """Test that verbose mode propagates to transcribe_audio."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"

        fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv", ["transcribe", str(audio), "-o", str(out_dir), "--verbose"]
            ):
                main()

            # Verify verbose was passed through
            mock_process_audio.assert_called_once()
            assert mock_process_audio.call_args.kwargs["verbose"] is True
            mock_transcribe.assert_called_once()
            assert mock_transcribe.call_args.kwargs["verbose"] is True

    def test_local_cleanup_removes_work_dir(self, tmp_path: Path) -> None:
        """Test that --cleanup flag removes the work directory."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "out"

        fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with (
            patch("transcribe_src.local_processor.process_audio") as mock_process_audio,
            patch("transcribe_src.local_processor.transcribe_audio") as mock_transcribe,
            patch("transcribe_src.local_processor.shutil.rmtree") as mock_rmtree,
        ):
            mock_process_audio.return_value = tmp_path / "processed.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv", ["transcribe", str(audio), "-o", str(out_dir), "--cleanup"]
            ):
                main()

            # Verify shutil.rmtree was called
            mock_rmtree.assert_called_once()
            work_dir = out_dir / "work" / "audio"
            assert str(work_dir) in str(mock_rmtree.call_args[0][0])

    def test_youtube_verbose_mode(self, tmp_path: Path) -> None:
        """Test verbose mode for YouTube pipeline."""
        out_dir = tmp_path / "out"

        fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                [
                    "transcribe",
                    "https://www.youtube.com/watch?v=abc123",
                    "--source",
                    "youtube",
                    "-o",
                    str(out_dir),
                    "--verbose",
                ],
            ):
                main()

            # Verify verbose propagated
            mock_transcribe.assert_called_once()
            assert mock_transcribe.call_args.kwargs["verbose"] is True

    def test_youtube_cleanup_removes_work_dir(self, tmp_path: Path) -> None:
        """Test cleanup flag for YouTube pipeline."""
        out_dir = tmp_path / "out"

        fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with (
            patch(
                "transcribe_src.youtube_processor.get_youtube_video_info"
            ) as mock_info,
            patch(
                "transcribe_src.youtube_processor.download_youtube_audio"
            ) as mock_download,
            patch(
                "transcribe_src.youtube_processor.transcribe_audio"
            ) as mock_transcribe,
            patch("transcribe_src.youtube_processor.shutil.rmtree") as mock_rmtree,
        ):
            mock_info.return_value = {
                "id": "abc123",
                "title": "Test",
                "url": "https://www.youtube.com/watch?v=abc123",
            }
            mock_download.return_value = tmp_path / "downloaded.wav"
            mock_transcribe.return_value = fake_segments

            with patch(
                "sys.argv",
                [
                    "transcribe",
                    "https://www.youtube.com/watch?v=abc123",
                    "--source",
                    "youtube",
                    "-o",
                    str(out_dir),
                    "--cleanup",
                ],
            ):
                main()

            # Verify shutil.rmtree was called
            mock_rmtree.assert_called_once()
            work_dir = out_dir / "work"
            assert str(work_dir) in str(mock_rmtree.call_args[0][0])


class TestMultiInput:
    """Test multi-input and auto-detection support."""

    def test_is_youtube_url_detection(self) -> None:
        """Test that _is_youtube_url correctly identifies YouTube URLs."""
        from transcribe_src.__main__ import _is_youtube_url

        assert _is_youtube_url("https://www.youtube.com/watch?v=abc123")
        assert _is_youtube_url("https://youtu.be/abc123")
        assert _is_youtube_url("https://youtube.com/shorts/abc123")
        assert not _is_youtube_url("/path/to/file.mp3")
        assert not _is_youtube_url("file.mp4")
        assert not _is_youtube_url("https://example.com/video")

    @patch("transcribe_src.__main__.process_local_file")
    def test_multiple_local_files(self, mock_process, tmp_path: Path) -> None:
        """Test processing multiple local files in one invocation."""
        audio1 = tmp_path / "audio1.mp3"
        audio1.write_text("fake audio")
        audio2 = tmp_path / "audio2.mp3"
        audio2.write_text("fake audio")
        out_dir = tmp_path / "output"

        mock_process.return_value = [out_dir / "audio1.txt"]

        with patch(
            "sys.argv",
            ["transcribe", str(audio1), str(audio2), "-o", str(out_dir)],
        ):
            main()

        assert mock_process.call_count == 2
        assert mock_process.call_args_list[0][0][1] == audio1
        assert mock_process.call_args_list[1][0][1] == audio2

    @patch("transcribe_src.__main__.process_youtube_video")
    @patch("transcribe_src.__main__.process_local_file")
    def test_mixed_inputs_auto_detect(
        self, mock_local, mock_youtube, tmp_path: Path
    ) -> None:
        """Test that mixed local/YouTube inputs are auto-detected."""
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake audio")
        out_dir = tmp_path / "output"

        mock_local.return_value = [out_dir / "audio.txt"]
        mock_youtube.return_value = [out_dir / "yt.txt"]

        with patch(
            "sys.argv",
            [
                "transcribe",
                str(audio),
                "https://www.youtube.com/watch?v=abc123",
                "-o",
                str(out_dir),
            ],
        ):
            main()

        mock_local.assert_called_once()
        mock_youtube.assert_called_once()

    @patch("transcribe_src.__main__.process_local_file")
    def test_explicit_source_applies_to_all(self, mock_process, tmp_path: Path) -> None:
        """Test that --source forces all inputs to that type."""
        f1 = tmp_path / "audio1.mp3"
        f1.write_text("fake")
        f2 = tmp_path / "audio2.mp3"
        f2.write_text("fake")
        out_dir = tmp_path / "output"

        mock_process.return_value = [out_dir / "out.txt"]

        with patch(
            "sys.argv",
            ["transcribe", str(f1), str(f2), "--source", "local", "-o", str(out_dir)],
        ):
            main()

        assert mock_process.call_count == 2
