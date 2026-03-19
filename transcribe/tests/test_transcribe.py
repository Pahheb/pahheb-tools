"""Tests for transcribe tool."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_src.config import Config
from transcribe_src.file_writer import (
    format_srt_timestamp,
    sanitize_filename,
    write_transcript_srt,
    write_transcript_txt,
)
from transcribe_src.progress import ProgressTracker
from transcribe_src.youtube_downloader import (
    YouTubeDownloadError,
    download_youtube_audio,
    get_youtube_video_info,
)


class TestConfig:
    """Tests for Config class."""

    def test_config_local_defaults(self):
        """Test default config for local source."""
        config = Config(source="local", input_path="test.mp3")
        assert config.source == "local"
        assert config.model_size == "small"
        assert config.device == "auto"
        assert config.denoise is False
        assert config.vad is False
        assert config.audio_enhance is False
        assert config.srt is False

    def test_config_youtube_with_options(self):
        """Test config with various options."""
        config = Config(
            source="youtube",
            input_path="https://youtube.com/watch?v=test",
            output_dir=Path("/tmp/output"),
            language="en",
            model_size="medium",
            device="cuda",
            denoise=True,
            vad=True,
            srt=True,
        )
        assert config.source == "youtube"
        assert config.language == "en"
        assert config.model_size == "medium"
        assert config.denoise is True
        assert config.vad is True
        assert config.srt is True

    def test_config_invalid_source(self):
        """Test config with invalid source."""
        with pytest.raises(ValueError, match="Invalid source"):
            Config(source="invalid", input_path="test.mp3")

    def test_config_invalid_model(self):
        """Test config with invalid model size."""
        with pytest.raises(ValueError, match="Invalid model_size"):
            Config(source="local", input_path="test.mp3", model_size="invalid")

    def test_config_output_dir_path(self):
        """Test output_dir is converted to Path."""
        config = Config(source="local", input_path="test.mp3", output_dir="/tmp/out")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/out")


class TestFileWriter:
    """Tests for file writer functions."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test file") == "test file"
        assert sanitize_filename("test<>file") == "test__file"
        assert sanitize_filename("  spaces  ") == "spaces"
        assert sanitize_filename("a" * 150, max_length=10) == "aaaaaaaaaa"

    def test_sanitize_filename_empty(self):
        """Test sanitization with only special chars."""
        result = sanitize_filename("<>:/\\|?*")
        assert result == "________"

    def test_format_srt_timestamp(self):
        """Test SRT timestamp formatting."""
        assert format_srt_timestamp(0) == "00:00:00,000"
        assert format_srt_timestamp(61.5) == "00:01:01,500"
        assert format_srt_timestamp(3661.123) == "01:01:01,123"
        assert format_srt_timestamp(-1) == "00:00:00,000"

    def test_write_transcript_txt(self, tmp_path):
        """Test writing TXT transcript."""
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
            {"text": "This is a test", "start": 1.0, "end": 3.0},
        ]
        output_path = tmp_path / "test.txt"

        result = write_transcript_txt(segments, output_path)

        assert result == output_path
        content = output_path.read_text()
        assert "Hello world" in content
        assert "This is a test" in content

    def test_write_transcript_txt_with_metadata(self, tmp_path):
        """Test writing TXT with metadata."""
        segments = [{"text": "Hello", "start": 0.0, "end": 1.0}]
        output_path = tmp_path / "test.txt"
        metadata = {"source": "local", "title": "Test Video"}

        write_transcript_txt(segments, output_path, metadata)

        content = output_path.read_text()
        assert "source: local" in content
        assert "title: Test Video" in content

    def test_write_transcript_srt(self, tmp_path):
        """Test writing SRT transcript."""
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.5},
            {"text": "Second line", "start": 1.5, "end": 3.0},
        ]
        output_path = tmp_path / "test.srt"

        result = write_transcript_srt(segments, output_path)

        assert result == output_path
        content = output_path.read_text()
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:01,500" in content
        assert "2\n" in content
        assert "00:00:01,500 --> 00:00:03,000" in content


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(10, "Test")
        assert tracker.total == 10
        assert tracker.current == 0
        assert tracker.description == "Test"

    def test_progress_update(self):
        """Test progress update."""
        tracker = ProgressTracker(10, "Test")
        tracker.update(5)
        assert tracker.current == 5
        tracker.update()
        assert tracker.current == 6


class TestYouTubeDownloader:
    """Tests for YouTube downloader functions."""

    @patch("subprocess.run")
    def test_download_returns_newly_downloaded_file(self, mock_run, tmp_path):
        """Test that download returns the newly downloaded file, not existing ones.

        This is a regression test for the bug where downloading a second video
        would return the first video's audio file if it already existed in
        the work directory.
        """
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        existing_audio = tmp_path / "existing_video.wav"
        existing_audio.write_text("existing audio content")

        import time

        time.sleep(0.01)

        new_audio = tmp_path / "new_video_title_newvideo.wav"
        new_audio.write_text("new audio content")

        result = download_youtube_audio(
            "https://youtube.com/watch?v=newvideo",
            tmp_path,
            verbose=False,
        )

        assert "newvideo" in str(result)
        assert "existing" not in str(result)

    @patch("subprocess.run")
    def test_download_returns_first_file_when_only_one_exists(self, mock_run, tmp_path):
        """Test return first file when only one exists (backward compatibility)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        audio_file = tmp_path / "some_video.wav"
        audio_file.write_text("audio content")

        result = download_youtube_audio(
            "https://youtube.com/watch?v=somevideo",
            tmp_path,
            verbose=False,
        )

        assert result == audio_file

    @patch("subprocess.run")
    def test_download_raises_on_empty_directory(self, mock_run, tmp_path):
        """Test that download raises error when no audio file is downloaded."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with pytest.raises(YouTubeDownloadError, match="No audio file"):
            download_youtube_audio(
                "https://youtube.com/watch?v=test",
                tmp_path,
                verbose=False,
            )

    @patch("subprocess.run")
    def test_download_raises_on_yt_dlp_failure(self, mock_run, tmp_path):
        """Test that download raises error when yt-dlp fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="ERROR: Video not found")

        with pytest.raises(YouTubeDownloadError, match="yt-dlp failed"):
            download_youtube_audio(
                "https://youtube.com/watch?v=invalid",
                tmp_path,
                verbose=False,
            )

    @patch("subprocess.run")
    def test_download_raises_when_yt_dlp_not_found(self, mock_run, tmp_path):
        """Test error when yt-dlp is not installed."""
        mock_run.side_effect = FileNotFoundError("yt-dlp")

        with pytest.raises(YouTubeDownloadError, match="yt-dlp not found"):
            download_youtube_audio(
                "https://youtube.com/watch?v=test",
                tmp_path,
                verbose=False,
            )

    @patch("subprocess.run")
    def test_get_youtube_video_info_parses_json(self, mock_run, tmp_path):
        """Test that get_youtube_video_info correctly parses video info."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"id": "abc123", "title": "Test Video", "webpage_url": "https://youtube.com/watch?v=abc123", "duration": 120}',
            stderr="",
        )

        result = get_youtube_video_info("https://youtube.com/watch?v=abc123")

        assert result["id"] == "abc123"
        assert result["title"] == "Test Video"
        assert result["url"] == "https://youtube.com/watch?v=abc123"
        assert result["duration"] == 120

    @patch("subprocess.run")
    def test_get_youtube_video_info_raises_on_invalid_json(self, mock_run, tmp_path):
        """Test error handling for invalid JSON response."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not valid json", stderr=""
        )

        with pytest.raises(YouTubeDownloadError, match="Failed to parse video info"):
            get_youtube_video_info("https://youtube.com/watch?v=test")

    @patch("subprocess.run")
    def test_get_youtube_video_info_json_import_available(self, mock_run, tmp_path):
        """Test that json module is available for exception handling."""
        import transcribe_src.youtube_downloader as youtube_dl

        assert hasattr(youtube_dl, "json")
        assert youtube_dl.json.__name__ == "json"

    @patch("subprocess.run")
    def test_get_youtube_video_info_handles_yt_dlp_error(self, mock_run, tmp_path):
        """Test error handling when yt-dlp fails."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: Unable to extract video data"
        )

        with pytest.raises(YouTubeDownloadError, match="yt-dlp failed"):
            get_youtube_video_info("https://youtube.com/watch?v=test")

    @patch("subprocess.run")
    def test_download_creates_output_directory(self, mock_run, tmp_path):
        """Test that download creates output directory if it doesn't exist."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        output_dir = tmp_path / "new" / "nested" / "dir"
        audio_file = output_dir / "test_video.wav"
        audio_file.parent.mkdir(parents=True)
        audio_file.write_text("content")

        download_youtube_audio(
            "https://youtube.com/watch?v=test",
            output_dir,
            verbose=False,
        )

        assert output_dir.exists()


class TestConfigEdgeCases:
    """Tests for Config validation edge cases."""

    def test_config_invalid_device(self):
        """Test config with invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            Config(source="local", input_path="test.mp3", device="tpu")

    def test_config_valid_devices(self):
        """Test all valid devices are accepted."""
        for device in ("auto", "cuda", "cpu", "mps"):
            config = Config(source="local", input_path="test.mp3", device=device)
            assert config.device == device


class TestSanitizeFilenameEdgeCases:
    """Tests for sanitize_filename edge cases."""

    def test_sanitize_empty_string(self):
        """Test that empty string returns 'untitled'."""
        assert sanitize_filename("") == "untitled"

    def test_sanitize_only_dots_and_spaces(self):
        """Test that only dots and spaces returns 'untitled'."""
        assert sanitize_filename("...   ") == "untitled"
        assert sanitize_filename("   ...   ") == "untitled"

    def test_sanitize_unicode_characters(self):
        """Test that unicode characters are preserved."""
        assert sanitize_filename("résumé.mp4") == "résumé.mp4"
        assert sanitize_filename("日本語テスト") == "日本語テスト"

    def test_sanitize_truncation_at_word_boundary(self):
        """Test truncation with word boundary (rsplit behavior)."""
        result = sanitize_filename("hello world this is a long filename", max_length=12)
        # With rsplit(" ", 1), "hello world " → "hello world"
        assert result == "hello world"

    def test_sanitize_exact_max_length(self):
        """Test string exactly at max_length is unchanged."""
        assert sanitize_filename("abcdefghij", max_length=10) == "abcdefghij"

    def test_sanitize_whitespace_only(self):
        """Test that whitespace-only string returns 'untitled'."""
        assert sanitize_filename("   ") == "untitled"


class TestSrtTimestampEdgeCases:
    """Tests for format_srt_timestamp edge cases."""

    def test_fractional_millisecond_rounding_up(self):
        """Test rounding of fractional milliseconds."""
        result = format_srt_timestamp(0.0005)
        # 0.0005 * 1000 = 0.5, Python banker's rounding → 0
        assert result == "00:00:00,000"

    def test_fractional_millisecond_rounding_down(self):
        """Test rounding down near boundary."""
        result = format_srt_timestamp(0.0004)
        # 0.0004 * 1000 = 0.4, round → 0ms
        assert result == "00:00:00,000"

    def test_near_one_second_rounding(self):
        """Test rounding near 1.0 second boundary."""
        result = format_srt_timestamp(0.9995)
        # 0.9995 * 1000 = 999.5, round → 1000ms = 1s
        assert result == "00:00:01,000"

    def test_large_hour_value(self):
        """Test timestamps exceeding 99 hours."""
        result = format_srt_timestamp(100.0 * 3600)
        assert result == "100:00:00,000"


class TestTranscriptFileWriterEdgeCases:
    """Tests for file writer edge cases."""

    def test_write_transcript_txt_empty_segments(self, tmp_path):
        """Test writing empty segments list produces valid file."""
        output_path = tmp_path / "empty.txt"
        result = write_transcript_txt([], output_path)
        assert result == output_path
        assert output_path.exists()

    def test_write_transcript_txt_strips_whitespace(self, tmp_path):
        """Test that segment text with leading/trailing whitespace is stripped."""
        segments = [{"text": "  hello world  ", "start": 0.0, "end": 1.0}]
        output_path = tmp_path / "whitespace.txt"
        write_transcript_txt(segments, output_path)
        content = output_path.read_text()
        assert "hello world" in content
        assert "  hello world  " not in content

    def test_write_transcript_srt_empty_segments(self, tmp_path):
        """Test writing empty SRT segments produces valid file."""
        output_path = tmp_path / "empty.srt"
        result = write_transcript_srt([], output_path)
        assert result == output_path
        assert output_path.exists()


class TestProgressTrackerEdgeCases:
    """Tests for ProgressTracker edge cases."""

    def test_progress_zero_total_no_division_error(self):
        """Test that total=0 doesn't cause division by zero."""
        tracker = ProgressTracker(0, "Empty")
        tracker.update(1)
        assert tracker.current == 1

    def test_progress_update_with_message(self, capsys):
        """Test update with custom message."""
        tracker = ProgressTracker(10, "Processing")
        tracker.update(3, message="3 items done")
        captured = capsys.readouterr()
        assert "3 items done" in captured.out

    def test_progress_complete(self, capsys):
        """Test complete method."""
        tracker = ProgressTracker(10, "Task")
        tracker.complete()
        captured = capsys.readouterr()
        assert "Task: Complete" in captured.out

    def test_progress_update_custom_message(self, capsys):
        """Test complete with custom message."""
        tracker = ProgressTracker(10, "Task")
        tracker.complete(message="All done!")
        captured = capsys.readouterr()
        assert "Task: All done!" in captured.out


class TestYouTubeDownloaderEdgeCases:
    """Tests for YouTube downloader edge cases."""

    @patch("subprocess.run")
    def test_get_youtube_video_info_timeout(self, mock_run):
        """Test timeout handling in get_youtube_video_info."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["yt-dlp"], timeout=60)

        with pytest.raises(YouTubeDownloadError, match="timed out"):
            get_youtube_video_info("https://youtube.com/watch?v=test")

    @patch("subprocess.run")
    def test_get_youtube_video_info_yt_dlp_not_found(self, mock_run):
        """Test FileNotFoundError in get_youtube_video_info (latent bug fix)."""
        mock_run.side_effect = FileNotFoundError("yt-dlp")

        with pytest.raises(YouTubeDownloadError, match="yt-dlp not found"):
            get_youtube_video_info("https://youtube.com/watch?v=test")

    @patch("subprocess.run")
    def test_get_youtube_video_info_verbose(self, mock_run):
        """Test verbose mode in get_youtube_video_info doesn't include --quiet."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"id": "x", "title": "T", "webpage_url": "https://u", "duration": 1}',
            stderr="",
        )

        get_youtube_video_info("https://youtube.com/watch?v=x", verbose=True)

        call_args = mock_run.call_args[0][0]
        assert "--quiet" not in call_args


try:
    import torch  # noqa: F401

    _torch_available = True
except ImportError:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestWhisperHelpers:
    """Tests for whisper helper functions (skipped if torch not available)."""

    def test_get_compute_type_cpu_default(self):
        """Test default compute type for CPU."""
        from transcribe_src.whisper import get_compute_type

        assert get_compute_type("cpu") == "int8"

    def test_get_compute_type_cuda_default(self):
        """Test default compute type for CUDA."""
        from transcribe_src.whisper import get_compute_type

        assert get_compute_type("cuda") == "float16"

    def test_get_compute_type_mps_default(self):
        """Test default compute type for MPS."""
        from transcribe_src.whisper import get_compute_type

        assert get_compute_type("mps") == "float16"

    def test_get_compute_type_unknown_default(self):
        """Test default compute type for unknown device."""
        from transcribe_src.whisper import get_compute_type

        assert get_compute_type("tpu") == "int8"

    def test_get_compute_type_explicit_override(self):
        """Test explicit compute_type overrides default."""
        from transcribe_src.whisper import get_compute_type

        assert get_compute_type("cpu", "float32") == "float32"
        assert get_compute_type("cuda", "int8") == "int8"


class TestAudioProcessorHelpers:
    """Tests for audio_processor helper functions."""

    def test_find_denoise_model_finds_std_rnnn(self, tmp_path):
        """Test finding std.rnnn in search path."""
        from transcribe_src.audio_processor import find_denoise_model

        model = tmp_path / "std.rnnn"
        model.write_text("fake model")

        result = find_denoise_model([tmp_path])
        assert result == model

    def test_find_denoise_model_finds_model_rnnn(self, tmp_path):
        """Test finding model.rnnn in search path."""
        from transcribe_src.audio_processor import find_denoise_model

        model = tmp_path / "model.rnnn"
        model.write_text("fake model")

        result = find_denoise_model([tmp_path])
        assert result == model

    def test_find_denoise_model_searches_directories(self, tmp_path):
        """Test that find_denoise_model searches inside directories."""
        from transcribe_src.audio_processor import find_denoise_model

        subdir = tmp_path / "models"
        subdir.mkdir()
        model = subdir / "std.rnnn"
        model.write_text("fake model")

        result = find_denoise_model([tmp_path / "missing", subdir])
        assert result == model

    def test_find_denoise_model_not_found(self, tmp_path):
        """Test that find_denoise_model returns None when no model exists."""
        from transcribe_src.audio_processor import find_denoise_model

        (tmp_path / "not_a_model.txt").write_text("not a model")

        result = find_denoise_model([tmp_path])
        assert result is None

    def test_find_denoise_model_nonexistent_paths(self, tmp_path):
        """Test that nonexistent search paths are skipped."""
        from transcribe_src.audio_processor import find_denoise_model

        result = find_denoise_model([tmp_path / "nonexistent"])
        assert result is None

    def test_get_default_model_search_paths(self):
        """Test get_default_model_search_paths returns expected paths."""
        from transcribe_src.audio_processor import get_default_model_search_paths

        paths = get_default_model_search_paths()
        assert len(paths) == 2
        assert paths[0] == Path("./models")
        assert "transcribe" in str(paths[1])
        assert "models" in str(paths[1])

    def test_find_denoise_model_direct_file_match(self, tmp_path):
        """Test finding model when search path IS the model file."""
        from transcribe_src.audio_processor import find_denoise_model

        model = tmp_path / "std.rnnn"
        model.write_text("fake model")

        result = find_denoise_model([model])
        assert result == model

    def test_find_denoise_model_priority(self, tmp_path):
        """Test that first found model wins (std.rnnn before model.rnnn)."""
        from transcribe_src.audio_processor import find_denoise_model

        std_model = tmp_path / "std.rnnn"
        std_model.write_text("std")
        other_model = tmp_path / "model.rnnn"
        other_model.write_text("other")

        result = find_denoise_model([tmp_path])
        assert result == std_model


class TestAudioProcessor:
    """Tests for audio_processor subprocess calls (mocked)."""

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_success(self, mock_run, tmp_path):
        """Test successful audio processing returns output path."""
        from transcribe_src.audio_processor import process_audio

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create the output file that ffmpeg would create
        output_dir = tmp_path / "work"
        output_dir.mkdir(parents=True)
        output_file = output_dir / "audio_16k_mono.wav"

        def side_effect(*args, **kwargs):
            # Simulate ffmpeg creating the output file
            output_file.write_text("fake wav")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")

        result = process_audio(input_file, output_dir)
        assert result == output_file
        assert "ffmpeg" in mock_run.call_args[0][0][0]

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_failure_raises_error(self, mock_run, tmp_path):
        """Test that ffmpeg failure raises AudioProcessingError."""
        from transcribe_src.audio_processor import AudioProcessingError, process_audio

        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="some ffmpeg error"
        )

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")

        with pytest.raises(AudioProcessingError, match="ffmpeg failed"):
            process_audio(input_file, tmp_path / "work")

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_timeout_raises_error(self, mock_run, tmp_path):
        """Test that ffmpeg timeout raises AudioProcessingError."""
        from transcribe_src.audio_processor import AudioProcessingError, process_audio

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600)

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")

        with pytest.raises(AudioProcessingError, match="timed out"):
            process_audio(input_file, tmp_path / "work")

    @patch("transcribe_src.audio_processor.check_ffmpeg_arnndn_support")
    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_denoise_with_model(self, mock_run, mock_arnndn, tmp_path):
        """Test denoise with model includes arnndn filter in command."""
        from transcribe_src.audio_processor import process_audio

        mock_arnndn.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        model = tmp_path / "std.rnnn"
        model.write_text("fake model")
        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"
        (output_dir / "audio_16k_mono.wav").parent.mkdir(parents=True, exist_ok=True)

        def side_effect(*args, **kwargs):
            (output_dir / "audio_16k_mono.wav").write_text("fake")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        process_audio(input_file, output_dir, denoise=True, denoise_model=str(model))

        cmd = mock_run.call_args[0][0]
        af_idx = cmd.index("-af") + 1
        assert "arnndn" in cmd[af_idx]

    @patch("transcribe_src.audio_processor.check_ffmpeg_arnndn_support")
    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_denoise_without_model_no_crash(
        self, mock_run, mock_arnndn, tmp_path
    ):
        """Test denoise=True without model found doesn't crash."""
        from transcribe_src.audio_processor import process_audio

        mock_arnndn.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"
        (output_dir / "audio_16k_mono.wav").parent.mkdir(parents=True, exist_ok=True)

        def side_effect(*args, **kwargs):
            (output_dir / "audio_16k_mono.wav").write_text("fake")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        # No model file exists, but arnndn is supported — should not crash
        process_audio(input_file, output_dir, denoise=True)

        # Should still call ffmpeg, just without arnndn filter
        mock_run.assert_called_once()

    @patch("transcribe_src.audio_processor.check_ffmpeg_arnndn_support")
    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_denoise_no_arnndn_support(
        self, mock_run, mock_arnndn, tmp_path, capsys
    ):
        """Test denoise=True without arnndn support prints warning."""
        from transcribe_src.audio_processor import process_audio

        mock_arnndn.return_value = False
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"
        (output_dir / "audio_16k_mono.wav").parent.mkdir(parents=True, exist_ok=True)

        def side_effect(*args, **kwargs):
            (output_dir / "audio_16k_mono.wav").write_text("fake")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        process_audio(input_file, output_dir, denoise=True, verbose=True)

        captured = capsys.readouterr()
        assert "arnndn" in captured.out.lower()

    @patch("transcribe_src.audio_processor.check_ffmpeg_arnndn_support")
    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_denoise_model_not_found_error(
        self, mock_run, mock_arnndn, tmp_path
    ):
        """Test denoise with non-existent model path raises AudioProcessingError."""
        from transcribe_src.audio_processor import AudioProcessingError, process_audio

        mock_arnndn.return_value = True

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"

        with pytest.raises(AudioProcessingError, match="Denoise model not found"):
            process_audio(
                input_file,
                output_dir,
                denoise=True,
                denoise_model="/nonexistent/model.rnnn",
            )

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_enhance_filters(self, mock_run, tmp_path):
        """Test audio_enhance=True adds enhancement filters."""
        from transcribe_src.audio_processor import process_audio

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"
        (output_dir / "audio_16k_mono.wav").parent.mkdir(parents=True, exist_ok=True)

        def side_effect(*args, **kwargs):
            (output_dir / "audio_16k_mono.wav").write_text("fake")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        process_audio(input_file, output_dir, audio_enhance=True)

        cmd = mock_run.call_args[0][0]
        af_idx = cmd.index("-af") + 1
        assert "highpass" in cmd[af_idx]
        assert "lowpass" in cmd[af_idx]
        assert "compand" in cmd[af_idx]
        assert "loudnorm" in cmd[af_idx]

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_output_not_created(self, mock_run, tmp_path):
        """Test error when ffmpeg succeeds but output file is missing."""
        from transcribe_src.audio_processor import AudioProcessingError, process_audio

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"

        with pytest.raises(AudioProcessingError, match="Output file not created"):
            process_audio(input_file, output_dir)

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_process_audio_verbose_shows_command(self, mock_run, tmp_path, capsys):
        """Test verbose mode prints ffmpeg command."""
        from transcribe_src.audio_processor import process_audio

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_file = tmp_path / "input.mp3"
        input_file.write_text("fake audio")
        output_dir = tmp_path / "work"
        (output_dir / "audio_16k_mono.wav").parent.mkdir(parents=True, exist_ok=True)

        def side_effect(*args, **kwargs):
            (output_dir / "audio_16k_mono.wav").write_text("fake")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        process_audio(input_file, output_dir, verbose=True)

        captured = capsys.readouterr()
        assert "Running:" in captured.out
        assert "ffmpeg" in captured.out

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_check_ffmpeg_arnndn_support_true(self, mock_run):
        """Test check_ffmpeg returns True when arnndn filter is present."""
        from transcribe_src.audio_processor import check_ffmpeg_arnndn_support

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="AR a arnndn audio RNNoise based denoiser",
            stderr="",
        )
        assert check_ffmpeg_arnndn_support() is True

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_check_ffmpeg_arnndn_support_false(self, mock_run):
        """Test check_ffmpeg returns False when arnndn filter is absent."""
        from transcribe_src.audio_processor import check_ffmpeg_arnndn_support

        mock_run.return_value = MagicMock(
            returncode=0, stdout="some other filters", stderr=""
        )
        assert check_ffmpeg_arnndn_support() is False

    @patch("transcribe_src.audio_processor.subprocess.run")
    def test_check_ffmpeg_arnndn_support_exception(self, mock_run):
        """Test check_ffmpeg returns False when ffmpeg raises exception."""
        from transcribe_src.audio_processor import check_ffmpeg_arnndn_support

        mock_run.side_effect = OSError("ffmpeg not found")
        assert check_ffmpeg_arnndn_support() is False
