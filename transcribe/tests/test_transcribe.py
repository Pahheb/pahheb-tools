"""Tests for transcribe tool."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from transcribe.config import Config
from transcribe.file_writer import (
    sanitize_filename,
    format_srt_timestamp,
    write_transcript_txt,
    write_transcript_srt,
)
from transcribe.progress import ProgressTracker


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
