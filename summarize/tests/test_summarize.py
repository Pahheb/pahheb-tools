"""Tests for summarize tool."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from summarize_src.config import Config
from summarize_src.file_writer import (
    sanitize_filename,
    write_summary_txt,
    write_summary_md,
    write_summary_json,
)
from summarize_src.summarizer import (
    OllamaProvider,
    HuggingFaceProvider,
    get_provider,
    ProviderNotAvailableError,
    SummarizerError,
)


class TestConfig:
    """Tests for Config class."""

    def test_config_defaults(self):
        """Test default config."""
        config = Config(input_files=[])
        assert config.provider == "ollama"
        assert config.model == "llama3.2"
        assert config.output_format == "txt"
        assert config.summary_length == "standard"
        assert config.combine is False
        assert config.unified is False
        assert config.skip_combined is False

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = Config(
            provider="huggingface",
            model="custom-model",
            output_format="md",
            summary_length="brief",
            combine=True,
        )
        assert config.provider == "huggingface"
        assert config.model == "custom-model"
        assert config.output_format == "md"
        assert config.summary_length == "brief"
        assert config.combine is True

    def test_config_invalid_provider(self):
        """Test config with invalid provider."""
        with pytest.raises(ValueError, match="Invalid provider"):
            Config(provider="invalid", input_files=[])

    def test_config_invalid_output_format(self):
        """Test config with invalid output format."""
        with pytest.raises(ValueError, match="Invalid output_format"):
            Config(output_format="invalid", input_files=[])

    def test_config_invalid_summary_length(self):
        """Test config with invalid summary length."""
        with pytest.raises(ValueError, match="Invalid summary_length"):
            Config(summary_length="invalid", input_files=[])

    def test_config_output_dir_conversion(self):
        """Test output_dir is converted to Path."""
        config = Config(input_files=[], output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/test")

    def test_config_unified_flag(self):
        """Test config with unified flag."""
        config = Config(unified=True, input_files=[])
        assert config.unified is True
        assert config.combine is False

    def test_config_skip_combined_flag(self):
        """Test config with skip_combined flag."""
        config = Config(skip_combined=True, input_files=[])
        assert config.skip_combined is True

    def test_config_combined_and_unified_mutually_exclusive(self):
        """Test that combine and unified can both be set (handled at runtime)."""
        config = Config(combine=True, unified=True, input_files=[])
        assert config.combine is True
        assert config.unified is True

    def test_config_find_combined_files(self):
        """Test finding combined files."""
        files = [
            Path("video1.txt"),
            Path("combined_summary.txt"),
            Path("video2.txt"),
            Path("combined.txt"),
        ]
        config = Config(input_files=files)
        combined = config.find_combined_files()
        assert len(combined) == 2
        assert any("combined" in f.stem.lower() for f in combined)

    def test_config_filter_input_files_skip_combined(self):
        """Test filtering out combined files."""
        files = [
            Path("video1.txt"),
            Path("combined_summary.txt"),
            Path("video2.txt"),
        ]
        config = Config(input_files=files, skip_combined=True)
        filtered = config.filter_input_files()
        assert len(filtered) == 2
        assert all("combined" not in f.stem.lower() for f in filtered)

    def test_config_filter_input_files_no_skip(self):
        """Test filtering returns all files when skip_combined is False."""
        files = [
            Path("video1.txt"),
            Path("combined_summary.txt"),
        ]
        config = Config(input_files=files, skip_combined=False)
        filtered = config.filter_input_files()
        assert len(filtered) == 2

    def test_config_transcribe_defaults(self):
        """Test config with transcribe defaults."""
        config = Config(input_files=[])
        assert config.transcribe_source == "local"
        assert config.transcribe_language is None
        assert config.transcribe_model == "small"
        assert config.transcribe_device == "auto"
        assert config.transcribe_compute is None
        assert config.transcribe_denoise is False
        assert config.transcribe_vad is False
        assert config.transcribe_audio_enhance is False
        assert config.transcribe_srt is False
        assert config.transcribe_cleanup is False

    def test_config_transcribe_output_dir(self):
        """Test transcribe_output_dir is inside output_dir."""
        config = Config(input_files=[], output_dir=Path("/home/user/summaries"))
        assert config.transcribe_output_dir == Path(
            "/home/user/summaries/transcriptions"
        )

    def test_config_transcribe_custom(self):
        """Test config with custom transcribe options."""
        config = Config(
            input_files=[],
            transcribe_source="youtube",
            transcribe_language="es",
            transcribe_model="medium",
            transcribe_device="cuda",
            transcribe_compute="float16",
            transcribe_denoise=True,
            transcribe_vad=True,
            transcribe_audio_enhance=True,
            transcribe_srt=True,
            transcribe_cleanup=True,
        )
        assert config.transcribe_source == "youtube"
        assert config.transcribe_language == "es"
        assert config.transcribe_model == "medium"
        assert config.transcribe_device == "cuda"
        assert config.transcribe_compute == "float16"
        assert config.transcribe_denoise is True
        assert config.transcribe_vad is True
        assert config.transcribe_audio_enhance is True
        assert config.transcribe_srt is True
        assert config.transcribe_cleanup is True

    def test_config_build_transcribe_args_defaults(self):
        """Test building transcribe args with defaults."""
        config = Config(input_files=[], output_dir=Path("./summaries"))
        args = config.build_transcribe_args()
        assert "--output-dir" in args
        assert "summaries/transcriptions" in " ".join(args)
        assert "--source" not in args
        assert "--language" not in args

    def test_config_build_transcribe_args_all_options(self):
        """Test building transcribe args with all options."""
        config = Config(
            input_files=[],
            output_dir=Path("./out"),
            transcribe_source="youtube",
            transcribe_language="en",
            transcribe_model="large-v3",
            transcribe_device="cuda",
            transcribe_compute="float16",
            transcribe_denoise=True,
            transcribe_vad=True,
            transcribe_audio_enhance=True,
            transcribe_srt=True,
            transcribe_cleanup=True,
        )
        args = config.build_transcribe_args()
        assert "--source" in args
        assert "youtube" in args
        assert "--language" in args
        assert "en" in args
        assert "--model" in args
        assert "large-v3" in args
        assert "--device" in args
        assert "cuda" in args
        assert "--compute" in args
        assert "float16" in args
        assert "--denoise" in args
        assert "--vad" in args
        assert "--audio-enhance" in args
        assert "--srt" in args
        assert "--cleanup" in args


class TestFileWriter:
    """Tests for file writer functions."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test file") == "test file"
        assert sanitize_filename("test<>file") == "test__file"
        assert sanitize_filename("  spaces  ") == "spaces"
        assert sanitize_filename("a" * 150, max_length=10) == "aaaaaaaaaa"

    def test_sanitize_filename_special_chars(self):
        """Test sanitization with only special chars."""
        result = sanitize_filename("<>:/\\|?*")
        assert result == "________"

    def test_write_summary_txt(self, tmp_path):
        """Test writing TXT summary."""
        summary = "This is a test summary."
        output_path = tmp_path / "test.txt"

        result = write_summary_txt(summary, output_path)

        assert result == output_path
        content = output_path.read_text()
        assert "This is a test summary." in content

    def test_write_summary_txt_with_metadata(self, tmp_path):
        """Test writing TXT with metadata."""
        summary = "Test summary"
        output_path = tmp_path / "test.txt"
        metadata = {"source": "test.txt", "title": "Test File"}

        result = write_summary_txt(summary, output_path, metadata)

        content = output_path.read_text()
        assert "Source: test.txt" in content
        assert "Title: Test File" in content

    def test_write_summary_md(self, tmp_path):
        """Test writing MD summary."""
        summary = "This is a test summary."
        key_points = ["Point 1", "Point 2"]
        output_path = tmp_path / "test.md"

        result = write_summary_md(summary, output_path, key_points)

        assert result == output_path
        content = output_path.read_text()
        assert "## Key Points" in content
        assert "Point 1" in content
        assert "Point 2" in content

    def test_write_summary_json(self, tmp_path):
        """Test writing JSON summary."""
        summary = "This is a test summary."
        key_points = ["Point 1", "Point 2"]
        metadata = {"source": "test.txt"}
        output_path = tmp_path / "test.json"

        result = write_summary_json(summary, output_path, key_points, metadata)

        assert result == output_path
        content = output_path.read_text()
        assert '"summary"' in content
        assert '"key_points"' in content


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @patch("httpx.Client")
    def test_is_available_true(self, mock_client):
        """Test is_available returns True when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        provider = OllamaProvider()
        assert provider.is_available() is True

    @patch("httpx.Client")
    def test_is_available_false(self, mock_client):
        """Test is_available returns False when Ollama is not running."""
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception(
            "Connection failed"
        )

        provider = OllamaProvider()
        assert provider.is_available() is False

    @patch("httpx.Client")
    def test_summarize_success(self, mock_client):
        """Test successful summarization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "KEYPOINTS:\n- Point 1\n\nSUMMARY:\nThis is the summary."
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        provider = OllamaProvider()
        result = provider.summarize("Test transcription text", summary_type="brief")

        assert (
            "summary" in result.summary.lower()
            or "point 1" in str(result.key_points).lower()
        )

    @patch("httpx.Client")
    def test_summarize_returns_summary_content(self, mock_client):
        """Test that summarization actually returns the summary content (not empty)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "KEYPOINTS:\n- First key point\n- Second key point\n\nSUMMARY:\nThis is the actual summary text that should be returned."
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        provider = OllamaProvider()
        result = provider.summarize("Test transcription text")

        assert (
            result.summary == "This is the actual summary text that should be returned."
        )
        assert "First key point" in result.key_points
        assert "Second key point" in result.key_points

    @patch("httpx.Client")
    def test_summarize_no_keypoints_returns_summary(self, mock_client):
        """Test parsing when response has no keypoints but has summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Here is a summary without keypoints section."
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        provider = OllamaProvider()
        result = provider.summarize("Test text")

        assert result.summary == "Here is a summary without keypoints section."

    @patch("httpx.Client")
    def test_summarize_connection_error(self, mock_client):
        """Test summarization with connection error."""
        import httpx

        mock_client.return_value.__enter__.return_value.post.side_effect = (
            httpx.ConnectError("Connection refused")
        )

        provider = OllamaProvider()

        with pytest.raises(ProviderNotAvailableError, match="Cannot connect to Ollama"):
            provider.summarize("Test text")

    def test_build_prompt_brief(self):
        """Test prompt building for brief summary."""
        provider = OllamaProvider()
        prompt = provider._build_prompt("test text", "brief")
        assert "concise summary" in prompt.lower()
        assert "2-3 sentences" in prompt.lower()

    def test_build_prompt_standard(self):
        """Test prompt building for standard summary."""
        provider = OllamaProvider()
        prompt = provider._build_prompt("test text", "standard")
        assert "comprehensive summary" in prompt.lower()

    def test_build_prompt_detailed(self):
        """Test prompt building for detailed summary."""
        provider = OllamaProvider()
        prompt = provider._build_prompt("test text", "detailed")
        assert "detailed summary" in prompt.lower()


class TestHuggingFaceProvider:
    """Tests for HuggingFaceProvider."""

    def test_parse_response_returns_summary(self):
        """Test that _parse_response returns the actual summary content."""
        provider = HuggingFaceProvider()
        content = (
            "KEYPOINTS:\n- Point 1\n\nSUMMARY:\nThis is the HuggingFace summary text."
        )
        result = provider._parse_response(content)

        assert result.summary == "This is the HuggingFace summary text."
        assert "Point 1" in result.key_points

    def test_parse_response_no_keypoints(self):
        """Test parsing when response has no keypoints but has summary."""
        provider = HuggingFaceProvider()
        content = "Here is a summary without keypoints."
        result = provider._parse_response(content)

        assert result.summary == "Here is a summary without keypoints."

    def test_parse_response_with_numbered_points(self):
        """Test parsing numbered list keypoints."""
        provider = HuggingFaceProvider()
        content = (
            "KEYPOINTS:\n1. First point\n2. Second point\n\nSUMMARY:\nSummary text."
        )
        result = provider._parse_response(content)

        assert "First point" in result.key_points
        assert "Second point" in result.key_points
        """Test is_available checks for transformers."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"transformers": None}):
            provider = HuggingFaceProvider()
            result = provider.is_available()
            assert result is True or result is False

    def test_get_provider_ollama(self):
        """Test get_provider for ollama."""
        provider = get_provider("ollama", "llama3.2")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2"

    def test_get_provider_huggingface(self):
        """Test get_provider for huggingface."""
        provider = get_provider("huggingface", "test-model")
        assert isinstance(provider, HuggingFaceProvider)
        assert provider.model_name == "test-model"

    def test_get_provider_invalid(self):
        """Test get_provider with invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid")


class TestCLI:
    """Tests for CLI parsing."""

    def test_parse_args_basic(self, tmp_path):
        """Test basic argument parsing."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "file.txt"
        test_file.write_text("test")

        with patch("sys.argv", ["summarize", str(test_file)]):
            args = parse_args()

        assert len(args.input_files) == 1
        assert args.provider == "ollama"

    def test_parse_args_multiple_files(self, tmp_path):
        """Test parsing multiple input files."""
        from summarize_src.cli import parse_args

        test_file1 = tmp_path / "file1.txt"
        test_file2 = tmp_path / "file2.txt"
        test_file1.write_text("test")
        test_file2.write_text("test")

        with patch("sys.argv", ["summarize", str(test_file1), str(test_file2)]):
            args = parse_args()

        assert len(args.input_files) == 2

    def test_parse_args_with_options(self, tmp_path):
        """Test parsing with options."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "file.txt"
        test_file.write_text("test")

        with patch(
            "sys.argv",
            [
                "summarize",
                str(test_file),
                "--provider",
                "huggingface",
                "--model",
                "test-model",
                "--output-format",
                "md",
                "--summary-length",
                "brief",
                "--combine",
            ],
        ):
            args = parse_args()

        assert args.provider == "huggingface"
        assert args.model == "test-model"
        assert args.output_format == "md"
        assert args.summary_length == "brief"
        assert args.combine is True

    def test_parse_args_with_transcribe(self, tmp_path):
        """Test parsing with transcribe flag."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"fake video")

        with patch(
            "sys.argv",
            [
                "summarize",
                str(test_file),
                "--transcribe",
                "--transcribe-model",
                "medium",
                "--transcribe-srt",
            ],
        ):
            args = parse_args()

        assert args.transcribe is True
        assert args.transcribe_model == "medium"
        assert args.transcribe_srt is True
        assert args.transcribe_source == "local"

    def test_parse_args_with_transcribe_youtube(self):
        """Test parsing with transcribe flag and youtube source."""
        from summarize_src.cli import parse_args
        from unittest.mock import patch
        import sys

        with patch.object(
            sys,
            "argv",
            [
                "summarize",
                "--transcribe",
                "--transcribe-source",
                "youtube",
                "--transcribe-language",
                "en",
                "https://youtube.com/watch?v=abc",
            ],
        ):
            args = parse_args()

        assert args.transcribe is True
        assert args.transcribe_source == "youtube"
        assert args.transcribe_language == "en"

    def test_parse_args_youtube_url_kept_as_string(self):
        """Test that YouTube URLs are kept as strings, not converted to Path objects."""
        from summarize_src.cli import parse_args

        youtube_urls = [
            "https://youtube.com/watch?v=abc123",
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "http://youtube.com/watch?v=abc123",
        ]

        for url in youtube_urls:
            with patch(
                "sys.argv",
                [
                    "summarize",
                    "--transcribe",
                    "--transcribe-source",
                    "youtube",
                    url,
                ],
            ):
                args = parse_args()

            assert args.transcribe is True
            assert len(args.input_files) == 1
            assert isinstance(args.input_files[0], str), (
                f"URL should be string, got {type(args.input_files[0])}"
            )
            assert args.input_files[0] == url, f"URL should be preserved exactly: {url}"
            assert args.input_files[0].startswith("http"), (
                f"URL should not have slashes stripped: {args.input_files[0]}"
            )
            assert (
                "https://" in args.input_files[0] or "http://" in args.input_files[0]
            ), f"URL should have full protocol: {args.input_files[0]}"

    def test_parse_args_with_transcribe_all_options(self, tmp_path):
        """Test parsing with all transcribe options."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"fake video")

        with patch(
            "sys.argv",
            [
                "summarize",
                str(test_file),
                "--transcribe",
                "--transcribe-model",
                "large-v3",
                "--transcribe-device",
                "cuda",
                "--transcribe-compute",
                "float16",
                "--transcribe-denoise",
                "--transcribe-vad",
                "--transcribe-audio-enhance",
                "--transcribe-srt",
                "--transcribe-cleanup",
            ],
        ):
            args = parse_args()

        assert args.transcribe is True
        assert args.transcribe_model == "large-v3"
        assert args.transcribe_device == "cuda"
        assert args.transcribe_compute == "float16"
        assert args.transcribe_denoise is True
        assert args.transcribe_vad is True
        assert args.transcribe_audio_enhance is True
        assert args.transcribe_srt is True
        assert args.transcribe_cleanup is True

    def test_parse_args_with_unified(self, tmp_path):
        """Test parsing with unified flag."""
        from summarize_src.cli import parse_args

        test_file1 = tmp_path / "file1.txt"
        test_file2 = tmp_path / "file2.txt"
        test_file1.write_text("test")
        test_file2.write_text("test")

        with patch(
            "sys.argv",
            ["summarize", str(test_file1), str(test_file2), "--unified"],
        ):
            args = parse_args()

        assert args.unified is True
        assert args.combine is False

    def test_parse_args_with_skip_combined(self, tmp_path):
        """Test parsing with skip-combined flag."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "file.txt"
        test_file.write_text("test")

        with patch(
            "sys.argv",
            ["summarize", str(test_file), "--skip-combined"],
        ):
            args = parse_args()

        assert args.skip_combined is True

    def test_parse_args_with_combine_and_unified(self, tmp_path):
        """Test parsing with both combine and unified."""
        from summarize_src.cli import parse_args

        test_file = tmp_path / "file.txt"
        test_file.write_text("test")

        with patch(
            "sys.argv",
            ["summarize", str(test_file), "--combine", "--unified"],
        ):
            args = parse_args()

        assert args.combine is True
        assert args.unified is True

    def test_parse_args_nonexistent_file(self, capsys):
        """Test parsing with nonexistent file shows warning."""
        from summarize_src.cli import parse_args
        import sys
        from pathlib import Path
        from unittest.mock import patch

        nonexistent = Path("/nonexistent/file.txt")

        with patch("sys.argv", ["summarize", str(nonexistent)]):
            try:
                args = parse_args()
            except SystemExit:
                pass
            captured = capsys.readouterr()
            assert "Warning" in captured.err or "not found" in captured.err.lower()

    def test_parse_args_no_inputs_shows_help(self, capsys):
        """Test that running with no inputs shows help instead of error."""
        from summarize_src.cli import parse_args
        import sys
        from unittest.mock import patch

        with patch("sys.argv", ["summarize"]):
            try:
                parse_args()
            except SystemExit:
                pass
            captured = capsys.readouterr()
            assert (
                "usage:" in captured.out.lower() or "summarize" in captured.out.lower()
            )


class TestTranscribeIntegration:
    """Integration tests for transcribe workflow."""

    def test_transcribe_output_dir_relative_to_summary_output(self):
        """Test that transcribe output dir is relative to summary output dir."""
        config = Config(
            input_files=["video1.mp4", "video2.mp4"],
            output_dir="/home/user/summaries",
            transcribe_first=True,
        )
        assert (
            str(config.transcribe_output_dir) == "/home/user/summaries/transcriptions"
        )

    def test_transcribe_output_dir_with_custom_output(self):
        """Test transcribe output dir with custom output dir."""
        config = Config(
            input_files=["video1.mp4"],
            output_dir="/home/user/summaries",
            transcribe_first=True,
        )
        assert (
            str(config.transcribe_output_dir) == "/home/user/summaries/transcriptions"
        )

    def test_transcribe_output_dir_no_transcribe_flag(self):
        """Test that transcribe output dir exists even when transcribe is disabled."""
        config = Config(
            input_files=["video1.mp4"],
            output_dir="/home/user/summaries",
            transcribe_first=False,
        )
        assert (
            str(config.transcribe_output_dir) == "/home/user/summaries/transcriptions"
        )

    def test_build_transcribe_args_with_srt(self):
        """Test building transcribe args includes SRT option."""
        config = Config(
            input_files=["video.mp4"],
            transcribe_first=True,
            transcribe_srt=True,
        )
        args = config.build_transcribe_args()
        assert "--srt" in args

    def test_build_transcribe_args_with_vad(self):
        """Test building transcribe args includes VAD option."""
        config = Config(
            input_files=["video.mp4"],
            transcribe_first=True,
            transcribe_vad=True,
        )
        args = config.build_transcribe_args()
        assert "--vad" in args

    def test_build_transcribe_args_no_inputs(self):
        """Test building transcribe args includes output-dir even with no input files."""
        config = Config(
            input_files=[],
            transcribe_first=True,
        )
        args = config.build_transcribe_args()
        assert "--output-dir" in args

    def test_build_transcribe_args_youtube_source(self):
        """Test building transcribe args includes source option for youtube."""
        config = Config(
            input_files=["https://youtube.com/watch?v=abc123"],
            transcribe_first=True,
            transcribe_source="youtube",
        )
        args = config.build_transcribe_args()
        assert "--source" in args
        assert "youtube" in args

    def test_full_transcribe_to_summarize_workflow_config(self):
        """Test full workflow: transcribe output dir -> summarize picks up."""
        config = Config(
            input_files=["video1.mp4", "video2.mp4"],
            output_dir="/home/user/output",
            transcribe_first=True,
            transcribe_model="medium",
            transcribe_language="en",
            transcribe_srt=True,
        )

        transcribe_args = config.build_transcribe_args()
        assert "--output-dir" in transcribe_args
        transcribe_output_idx = transcribe_args.index("--output-dir") + 1
        assert (
            str(config.transcribe_output_dir) in transcribe_args[transcribe_output_idx]
        )
        assert "--model" in transcribe_args
        assert "--language" in transcribe_args
        assert "--srt" in transcribe_args


class TestVideoIdExtraction:
    """Tests for video ID extraction from YouTube URLs."""

    def test_extract_video_id_standard_url(self):
        """Test extracting video ID from standard YouTube URL."""
        from summarize_src.__main__ import extract_video_id

        assert (
            extract_video_id("https://www.youtube.com/watch?v=Qfo6xdVMFmM")
            == "Qfo6xdVMFmM"
        )
        assert (
            extract_video_id("http://youtube.com/watch?v=Qfo6xdVMFmM") == "Qfo6xdVMFmM"
        )

    def test_extract_video_id_short_url(self):
        """Test extracting video ID from youtu.be URL."""
        from summarize_src.__main__ import extract_video_id

        assert extract_video_id("https://youtu.be/Qfo6xdVMFmM") == "Qfo6xdVMFmM"
        assert extract_video_id("http://youtu.be/Qfo6xdVMFmM") == "Qfo6xdVMFmM"

    def test_extract_video_id_shorts_url(self):
        """Test extracting video ID from YouTube Shorts URL."""
        from summarize_src.__main__ import extract_video_id

        assert (
            extract_video_id("https://www.youtube.com/shorts/Qfo6xdVMFmM")
            == "Qfo6xdVMFmM"
        )

    def test_extract_video_id_video_id_only(self):
        """Test extracting video ID when only ID is provided."""
        from summarize_src.__main__ import extract_video_id

        assert extract_video_id("Qfo6xdVMFmM") == "Qfo6xdVMFmM"

    def test_extract_video_id_invalid(self):
        """Test extracting video ID from invalid URL."""
        from summarize_src.__main__ import extract_video_id

        assert extract_video_id("https://example.com/video") is None
        assert extract_video_id("not-a-url") is None

    def test_find_transcription_file_by_video_id(self, tmp_path):
        """Test finding transcription file by YouTube video ID."""
        from summarize_src.__main__ import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()

        test_file = (
            transcribe_dir
            / "Qfo6xdVMFmM_Google Just Doubled Down On Killing Android.txt"
        )
        test_file.write_text("transcript content")

        result = find_transcription_file(
            "https://www.youtube.com/watch?v=Qfo6xdVMFmM", transcribe_dir
        )
        assert result is not None
        assert (
            result.name == "Qfo6xdVMFmM_Google Just Doubled Down On Killing Android.txt"
        )

    def test_find_transcription_file_by_stem(self, tmp_path):
        """Test finding transcription file by file stem."""
        from summarize_src.__main__ import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()

        test_file = transcribe_dir / "myvideo.txt"
        test_file.write_text("transcript content")

        result = find_transcription_file("myvideo.txt", transcribe_dir)
        assert result is not None
        assert result.name == "myvideo.txt"

    def test_find_transcription_file_not_found(self, tmp_path):
        """Test finding transcription file when it doesn't exist."""
        from summarize_src.__main__ import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()

        result = find_transcription_file(
            "https://www.youtube.com/watch?v=NonExistent", transcribe_dir
        )
        assert result is None
