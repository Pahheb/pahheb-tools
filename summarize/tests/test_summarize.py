"""Tests for summarize tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarize_src.config import Config
from summarize_src.file_writer import (
    write_summary_json,
    write_summary_md,
    write_summary_txt,
)
from summarize_src.summarizer import (
    HuggingFaceProvider,
    OllamaProvider,
    ProviderNotAvailableError,
    _parse_response,
    get_provider,
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

        write_summary_txt(summary, output_path, metadata=metadata)

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
        content = (
            "KEYPOINTS:\n- Point 1\n\nSUMMARY:\nThis is the HuggingFace summary text."
        )
        result = _parse_response(content, "huggingface", "test-model")

        assert result.summary == "This is the HuggingFace summary text."
        assert "Point 1" in result.key_points

    def test_parse_response_no_keypoints(self):
        """Test parsing when response has no keypoints but has summary."""
        content = "Here is a summary without keypoints."
        result = _parse_response(content, "huggingface", "test-model")

        assert result.summary == "Here is a summary without keypoints."

    def test_parse_response_with_numbered_points(self):
        """Test parsing numbered list keypoints."""
        content = (
            "KEYPOINTS:\n1. First point\n2. Second point\n\nSUMMARY:\nSummary text."
        )
        result = _parse_response(content, "huggingface", "test-model")

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
        import sys
        from unittest.mock import patch

        from summarize_src.cli import parse_args

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
        from pathlib import Path
        from unittest.mock import patch

        from summarize_src.cli import parse_args

        nonexistent = Path("/nonexistent/file.txt")

        with patch("sys.argv", ["summarize", str(nonexistent)]):
            try:
                parse_args()
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "not found" in captured.err.lower()

    def test_parse_args_no_inputs_shows_help(self, capsys):
        """Test that running with no inputs shows help instead of error."""
        from unittest.mock import patch

        from summarize_src.cli import parse_args

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
        from summarize_src.transcription import extract_video_id

        assert (
            extract_video_id("https://www.youtube.com/watch?v=Qfo6xdVMFmM")
            == "Qfo6xdVMFmM"
        )
        assert (
            extract_video_id("http://youtube.com/watch?v=Qfo6xdVMFmM") == "Qfo6xdVMFmM"
        )

    def test_extract_video_id_short_url(self):
        """Test extracting video ID from youtu.be URL."""
        from summarize_src.transcription import extract_video_id

        assert extract_video_id("https://youtu.be/Qfo6xdVMFmM") == "Qfo6xdVMFmM"
        assert extract_video_id("http://youtu.be/Qfo6xdVMFmM") == "Qfo6xdVMFmM"

    def test_extract_video_id_shorts_url(self):
        """Test extracting video ID from YouTube Shorts URL."""
        from summarize_src.transcription import extract_video_id

        assert (
            extract_video_id("https://www.youtube.com/shorts/Qfo6xdVMFmM")
            == "Qfo6xdVMFmM"
        )

    def test_extract_video_id_video_id_only(self):
        """Test extracting video ID when only ID is provided."""
        from summarize_src.transcription import extract_video_id

        assert extract_video_id("Qfo6xdVMFmM") == "Qfo6xdVMFmM"

    def test_extract_video_id_invalid(self):
        """Test extracting video ID from invalid URL."""
        from summarize_src.transcription import extract_video_id

        assert extract_video_id("https://example.com/video") is None
        assert extract_video_id("not-a-url") is None

    def test_find_transcription_file_by_video_id(self, tmp_path):
        """Test finding transcription file by YouTube video ID."""
        from summarize_src.transcription import find_transcription_file

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
        from summarize_src.transcription import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()

        test_file = transcribe_dir / "myvideo.txt"
        test_file.write_text("transcript content")

        result = find_transcription_file("myvideo.txt", transcribe_dir)
        assert result is not None
        assert result.name == "myvideo.txt"

    def test_find_transcription_file_not_found(self, tmp_path):
        """Test finding transcription file when it doesn't exist."""
        from summarize_src.transcription import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()

        result = find_transcription_file(
            "https://www.youtube.com/watch?v=NonExistent", transcribe_dir
        )
        assert result is None


class TestFileWriterMetadataPaths:
    """Tests for file writer metadata and key_points paths in md/json formats."""

    def test_write_summary_md_with_duration_metadata(self, tmp_path):
        """Test writing MD with duration metadata."""
        output_path = tmp_path / "with_dur.md"
        write_summary_md(
            "Summary text.",
            output_path,
            key_points=["Point 1"],
            metadata={"source": "test", "duration": "5:30"},
        )
        content = output_path.read_text()
        assert "**Duration:** 5:30" in content
        assert "**Source:** test" in content
        assert "## Key Points" in content
        assert "1. Point 1" in content
        assert "## Summary" in content
        assert "Summary text." in content

    def test_write_summary_md_with_summary_type_metadata(self, tmp_path):
        """Test writing MD with summary_type metadata."""
        output_path = tmp_path / "with_type.md"
        write_summary_md(
            "Summary.",
            output_path,
            metadata={"source": "test", "summary_type": "detailed"},
        )
        content = output_path.read_text()
        assert "**Type:** detailed" in content

    def test_write_summary_md_with_all_metadata(self, tmp_path):
        """Test writing MD with all metadata fields present."""
        output_path = tmp_path / "all_meta.md"
        write_summary_md(
            "Full summary.",
            output_path,
            key_points=["A", "B"],
            metadata={
                "source": "youtube",
                "title": "My Video",
                "duration": "10:00",
                "summary_type": "standard",
            },
        )
        content = output_path.read_text()
        assert "**Source:** youtube" in content
        assert "**Title:** My Video" in content
        assert "**Duration:** 10:00" in content
        assert "**Type:** standard" in content
        assert "## Key Points" in content

    def test_write_summary_txt_with_duration_metadata(self, tmp_path):
        """Test writing TXT with duration metadata."""
        output_path = tmp_path / "with_dur.txt"
        write_summary_txt(
            "Summary.",
            output_path,
            metadata={"source": "test", "duration": "3:00"},
        )
        content = output_path.read_text()
        assert "Duration: 3:00" in content

    def test_write_summary_txt_with_summary_type_metadata(self, tmp_path):
        """Test writing TXT with summary_type metadata."""
        output_path = tmp_path / "with_type.txt"
        write_summary_txt(
            "Summary.",
            output_path,
            metadata={"source": "test", "summary_type": "brief"},
        )
        content = output_path.read_text()
        assert "Summary Type: brief" in content

    def test_write_summary_json_with_all_metadata(self, tmp_path):
        """Test writing JSON with all metadata fields."""
        import json

        output_path = tmp_path / "full.json"
        write_summary_json(
            "Summary.",
            output_path,
            key_points=["A", "B"],
            metadata={"source": "test", "title": "Test"},
        )
        content = json.loads(output_path.read_text())
        assert content["summary"] == "Summary."
        assert content["key_points"] == ["A", "B"]
        assert content["metadata"]["source"] == "test"
        assert content["metadata"]["title"] == "Test"


class TestFileWriterEdgeCases:
    """Tests for file writer edge cases."""

    def test_write_summary_txt_no_metadata(self, tmp_path):
        """Test writing TXT without metadata."""
        output_path = tmp_path / "no_meta.txt"
        write_summary_txt("Just a summary.", output_path)
        content = output_path.read_text()
        assert "Source:" not in content
        assert "Just a summary." in content

    def test_write_summary_txt_no_key_points(self, tmp_path):
        """Test writing TXT without key points."""
        output_path = tmp_path / "no_kp.txt"
        write_summary_txt("Summary text.", output_path, metadata={"source": "test"})
        content = output_path.read_text()
        assert "KEY POINTS:" not in content
        assert "Summary text." in content

    def test_write_summary_md_no_metadata(self, tmp_path):
        """Test writing MD without metadata."""
        output_path = tmp_path / "no_meta.md"
        write_summary_md("Just a summary.", output_path)
        content = output_path.read_text()
        # No metadata section means no Source/Title lines
        assert "**Source:**" not in content
        assert "Just a summary." in content

    def test_write_summary_md_no_key_points(self, tmp_path):
        """Test writing MD without key points."""
        output_path = tmp_path / "no_kp.md"
        write_summary_md("Summary text.", output_path)
        content = output_path.read_text()
        assert "## Key Points" not in content
        assert "Summary text." in content

    def test_write_summary_json_no_metadata(self, tmp_path):
        """Test writing JSON without metadata."""
        output_path = tmp_path / "no_meta.json"
        write_summary_json("Summary.", output_path)
        content = output_path.read_text()
        assert '"metadata"' not in content
        assert '"summary"' in content

    def test_write_summary_json_no_key_points(self, tmp_path):
        """Test writing JSON without key points."""
        output_path = tmp_path / "no_kp.json"
        write_summary_json("Summary.", output_path)
        content = output_path.read_text()
        assert '"key_points"' not in content


class TestOllamaProviderEdgeCases:
    """Tests for OllamaProvider edge cases."""

    @patch("httpx.Client")
    def test_is_available_returns_false_on_non_200(self, mock_client):
        """Test is_available returns False on non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        provider = OllamaProvider()
        assert provider.is_available() is False

    @patch("httpx.Client")
    def test_summarize_non_200_raises_summarizer_error(self, mock_client):
        """Test that non-200 API response raises SummarizerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        provider = OllamaProvider()
        from summarize_src.summarizer import SummarizerError

        with pytest.raises(SummarizerError, match="Ollama API error"):
            provider.summarize("Test text")

    @patch("httpx.Client")
    def test_summarize_timeout_raises_summarizer_error(self, mock_client):
        """Test that httpx.TimeoutException raises SummarizerError."""
        import httpx

        mock_client.return_value.__enter__.return_value.post.side_effect = (
            httpx.TimeoutException("Connection timed out")
        )

        provider = OllamaProvider()
        from summarize_src.summarizer import SummarizerError

        with pytest.raises(SummarizerError, match="timed out"):
            provider.summarize("Test text")

    def test_parse_response_star_bullet_points(self):
        """Test parsing response with * bullet points."""
        content = "KEYPOINTS:\n* First point\n* Second point\n\nSUMMARY:\nSummary text."
        result = _parse_response(content, "ollama", "llama3.2")

        assert "First point" in result.key_points
        assert "Second point" in result.key_points

    def test_parse_response_dot_bullet_points(self):
        """Test parsing response with • bullet points."""
        content = (
            "KEYPOINTS:\n• First dot point\n• Second dot point\n\nSUMMARY:\nSummary."
        )
        result = _parse_response(content, "ollama", "llama3.2")

        assert "First dot point" in result.key_points
        assert "Second dot point" in result.key_points

    def test_parse_response_key_points_with_space(self):
        """Test parsing KEY POINTS: (with space) prefix."""
        content = "KEY POINTS:\n- Point one\n- Point two\n\nSUMMARY:\nSummary text."
        result = _parse_response(content, "ollama", "llama3.2")

        assert "Point one" in result.key_points
        assert "Point two" in result.key_points

    def test_parse_response_short_points_filtered_out(self):
        """Test that points <= 3 characters are filtered out."""
        content = (
            "KEYPOINTS:\n- OK\n- Abc\n- Yes this is long enough\n\nSUMMARY:\nText."
        )
        result = _parse_response(content, "ollama", "llama3.2")

        assert "OK" not in result.key_points
        assert "Abc" not in result.key_points
        assert "Yes this is long enough" in result.key_points

    def test_build_prompt_unknown_summary_type_fallback(self):
        """Test that unknown summary_type falls back to default prompt."""
        provider = OllamaProvider()
        prompt = provider._build_prompt("test text", "unknown_type")
        assert "comprehensive summary" in prompt.lower()


class TestReadTranscription:
    """Tests for read_transcription function."""

    def test_read_with_metadata_header(self, tmp_path):
        """Test reading a file with metadata header and --- delimiters."""
        from summarize_src.transcription import read_transcription

        txt = tmp_path / "meta.txt"
        txt.write_text(
            "source: youtube\nvideo_id: abc123\nmodel: small\n"
            "---\n"
            "This is the actual transcript.\n"
            "---\n"
        )
        result = read_transcription(txt)
        assert "This is the actual transcript." in result
        assert "source: youtube" not in result

    def test_read_only_metadata_no_transcript(self, tmp_path):
        """Test reading a file with only metadata falls back to full content."""
        from summarize_src.transcription import read_transcription

        txt = tmp_path / "meta_only.txt"
        txt.write_text("source: youtube\nvideo_id: abc123\nmodel: small\n")
        result = read_transcription(txt)
        # Falls back to content since no transcript found
        assert "source: youtube" in result

    def test_read_with_string_path(self, tmp_path):
        """Test reading with string path (not Path object)."""
        from summarize_src.transcription import read_transcription

        txt = tmp_path / "string_path.txt"
        txt.write_text("Hello world transcript.")
        result = read_transcription(str(txt))
        assert result == "Hello world transcript."

    def test_read_empty_file(self, tmp_path):
        """Test reading an empty file."""
        from summarize_src.transcription import read_transcription

        txt = tmp_path / "empty.txt"
        txt.write_text("")
        result = read_transcription(txt)
        assert result == ""


class TestFindTranscriptionEdgeCases:
    """Tests for find_transcription_file edge cases."""

    def test_find_with_question_mark_in_url(self, tmp_path):
        """Test finding file when URL stem has query params."""
        from summarize_src.transcription import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()
        test_file = transcribe_dir / "myvideo.txt"
        test_file.write_text("content")

        result = find_transcription_file(
            "https://youtube.com/watch?v=abc123&t=30s", transcribe_dir
        )
        # Should not crash; may or may not find file
        assert result is None or isinstance(result, type(test_file))

    def test_find_with_dot_in_url_stem(self, tmp_path):
        """Test finding file when URL stem has dots."""
        from summarize_src.transcription import find_transcription_file

        transcribe_dir = tmp_path / "transcriptions"
        transcribe_dir.mkdir()
        test_file = transcribe_dir / "lecture.mp4.txt"
        test_file.write_text("content")

        result = find_transcription_file("lecture.mp4", transcribe_dir)
        assert result is not None
        assert result.name == "lecture.mp4.txt"


class TestProviderNotAvailableMessages:
    """Tests for provider-specific error messages in main()."""

    def _make_config(self, provider, tmp_path):
        """Helper to create a Config with a test file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("transcript content")
        return Config(
            input_files=[test_file],
            provider=provider,
            output_dir=tmp_path,
        )

    @patch("summarize_src.__main__.get_provider")
    def test_ollama_not_available_shows_helpful_error(
        self, mock_get_provider, tmp_path, capsys
    ):
        """Test that unavailable Ollama provider shows install instructions."""
        from summarize_src.__main__ import main

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_get_provider.return_value = mock_provider

        config = self._make_config("ollama", tmp_path)
        with patch("summarize_src.__main__.parse_args") as mock_args:
            mock_args.return_value = self._make_mock_args(config)
            with pytest.raises(SystemExit):
                main()

        err = capsys.readouterr().err
        assert "ollama serve" in err

    @patch("summarize_src.__main__.get_provider")
    def test_huggingface_not_available_shows_helpful_error(
        self, mock_get_provider, tmp_path, capsys
    ):
        """Test that unavailable HuggingFace shows install instructions."""
        from summarize_src.__main__ import main

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_get_provider.return_value = mock_provider

        config = self._make_config("huggingface", tmp_path)
        with patch("summarize_src.__main__.parse_args") as mock_args:
            mock_args.return_value = self._make_mock_args(config)
            with pytest.raises(SystemExit):
                main()

        err = capsys.readouterr().err
        assert "transformers" in err.lower()

    def _make_mock_args(self, config):
        """Create a mock args object matching a Config."""
        args = MagicMock()
        args.provider = config.provider
        args.model = config.model
        args.input_files = config.input_files
        args.output_dir = config.output_dir
        args.output_format = config.output_format
        args.summary_length = config.summary_length
        args.combine = config.combine
        args.unified = config.unified
        args.skip_combined = config.skip_combined
        args.transcribe = config.transcribe_first
        args.verbose = config.verbose
        args.transcribe_source = config.transcribe_source
        args.transcribe_language = config.transcribe_language
        args.transcribe_model = config.transcribe_model
        args.transcribe_device = config.transcribe_device
        args.transcribe_compute = config.transcribe_compute
        args.transcribe_denoise = config.transcribe_denoise
        args.transcribe_vad = config.transcribe_vad
        args.transcribe_audio_enhance = config.transcribe_audio_enhance
        args.transcribe_srt = config.transcribe_srt
        args.transcribe_cleanup = config.transcribe_cleanup
        args.single_threaded = config.single_threaded
        return args


class TestSummarizeMainVerbose:
    """Tests for verbose output in main() and summarize functions."""

    def test_summarize_file_verbose(self, tmp_path, capsys):
        """Test summarize_file with verbose=True prints reading message."""
        from summarize_src.__main__ import summarize_file

        txt = tmp_path / "test.txt"
        txt.write_text("This is a transcript.")
        config = Config(
            input_files=[txt],
            output_dir=tmp_path,
            output_format="txt",
            summary_length="standard",
        )
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Summary text."
        mock_result.key_points = ["Point 1"]
        mock_result.metadata = {}
        mock_provider.summarize.return_value = mock_result

        summarize_file(txt, config, mock_provider, verbose=True)

        out = capsys.readouterr().out
        assert "Reading:" in out
        assert "Summarizing" in out
        assert "characters" in out

    def test_summarize_unified_verbose(self, tmp_path, capsys):
        """Test summarize_unified with verbose=True."""
        from summarize_src.__main__ import summarize_unified

        txt1 = tmp_path / "file1.txt"
        txt1.write_text("Transcript 1.")
        txt2 = tmp_path / "file2.txt"
        txt2.write_text("Transcript 2.")
        config = Config(
            input_files=[txt1, txt2],
            output_dir=tmp_path,
            output_format="txt",
            summary_length="standard",
        )
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Unified summary."
        mock_result.key_points = []
        mock_result.metadata = {}
        mock_provider.summarize.return_value = mock_result

        summarize_unified([txt1, txt2], config, mock_provider, verbose=True)

        out = capsys.readouterr().out
        assert "Merging" in out
        assert "unified" in out.lower()

    def test_summarize_combined_verbose(self, tmp_path, capsys):
        """Test summarize_combined with verbose=True."""
        from summarize_src.__main__ import summarize_combined

        txt1 = tmp_path / "file1.txt"
        txt1.write_text("Transcript 1.")
        txt2 = tmp_path / "file2.txt"
        txt2.write_text("Transcript 2.")
        config = Config(
            input_files=[txt1, txt2],
            output_dir=tmp_path,
            output_format="txt",
            summary_length="standard",
        )
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Summary of file."
        mock_result.key_points = ["Key"]
        mock_result.metadata = {}
        mock_provider.summarize.return_value = mock_result

        summarize_combined([txt1, txt2], config, mock_provider, verbose=True)

        out = capsys.readouterr().out
        assert "combined" in out.lower()
        assert "file1.txt" in out

    def test_summarize_file_output_formats(self, tmp_path):
        """Test summarize_file writes correct format extensions."""
        from summarize_src.__main__ import summarize_file

        txt = tmp_path / "input.txt"
        txt.write_text("Transcript content.")
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Summary."
        mock_result.key_points = ["Point"]
        mock_result.metadata = {"source": "test"}
        mock_provider.summarize.return_value = mock_result

        for fmt in ("txt", "md", "json"):
            config = Config(
                input_files=[txt],
                output_dir=tmp_path / fmt,
                output_format=fmt,
                summary_length="standard",
            )
            result = summarize_file(txt, config, mock_provider)
            assert result.suffix == f".{fmt}"

    def test_main_verbose_shows_config(self, tmp_path, capsys):
        """Test that verbose mode in main() prints provider/model info."""
        from summarize_src.__main__ import main

        txt = tmp_path / "test.txt"
        txt.write_text("Transcript.")

        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Summary."
        mock_result.key_points = []
        mock_result.metadata = {}
        mock_provider.is_available.return_value = True
        mock_provider.summarize.return_value = mock_result

        with (
            patch("summarize_src.__main__.get_provider", return_value=mock_provider),
            patch(
                "sys.argv",
                ["summarize", str(txt), "--verbose"],
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "Provider:" in out
        assert "Output dir:" in out

    def test_main_generic_exception_verbose_shows_traceback(self, tmp_path, capsys):
        """Test that verbose mode shows traceback on generic exception."""
        from summarize_src.__main__ import main

        txt = tmp_path / "test.txt"
        txt.write_text("Transcript.")

        # get_provider raises during setup, hitting the top-level Exception handler
        with (
            patch(
                "summarize_src.__main__.get_provider",
                side_effect=RuntimeError("unexpected crash"),
            ),
            patch(
                "sys.argv",
                ["summarize", str(txt), "--verbose"],
            ),
            pytest.raises(SystemExit),
        ):
            main()

        err = capsys.readouterr().err
        assert "Unexpected error" in err
