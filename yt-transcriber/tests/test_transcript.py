"""Tests for transcript module."""

import pytest
from unittest.mock import patch, MagicMock
from yt_transcriber.transcript import (
    get_official_transcript,
    get_available_transcripts,
    sanitize_text,
    NoTranscriptError,
)


class TestSanitizeText:
    """Tests for text sanitization."""

    def test_sanitize_basic(self):
        text = "Hello   World"
        assert sanitize_text(text) == "Hello World"

    def test_sanitize_whitespace(self):
        text = "  Hello  World  "
        assert sanitize_text(text) == "Hello World"

    def test_sanitize_newlines(self):
        text = "Hello\nWorld"
        assert sanitize_text(text) == "Hello World"


class TestGetOfficialTranscript:
    """Tests for official transcript extraction."""

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_transcript_success(self, mock_api):
        """Test successful transcript extraction."""
        # Mock transcript data
        mock_transcript = [
            {"text": "Hello world"},
            {"text": "This is a test"},
        ]
        mock_api.get_transcript.return_value = mock_transcript

        # Mock transcript list for language detection
        mock_transcript_list = [MagicMock(language_code="en")]
        mock_api.list_transcripts.return_value = mock_transcript_list

        transcript, lang = get_official_transcript("abc123")

        assert transcript == "Hello world This is a test"
        assert lang == "en"
        mock_api.get_transcript.assert_called_once()

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_transcript_with_language(self, mock_api):
        """Test transcript extraction with specific language."""
        mock_transcript = [{"text": "Hola mundo"}]
        mock_api.get_transcript.return_value = mock_transcript

        mock_transcript_list = [MagicMock(language_code="es")]
        mock_api.list_transcripts.return_value = mock_transcript_list

        transcript, lang = get_official_transcript("abc123", language="es")

        assert transcript == "Hola mundo"
        assert lang == "es"
        mock_api.get_transcript.assert_called_once_with(
            "abc123", languages=["es"], preserve_formatting=False
        )

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_transcript_no_transcript(self, mock_api):
        """Test error when no transcript is available."""
        from youtube_transcript_api import NoTranscriptFound

        # NoTranscriptFound requires video_id, requested_language_codes, and transcript_data
        mock_api.get_transcript.side_effect = NoTranscriptFound(
            video_id="abc123", requested_language_codes=["en"], transcript_data=[]
        )

        with pytest.raises(NoTranscriptError):
            get_official_transcript("abc123")

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_transcript_disabled(self, mock_api):
        """Test error when transcripts are disabled."""
        from youtube_transcript_api import TranscriptsDisabled

        # TranscriptsDisabled requires video_id
        mock_api.get_transcript.side_effect = TranscriptsDisabled(video_id="abc123")

        with pytest.raises(NoTranscriptError):
            get_official_transcript("abc123")


class TestGetAvailableTranscripts:
    """Tests for listing available transcripts."""

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_available_transcripts(self, mock_api):
        """Test getting list of available transcripts."""
        mock_transcript = MagicMock()
        mock_transcript.language = "English"
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False

        mock_api.list_transcripts.return_value = [mock_transcript]

        transcripts = get_available_transcripts("abc123")

        assert len(transcripts) == 1
        assert transcripts[0]["language"] == "English"
        assert transcripts[0]["language_code"] == "en"
        assert transcripts[0]["is_generated"] is False

    @patch("yt_transcriber.transcript.YouTubeTranscriptApi")
    def test_get_available_transcripts_error(self, mock_api):
        """Test error handling when API fails."""
        mock_api.list_transcripts.side_effect = Exception("API error")

        transcripts = get_available_transcripts("abc123")

        assert transcripts == []
