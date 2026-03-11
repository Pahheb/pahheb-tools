"""Tests for playlist processing."""

import pytest
from unittest.mock import patch, MagicMock
from yt_transcriber.playlist import get_playlist_videos, PlaylistError


class TestGetPlaylistVideos:
    """Tests for getting playlist videos."""

    @patch("yt_transcriber.playlist.subprocess.run")
    def test_get_playlist_videos_success(self, mock_run):
        """Test successful playlist video extraction."""
        # Mock yt-dlp JSON output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n".join(
            [
                '{"_type": "url", "id": "abc123", "url": "https://www.youtube.com/watch?v=abc123", "title": "Video 1"}',
                '{"_type": "url", "id": "def456", "url": "https://www.youtube.com/watch?v=def456", "title": "Video 2"}',
                '{"_type": "url", "id": "ghi789", "url": "https://www.youtube.com/watch?v=ghi789", "title": "Video 3"}',
            ]
        )
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        videos = get_playlist_videos("https://www.youtube.com/playlist?list=PLtest")

        assert len(videos) == 3
        assert videos[0]["video_id"] == "abc123"
        assert videos[1]["video_id"] == "def456"
        assert videos[2]["video_id"] == "ghi789"

    @patch("yt_transcriber.playlist.subprocess.run")
    def test_get_playlist_videos_empty(self, mock_run):
        """Test empty playlist."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        videos = get_playlist_videos("https://www.youtube.com/playlist?list=PLempty")

        assert len(videos) == 0

    @patch("yt_transcriber.playlist.subprocess.run")
    def test_get_playlist_videos_error(self, mock_run):
        """Test error when playlist cannot be accessed."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Playlist not found"
        mock_run.return_value = mock_result

        with pytest.raises(PlaylistError):
            get_playlist_videos("https://www.youtube.com/playlist?list=PLinvalid")
