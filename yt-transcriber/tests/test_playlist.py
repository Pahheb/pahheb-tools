"""Tests for playlist processing."""

import pytest
from unittest.mock import patch, MagicMock
from yt_transcriber.playlist import get_playlist_videos, PlaylistError


class TestGetPlaylistVideos:
    """Tests for getting playlist videos."""

    @patch("yt_transcriber.playlist.Playlist")
    def test_get_playlist_videos_success(self, mock_playlist_class):
        """Test successful playlist video extraction."""
        # Mock Playlist object
        mock_playlist = MagicMock()
        mock_playlist.video_urls = [
            "https://www.youtube.com/watch?v=abc123",
            "https://www.youtube.com/watch?v=def456",
            "https://www.youtube.com/watch?v=ghi789",
        ]
        mock_playlist_class.return_value = mock_playlist

        videos = get_playlist_videos("https://www.youtube.com/playlist?list=PLtest")

        assert len(videos) == 3
        assert videos[0]["video_id"] == "abc123"
        assert videos[1]["video_id"] == "def456"
        assert videos[2]["video_id"] == "ghi789"

    @patch("yt_transcriber.playlist.Playlist")
    def test_get_playlist_videos_empty(self, mock_playlist_class):
        """Test empty playlist."""
        mock_playlist = MagicMock()
        mock_playlist.video_urls = []
        mock_playlist_class.return_value = mock_playlist

        videos = get_playlist_videos("https://www.youtube.com/playlist?list=PLempty")

        assert len(videos) == 0

    @patch("yt_transcriber.playlist.Playlist")
    def test_get_playlist_videos_error(self, mock_playlist_class):
        """Test error when playlist cannot be accessed."""
        mock_playlist_class.side_effect = Exception("Playlist not found")

        with pytest.raises(PlaylistError):
            get_playlist_videos("https://www.youtube.com/playlist?list=PLinvalid")
