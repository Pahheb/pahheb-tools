"""Tests for URL parsing module."""

import pytest
from yt_transcriber.url_parser import (
    is_youtube_url,
    extract_video_id,
    is_playlist_url,
    extract_playlist_id,
    is_file_with_urls,
    parse_input,
    read_urls_from_file,
)
from pathlib import Path
import tempfile


class TestYouTubeURL:
    """Tests for YouTube URL detection."""

    def test_is_youtube_url_true(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("youtube.com/watch?v=abc123")

    def test_is_youtube_url_false(self):
        assert not is_youtube_url("https://google.com")
        assert not is_youtube_url("not a url")
        assert not is_youtube_url("")


class TestVideoIDExtraction:
    """Tests for video ID extraction."""

    def test_extract_video_id_standard(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_short(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_mobile(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_direct(self):
        video_id = "dQw4w9WgXcQ"
        assert extract_video_id(video_id) == video_id

    def test_extract_video_id_invalid(self):
        assert extract_video_id("https://google.com") is None
        assert extract_video_id("invalid") is None


class TestPlaylistURL:
    """Tests for playlist URL detection."""

    def test_is_playlist_url_true(self):
        assert is_playlist_url("https://www.youtube.com/playlist?list=PLabc123")
        assert is_playlist_url("https://youtube.com/playlist?list=PLabc123")
        assert is_playlist_url("https://www.youtube.com/watch?v=xyz&list=PLabc123")

    def test_is_playlist_url_false(self):
        assert not is_playlist_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert not is_playlist_url("https://google.com")


class TestPlaylistIDExtraction:
    """Tests for playlist ID extraction."""

    def test_extract_playlist_id_standard(self):
        url = "https://www.youtube.com/playlist?list=PLabc123def456"
        assert extract_playlist_id(url) == "PLabc123def456"

    def test_extract_playlist_id_in_watch_url(self):
        url = "https://www.youtube.com/watch?v=xyz&list=PLabc123"
        assert extract_playlist_id(url) == "PLabc123"

    def test_extract_playlist_id_invalid(self):
        assert extract_playlist_id("https://google.com") is None


class TestParseInput:
    """Tests for input parsing."""

    def test_parse_video_url(self):
        input_type, identifier, _ = parse_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert input_type == "video"
        assert identifier == "dQw4w9WgXcQ"

    def test_parse_video_id(self):
        input_type, identifier, _ = parse_input("dQw4w9WgXcQ")
        assert input_type == "video"
        assert identifier == "dQw4w9WgXcQ"

    def test_parse_playlist_url(self):
        input_type, identifier, _ = parse_input("https://www.youtube.com/playlist?list=PLabc123")
        assert input_type == "playlist"
        assert identifier == "PLabc123"

    def test_parse_invalid_input(self):
        with pytest.raises(ValueError):
            parse_input("not a valid input")

    def test_parse_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
            f.write("https://youtu.be/abc123\n")
            temp_path = f.name

        try:
            input_type, identifier, _ = parse_input(temp_path)
            assert input_type == "file"
            assert identifier == temp_path
        finally:
            Path(temp_path).unlink()


class TestReadURLsFromFile:
    """Tests for reading URLs from file."""

    def test_read_urls_from_valid_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
            f.write("https://youtu.be/abc123\n")
            f.write("not a url\n")
            f.write("https://google.com\n")
            temp_path = f.name

        try:
            urls = read_urls_from_file(temp_path)
            assert len(urls) == 2
            assert "dQw4w9WgXcQ" in urls[0]
            assert "abc123" in urls[1]
        finally:
            Path(temp_path).unlink()

    def test_read_urls_from_nonexistent_file(self):
        with pytest.raises(ValueError):
            read_urls_from_file("/nonexistent/file.txt")
