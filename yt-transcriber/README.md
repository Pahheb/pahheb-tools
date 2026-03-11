# YouTube Transcriber

A powerful command-line tool for transcribing YouTube videos. It automatically extracts official transcripts when available and falls back to local Whisper transcription for videos without captions.

## Features

- **Automatic Official Transcript Extraction**: Uses `youtube-transcript-api` to fetch available transcripts
- **Local Whisper Transcription**: Falls back to faster-whisper for videos without transcripts
- **AMD GPU Support**: Automatically detects and uses ROCm for GPU acceleration
- **Playlist Processing**: Handle playlists with thousands of videos with rate limiting
- **Combined Output**: Option to merge all transcriptions into a single file
- **Progress Tracking**: Real-time progress with source indication (official vs Whisper)

## Installation

### Prerequisites

- Python 3.12+
- FFmpeg (for audio extraction)
- AMD ROCm (optional, for GPU acceleration)

### Install Dependencies

```bash
cd yt-transcriber
pip install -e .
```

For development (includes tests and linting):

```bash
pip install -e ".[dev]"
```

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Usage

### Basic Usage

Transcribe a single video:

```bash
yt-transcribe https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory for files | `./transcriptions` |
| `--combine` | Create combined.txt with all transcriptions | False |
| `--language` | Language code (en, es, fr, etc.) | Auto-detect |
| `--model-size` | Whisper model (tiny, base, small, medium, large-v3) | `small` |
| `--device` | Device (auto, cuda, cpu, mps) | `auto` |
| `--sleep` | Seconds between videos | 2 |
| `--verbose` | Show detailed progress | False |

### Examples

**Transcribe a playlist:**

```bash
yt-transcribe https://www.youtube.com/playlist?list=PLabc123...
```

**Transcribe with combined output:**

```bash
yt-transcribe playlist_url --combine --output-dir ./my_transcriptions
```

**Force specific Whisper model:**

```bash
yt-transcribe video_url --model-size medium
```

**Use CPU only (no GPU):**

```bash
yt-transcribe video_url --device cpu
```

**Transcribe with language specification:**

```bash
yt-transcribe video_url --language en
```

**Process URLs from a file:**

Create a file `videos.txt` with one URL per line:

```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://youtu.be/VIDEO_ID_3
```

Then run:

```bash
yt-transcribe videos.txt --combine
```

## Output Format

### Individual Files

Each transcription is saved as `{video_id}_{title_safe}.txt` with the following header:

```
Video ID: dQw4w9WgXcQ
Title: Rick Astley - Never Gonna Give You Up
Language: en
Source: official
--------------------------------------------------

[transcript text here]
```

### Combined File

When using `--combine`, all transcriptions are merged into `combined.txt` with separators:

```
======================================================================
=== Rick Astley - Never Gonna Give You Up (dQw4w9WgXcQ) ===
======================================================================

[transcript text]

======================================================================
=== Another Video Title (VIDEO_ID) ===
======================================================================

[transcript text]
```

## Processing Flow

1. **Check Official Transcript**: Try to fetch available transcript using `youtube-transcript-api`
2. **Download Audio**: If no official transcript, download audio stream using `pytube`
3. **Transcribe with Whisper**: Use faster-whisper for local transcription
4. **Save File**: Write transcript to `{video_id}_{title}.txt`
5. **Rate Limiting**: Sleep between videos to avoid YouTube rate limits

## AMD GPU Acceleration

The tool automatically detects AMD GPUs with ROCm support and uses GPU acceleration for Whisper transcription when available. To verify ROCm is working:

```bash
python -c "import torch; print(f'ROCm available: {torch.version.hip is not None}')"
```

If ROCm is not detected, the tool will fall back to CPU transcription.

## Whisper Model Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~75 MB | Fastest | Lower | Quick transcription |
| `base` | ~300 MB | Fast | Medium | Balanced |
| `small` | ~460 MB | Medium | Good | **Default - Balanced** |
| `medium` | ~1.5 GB | Slower | Better | Higher accuracy |
| `large-v3` | ~3 GB | Slowest | Best | Maximum accuracy |

**Recommendation**: Use `small` for balanced performance, `medium` for better accuracy.

## Rate Limiting

YouTube has API rate limits. The tool handles this by:

- Default 2-second delay between videos
- Exponential backoff on rate limit errors (2s, 4s, 8s)
- Respect for `Retry-After` headers

For playlists with 1000+ videos, consider increasing the sleep delay:

```bash
yt-transcribe large_playlist_url --sleep 5
```

## Troubleshooting

### "No transcript found"

Some videos don't have auto-generated captions. The tool will automatically fall back to Whisper transcription.

### "Video unavailable"

The video may be private, deleted, or region-restricted. The tool will skip it and continue with others.

### Slow transcription

- Use a smaller Whisper model (`tiny` or `base`)
- Use CPU-only mode if GPU has issues: `--device cpu`
- Increase sleep delay for large playlists

### GPU not detected

- Verify ROCm is installed: `rocminfo`
- Check PyTorch ROCm: `python -c "import torch; print(torch.version.hip)"`
- Fall back to CPU: `--device cpu`

## Development

### Project Structure

```
yt-transcriber/
├── yt_transcriber/
│   ├── __main__.py        # CLI entry point
│   ├── cli.py             # Argument parsing
│   ├── downloader.py      # Audio download
│   ├── transcript.py      # Official transcript extraction
│   ├── whisper.py         # Whisper transcription
│   ├── playlist.py        # Playlist processing
│   ├── url_parser.py      # URL parsing
│   ├── file_writer.py     # File output
│   └── config.py          # Configuration
├── tests/                 # Unit tests
├── pyproject.toml         # Dependencies
└── README.md
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Quality

```bash
# Linting
ruff check yt_transcriber/

# Type checking
mypy yt_transcriber/
```

## License

MIT License - see LICENSE file for details.

## Credits

- `youtube-transcript-api` for official transcript extraction
- `pytube` for YouTube video downloading
- `faster-whisper` for local transcription
- PyTorch for GPU acceleration support
