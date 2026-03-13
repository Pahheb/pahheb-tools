# Transcribe Tool

A command-line tool for transcribing video and audio files using Whisper and RNNoise.

## Features

- **Multi-format Support**: Handles MP3, WAV, M4A, MP4, and other common media formats.
- **YouTube Support**: Transcribe YouTube videos directly from URLs.
- **Noise Reduction**: Uses RNNoise (via FFmpeg) to reduce background noise and hiss.
- **VAD Filtering**: Applies Voice Activity Detection to reduce silence and garbage text.
- **Dual Output**: Generates both plain text (`.txt`) and SubRip (`.srt`) files.

## Prerequisites

- **Python 3.12+**
- **FFmpeg**: Must be installed and accessible in your system PATH.
  - *Note:* For noise reduction, FFmpeg must be compiled with the `arnndn` filter.
- **uv**: Recommended package manager (install from https://github.com/astral-sh/uv)

## Installation

```bash
cd transcribe
./setup.sh
```

Or manually:

```bash
cd transcribe
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

Basic usage to transcribe an audio file:

```bash
transcribe input.mp3
```

Or use the wrapper script:

```bash
./transcribe input.mp3
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Audio/video file paths or YouTube URLs | Required |
| `--source`, `-s` | Source type (`local` or `youtube`) | `local` |
| `--output-dir`, `-o` | Output directory | `./transcriptions` |
| `--language`, `-l` | Language code (`en`, `es`, etc.) or empty for auto-detect | `None` |
| `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `small` |
| `--device` | Device to run inference on (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `--compute` | Compute type (`int8`, `float16`, etc.) | Auto-selected |
| `--denoise` | Enable RNNoise denoising | `False` |
| `--vad` | Enable VAD filtering | `False` |
| `--audio-enhance` | Apply audio enhancement filters | `False` |
| `--srt` | Generate SRT subtitle file | `False` |
| `--cleanup` | Delete intermediate files after transcription | `False` |
| `--verbose`, `-v` | Show detailed progress | `False` |

## Output

The tool generates files in the specified output directory:
1. `<basename>.txt`: Plain text transcription.
2. `<basename>.srt`: Subtitle file with timestamps (if `--srt` is used).

## Notes

- If FFmpeg does not support the `arnndn` filter, the denoising step will be skipped, but VAD filtering will still be applied.
- The `--device` auto setting will use CUDA if available, otherwise it falls back to CPU.
