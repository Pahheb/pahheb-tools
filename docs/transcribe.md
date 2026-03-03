# Transcribe Tool

A command-line tool for transcribing video and audio files using Whisper and RNNoise.

## Features

- **Multi-format Support**: Handles MP3, WAV, M4A, MP4, and other common media formats.
- **Noise Reduction**: Uses RNNoise (via FFmpeg) to reduce background noise and hiss.
- **VAD Filtering**: Applies Voice Activity Detection to reduce silence and garbage text.
- **Dual Output**: Generates both plain text (`.txt`) and SubRip (`.srt`) files.

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Must be installed and accessible in your system PATH.
  - *Note:* For noise reduction, FFmpeg must be compiled with the `arnndn` filter.
- **Libraries**:
  ```bash
  pip install faster-whisper tqdm
  ```

## Usage

Basic usage to transcribe an audio file:

```bash
python transcribe.py input.mp3
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Audio/video file paths (e.g., `video.mp4`, `audio.wav`) | Required |
| `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `large-v3` |
| `--language` | Language code (`en`, `es`, etc.) or empty for auto-detect | `None` |
| `--device` | Device to run inference on (`auto`, `cpu`, `cuda`) | `auto` |
| `--compute` | Compute type (`int8`, `float16`, etc.) | Auto-selected |
| `--no-denoise` | Disable RNNoise denoising step | `False` |
| `--no-vad` | Disable VAD filtering (not recommended for noisy audio) | `False` |
| `--out` | Custom output base path (without extension) | Same as input name |

## Output

The tool generates two files in the same directory as the input:
1. `<basename>.txt`: Plain text transcription.
2. `<basename>.srt`: Subtitle file with timestamps.

## Notes

- If FFmpeg does not support the `arnndn` filter, the denoising step will be skipped, but VAD filtering will still be applied.
- The `--device` auto setting will use CUDA if available, otherwise it falls back to CPU.
