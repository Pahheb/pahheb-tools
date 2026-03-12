# transcribe

Unified transcription tool for local audio/video files and YouTube videos.

## Features

- **Local files**: Transcribe MP3, WAV, M4A, MP4, AVI, MKV, MOV, FLAC, AAC, OGG, WMA, WebM
- **YouTube**: Download and transcribe YouTube videos
- **Whisper models**: tiny, base, small, medium, large-v3 (default: small)
- **GPU support**: CUDA, AMD ROCm, Apple MPS, CPU
- **Optional processing**:
  - RNNoise denoising (`--denoise`)
  - Voice activity detection (`--vad`)
  - Audio enhancement filters (`--audio-enhance`)
- **Output formats**: TXT (default), SRT subtitles (`--srt`)
- **Cleanup**: Remove intermediate files (`--cleanup`)

## Installation

```bash
./setup.sh
```

Or manually:

```bash
uv venv venv
source venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

### Local file transcription

```bash
transcribe /path/to/audio.mp3
transcribe video.mov --model medium --srt
transcribe audio.mp3 --audio-enhance --vad
```

### YouTube transcription

```bash
transcribe https://www.youtube.com/watch?v=VIDEO_ID --source youtube
transcribe https://youtu.be/VIDEO_ID -s youtube --srt
```

### With RNNoise denoising

```bash
transcribe audio.mp3 --denoise
```

### All options

```bash
transcribe input.mp3 \
  --source local \
  --output-dir ./transcriptions \
  --language en \
  --model small \
  --device auto \
  --vad \
  --srt \
  --cleanup \
  --verbose
```

## Options

- `--source`, `-s`: Input source type (local or youtube)
- `--output-dir`, `-o`: Output directory
- `--language`, `-l`: Language code (auto-detect if not specified)
- `--model`: Whisper model size (tiny/base/small/medium/large-v3)
- `--device`: Device (auto/cuda/cpu/mps)
- `--compute`: Compute type (int8, float16, etc.)
- `--denoise`: Apply RNNoise denoising
- `--denoise-model`: Path to RNNoise model
- `--vad`: Apply voice activity detection
- `--audio-enhance`: Apply audio enhancement filters
- `--srt`: Generate SRT subtitle file
- `--cleanup`: Delete intermediate files after transcription
- `--verbose`, `-v`: Show detailed progress
