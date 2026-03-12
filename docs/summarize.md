# Summarize Tool

AI-powered summarization of video and audio transcriptions using local AI models.

## Features

- **Multiple AI Providers**: Ollama (local), HuggingFace (local), watsonx.ai (cloud - coming soon)
- **Flexible Input**: Accept pre-transcribed `.txt` files or transcribe on-the-fly
- **Integrated Transcription**: Transcribe and summarize in one command
- **Combined Summaries**: Merge multiple transcriptions into one summary with `--combine`
- **Unified Summaries**: Merge transcriptions first, then create ONE summary with `--unified`
- **Smart Filtering**: Skip combined files automatically with `--skip-combined`
- **Multiple Output Formats**: TXT, Markdown, or JSON output
- **Configurable Length**: Brief, standard, or detailed summaries

## Prerequisites

### Required

- **Python 3.12+**
- **uv** package manager - Install from https://github.com/astral-sh/uv

### For Ollama Provider (Recommended)

1. Install Ollama: https://github.com/ollama/ollama
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2`

### For HuggingFace Provider (Alternative)

Install transformers and torch:

```bash
pip install transformers torch
```

## Installation

```bash
cd summarize
./setup.sh
```

## Usage

### Basic Usage

```bash
# Summarize a single transcription
summarize transcription.txt

# Summarize multiple files
summarize file1.txt file2.txt

# Transcribe and summarize in one step
summarize video.mp4 --transcribe
summarize https://youtube.com/watch?v=ID --transcribe --transcribe-source youtube
```

### Options

#### General Options

| Option | Description | Default |
|--------|-------------|---------|
| `--transcribe` | Transcribe video/audio before summarizing | False |
| `--provider` | AI provider (ollama, huggingface) | ollama |
| `--model` | Model to use for summarization | Provider-specific |
| `--combine` | Combine per-video summaries into one file (post-processing) | False |
| `--unified` | Merge transcriptions first, then create one unified summary | False |
| `--skip-combined` | Skip combined*.txt files from input | False |
| `--output-dir` | Output directory | ./summaries |
| `--output-format` | Output format (txt, md, json) | txt |
| `--summary-length` | brief, standard, detailed | standard |

#### Transcribe Options (only used with `--transcribe`)

| Option | Description | Default |
|--------|-------------|---------|
| `--transcribe-source` | Input source (local, youtube) | local |
| `--transcribe-language` | Language code (e.g., 'en', 'es') | auto-detect |
| `--transcribe-model` | Whisper model (tiny, base, small, medium, large-v3) | small |
| `--transcribe-device` | Device (auto, cuda, cpu, mps) | auto |
| `--transcribe-compute` | Compute type | auto |
| `--transcribe-denoise` | Apply RNNoise denoising | False |
| `--transcribe-vad` | Apply voice activity detection | False |
| `--transcribe-audio-enhance` | Apply audio enhancement | False |
| `--transcribe-srt` | Generate SRT subtitle file | False |
| `--transcribe-cleanup` | Delete intermediate files | False |

## Examples

### Summarize a Transcription

```bash
summarize ./transcriptions/video_abc123.txt
```

### Transcribe and Summarize

```bash
# Local file
summarize video.mp4 --transcribe
summarize video.mp4 --transcribe --transcribe-model medium --transcribe-srt

# YouTube video
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube
```

### Combine Multiple Transcriptions

```bash
summarize *.txt --combine --output-format md
```

### Unified Summary

```bash
# Merge transcriptions first, then summarize as one
summarize video1.txt video2.txt --unified
```

### Skip Combined Files

When the transcribe tool is used with `--combine`, it creates both individual files and `combined.txt`. Use `--skip-combined` to avoid duplicates:

```bash
summarize ./transcriptions/*.txt --unified --skip-combined
```

## Output Formats

### Markdown Output (Recommended)

Includes key points and detailed summary:

```markdown
# Summary

**Source:** ./transcriptions/video.txt

## Key Points

1. First main point
2. Second main point
3. Third main point

## Summary

[Detailed summary text]
```

## Troubleshooting

### "Cannot connect to Ollama"

Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found"

Pull the model:
```bash
ollama pull llama3.2
```
