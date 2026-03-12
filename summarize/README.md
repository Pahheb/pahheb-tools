# summarize

AI-powered summarization of video and audio transcriptions.

## Features

- **Multiple AI Providers**: Ollama (local), HuggingFace (local), watsonx.ai (cloud - coming soon)
- **Flexible Input**: Accept pre-transcribed `.txt` files or transcribe on-the-fly
- **Integrated Transcription**: Transcribe and summarize in one command with direct transcribe options
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

### For watsonx.ai Provider (Coming Soon)

Set environment variables:

```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
```

## Installation

```bash
cd summarize
./setup.sh
```

Or manually:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Summarize Pre-Transcribed Files

```bash
summarize transcription1.txt
summarize ./transcriptions/*.txt
```

### Transcribe and Summarize

```bash
# Simple transcription
summarize video.mp4 --transcribe
summarize audio.wav --transcribe --combine

# With transcription options
summarize video.mp4 --transcribe --transcribe-model medium --transcribe-srt

# YouTube transcription
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube --transcribe-language en
```

### Combine Multiple Transcriptions (Post-Processing Merge)

```bash
summarize file1.txt file2.txt --combine
```

### Unified Summary (Merge Transcriptions First)

```bash
summarize ./transcriptions/*.txt --unified
```

### Skip Combined Files

When using the transcribe tool with `--combine`, it creates both individual files and a `combined.txt`. Use `--skip-combined` to exclude the combined file:

```bash
summarize ./transcriptions/*.txt --unified --skip-combined
summarize ./transcriptions/*.txt --skip-combined
```

### Use Different AI Provider

```bash
summarize file.txt --provider ollama --model llama3.2
summarize file.txt --provider huggingface --model microsoft/Phi-3.5-mini-instruct
```

### Control Output

```bash
summarize file.txt --output-dir ./summaries --output-format md
summarize file.txt --summary-length brief
```

## Options

### General Options

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Input files (.txt transcriptions or video/audio with --transcribe) | Required |
| `--transcribe`, `-t` | Transcribe video/audio before summarizing | False |
| `--provider`, `-p` | AI provider (ollama, huggingface, watsonx) | `ollama` |
| `--model`, `-m` | Model to use for summarization | Provider-specific |
| `--combine`, `-c` | Combine per-video summaries into one file (post-processing) | False |
| `--unified`, `-u` | Merge transcriptions first, then create one unified summary | False |
| `--skip-combined` | Skip combined*.txt files from input | False |
| `--output-dir`, `-o` | Output directory | `./summaries` |
| `--output-format`, `-f` | Output format (txt, md, json) | `txt` |
| `--summary-length`, `-l` | Summary length (brief, standard, detailed) | `standard` |
| `--verbose`, `-v` | Show detailed progress | False |

### Transcribe Options (only used with `--transcribe`)

| Option | Description | Default |
|--------|-------------|---------|
| `--transcribe-source`, `-ts` | Input source type (local, youtube) | `local` |
| `--transcribe-language`, `-tl` | Language code (e.g., 'en', 'es') | auto-detect |
| `--transcribe-model`, `-tm` | Whisper model (tiny, base, small, medium, large-v3) | `small` |
| `--transcribe-device`, `-td` | Device (auto, cuda, cpu, mps) | `auto` |
| `--transcribe-compute` | Compute type (int8, int8_float16, float16, float32) | auto |
| `--transcribe-denoise` | Apply RNNoise denoising | False |
| `--transcribe-vad` | Apply voice activity detection | False |
| `--transcribe-audio-enhance` | Apply audio enhancement | False |
| `--transcribe-srt` | Generate SRT subtitle file | False |
| `--transcribe-cleanup` | Delete intermediate files | False |

## Examples

### Summarize a Single Transcription

```bash
summarize ./transcriptions/video_abc123.txt
```

### Summarize Multiple Files

```bash
summarize video1.txt video2.txt video3.txt
```

### Transcribe and Summarize

```bash
# Simple - transcribe local file then summarize
summarize video.mp4 --transcribe

# With model selection and SRT output
summarize video.mp4 --transcribe --transcribe-model medium --transcribe-srt

# YouTube video
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube
```

### Combine and Summarize

```bash
summarize *.txt --combine --output-format md
```

### Unified Summary

```bash
summarize video1.txt video2.txt video3.txt --unified
```

### Skip Combined Files

```bash
# If transcribe was run with --combine, you get both individual files
# and combined.txt. Skip the combined file to avoid duplicates:
summarize ./transcriptions/*.txt --unified --skip-combined
```

### Brief Summary

```bash
summarize long_transcript.txt --summary-length brief
```

### Detailed Summary

```bash
summarize long_transcript.txt --summary-length detailed --output-format md
```

### Full Pipeline: Transcribe YouTube Video and Summarize

```bash
# Transcribe and summarize in one command
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube

# Or with custom output directory (transcriptions go in ./my_summaries/transcriptions/)
summarize https://youtube.com/watch?v=VIDEO_ID --transcribe --transcribe-source youtube --output-dir ./my_summaries
```

## Output Format

### TXT Output

```
Source: ./transcriptions/video.txt
Summary Type: standard
--------------------------------------------------

[Summary text here]
```

### Markdown Output

```markdown
# Summary

**Source:** ./transcriptions/video.txt
**Type:** standard

## Key Points

1. Point 1
2. Point 2
3. Point 3

## Summary

[Detailed summary text]
```

### JSON Output

```json
{
  "summary": "...",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "metadata": {
    "source": "...",
    "summary_type": "standard",
    "provider": "ollama"
  }
}
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

### "HuggingFace model not available"

Make sure transformers and torch are installed, or try a different model:
```bash
pip install transformers torch
```
