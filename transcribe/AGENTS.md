# AGENTS.md — transcribe

Local + YouTube audio transcription using faster-whisper (OpenAI Whisper via CTranslate2).

## Flow

1. `cli.py` parses args → `config.py` stores config
2. `__main__.py` loops over `args.inputs`, auto-detects source per input (YouTube URL vs local file)
3. If YouTube: `youtube_processor.py` → `youtube_downloader.py` (yt-dlp) → `audio_processor.py` (FFmpeg) → `whisper.py`
4. If local: `local_processor.py` → `audio_processor.py` → `whisper.py`
5. `file_writer.py` writes TXT and optional SRT output

## Pipeline: YouTube

```
youtube_processor.py
  └─ youtube_downloader.download_youtube_audio(url) → Path
  └─ audio_processor.process_audio(wav, denoise, vad, enhance) → Path
  └─ whisper.transcribe_audio(wav, model, language, device) → list[dict]
  └─ file_writer.write_transcripts(segments, base_path, metadata, srt)
```

## Pipeline: Local

```
local_processor.py
  └─ audio_processor.process_audio(input, denoise, vad, enhance) → Path
  └─ whisper.transcribe_audio(wav, model, language, device) → list[dict]
  └─ file_writer.write_transcripts(segments, base_path, metadata, srt)
```

## Module Relationships

```
__main__.py  →  cli.py → config.py
            →  _is_youtube_url() (auto-detect source)
            →  _process_input() (dispatches to processor based on source)
            →  local_processor.py → audio_processor.py → whisper.py → file_writer.py
            →  youtube_processor.py → youtube_downloader.py → audio_processor.py → whisper.py → file_writer.py
```

## Key Functions

- `__main__._is_youtube_url(text)` — detects YouTube URLs via regex
- `__main__._process_input(config, input_arg, force_source)` — dispatches one input to local or YouTube processor
- `whisper.transcribe_audio()` — loads model, runs inference, returns `[{"text", "start", "end"}]`
- `whisper.get_compute_type(device)` — returns int8/float16 based on device
- `audio_processor.process_audio()` — calls FFmpeg for WAV conversion + optional denoise/VAD/enhance
- `youtube_downloader.download_youtube_audio()` — yt-dlp wrapper, returns WAV path
- `file_writer.write_transcripts()` — writes TXT + optional SRT
- `file_writer.format_srt_timestamp()` — converts float seconds to SRT format

## Known Issues

- `torch` imported unconditionally at module level in `whisper.py` — crashes if torch not installed
- FFmpeg arnndn support check runs subprocess on every `process_audio(denoise=True)` call (no caching)

## Commands

| Task | Command |
|------|---------|
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Typecheck | `python -m mypy transcribe_src/ --config-file mypy.ini` |
| Test | `python -m pytest tests/ -q` |
| Test single | `python -m pytest tests/test_transcribe.py -q` |
    | Run | `transcribe <file> [file2] [--source youtube] [--model small]` |

## Test Patterns

- Unit tests mock `subprocess.run` for FFmpeg/yt-dlp calls
- Integration tests mock `whisper.transcribe_audio` to avoid loading models
- `TestWhisperHelpers` tests are skipped if torch not installed
