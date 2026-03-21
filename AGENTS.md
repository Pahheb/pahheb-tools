# AGENTS.md

Monorepo with two Python CLI tools: `transcribe` and `summarize`. Each is a standalone installable package.

## Package Manager
Use **uv**: `uv venv --python 3.12 .venv && uv pip install -e . --python .venv/bin/python`

## File-Scoped Commands

| Task | Command |
|------|---------|
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Typecheck | `python -m mypy <pkg>_src/ --config-file mypy.ini` |
| Test | `python -m pytest tests/ -q` |
| Test single file | `python -m pytest tests/test_<name>.py -q` |

Always run from the project directory (e.g. `summarize/` or `transcribe/`).

## Project Structure

```
summarize/           AI transcription summarizer (Ollama/HuggingFace providers)
  summarize_src/
    __main__.py      Entry point, summarization handlers
    transcription.py File search, transcription subprocess, transcript reading
    pipeline.py      Threading pipeline (transcribe + summarize in parallel)
    summarizer.py    Provider abstraction (OllamaProvider, HuggingFaceProvider)
    file_writer.py   Output writers (txt/md/json), write_summary() dispatcher
    config.py        Config dataclass, build_transcribe_args()
    cli.py           Argparse CLI
  tests/
    test_summarize.py      Unit tests
    test_integration.py    Integration tests (mocked pipelines)

transcribe/          Local + YouTube audio transcription (faster-whisper)
  transcribe_src/
    __main__.py      Entry point (thin — 60 lines)
    whisper.py       transcribe_audio() + get_compute_type()
    audio_processor.py   FFmpeg subprocess wrapper
    youtube_processor.py YouTube download + transcription pipeline
    youtube_downloader.py yt-dlp wrapper
    local_processor.py   Local file transcription pipeline
    file_writer.py   TXT/SRT output writers
    config.py        Config dataclass
    cli.py           Argparse CLI
  tests/
    test_transcribe.py     Unit tests
    test_integration.py    Integration tests (mocked pipelines)
```

## CI

Single workflow: `.github/workflows/test.yml`. Runs on push/PR to master for changes in `summarize/**` or `transcribe/**`. Per-project jobs: ruff, ruff format, mypy, ty (advisory), pytest+coverage.

## Commit Format

```
<type>(<scope>): <subject

<body>

<footer>
```

Types: `feat`, `fix`, `ref`, `test`, `ci`, `docs`, `chore`. Scope is optional. Imperative mood. No period. Under 70 chars.

## Known Design Debts

- `sanitize_filename` in `transcribe/youtube_processor.py` has `import re` inside function body
- `Config.input_files` in summarize is `list[str | Path]` — non-transcribe path passes mixed types to functions expecting `Path` (has `# type: ignore[assignment]`)
- `--single-threaded` path in summarize `__main__.py` duplicates transcription loop logic from `pipeline.py`

## Key Conventions

- No `__init__.py` imports beyond version strings
- Tests use `unittest.mock.patch` on the module that imports the target (not the definition module)
- Both projects use `ruff.toml` (not pyproject.toml) for lint config
- Both projects use `mypy.ini` for type check config
