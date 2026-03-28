# AGENTS.md — summarize

AI-powered transcription summarizer. Takes transcription files (or YouTube URLs with `--transcribe`) and produces summaries via Ollama or HuggingFace.

## Flow

1. `cli.py` parses args → `config.py` stores config
2. If `--transcribe`, calls `transcribe` CLI subprocess to produce `.txt` files
3. `summarizer.py` sends text to LLM provider, returns `SummaryResult(summary, key_points, metadata)`
4. `file_writer.py` writes output in txt/md/json format via `write_summary()` dispatcher

## Threading Pipeline (default with `--transcribe`)

`pipeline.py` implements a two-thread producer-consumer pattern:
- Main thread: iterates input files, checks for existing transcriptions via `transcribe_file()`
- Daemon thread: consumes from queue, calls `summarize_file()`
- Sentinel `_SENTINEL` signals end of work
- `--single-threaded` / `--st` flag disables threading

## Module Relationships

```
__main__.py  →  transcription.py (read_transcription, transcribe_file, find_transcription_file)
            →  pipeline.py (_transcribe_then_summarize_thread, _transcribe_then_combine)
            →  summarizer.py (get_provider, ProviderNotAvailableError, SummarizerError)
            →  file_writer.py (write_summary)
            →  cli.py (parse_args)
            →  config.py (Config)
```

## Provider Architecture

- `BaseProvider` (ABC) → `OllamaProvider`, `HuggingFaceProvider`
- Shared `_parse_response(content, source, model)` function parses KEYPOINTS/SUMMARY format
- `get_provider(name, model)` factory function
- Providers check availability via `is_available()` before use

## Known Issues

- `Config.input_files: list[str | Path]` — YouTube URLs are strings, files are Paths. `filter_input_files()` returns mixed types, `# type: ignore[assignment]` in `__main__.py` (Python list invariance issue)

## Commands

| Task | Command |
|------|---------|
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Typecheck | `python -m mypy summarize_src/ --config-file mypy.ini` |
| Test | `python -m pytest tests/ -q` |
| Test single | `python -m pytest tests/test_summarize.py -q` |
| Run | `summarize <file_or_url> [--transcribe] [--provider ollama]` |

## Test Patterns

- Unit tests mock at the module boundary (e.g. `patch("summarize_src.summarizer.httpx.Client")`)
- Integration tests mock `get_provider`, `summarize_file`, `transcribe_file`, `find_transcription_file` on `__main__` or `pipeline`
- Threaded tests use `_transcribe_then_summarize_thread` with mocked `summarize_fn` callback
