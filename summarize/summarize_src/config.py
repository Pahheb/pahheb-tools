"""Configuration for summarize tool."""

from dataclasses import dataclass, field
from pathlib import Path


def _stem(f: str | Path) -> str:
    """Extract filename stem from a string path or Path object."""
    if isinstance(f, Path):
        return f.stem.lower()
    name = f.split("/")[-1].split("\\")[-1]
    if "?" in name:
        name = name.split("?")[0]
    return Path(name).stem.lower()


@dataclass
class Config:
    """Configuration for summarize tool."""

    provider: str = "ollama"
    model: str = "llama3.2"
    input_files: list[str | Path] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("./summaries"))
    output_format: str = "txt"
    summary_length: str = "standard"
    combine: bool = False
    unified: bool = False
    skip_combined: bool = False
    transcribe_first: bool = False
    verbose: bool = False
    single_threaded: bool = False

    transcribe_source: str = "local"
    transcribe_language: str | None = None
    transcribe_model: str = "small"
    transcribe_device: str = "auto"
    transcribe_compute: str | None = None
    transcribe_denoise: bool = False
    transcribe_vad: bool = False
    transcribe_audio_enhance: bool = False
    transcribe_srt: bool = False
    transcribe_cleanup: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir).expanduser().resolve()

    @property
    def transcribe_output_dir(self) -> Path:
        """Return the transcribe output directory (inside summary output dir)."""
        return self.output_dir / "transcriptions"

    def build_transcribe_args(self) -> list[str]:
        """Build argument list for transcribe command."""
        args = []

        if self.transcribe_source != "local":
            args.extend(["--source", self.transcribe_source])

        args.extend(["--output-dir", str(self.transcribe_output_dir)])

        if self.transcribe_language:
            args.extend(["--language", self.transcribe_language])

        if self.transcribe_model != "small":
            args.extend(["--model", self.transcribe_model])

        if self.transcribe_device != "auto":
            args.extend(["--device", self.transcribe_device])

        if self.transcribe_compute:
            args.extend(["--compute", self.transcribe_compute])

        if self.transcribe_denoise:
            args.append("--denoise")

        if self.transcribe_vad:
            args.append("--vad")

        if self.transcribe_audio_enhance:
            args.append("--audio-enhance")

        if self.transcribe_srt:
            args.append("--srt")

        if self.transcribe_cleanup:
            args.append("--cleanup")

        return args

    def find_combined_files(self) -> list[str | Path]:
        """Find any combined*.txt files in input files."""
        return [
            f for f in self.input_files if "combined" in _stem(f) and _stem(f) != ""
        ]

    def filter_input_files(self) -> list[str | Path]:
        """Filter input files based on skip_combined setting."""
        if self.skip_combined:
            return [f for f in self.input_files if "combined" not in _stem(f)]
        return self.input_files
