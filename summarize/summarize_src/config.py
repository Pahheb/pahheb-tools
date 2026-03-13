"""Configuration for summarize tool."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for summarize tool."""

    provider: str = "ollama"
    model: str = "llama3.2"
    input_files: list[Path] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("./summaries"))
    output_format: str = "txt"
    summary_length: str = "standard"
    combine: bool = False
    unified: bool = False
    skip_combined: bool = False
    transcribe_first: bool = False
    verbose: bool = False

    transcribe_source: str = "local"
    transcribe_language: Optional[str] = None
    transcribe_model: str = "small"
    transcribe_device: str = "auto"
    transcribe_compute: Optional[str] = None
    transcribe_denoise: bool = False
    transcribe_vad: bool = False
    transcribe_audio_enhance: bool = False
    transcribe_srt: bool = False
    transcribe_cleanup: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.provider not in ["ollama", "huggingface", "watsonx"]:
            raise ValueError(
                f"Invalid provider: {self.provider}. Must be: ollama, huggingface, or watsonx"
            )

        if self.output_format not in ["txt", "md", "json"]:
            raise ValueError(
                f"Invalid output_format: {self.output_format}. Must be: txt, md, or json"
            )

        if self.summary_length not in ["brief", "standard", "detailed"]:
            raise ValueError(
                f"Invalid summary_length: {self.summary_length}. Must be: brief, standard, or detailed"
            )

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

    def find_combined_files(self) -> list[Path]:
        """Find any combined*.txt files in input files."""
        combined = []
        for f in self.input_files:
            if isinstance(f, str):
                stem = f.split("/")[-1].split("\\")[-1]
                if "?" in stem:
                    stem = stem.split("?")[0]
                if "combined" in stem.lower() and f.endswith(".txt"):
                    combined.append(f)
            else:
                if "combined" in f.stem.lower() and f.suffix == ".txt":
                    combined.append(f)
        return combined

    def filter_input_files(self) -> list[Path]:
        """Filter input files based on skip_combined setting."""
        if self.skip_combined:
            result = []
            for f in self.input_files:
                if isinstance(f, str):
                    stem = f.split("/")[-1].split("\\")[-1]
                    if "?" in stem:
                        stem = stem.split("?")[0]
                else:
                    stem = f.stem.lower()
                if "combined" not in stem:
                    result.append(f)
            return result
        return self.input_files
