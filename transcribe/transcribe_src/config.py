"""Configuration management for transcribe tool."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration for transcribe tool."""

    source: str
    output_dir: Path = field(default_factory=lambda: Path("./transcriptions"))

    language: str | None = None
    model_size: str = "small"
    device: str = "auto"
    compute_type: str | None = None

    denoise: bool = False
    denoise_model: str | None = None
    vad: bool = False
    audio_enhance: bool = False
    srt: bool = False

    cleanup: bool = False

    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
