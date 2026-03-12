"""Configuration management for transcribe tool."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for transcribe tool."""

    source: str
    input_path: str
    output_dir: Path = field(default_factory=lambda: Path("./transcriptions"))

    language: Optional[str] = None
    model_size: str = "small"
    device: str = "auto"
    compute_type: Optional[str] = None

    denoise: bool = False
    denoise_model: Optional[str] = None
    vad: bool = False
    audio_enhance: bool = False
    srt: bool = False

    cleanup: bool = False

    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.source not in ("local", "youtube"):
            raise ValueError(f"Invalid source: {self.source}. Must be 'local' or 'youtube'")

        valid_models = {"tiny", "base", "small", "medium", "large-v3"}
        if self.model_size not in valid_models:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. Must be one of: {', '.join(valid_models)}"
            )

        valid_devices = {"auto", "cuda", "cpu", "mps"}
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device: {self.device}. Must be one of: {', '.join(valid_devices)}"
            )
