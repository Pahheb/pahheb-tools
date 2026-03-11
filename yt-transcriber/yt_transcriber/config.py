"""Configuration management for yt-transcriber."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for yt-transcriber."""

    # Input/output
    input_source: str
    output_dir: Path
    combine: bool = False

    # Transcription settings
    language: Optional[str] = None
    model_size: str = "small"
    device: str = "auto"

    # Processing settings
    sleep_delay: int = 2
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory is a Path
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate model size
        valid_models = {"tiny", "base", "small", "medium", "large-v3"}
        if self.model_size not in valid_models:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. Must be one of: {', '.join(valid_models)}"
            )

        # Validate device
        valid_devices = {"auto", "cuda", "cpu", "mps"}
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device: {self.device}. Must be one of: {', '.join(valid_devices)}"
            )

        # Validate sleep delay
        if self.sleep_delay < 0:
            raise ValueError("sleep_delay must be non-negative")
