"""Progress tracking utilities."""

from typing import Optional


class ProgressTracker:
    """Simple progress tracker for transcription tasks."""

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Progress description
        """
        self.total = total
        self.description = description
        self.current = 0

    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        msg = message or f"{self.current}/{self.total}"
        print(f"{self.description}: {msg} ({percentage:.0f}%)")

    def complete(self, message: str = "Complete"):
        """Mark as complete."""
        print(f"{self.description}: {message}")
