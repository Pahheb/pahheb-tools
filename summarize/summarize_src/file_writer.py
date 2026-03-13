"""File writing utilities for summarize tool."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    filename = filename.strip(". ")

    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename if filename else "unnamed"


def write_summary_txt(
    summary: str,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write summary as plain text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_parts = []

    if metadata:
        content_parts.append(f"Source: {metadata.get('source', 'unknown')}")
        if "title" in metadata:
            content_parts.append(f"Title: {metadata['title']}")
        if "duration" in metadata:
            content_parts.append(f"Duration: {metadata['duration']}")
        if "summary_type" in metadata:
            content_parts.append(f"Summary Type: {metadata['summary_type']}")
        content_parts.append("-" * 50)
        content_parts.append("")

    content_parts.append(summary)

    output_path.write_text("\n".join(content_parts), encoding="utf-8")
    return output_path


def write_summary_md(
    summary: str,
    output_path: Path,
    key_points: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write summary as Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_parts = []

    if metadata:
        content_parts.append("# Summary")
        content_parts.append("")
        if "source" in metadata:
            content_parts.append(f"**Source:** {metadata['source']}")
        if "title" in metadata:
            content_parts.append(f"**Title:** {metadata['title']}")
        if "duration" in metadata:
            content_parts.append(f"**Duration:** {metadata['duration']}")
        if "summary_type" in metadata:
            content_parts.append(f"**Type:** {metadata['summary_type']}")
        content_parts.append("")

    if key_points:
        content_parts.append("## Key Points")
        content_parts.append("")
        for i, point in enumerate(key_points, 1):
            content_parts.append(f"{i}. {point}")
        content_parts.append("")

    content_parts.append("## Summary")
    content_parts.append("")
    content_parts.append(summary)

    output_path.write_text("\n".join(content_parts), encoding="utf-8")
    return output_path


def write_summary_json(
    summary: str,
    output_path: Path,
    key_points: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write summary as JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": summary,
    }

    if key_points:
        data["key_points"] = key_points

    if metadata:
        data["metadata"] = metadata

    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
