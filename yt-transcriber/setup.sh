#!/usr/bin/env bash
# Setup script for yt-transcriber
# Creates virtual environment and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== yt-transcriber Setup ==="
echo

# Check if uv is installed
if ! command -v uv &>/dev/null; then
	echo "Error: 'uv' package manager is not installed."
	echo
	echo "Please install uv:"
	echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
	echo
	echo "Or install dependencies manually with:"
	echo "  pip install -e ."
	exit 1
fi

# Check if yt-dlp is installed
if ! command -v yt-dlp &>/dev/null; then
	echo "Warning: 'yt-dlp' is not installed globally."
	echo "It will be installed in the virtual environment."
fi

# Create virtual environment
echo "Creating virtual environment with uv..."
uv venv --python 3.12 .venv

# Activate venv temporarily
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo
echo "=== Setup Complete ==="
echo
echo "To use yt-transcribe:"
echo "  1. Run: source .venv/bin/activate"
echo "  2. Then: yt-transcribe [OPTIONS] INPUT"
echo
echo "Or use the wrapper script:"
echo "  ./yt-transcribe [OPTIONS] INPUT"
echo
echo "For help:"
echo "  ./yt-transcribe --help"
