#!/bin/bash
# Setup script for transcribe tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up transcribe tool..."

# Check for uv
if ! command -v uv &>/dev/null; then
	echo "Error: uv not found. Install with: pip install uv"
	exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
	echo "Creating virtual environment..."
	uv venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo ""
echo "Setup complete!"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: transcribe --help"
