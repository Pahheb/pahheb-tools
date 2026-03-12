#!/usr/bin/env bash
# Setup script for summarize tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up summarize tool..."

if ! command -v uv &>/dev/null; then
	echo "Error: 'uv' package manager is not installed"
	echo "Please install uv: https://github.com/astral-sh/uv#installation"
	exit 1
fi

uv venv --python 3.12 "$VENV_DIR"

echo "Installing dependencies..."
uv pip install -e "$SCRIPT_DIR" --python "$VENV_DIR/bin/python"

echo ""
echo "Setup complete!"
echo "Virtual environment: $VENV_DIR"
echo ""
echo "Usage:"
echo "  ./summarize [options] input_files..."
echo ""
echo "For more options, run:"
echo "  ./summarize --help"
