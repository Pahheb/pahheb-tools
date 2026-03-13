#!/usr/bin/env bash
# Test runner script for summarize tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

setup_venv() {
	echo "Setting up virtual environment..."

	if ! command -v uv &>/dev/null; then
		echo "Error: 'uv' package manager is not installed"
		echo "Please install uv: https://github.com/astral-sh/uv#installation"
		exit 1
	fi

	uv venv --python 3.12 "$VENV_DIR"
	uv pip install -e "$SCRIPT_DIR" --python "$VENV_DIR/bin/python"
	uv pip install pytest --python "$VENV_DIR/bin/python"

	echo "Virtual environment created at $VENV_DIR"
}

if [ ! -d "$VENV_DIR" ]; then
	setup_venv
elif [ ! -f "$VENV_DIR/bin/python" ]; then
	echo "Virtual environment is corrupted. Recreating..."
	rm -rf "$VENV_DIR"
	setup_venv
fi

export PATH="$VENV_DIR/bin:$PATH"

echo "Running tests..."
pytest -v "$SCRIPT_DIR/tests/"
