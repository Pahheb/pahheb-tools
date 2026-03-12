#!/bin/bash
# Test runner script for transcribe tool
# Runs tests inside the virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

check_uv() {
	if ! command -uv &>/dev/null; then
		if ! command -v uv &>/dev/null; then
			echo "Error: 'uv' package manager is not installed"
			echo "Please install uv: https://github.com/astral-sh/uv#installation"
			exit 1
		fi
	fi
}

setup_venv() {
	echo "Setting up virtual environment..."
	check_uv

	uv venv --python 3.12 "$VENV_DIR"

	echo "Installing dependencies..."
	uv pip install -e "$SCRIPT_DIR" --python "$VENV_DIR/bin/python"
	uv pip install pytest pytest-asyncio --python "$VENV_DIR/bin/python"

	echo "Virtual environment created at $VENV_DIR"
}

if [ ! -d "$VENV_DIR" ]; then
	echo "Virtual environment not found."
	setup_venv
elif [ ! -f "$VENV_DIR/bin/python" ]; then
	echo "Virtual environment is corrupted or incomplete."
	rm -rf "$VENV_DIR"
	setup_venv
fi

echo "Running tests in virtual environment..."
"$VENV_DIR/bin/python" -m pytest "$@"
