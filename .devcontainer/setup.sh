#!/bin/bash
set -e

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/home/vscode/.cargo/bin:$PATH"

echo "Setting up Python environment with uv..."
uv venv --clear
source .venv/bin/activate

echo "Installing dependencies..."
uv sync --dev


echo "Installing pre-commit hooks..."
pre-commit install

echo "Setup complete! Your development environment is ready."
