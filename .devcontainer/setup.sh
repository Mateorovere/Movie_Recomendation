#!/bin/bash
set -e

echo "Installing uv..."
pip install uv pre-commit

echo "Setting up Python environment with uv..."
uv venv --clear
source .venv/bin/activate

echo "Installing dependencies..."
uv sync --dev

echo "Installing pre-commit hooks..."
pre-commit install
echo "Setup complete! Your development environment is ready."
