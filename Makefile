.PHONY: help install test lint format type-check clean pre-commit

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make type-check   - Run type checking"
	@echo "  make pre-commit   - Run pre-commit on all files"
	@echo "  make clean        - Clean build artifacts"

install:
	uv sync
	uv run pre-commit install

test:
	uv run pytest --cov --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run black src tests
	uv run ruff check --fix src tests

type-check:
	uv run mypy src

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
