# My Python Project

A modern Python project with devcontainer, uv, and pre-commit hooks.

## Features

- 🐍 Python 3.12+
- 📦 Dependency management with [uv](https://github.com/astral-sh/uv)
- 🐳 Development container for consistent environments
- ✅ Pre-commit hooks with Black, Ruff, mypy, and pytest
- 🧪 Testing with pytest and coverage
- 🔍 Type checking with mypy
- 📝 Code formatting with Black
- 🚀 Fast linting with Ruff

## Getting Started

### Option 1: Using DevContainer (Recommended)

1. Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this folder in VS Code
4. Click "Reopen in Container" when prompted (or use Command Palette: "Dev Containers: Reopen in Container")

The container will automatically:
- Install Python 3.12
- Install uv
- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks

### Option 2: Local Setup

1. Install [uv](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate virtual environment:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_example.py
```

### Code Formatting

```bash
# Format code with Black
uv run black src tests

# Format with Ruff
uv run ruff format src tests
```

### Linting

```bash
# Run Ruff linter
uv run ruff check src tests

# Auto-fix issues
uv run ruff check --fix src tests
```

### Type Checking

```bash
uv run mypy src
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files
uv run pre-commit run
```

## Project Structure

```
.
├── .devcontainer/
│   └── devcontainer.json    # Dev container configuration
├── src/
│   └── my_python_project/   # Source code
│       └── __init__.py
├── tests/
│   └── test_example.py      # Test files
├── .gitignore
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── pyproject.toml           # Project configuration
├── README.md
└── uv.lock                  # Locked dependencies (auto-generated)
```

## Adding Dependencies

```bash
# Add a runtime dependency
uv add requests

# Add a development dependency
uv add --dev pytest-mock

# Remove a dependency
uv remove requests
```

## CI/CD

This project is ready for CI/CD. Example GitHub Actions workflow:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run pre-commit
        run: uv run pre-commit run --all-files
      - name: Run tests
        run: uv run pytest --cov
```

## License

MIT
