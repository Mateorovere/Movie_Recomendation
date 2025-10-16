"""Tests for the example module."""

import pytest
from my_python_project import add, hello


def test_hello() -> None:
    """Test the hello function."""
    assert hello("World") == "Hello, World!"
    assert hello("Python") == "Hello, Python!"


def test_add() -> None:
    """Test the add function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_add_negative() -> None:
    """Test adding negative numbers."""
    assert add(-5, -3) == -8


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (10, 20, 30),
        (100, 200, 300),
    ],
)
def test_add_parametrized(a: int, b: int, expected: int) -> None:
    """Test add with multiple parameters."""
    assert add(a, b) == expected
