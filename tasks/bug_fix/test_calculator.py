"""Test for calculator module — expects divide(1, 0) to raise ZeroDivisionError."""

import pytest
from tasks.bug_fix.calculator import add, subtract, multiply, divide, calculate


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12


def test_divide_normal():
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == 3.5


def test_divide_by_zero():
    """BUG: divide(1, 0) should raise ZeroDivisionError, not return None."""
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


def test_calculate():
    assert calculate("2 + 3") == 5
    assert calculate("10 / 2") == 5.0
