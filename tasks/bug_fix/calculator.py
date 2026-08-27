"""A calculator module with a bug: divide_by_zero returns None instead of raising."""


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float | None:
    """Divide a by b. Returns None if b is zero (BUG: should raise ZeroDivisionError)."""
    if b == 0:
        return None
    return a / b


def calculate(expr: str) -> float:
    """Parse a simple binary expression like '3 + 4' and return the result."""
    parts = expr.split()
    if len(parts) != 3:
        raise ValueError(f"Expected format 'a op b', got: {expr}")
    a, op, b = float(parts[0]), parts[1], float(parts[2])
    if op == "+":
        return add(a, b)
    elif op == "-":
        return subtract(a, b)
    elif op == "*":
        return multiply(a, b)
    elif op == "/":
        return divide(a, b)
    else:
        raise ValueError(f"Unknown operator: {op}")
