"""Least squares, small and dependency-free.

Normal equations with Gaussian elimination.  For the problem sizes here - never
more than four unknowns against a few thousand samples - the conditioning
penalty of the normal equations over a QR factorisation is irrelevant, and the
whole thing fits on a screen.
"""

__all__ = ["solve", "least_squares", "fit_line"]


class SingularMatrixError(ValueError):
    """The design matrix had no unique solution, usually from degenerate data."""


def solve(matrix, rhs):
    """Solve a square system by Gaussian elimination with partial pivoting."""
    size = len(matrix)
    if any(len(row) != size for row in matrix) or len(rhs) != size:
        raise ValueError("solve expects a square matrix and a matching right-hand side")
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise SingularMatrixError(f"no pivot in column {column}")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_row = augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot_row[column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * pivot_row[index]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for column in range(row + 1, size):
            total -= augmented[row][column] * solution[column]
        solution[row] = total / augmented[row][row]
    return solution


def least_squares(design, targets):
    """Minimise |design @ x - targets|, returning (coefficients, rms_residual).

    `design` is a sequence of rows, one per sample.
    """
    if not design:
        raise ValueError("least_squares needs at least one sample")
    width = len(design[0])
    if any(len(row) != width for row in design):
        raise ValueError("every design row must have the same width")
    if len(targets) != len(design):
        raise ValueError("design and targets must have the same length")
    if len(design) < width:
        raise ValueError(
            f"least_squares needs at least {width} samples for {width} unknowns"
        )

    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, target in zip(design, targets):
        for i in range(width):
            rhs[i] += row[i] * target
            for j in range(width):
                normal[i][j] += row[i] * row[j]

    coefficients = solve(normal, rhs)
    total = 0.0
    for row, target in zip(design, targets):
        predicted = sum(
            coefficient * value for coefficient, value in zip(coefficients, row)
        )
        total += (predicted - target) ** 2
    return coefficients, (total / len(design)) ** 0.5


def fit_line(xs, ys):
    """Ordinary least squares line fit, returning (slope, intercept, r_squared)."""
    if len(xs) != len(ys):
        raise ValueError("fit_line needs matching x and y")
    if len(xs) < 2:
        raise ValueError("fit_line needs at least two points")
    (slope, intercept), _ = least_squares([[x, 1.0] for x in xs], list(ys))
    mean = sum(ys) / len(ys)
    total = sum((y - mean) ** 2 for y in ys)
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return slope, intercept, r_squared
