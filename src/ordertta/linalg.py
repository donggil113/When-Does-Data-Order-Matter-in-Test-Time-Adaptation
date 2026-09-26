"""Minimal dense linear algebra on Python lists.

Vectors are ``list[float]``; matrices are ``list[list[float]]`` in row-major order.
Only what the fixtures need is implemented; sizes are tiny (d <= ~64).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

Vector = List[float]
Matrix = List[List[float]]


def zeros(n: int) -> Vector:
    return [0.0] * n


def eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[float], s: float) -> Vector:
    return [s * x for x in a]


def axpy(alpha: float, x: Sequence[float], y: Sequence[float]) -> Vector:
    """Return alpha * x + y."""
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def matvec(m: Matrix, v: Sequence[float]) -> Vector:
    return [math.fsum(mij * vj for mij, vj in zip(row, v)) for row in m]


def rmatvec(m: Matrix, v: Sequence[float]) -> Vector:
    """Return m^T v."""
    ncol = len(m[0])
    out = [0.0] * ncol
    for row, vi in zip(m, v):
        for j in range(ncol):
            out[j] += row[j] * vi
    return out


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    return [[math.fsum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def outer(a: Sequence[float], b: Sequence[float]) -> Matrix:
    return [[x * y for y in b] for x in a]


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    return [add(ra, rb) for ra, rb in zip(a, b)]


def mat_scale(a: Matrix, s: float) -> Matrix:
    return [scale(r, s) for r in a]


def random_vector(rng: random.Random, n: int, sd: float = 1.0) -> Vector:
    return [rng.gauss(0.0, sd) for _ in range(n)]


def random_matrix(rng: random.Random, rows: int, cols: int, sd: float = 1.0) -> Matrix:
    return [random_vector(rng, cols, sd) for _ in range(rows)]


def random_spd(rng: random.Random, n: int, min_eig: float = 0.1, max_eig: float = 2.0) -> Matrix:
    """Random symmetric positive definite matrix Q diag(lam) Q^T with eigenvalues in [min_eig, max_eig]."""
    q = orthonormalize(random_matrix(rng, n, n))  # rows orthonormal
    lam = [rng.uniform(min_eig, max_eig) for _ in range(n)]
    out = [[0.0] * n for _ in range(n)]
    for k in range(n):
        qk = q[k]
        for i in range(n):
            for j in range(n):
                out[i][j] += lam[k] * qk[i] * qk[j]
    return out


def orthonormalize(vectors: Sequence[Sequence[float]], tol: float = 1e-10) -> Matrix:
    """Modified Gram-Schmidt on a list of vectors; drops (near-)dependent vectors."""
    basis: Matrix = []
    for v in vectors:
        w = list(v)
        for _ in range(2):  # re-orthogonalise once for numerical stability
            for b in basis:
                w = axpy(-dot(w, b), b, w)
        n = norm(w)
        if n > tol:
            basis.append(scale(w, 1.0 / n))
    return basis


def max_abs(a: Sequence[float]) -> float:
    return max((abs(x) for x in a), default=0.0)
