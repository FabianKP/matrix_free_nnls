"""
Created by Fabian on 06.11.2022.
"""

import numpy as np

from matrix_free_nnls import solve_nnls


np.random.seed(42)


def test_returns_something():
    a = np.random.randn(10, 10)
    b = np.random.randn(10)
    x, res = solve_nnls(a, b)