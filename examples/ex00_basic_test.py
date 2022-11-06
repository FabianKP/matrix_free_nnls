"""
Created by Fabian on 06.11.2022.

Basic test for `nnls_solve`.
"""

import numpy as np
from scipy.optimize import nnls

from matrix_free_nnls import solve_nnls


np.random.seed(42)
m = 3000
n = 1000

a = np.random.randn(m, n)
b = np.random.randn(m)

atanorm = np.linalg.norm(a.T @ a)

# Solve with 'solve_nnls'.
x, res = solve_nnls(a, b, with_restart=False, verbose=True)
rnorm = np.linalg.norm(res)
# Solve with restart.
x_restart, res_restart = solve_nnls(a, b, with_restart=True, verbose=True)
rnorm_restart = np.linalg.norm(res_restart)

# Solve with Lawson-Hanson method.
x_lh, rnorm_lh = nnls(a, b)

print("solve_nnls:")
print(f"   Residual {rnorm}.")
print("solve_nnls (with restart):")
print(f"   Residual {rnorm_restart}.")
print("scipy.optimize.nnls:")
print(f"   Residual {rnorm_lh}.")
diff = rnorm_lh - rnorm_restart
if diff > 0:
    print(f"'solve_nnls' is better by {diff}.")
elif diff < 0:
    print(f"'scipy.optimize.nnls' is better by {-diff}.")
else:
    print("The computed solutions are equally good.")