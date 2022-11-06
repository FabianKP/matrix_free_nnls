"""
Created by Fabian on 06.11.2022.

Tests `solve_nnls` in case the user does not provide ||A.T A||.
Surprisingly, the solver converges much faster if we use the estimate of ||A.T A||.
Since the power method tends to underestimate the
"""

import numpy as np
from scipy.linalg.interpolative import estimate_spectral_norm
from time import time

from matrix_free_nnls import solve_nnls


#np.random.seed(65)
m = 3000
n = 3000

a = np.random.randn(m, n)
b = np.random.randn(m)

atanorm = np.linalg.norm(a.T @ a)
atanorm_estimate = estimate_spectral_norm(a.T @ a)

print(f"||A.T A|| = {atanorm}")
print(f"Estimate = {atanorm_estimate}")

# Solve using atanorm.
t0 = time()
x_scale, res_scale = solve_nnls(a, b, atanorm=atanorm, verbose=True)
t_scale = time() - t0
rnorm_scale = np.linalg.norm(res_scale)
# Solve without atanorm.
t0 = time()
x_noscale, res_noscale = solve_nnls(a, b, verbose=True)
t_noscale = time() - t0
rnorm_noscale = np.linalg.norm(res_noscale)

print("Using exact atanorm:")
print(f"   Computation time: {t_scale}")
print(f"   Objective : {rnorm_scale}")
print("Using estimated atanorm:")
print(f"   Computation time: {t_noscale}")
print(f"   Objective : {rnorm_noscale}")

