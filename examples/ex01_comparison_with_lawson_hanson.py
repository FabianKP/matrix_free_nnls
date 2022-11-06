"""
Created by Fabian on 06.11.2022.

Compares the `solve_nnls` function with scipy's implementation of the Lawson-Hanson method for
dense problems.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import nnls
from time import time

from matrix_free_nnls import solve_nnls


# Set to True if you want to rerun the numerical test. This takes a few minutes.
RERUN_COMPUTATIONS = False

import os
print(os.getcwd())

if RERUN_COMPUTATIONS:
    print("Starting computations.")
    np.random.seed(666)
    n_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    times_mfnnls = []
    times_lh = []
    for n in n_list:
        print(f"n = {n}.")
        a = np.random.randn(n, n)
        b = np.random.randn(n)
        # Solve with 'solve_nnls'.
        t0 = time()
        x_mfn, res = solve_nnls(a, b)
        t_mfn = time() - t0
        times_mfnnls.append(t_mfn)
        # Solve with Lawson-Hanson method.
        t0 = time()
        x_lh, rnorm = nnls(a, b)
        t_lh = time() - t0
        times_lh.append(t_lh)
    np.save("times_mfnnls.npy", np.array([n_list, times_mfnnls]))
    np.save("times_lh.npy", np.array([n_list, times_lh]))

# Make plot.
times_mfnnls = np.load("times_mfnnls.npy")
times_lh = np.load("times_lh.npy")

plt.plot(times_mfnnls[0], times_mfnnls[1], "r-", label="matrix_free_nnls.solve_nnls")
plt.plot(times_lh[0], times_lh[1], "b--", label="scipy.optimize.nnls")
plt.xlabel("Dimension")
plt.ylabel("Computation time [s]")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png")
plt.show()


