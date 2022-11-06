"""
Created by Fabian on 06.11.2022.
"""

from math import sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Union


def solve_nnls(a: Union[np.array, LinearOperator], b: np.array, with_restart: bool = False, max_iter: int = 1000,
               x0: np.array = None, verbose: bool = False)\
        -> Tuple[np.array, np.array]:
    """
    Solves non-negative least-squares problem
        minimize || A x - b ||_2^2 subject to x >= 0
    using an accelerated projected gradient descent with restart.
    Given an initial guess x_0 and 0 < alpha_0 < 1, we set
        y_0 = x_0,
        s = ||A.T A||,
        theta_1 = Id - A.T A / s,
        theta_2 = A.T b / s.
    Then the algorithm iterates
        x_{k+1} = [theta_1 y_k + theta_2]^+,
        alpha_{k+1} = 0.5 * (sqrt(alpha_k^4 + 4  alpha_k^2) - alpha_k^2),
        beta_{k+1} = alpha_k * (1 - alpha_k) / (alpha_k^2 + alpha_{k+1}),
        y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_{k}).

    Parameters
    ----------
    a :
        The design matrix. It can either be provided as `numpy.ndarray` or as `scipy.sparse.linalg.LinearOperator`,
        in which case only the forward and adjoint action have to be implemented.
    b : shape (M, )
        The regressand. Its dimension M must equal `a.shape[0]`.
    max_iter : int, optional
        The maximum number of iterations.
    x0 : shape (N, ), optional
        Initial guess for the minimizer.
    scale : float, optional
        The value of ||A.T A||. Can speed up convergence if provided.

    Returns
    -------
    x : shape (N, )
        The computed minimizer of the non-negative least-squares problem. A vector of dimension `a.shape[1]`.
    res : shape (M, )
        The residual vector a @ x - b.
    """
    # Check input.
    if not (isinstance(a, np.ndarray) or isinstance(a, LinearOperator)):
        raise ValueError("The design matrix 'a' must given as `np.ndarray` or `scipy.sparse.linalg.LinearOperator`.")
    m, n = a.shape
    if b.shape != (m, ):
        raise ValueError(f"The regressand 'b' must have shape ({m}, ).")
    # Set default initial guess to zero vector.
    if x0 is None:
        x0 = np.zeros(n, )
    if x0.shape != (n, ):
        raise ValueError(f"The initial guess 'x0' must have shape ({n}, ).")
    if not (isinstance(max_iter, int) and max_iter >= 1):
        raise ValueError(f"'max_iter' must be an integer greater or equal 1.")

    # Initialize variables
    y = x0
    x_old = x0
    x = x0
    alpha_old = 0.9
    ata = a.T @ a
    s = np.linalg.norm(ata)
    theta_1 = np.identity(n) - ata / s
    theta_2 = a.T @ b / s
    # Start the iteration.
    for k in range(max_iter):
        if verbose: print(f"Iteration {k + 1}/{max_iter}", end="")
        x = (theta_1 @ y + theta_2).clip(min=0.)
        alpha = 0.5 * (sqrt(alpha_old ** 4 + 4 * alpha_old ** 2) - alpha_old ** 2)
        beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 + alpha)
        y = x + beta * (x - x_old)
        x_old = x
        alpha_old = alpha
        if verbose: print("\r", end="")
    rnorm = np.linalg.norm(a @ x - b)
    return x, rnorm
