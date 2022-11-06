"""
Created by Fabian on 06.11.2022.
"""

from math import sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.linalg.interpolative import estimate_spectral_norm
from typing import Tuple, Union

CTOL = 1e-15


def solve_nnls(a: Union[np.array, LinearOperator], b: np.array, with_restart: bool = True, atanorm: float = None,
               max_iter: int = 3000, x0: np.array = None, gtol: float = 1e-10, alpha0: float = 0.9,
               verbose: bool = False) -> Tuple[np.array, np.array]:
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
    a : shape (M, N)
        The design matrix. It can either be provided as `numpy.ndarray` or as `scipy.sparse.linalg.LinearOperator`,
        in which case only the forward and adjoint action have to be implemented.
    b : shape (M, )
        The regressand. Its dimension M must equal `a.shape[0]`.
    with_restart: bool
        If True, restarts the iteration if the objective increases.
    atanorm : float
        Guess for ||A.T A||. A good guess can significantly speed up convergence.
        If it is not provided, it is estimated using `scipy.linalg.interpolative.estimate_spectral_norm`.
    max_iter : int
        The maximum number of iterations.
    x0 : shape (N, ), optional
        Initial guess for the minimizer.
    gtol : float
        The tolerance for the projected gradient. The iteration is stopped once the l1-norm of the projected gradient
        is less or equal 'gtol * M * N'.
    alpha0 : float
        The starting value for alpha. Must be between 0 and 1.
    verbose : bool
        If True, information is printed to the console during the iteration.

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
    # If A is given as a dense matrix, we create a LinearOperator.
    if isinstance(a, np.ndarray):
        a_op = aslinearoperator(a)
    else:
        a_op = a
    if atanorm is None:
        # If 'atanorm' is not provided, we estimate it using the power iteration.
        ata_matvec = lambda x : a_op.rmatvec(a_op.matvec(x))
        ata_op = LinearOperator(shape=(n, n), matvec=ata_matvec, rmatvec=ata_matvec)
        atanorm = estimate_spectral_norm(ata_op)

    # Select APGD with or without restart.
    if with_restart:
        x, res = apgd_with_restart(a=a_op, b=b, max_iter=max_iter, x0=x0, atanorm=atanorm, gtol=gtol, alpha0=alpha0,
                                   verbose=verbose)
    else:
        x, res = apgd(a=a_op, b=b, max_iter=max_iter, x0=x0, atanorm=atanorm, gtol=gtol, alpha0=alpha0, verbose=verbose)

    return x, res


def apgd(a: LinearOperator, b: np.array, max_iter: int, x0: np.array, atanorm: float, gtol: float,
         alpha0: float, verbose: bool) -> Tuple[np.array, np.array]:
    # Initialize variables
    y = x0
    x_old = x0
    x = x0
    alpha_old = alpha0
    s = atanorm
    theta_1 = LinearOperator(shape=(x0.size, x0.size), matvec=(lambda x : x - a.rmatvec(a.matvec(x)) / s))
    theta_2 = a.rmatvec(b) / s
    # Start the iteration.
    for k in range(max_iter):
        if verbose: print("\r", end="")
        if verbose: print(f"Iteration {k + 1}/{max_iter}.   ", end="")
        x = (theta_1.matvec(y) + theta_2).clip(min=0.)
        if _is_converged(a, x, atanorm * theta_2, gtol):
            if verbose: print(f"Convergence achieved, projected gradient below gtol={gtol}.")
            break
        alpha = 0.5 * (sqrt(alpha_old ** 4 + 4 * alpha_old ** 2) - alpha_old ** 2)
        beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 + alpha)
        y = x + beta * (x - x_old)
        x_old = x
        alpha_old = alpha
    res = a @ x - b
    return x, res


def apgd_with_restart(a: LinearOperator, b: np.array, max_iter: int, x0: np.array, atanorm: float, gtol: float,
                      alpha0: float, verbose: bool) -> Tuple[np.array, np.array]:
    # Initialize variables
    y = x0
    x_old = x0
    x = x0
    res = a.matvec(x) - b
    rnorm_old = np.linalg.norm(res)
    alpha_old = alpha0
    s = atanorm
    theta_1 = LinearOperator(shape=(x0.size, x0.size), matvec=(lambda x: x - a.rmatvec(a.matvec(x)) / s))
    theta_2 = a.rmatvec(b) / s
    for k in range(max_iter):
        if verbose: print("\r", end="")
        if verbose: print(f"Iteration {k + 1}/{max_iter}. Objective: {rnorm_old}.   ", end="")
        x = (theta_1.matvec(y) + theta_2).clip(min=0.)
        res = a.matvec(x) - b
        rnorm = np.linalg.norm(res)
        if not np.isfinite(rnorm):
            raise RuntimeError("Infinite objective. Try to re-run with increased 'scale'.")
        # Check convergence.
        if _is_converged(a, x, atanorm * theta_2, gtol):
            if verbose: print(f"Convergence achieved, projected gradient below gtol={gtol}.")
            break
        # If error increases, we need to restart.
        if rnorm > rnorm_old:
            # restart
            x = (theta_1.matvec(x_old) + theta_2).clip(min=0.)
            y = x
            alpha_old = alpha0
        else:
            alpha = 0.5 * (sqrt(alpha_old ** 4 + 4 * alpha_old ** 2) - alpha_old ** 2)
            beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 + alpha)
            y = x + beta * (x - x_old)
            x_old = x
            alpha_old = alpha
        rnorm_old = rnorm
    return x, res

def _is_converged(a, x, atb, gtol):
    """
    Convergence is achieved if the projected gradient is zero.
    """
    m, n = a.shape
    # Compute gradient.
    g = - (a.rmatvec(a.matvec(x)) - atb)
    # Project gradient
    g[x <= CTOL] = g[x <= CTOL].clip(min=0.)
    # Check if ||projected_gradient||_1 <= gtol
    gnorm = np.linalg.norm(g, ord=1)
    return (gnorm <= gtol * m * n)

