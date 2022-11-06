"""
Created by Fabian on 06.11.2022.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Union


def solve_nnls(a: Union[np.array, LinearOperator], b: np.array, max_iter: int) -> Tuple[np.array, np.array]:
    """
    Solves non-negative least-squares problem
        minimize || A x - b ||_2^2 subject to x >= 0
    using an accelerated projected gradient descent with restart.

    Parameters
    ----------
    a :
        The design matrix. It can either be provided as `numpy.ndarray` or as `scipy.sparse.linalg.LinearOperator`,
        in which case only the forward and adjoint action have to be implemented.
    b : shape (M, )
        The regressand. Its dimension M must equal `a.shape[0]`.
    max_iter :
        The maximum number of iterations.

    Returns
    -------
    x : shape (N, )
        The computed minimizer of the non-negative least-squares problem. A vector of dimension `a.shape[1]`.
    res : shape (M, )
        The residual vector a @ x - b.
    """
    raise NotImplementedError
