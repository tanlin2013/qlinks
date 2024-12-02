from __future__ import annotations

from collections import deque
from functools import reduce

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


def kron(operator_list: list[npt.NDArray | sp.spmatrix], shift: int = 0) -> sp.spmatrix:
    """Calculate the Kronecker product of a list of operators with cyclic permutation.

    Args:
        operator_list: A list of operators.
        shift: The number of cyclic shifts.

    Returns:
        The Kronecker product of the operators.
    """
    cyclic_list = deque(operator_list)
    cyclic_list.rotate(shift)
    return reduce(sp.kron, list(cyclic_list))


def sparse_real_if_close(sp_mat: sp.sparray | sp.spmatrix, tol: int | float = 1e-14) -> sp.spmatrix:
    """Return a real matrix if the imaginary part is close to zero.

    Args:
        sp_mat: A sparse matrix.
        tol: The tolerance for the imaginary part.

    Returns:
        A real matrix if the imaginary part is close to zero.
    """
    if not sp.issparse(sp_mat):
        raise ValueError("Input must be a scipy sparse matrix.")
    real_data = np.real_if_close(sp_mat.data, tol=tol)
    return sp_mat.__class__((real_data, sp_mat.indices, sp_mat.indptr), shape=sp_mat.shape)
