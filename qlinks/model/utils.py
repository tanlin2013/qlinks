from __future__ import annotations

from collections import deque
from functools import reduce

import numpy as np
import scipy.sparse as sp
import numpy.typing as npt


def kron(operator_list: list[npt.NDArray | sp.sparray], shift: int = 0):
    cyclic_list = deque(operator_list)
    cyclic_list.rotate(shift)
    return reduce(sp.kron, list(cyclic_list))

def sparse_real_if_close(sp_mat, tol: int | float = 1e-14):
    if not sp.issparse(sp_mat):
        raise ValueError("Input must be a scipy sparse matrix.")
    real_data = np.real_if_close(sp_mat.data, tol=tol)
    return sp_mat.__class__(
        (real_data, sp_mat.indices, sp_mat.indptr), shape=sp_mat.shape
    )
