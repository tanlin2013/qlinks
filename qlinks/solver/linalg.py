from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy as sp
import numpy.typing as npt


def null_space(mat: sp.sparse.sparray, k: Optional[int] = None) -> npt.NDArray[np.float64]:
    if np.prod(mat.shape) > 2**24:
        k = min(mat.shape) - 1 if k is None else min(k, min(mat.shape) - 1)
        u, s, vh = sp.sparse.linalg.svds(mat, k=k, which="SM")
        tol = np.finfo(mat.dtype).eps * mat.nnz
        null_vecs = vh.compress(s <= tol, axis=0).T
    else:
        null_vecs = sp.linalg.null_space(mat.toarray())
    return null_vecs


def eigh(
    mat: sp.sparse.sparray, k: Optional[int] = None, sigma: Optional[float] = None, **kwargs
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if np.prod(mat.shape) > 2**24 or k is not None:
        k = 6 if k is None else k
        return sp.sparse.linalg.eigsh(mat, k, sigma=sigma, **kwargs)
    else:
        return sp.linalg.eigh(mat.toarray())
