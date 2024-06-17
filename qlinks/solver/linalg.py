import numpy as np
import scipy as sp


def null_space(mat: sp.sparse.sparray):
    if np.prod(mat.shape) > 2**24:
        u, s, vh = sp.sparse.linalg.svds(mat, k=min(mat.shape) - 1, which="SM")
        tol = np.finfo(mat.dtype).eps * mat.nnz
        null_vecs = vh.compress(s <= tol, axis=0).T
    else:
        null_vecs = sp.linalg.null_space(mat.toarray())
    return null_vecs
