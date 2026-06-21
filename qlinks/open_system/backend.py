from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg

OpenSystemBackendName = Literal["scipy", "cupy"]


@dataclass(frozen=True, slots=True)
class OpenSystemBackend:
    """Array/sparse backend bundle for open-system solvers.

    Attributes:
        name: Backend name.
        array_module: Dense array module, such as NumPy or CuPy.
        sparse_module: Sparse matrix module.
        sparse_linalg_module: Sparse linear-algebra module, if available.
        supports_expm_multiply: Whether the backend can run Krylov
            ``expm_multiply`` evolution.
    """

    name: OpenSystemBackendName
    array_module: Any
    sparse_module: Any
    sparse_linalg_module: Any | None
    supports_expm_multiply: bool

    def asarray(self, value: Any, *, dtype=np.complex128):
        return self.array_module.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.name == "scipy":
            return np.asarray(value)
        return self.array_module.asnumpy(value)

    def sparse_identity(self, dim: int, *, format: str = "csr", dtype=np.complex128):
        return self.sparse_module.identity(dim, format=format, dtype=dtype)

    def sparse_kron(self, left: Any, right: Any, *, format: str = "csr"):
        return self.sparse_module.kron(left, right, format=format)

    def norm(self, value: Any) -> float:
        return float(self.array_module.linalg.norm(value))


def get_open_system_backend(
    backend: OpenSystemBackendName | OpenSystemBackend,
) -> OpenSystemBackend:
    """Resolve an open-system backend name or backend object.

    Args:
        backend: ``"scipy"``, ``"cupy"``, or an existing backend object.

    Returns:
        Backend object used by open-system operators and solvers.

    Raises:
        ImportError: If ``backend="cupy"`` is requested but CuPy is not
            installed.
        ValueError: If the backend name is unknown.
    """
    if isinstance(backend, OpenSystemBackend):
        return backend

    if backend == "scipy":
        return OpenSystemBackend(
            name="scipy",
            array_module=np,
            sparse_module=scipy_sparse,
            sparse_linalg_module=scipy_sparse_linalg,
            supports_expm_multiply=True,
        )

    if backend == "cupy":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse
            import cupyx.scipy.sparse.linalg as cupy_sparse_linalg
        except ImportError as exc:
            raise ImportError("backend='cupy' requires cupy and cupyx.scipy.sparse.") from exc

        return OpenSystemBackend(
            name="cupy",
            array_module=cp,
            sparse_module=cupy_sparse,
            sparse_linalg_module=cupy_sparse_linalg,
            supports_expm_multiply=hasattr(cupy_sparse_linalg, "expm_multiply"),
        )

    raise ValueError(f"Unsupported open-system backend: {backend!r}")


def as_backend_sparse_matrix(
    matrix: Any,
    *,
    backend: OpenSystemBackend,
    format: str = "csr",
    dtype=np.complex128,
):
    """Convert a dense or sparse matrix to a backend sparse matrix.

    Args:
        matrix: Input matrix in dense, SciPy sparse, or backend sparse form.
        backend: Target backend.
        format: Sparse format requested from the backend.
        dtype: Complex dtype for the converted matrix.

    Returns:
        Sparse matrix on the requested backend.
    """
    if backend.name == "scipy":
        if scipy_sparse.issparse(matrix):
            return matrix.asformat(format).astype(dtype)
        if hasattr(matrix, "asformat"):
            return matrix.asformat(format).astype(dtype)
        if hasattr(matrix, "tocsr"):
            return matrix.tocsr().asformat(format).astype(dtype)
        return scipy_sparse.csr_array(matrix, dtype=dtype).asformat(format)

    # cupy path
    if hasattr(matrix, "get"):
        return backend.sparse_module.csr_matrix(matrix, dtype=dtype).asformat(format)

    if scipy_sparse.issparse(matrix):
        return backend.sparse_module.csr_matrix(matrix.astype(dtype)).asformat(format)

    if hasattr(matrix, "tocsr"):
        sparse_matrix = matrix.tocsr()
        return backend.sparse_module.csr_matrix(sparse_matrix.astype(dtype)).asformat(format)

    return backend.sparse_module.csr_matrix(
        backend.asarray(matrix, dtype=dtype),
    ).asformat(format)


def as_backend_dense_array(
    matrix: Any,
    *,
    backend: OpenSystemBackend,
    dtype=np.complex128,
):
    """Convert a dense or sparse matrix to a backend dense array.

    Args:
        matrix: Input matrix in dense, SciPy sparse, or backend sparse form.
        backend: Target backend.
        dtype: Complex dtype for the converted array.

    Returns:
        Dense array on the requested backend.
    """
    if backend.name == "scipy":
        if scipy_sparse.issparse(matrix):
            return matrix.toarray().astype(dtype)
        if hasattr(matrix, "toarray"):
            return matrix.toarray().astype(dtype)
        if hasattr(matrix, "tocsr"):
            return matrix.tocsr().toarray().astype(dtype)
        return np.asarray(matrix, dtype=dtype)

    if scipy_sparse.issparse(matrix):
        return backend.array_module.asarray(matrix.toarray(), dtype=dtype)

    if hasattr(matrix, "toarray"):
        return backend.array_module.asarray(matrix.toarray(), dtype=dtype)

    if hasattr(matrix, "tocsr"):
        return backend.array_module.asarray(matrix.tocsr().toarray(), dtype=dtype)

    return backend.asarray(matrix, dtype=dtype)
