from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg

OpenSystemBackendName = Literal["scipy", "cupy"]


@dataclass(frozen=True, slots=True)
class OpenSystemBackend:
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
