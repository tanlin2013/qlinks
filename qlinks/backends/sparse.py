from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse


SparseBackendName = Literal["scipy", "cupy", "auto"]


class SparseBackend(Protocol):
    """
    Minimal backend protocol for sparse Hamiltonian construction.
    """

    name: str

    def as_data_array(self, data: list[complex], dtype: npt.DTypeLike) -> Any:
        ...

    def as_index_array(self, indices: list[int] | npt.NDArray[np.integer]) -> Any:
        ...

    def coo_matrix(
        self,
        data: Any,
        rows: Any,
        cols: Any,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> Any:
        ...

    def empty_csr(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> Any:
        ...

    def to_cpu_array(self, array: Any) -> npt.NDArray:
        ...

    def max_abs_data(self, sparse_matrix: Any) -> float:
        ...


@dataclass(frozen=True, slots=True)
class ScipySparseBackend:
    name: str = "scipy"

    def as_data_array(self, data: list[complex], dtype: npt.DTypeLike) -> npt.NDArray:
        return np.asarray(data, dtype=dtype)

    def as_index_array(self, indices: list[int] | npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
        return np.asarray(indices, dtype=np.int64)

    def coo_matrix(
        self,
        data: npt.NDArray,
        rows: npt.NDArray[np.int64],
        cols: npt.NDArray[np.int64],
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> scipy_sparse.csr_array:
        matrix = scipy_sparse.coo_array(
            (data, (rows, cols)),
            shape=shape,
            dtype=dtype,
        ).tocsr()

        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        return matrix

    def empty_csr(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> scipy_sparse.csr_array:
        return scipy_sparse.csr_array(shape, dtype=dtype)

    def to_cpu_array(self, array: Any) -> npt.NDArray:
        return np.asarray(array)

    def max_abs_data(self, sparse_matrix: Any) -> float:
        if sparse_matrix.nnz == 0:
            return 0.0

        data = sparse_matrix.data

        if data.size == 0:
            return 0.0

        return float(np.max(np.abs(data)))


@dataclass(frozen=True, slots=True)
class CupySparseBackend:
    """
    CuPy/CuPyX sparse backend.

    Imported lazily so qlinks does not require CuPy unless backend="cupy" is used.

    Note:
        cupyx.scipy.sparse currently follows the scipy sparse matrix-style API
        more closely than scipy's newer sparse array API. Therefore this backend
        creates cupyx.scipy.sparse.coo_matrix/csr_matrix, not csr_array.
    """

    name: str = "cupy"

    @property
    def cp(self):
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "backend='cupy' requires CuPy. Install the CuPy package matching "
                "your CUDA environment, e.g. cupy-cuda12x."
            ) from exc

        return cp

    @property
    def cupyx_sparse(self):
        try:
            import cupyx.scipy.sparse as cupyx_sparse
        except ImportError as exc:
            raise ImportError(
                "backend='cupy' requires cupyx.scipy.sparse, which is provided by CuPy."
            ) from exc

        return cupyx_sparse

    def as_data_array(self, data: list[complex], dtype: npt.DTypeLike):
        return self.cp.asarray(data, dtype=dtype)

    def as_index_array(self, indices: list[int] | npt.NDArray[np.integer]):
        return self.cp.asarray(indices, dtype=self.cp.int64)

    def coo_matrix(
        self,
        data: Any,
        rows: Any,
        cols: Any,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ):
        matrix = self.cupyx_sparse.coo_matrix(
            (data, (rows, cols)),
            shape=shape,
            dtype=dtype,
        ).tocsr()

        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        return matrix

    def empty_csr(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ):
        return self.cupyx_sparse.csr_matrix(shape, dtype=dtype)

    def to_cpu_array(self, array: Any) -> npt.NDArray:
        return self.cp.asnumpy(array)

    def max_abs_data(self, sparse_matrix: Any) -> float:
        if sparse_matrix.nnz == 0:
            return 0.0

        data = sparse_matrix.data

        if data.size == 0:
            return 0.0

        return float(self.cp.max(self.cp.abs(data)).item())


def get_sparse_backend(backend: SparseBackendName | SparseBackend = "scipy") -> SparseBackend:
    """
    Resolve a sparse backend.

    backend="auto" currently chooses scipy. This keeps behavior deterministic.
    Later, you can change "auto" to use cupy if an input basis lives on GPU.
    """

    if not isinstance(backend, str):
        return backend

    if backend == "scipy":
        return ScipySparseBackend()

    if backend == "cupy":
        return CupySparseBackend()

    if backend == "auto":
        return ScipySparseBackend()

    raise ValueError("backend must be one of 'scipy', 'cupy', or 'auto'.")
