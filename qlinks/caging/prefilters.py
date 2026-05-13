from __future__ import annotations

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray

from qlinks.caging.nullspace import nullspace_svd


def extract_subblocks(
    hamiltonian: object,
    support_indices: NDArray[np.int_],
) -> tuple[object, object, NDArray[np.int_]]:
    """
    Extract the internal and boundary blocks for a candidate support.

    Returns
    -------
    internal_matrix:
        ``H[support, support]``.

    boundary_matrix:
        ``H[outside, support]``.

    outside_indices:
        Indices outside the support.
    """
    support_indices = np.asarray(support_indices, dtype=np.int64)

    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square.")

    hilbert_size = hamiltonian.shape[0]

    if support_indices.ndim != 1:
        raise ValueError("support_indices must be a 1D array.")

    if np.any(support_indices < 0) or np.any(support_indices >= hilbert_size):
        raise ValueError("support_indices contains out-of-range indices.")

    outside_mask = np.ones(hilbert_size, dtype=bool)
    outside_mask[support_indices] = False
    outside_indices = np.nonzero(outside_mask)[0]

    internal_matrix = hamiltonian[support_indices, :][:, support_indices]
    boundary_matrix = hamiltonian[outside_indices, :][:, support_indices]

    return internal_matrix, boundary_matrix, outside_indices


def diagonal_values(
    hamiltonian: object,
    support_indices: NDArray[np.int_],
) -> NDArray[np.complex128]:
    """Return diagonal values of ``hamiltonian`` on ``support_indices``."""
    if scipy_sparse.issparse(hamiltonian):
        full_diagonal = hamiltonian.diagonal()
    else:
        full_diagonal = np.diag(hamiltonian)

    return np.asarray(full_diagonal[support_indices], dtype=np.complex128)


def has_uniform_diagonal(
    hamiltonian: object,
    support_indices: NDArray[np.int_],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check whether a candidate support has uniform diagonal values."""
    local_diagonal = diagonal_values(hamiltonian, support_indices)

    if local_diagonal.size == 0:
        return False

    return bool(
        np.allclose(
            local_diagonal,
            local_diagonal[0],
            atol=tolerance,
            rtol=0.0,
        )
    )


def boundary_nullity(
    boundary_matrix: object,
    *,
    tolerance: float = 1e-10,
) -> int:
    """Return the nullity of the boundary-leakage matrix."""
    nullspace_basis = nullspace_svd(boundary_matrix, tolerance=tolerance)
    return int(nullspace_basis.shape[1])


def passes_basic_prefilters(
    hamiltonian: object,
    support_indices: NDArray[np.int_],
    *,
    tolerance: float = 1e-10,
    min_size: int = 2,
    max_size: int | None = None,
    require_uniform_diagonal: bool = False,
) -> bool:
    """Run basic candidate prefilters."""
    support_indices = np.asarray(support_indices, dtype=np.int64)

    if support_indices.size < min_size:
        return False

    if max_size is not None and support_indices.size > max_size:
        return False

    if require_uniform_diagonal:
        if not has_uniform_diagonal(
            hamiltonian,
            support_indices,
            tolerance=tolerance,
        ):
            return False

    _internal_matrix, boundary_matrix, _outside_indices = extract_subblocks(
        hamiltonian,
        support_indices,
    )

    if boundary_nullity(boundary_matrix, tolerance=tolerance) == 0:
        return False

    return True
