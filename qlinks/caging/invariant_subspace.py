from __future__ import annotations

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray

from qlinks.caging.nullspace import nullspace_svd


def _vertical_stack(matrix_blocks: list[object]) -> object:
    """Vertically stack dense or sparse matrix blocks."""
    if any(scipy_sparse.issparse(matrix_block) for matrix_block in matrix_blocks):
        return scipy_sparse.vstack(matrix_blocks, format="csr")

    return np.vstack(matrix_blocks)


def invariant_boundary_nullspace(
    internal_matrix: object,
    boundary_matrix: object,
    *,
    tolerance: float = 1e-10,
    max_power: int | None = None,
    stabilization_rounds: int = 1,
) -> NDArray[np.complex128]:
    """
    Compute the stabilized subspace

        ker(C) >= ker([C; C A]) >= ker([C; C A; C A^2]) >= ...

    where ``internal_matrix`` is ``A`` and ``boundary_matrix`` is ``C``.

    The returned array has shape ``(support_size, subspace_dimension)``.
    Its columns form an orthonormal basis of the stabilized subspace.
    """
    support_size = internal_matrix.shape[0]

    if internal_matrix.shape != (support_size, support_size):
        raise ValueError("internal_matrix must be square.")

    if boundary_matrix.shape[1] != support_size:
        raise ValueError(
            "boundary_matrix must have the same number of columns as " "internal_matrix."
        )

    if stabilization_rounds < 1:
        raise ValueError("stabilization_rounds must be at least 1.")

    if max_power is None:
        max_power = support_size

    leakage_blocks: list[object] = []
    current_leakage_block = boundary_matrix

    previous_nullity: int | None = None
    stable_round_count = 0
    subspace_basis = np.zeros((support_size, 0), dtype=np.complex128)

    for power_index in range(max_power + 1):
        leakage_blocks.append(current_leakage_block)
        stacked_leakage_matrix = _vertical_stack(leakage_blocks)

        subspace_basis = nullspace_svd(
            stacked_leakage_matrix,
            tolerance=tolerance,
        )
        current_nullity = subspace_basis.shape[1]

        if current_nullity == 0:
            return subspace_basis

        if previous_nullity is not None and current_nullity == previous_nullity:
            stable_round_count += 1
            if stable_round_count >= stabilization_rounds:
                return subspace_basis
        else:
            stable_round_count = 0

        previous_nullity = current_nullity

        if power_index < max_power:
            current_leakage_block = current_leakage_block @ internal_matrix

    return subspace_basis
