"""Small linear-algebra helpers shared by stability symmetry diagnostics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _subspace_symmetry_representation(
    basis: npt.NDArray[np.complex128],
    permutation: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.complex128], float]:
    if basis.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.complex128), 0.0
    transformed = np.zeros_like(basis)
    transformed[permutation, :] = basis
    representation = basis.conj().T @ transformed
    residual = float(np.linalg.norm(transformed - basis @ representation))
    return np.asarray(representation, dtype=np.complex128), residual
