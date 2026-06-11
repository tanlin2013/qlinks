from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg


@dataclass
class IndependentColumnSelector:
    """Incrementally select linearly independent complex vectors.

    The selector keeps an orthonormal basis for the span of accepted vectors,
    using modified Gram-Schmidt.  This avoids repeatedly rebuilding dense
    matrices and calling ``matrix_rank`` when rank-selecting many candidate
    states.
    """

    tolerance: float
    columns: list[npt.NDArray[np.complex128]] = field(default_factory=list)
    _orthonormal_columns: list[npt.NDArray[np.complex128]] = field(default_factory=list)

    def accepts(self, candidate: npt.ArrayLike) -> bool:
        """Return whether ``candidate`` increases the selected span."""
        vector = _normalized_vector(candidate, tolerance=self.tolerance)
        residual = vector.copy()

        for basis_vector in self._orthonormal_columns:
            residual = residual - basis_vector * np.vdot(basis_vector, residual)

        residual_norm = float(scipy_linalg.norm(residual))
        return bool(residual_norm > self.tolerance)

    def add(self, candidate: npt.ArrayLike) -> bool:
        """Add ``candidate`` if it is independent and return whether it was added."""
        vector = _normalized_vector(candidate, tolerance=self.tolerance)
        residual = vector.copy()

        for basis_vector in self._orthonormal_columns:
            residual = residual - basis_vector * np.vdot(basis_vector, residual)

        residual_norm = float(scipy_linalg.norm(residual))
        if residual_norm <= self.tolerance:
            return False

        self.columns.append(vector)
        self._orthonormal_columns.append(
            np.asarray(residual / residual_norm, dtype=np.complex128),
        )
        return True

    @property
    def rank(self) -> int:
        """Return the number of accepted independent columns."""
        return len(self.columns)


def _normalized_vector(
    vector: npt.ArrayLike,
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    array = np.asarray(vector, dtype=np.complex128)

    if array.ndim != 1:
        raise ValueError("candidate vector must be one-dimensional.")

    norm = float(scipy_linalg.norm(array))
    if norm <= tolerance:
        return np.zeros(array.shape, dtype=np.complex128)

    return np.asarray(array / norm, dtype=np.complex128)
