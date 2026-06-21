from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CageState:
    """Validated interference-caged eigenstate on compact support.

    Attributes:
        energy: Eigenvalue associated with the cage state.
        local_state: State amplitudes restricted to ``support``.
        support: Global basis indices carrying nonzero amplitudes.
        boundary_residual: Norm of leakage outside the candidate support.
        eigen_residual: Eigen residual on the internal support.
        full_residual: Optional residual against the full Hamiltonian.
        metadata: Optional solver/search metadata.
    """

    energy: complex
    local_state: NDArray[np.complex128]
    support: NDArray[np.int_]
    boundary_residual: float
    eigen_residual: float
    full_residual: float | None = None
    metadata: dict[str, object] | None = None

    @property
    def support_size(self) -> int:
        """Number of vertices supporting the local state."""
        return int(self.support.size)


def cage_state_to_full_vector(
    cage_state: CageState,
    hilbert_size: int,
) -> np.ndarray:
    """Lift a compact cage state to the full Hilbert-space vector."""
    full_vector = np.zeros(hilbert_size, dtype=np.complex128)
    full_vector[cage_state.support] = cage_state.local_state

    return full_vector


def cage_states_to_full_matrix(
    cage_states: list[CageState],
    hilbert_size: int,
) -> np.ndarray:
    """Lift many compact cage states to dense full-Hilbert vectors.

    Args:
        cage_states: Compact cage states.
        hilbert_size: Full Hilbert-space dimension.

    Returns:
        Array with shape ``(n_states, hilbert_size)``.
    """
    full_matrix = np.zeros(
        (len(cage_states), hilbert_size),
        dtype=np.complex128,
    )

    for cage_index, cage_state in enumerate(cage_states):
        full_matrix[cage_index, cage_state.support] = cage_state.local_state

    return full_matrix
