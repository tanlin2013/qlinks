from __future__ import annotations

import numpy as np
import scipy.linalg as scipy_linalg

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.nullspace import as_dense_array
from qlinks.caging.prefilters import extract_subblocks
from qlinks.caging.results import CageState
from qlinks.caging.types import CageSolverConfig


def _apply_matrix(matrix: object, vector: np.ndarray) -> np.ndarray:
    """Apply a dense or sparse matrix to a vector."""
    return matrix @ vector


def _is_hermitian(matrix: np.ndarray, *, tolerance: float) -> bool:
    """Check whether a dense matrix is numerically Hermitian."""
    return bool(np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0.0))


def solve_candidate(
    hamiltonian: object,
    candidate: CandidateSubgraph,
    *,
    config: CageSolverConfig | None = None,
) -> list[CageState]:
    """Solve interference-caged eigenstates for one candidate support."""
    if config is None:
        config = CageSolverConfig()

    internal_matrix, boundary_matrix, _outside_indices = extract_subblocks(
        hamiltonian,
        candidate.vertices,
    )

    subspace_basis = invariant_boundary_nullspace(
        internal_matrix,
        boundary_matrix,
        tolerance=config.tolerance,
        max_power=config.max_power,
        stabilization_rounds=config.stabilization_rounds,
    )

    if subspace_basis.shape[1] == 0:
        return []

    dense_internal_matrix = as_dense_array(internal_matrix)
    dense_boundary_matrix = as_dense_array(boundary_matrix)

    projected_matrix = subspace_basis.conj().T @ dense_internal_matrix @ subspace_basis

    if _is_hermitian(projected_matrix, tolerance=config.tolerance):
        energy_values, projected_states = scipy_linalg.eigh(projected_matrix)
    else:
        energy_values, projected_states = scipy_linalg.eig(projected_matrix)

    cage_states: list[CageState] = []

    for state_index, energy_value in enumerate(energy_values):
        local_state = subspace_basis @ projected_states[:, state_index]

        if config.normalize_states:
            state_norm = scipy_linalg.norm(local_state)

            if state_norm <= config.tolerance:
                continue

            local_state = local_state / state_norm

        boundary_residual = float(scipy_linalg.norm(dense_boundary_matrix @ local_state))

        eigen_residual = float(
            scipy_linalg.norm(dense_internal_matrix @ local_state - energy_value * local_state)
        )

        full_residual: float | None = None

        if config.validate_full_residual:
            full_state = np.zeros(
                hamiltonian.shape[0],
                dtype=np.complex128,
            )
            full_state[candidate.vertices] = local_state

            full_residual = float(
                scipy_linalg.norm(
                    _apply_matrix(hamiltonian, full_state) - energy_value * full_state
                )
            )

        if boundary_residual <= config.tolerance and eigen_residual <= config.tolerance:
            cage_states.append(
                CageState(
                    energy=complex(energy_value),
                    local_state=local_state.astype(np.complex128, copy=False),
                    support=candidate.vertices,
                    boundary_residual=boundary_residual,
                    eigen_residual=eigen_residual,
                    full_residual=full_residual,
                    metadata={
                        "candidate_label": candidate.label,
                        "candidate_metadata": candidate.metadata,
                        "invariant_subspace_dim": subspace_basis.shape[1],
                    },
                )
            )

    return cage_states


def solve_candidates(
    hamiltonian: object,
    candidates: list[CandidateSubgraph],
    *,
    config: CageSolverConfig | None = None,
) -> list[CageState]:
    """Solve caged states for many candidate supports."""
    cage_states: list[CageState] = []

    for candidate in candidates:
        cage_states.extend(
            solve_candidate(
                hamiltonian,
                candidate,
                config=config,
            )
        )

    return cage_states
