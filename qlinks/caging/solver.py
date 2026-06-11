from __future__ import annotations

import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.localization import (
    IPRLocalizationConfig,
    localized_basis_by_many_start_ipr,
)
from qlinks.caging.nullspace import as_dense_array, nullspace_svd
from qlinks.caging.prefilters import extract_subblocks
from qlinks.caging.results import CageState
from qlinks.caging.types import CageSolverConfig


def _apply_matrix(matrix: object, vector: np.ndarray) -> np.ndarray:
    """Apply a dense or sparse matrix to a vector."""
    return matrix @ vector


def _is_hermitian(matrix: np.ndarray, *, tolerance: float) -> bool:
    """Check whether a dense matrix is numerically Hermitian."""
    return bool(np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0.0))


def _vertical_stack(matrix_blocks: list[object]) -> object:
    """Vertically stack dense or sparse matrix blocks."""
    if any(scipy_sparse.issparse(block) for block in matrix_blocks):
        return scipy_sparse.vstack(matrix_blocks, format="csr")

    return np.vstack([as_dense_array(block) for block in matrix_blocks])


def _energy_groups(
    energy_values: np.ndarray,
    *,
    tolerance: float,
) -> list[list[int]]:
    """Group nearly degenerate eigenvalue indices."""
    unused_indices = set(range(len(energy_values)))
    groups: list[list[int]] = []

    while unused_indices:
        reference_index = min(unused_indices)
        reference_energy = energy_values[reference_index]

        group = [
            index
            for index in sorted(unused_indices)
            if abs(energy_values[index] - reference_energy) <= tolerance
        ]

        for index in group:
            unused_indices.remove(index)

        groups.append(group)

    return groups


def _compact_support_from_local_state(
    parent_vertices: np.ndarray,
    local_state: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim zero-amplitude entries from a local state."""
    support_mask = np.abs(local_state) > tolerance

    if not np.any(support_mask):
        return parent_vertices, local_state

    compact_vertices = parent_vertices[support_mask]
    compact_state = local_state[support_mask]

    state_norm = scipy_linalg.norm(compact_state)
    if state_norm > tolerance:
        compact_state = compact_state / state_norm

    return compact_vertices, compact_state.astype(np.complex128, copy=False)


def _build_validated_cage_state(
    hamiltonian: object,
    parent_vertices: np.ndarray,
    local_state_on_parent: np.ndarray,
    energy_value: complex,
    *,
    candidate: CandidateSubgraph,
    dense_internal_matrix: np.ndarray,
    dense_boundary_matrix: np.ndarray,
    config: CageSolverConfig,
    metadata: dict[str, object],
) -> CageState | None:
    """Validate and build a CageState, trimming support if possible."""
    if config.normalize_states:
        state_norm = scipy_linalg.norm(local_state_on_parent)
        if state_norm <= config.tolerance:
            return None
        local_state_on_parent = local_state_on_parent / state_norm

    compact_vertices, compact_local_state = _compact_support_from_local_state(
        parent_vertices,
        local_state_on_parent,
        tolerance=(config.ipr_support_tolerance_factor * config.tolerance),
    )

    # Validate against the parent candidate first.
    boundary_residual = float(scipy_linalg.norm(dense_boundary_matrix @ local_state_on_parent))
    eigen_residual = float(
        scipy_linalg.norm(
            dense_internal_matrix @ local_state_on_parent - energy_value * local_state_on_parent
        )
    )

    full_residual: float | None = None
    if config.validate_full_residual:
        full_state = np.zeros(
            hamiltonian.shape[0],
            dtype=np.complex128,
        )
        full_state[parent_vertices] = local_state_on_parent
        full_residual = float(
            scipy_linalg.norm(_apply_matrix(hamiltonian, full_state) - energy_value * full_state)
        )

    if boundary_residual > config.tolerance:
        return None

    if eigen_residual > config.tolerance:
        return None

    if full_residual is not None and full_residual > config.tolerance:
        return None

    return CageState(
        energy=complex(energy_value),
        local_state=compact_local_state,
        support=compact_vertices,
        boundary_residual=boundary_residual,
        eigen_residual=eigen_residual,
        full_residual=full_residual,
        metadata={
            "candidate_label": candidate.label,
            "candidate_metadata": candidate.metadata,
            **metadata,
        },
    )


def _identity_like_internal_matrix(internal_matrix: object, support_size: int) -> object:
    if scipy_sparse.issparse(internal_matrix):
        return scipy_sparse.identity(
            support_size,
            dtype=np.complex128,
            format="csr",
        )

    return np.eye(support_size, dtype=np.complex128)


def _uniform_self_loop_value(
    self_loop_values: np.ndarray,
    vertices: np.ndarray,
    *,
    tolerance: float,
) -> complex | None:
    selected_values = np.asarray(self_loop_values[vertices], dtype=np.complex128)

    if selected_values.size == 0:
        return None

    reference_value = complex(selected_values[0])

    if not np.allclose(selected_values, reference_value, atol=tolerance, rtol=0.0):
        return None

    return reference_value


def solve_candidate_for_kinetic_targets(
    hamiltonian: object,
    kinetic_matrix: object,
    self_loop_values: np.ndarray,
    candidate: CandidateSubgraph,
    *,
    target_kappas: tuple[complex, ...],
    config: CageSolverConfig | None = None,
) -> list[CageState]:
    """
    Solve cage states with fixed kinetic eigenvalues.

    This is the fast path used by the high-level cage searcher for QDM/QLM
    Type-1 and Type-2 searches.  For candidates with a uniform diagonal
    potential value ``z`` on the support, a cage with kinetic eigenvalue
    ``kappa`` has full Hamiltonian energy ``kappa + z``.  We can therefore
    solve the fixed-kappa linear system directly,

        K_out,S psi = 0,
        (K_S - kappa I) psi = 0,

    instead of first computing the whole invariant boundary subspace and then
    diagonalizing the projected Hamiltonian.
    """
    if config is None:
        config = CageSolverConfig()

    if len(target_kappas) == 0:
        return []

    z_value = _uniform_self_loop_value(
        np.asarray(self_loop_values, dtype=np.complex128),
        candidate.vertices,
        tolerance=config.tolerance,
    )

    if z_value is None:
        return []

    internal_kinetic_matrix, boundary_matrix, _outside_indices = extract_subblocks(
        kinetic_matrix,
        candidate.vertices,
    )

    support_size = candidate.size
    identity_matrix = _identity_like_internal_matrix(internal_kinetic_matrix, support_size)

    dense_kinetic_internal = as_dense_array(internal_kinetic_matrix)
    dense_hamiltonian_internal = dense_kinetic_internal + z_value * np.eye(
        support_size,
        dtype=np.complex128,
    )
    dense_boundary_matrix = as_dense_array(boundary_matrix)

    cage_states: list[CageState] = []

    for target_kappa in target_kappas:
        shifted_matrix = internal_kinetic_matrix - target_kappa * identity_matrix
        combined_matrix = _vertical_stack([boundary_matrix, shifted_matrix])
        local_basis = nullspace_svd(combined_matrix, tolerance=config.tolerance)

        if local_basis.shape[1] == 0:
            continue

        fixed_kappa_subspace_dim = local_basis.shape[1]
        used_ipr = False

        if local_basis.shape[1] > 1 and config.degenerate_basis_strategy == "ipr":
            ipr_config = IPRLocalizationConfig(
                n_restarts=config.ipr_n_restarts,
                max_iter=config.ipr_max_iter,
                step_size=config.ipr_step_size,
                convergence_tolerance=config.ipr_convergence_tolerance,
                candidate_count=config.ipr_candidate_count,
                amplitude_tolerance=(config.ipr_support_tolerance_factor * config.tolerance),
                rank_tolerance=(config.ipr_rank_tolerance_factor * config.tolerance),
                random_seed=config.ipr_random_seed,
            )
            local_basis = localized_basis_by_many_start_ipr(
                local_basis.astype(np.complex128, copy=False),
                config=ipr_config,
            )
            used_ipr = True

        energy_value = complex(target_kappa) + z_value

        for localized_index in range(local_basis.shape[1]):
            local_state = local_basis[:, localized_index]
            cage_state = _build_validated_cage_state(
                hamiltonian,
                candidate.vertices,
                local_state,
                energy_value,
                candidate=candidate,
                dense_internal_matrix=dense_hamiltonian_internal,
                dense_boundary_matrix=dense_boundary_matrix,
                config=config,
                metadata={
                    "fixed_kappa_solver": True,
                    "target_kappa": complex(target_kappa),
                    "self_loop_value": z_value,
                    "invariant_subspace_dim": fixed_kappa_subspace_dim,
                    "degenerate_group_index": 0,
                    "degenerate_group_size": fixed_kappa_subspace_dim,
                    "degenerate_basis_strategy": ("ipr" if used_ipr else "none"),
                },
            )

            if cage_state is not None:
                cage_states.append(cage_state)

    return cage_states


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

    energy_groups = _energy_groups(
        energy_values,
        tolerance=config.tolerance,
    )

    for group_index, state_indices in enumerate(energy_groups):
        representative_energy = complex(energy_values[state_indices[0]])

        local_basis = subspace_basis @ projected_states[:, state_indices]

        used_ipr = False

        if len(state_indices) > 1 and config.degenerate_basis_strategy == "ipr":
            ipr_config = IPRLocalizationConfig(
                n_restarts=config.ipr_n_restarts,
                max_iter=config.ipr_max_iter,
                step_size=config.ipr_step_size,
                convergence_tolerance=config.ipr_convergence_tolerance,
                candidate_count=config.ipr_candidate_count,
                amplitude_tolerance=(config.ipr_support_tolerance_factor * config.tolerance),
                rank_tolerance=(config.ipr_rank_tolerance_factor * config.tolerance),
                random_seed=config.ipr_random_seed,
            )

            local_basis = localized_basis_by_many_start_ipr(
                local_basis.astype(np.complex128, copy=False),
                config=ipr_config,
            )
            used_ipr = True

        for localized_index in range(local_basis.shape[1]):
            local_state = local_basis[:, localized_index]

            cage_state = _build_validated_cage_state(
                hamiltonian,
                candidate.vertices,
                local_state,
                representative_energy,
                candidate=candidate,
                dense_internal_matrix=dense_internal_matrix,
                dense_boundary_matrix=dense_boundary_matrix,
                config=config,
                metadata={
                    "invariant_subspace_dim": subspace_basis.shape[1],
                    "degenerate_group_index": group_index,
                    "degenerate_group_size": len(state_indices),
                    "degenerate_basis_strategy": ("ipr" if used_ipr else "none"),
                },
            )

            if cage_state is not None:
                cage_states.append(cage_state)

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
