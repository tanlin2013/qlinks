from __future__ import annotations

import time

import numpy as np
import scipy.linalg as scipy_linalg

from qlinks.caging.candidate import (
    BOUNDARY_OVERLAP_MATRIX_METADATA_KEY,
    INTERNAL_KINETIC_MATRIX_METADATA_KEY,
    CandidateSubgraph,
)
from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.localization import (
    IPRLocalizationConfig,
    localized_basis_by_many_start_ipr,
)
from qlinks.caging.nullspace import as_dense_array, nullspace_from_gram
from qlinks.caging.prefilters import extract_subblocks
from qlinks.caging.results import CageState
from qlinks.caging.types import CageSolverConfig


def _add_timing(
    timing_collector: dict[str, float] | None,
    stage_name: str,
    elapsed_seconds: float,
) -> None:
    if timing_collector is None:
        return

    timing_collector[stage_name] = timing_collector.get(stage_name, 0.0) + elapsed_seconds


def _time_solver_stage(
    timing_collector: dict[str, float] | None,
    stage_name: str,
    func,
):
    start = time.perf_counter()
    result = func()
    _add_timing(timing_collector, stage_name, time.perf_counter() - start)
    return result


def _full_residual_from_column_block(
    hamiltonian_column_block: object,
    parent_vertices: np.ndarray,
    local_state: np.ndarray,
    energy_value: complex,
) -> float:
    """Return ||H[:, S] psi - E psi_S|| without forming a full state."""
    residual = np.asarray(
        hamiltonian_column_block @ local_state,
        dtype=np.complex128,
    ).reshape(-1)
    residual[parent_vertices] -= complex(energy_value) * local_state

    return float(scipy_linalg.norm(residual))


def _boundary_residual_from_column_block(
    operator_column_block: object,
    parent_vertices: np.ndarray,
    local_state: np.ndarray,
    dense_internal_matrix: np.ndarray,
) -> float:
    """Return outside-support residual from a full column block.

    This computes ``||O[outside, S] psi||`` from ``O[:, S]`` without forming
    an explicit outside-complement matrix.  It is numerically more reliable
    than recovering the residual from ``B^† B`` because it avoids taking the
    square root of small Gram-matrix roundoff errors.
    """
    residual = np.asarray(
        operator_column_block @ local_state,
        dtype=np.complex128,
    ).reshape(-1)
    residual[parent_vertices] -= dense_internal_matrix @ local_state

    return float(scipy_linalg.norm(residual))


def _boundary_residual_from_overlap(
    boundary_overlap_matrix: np.ndarray,
    local_state: np.ndarray,
) -> float:
    """Return ||B psi|| from the boundary-overlap matrix B^† B."""
    residual_squared = complex(
        np.vdot(
            local_state,
            boundary_overlap_matrix @ local_state,
        )
    )

    # Numerical roundoff can leave a tiny imaginary or negative-real part for a
    # positive-semidefinite Gram expectation value.
    residual_squared_real = max(float(np.real(residual_squared)), 0.0)
    return float(np.sqrt(residual_squared_real))


def _candidate_kinetic_blocks_for_fixed_kappa(
    kinetic_matrix: object,
    candidate: CandidateSubgraph,
) -> tuple[object, np.ndarray]:
    """Return K[S,S] and the dense boundary overlap K[out,S]^† K[out,S]."""
    cached_internal_matrix = candidate.metadata.get(INTERNAL_KINETIC_MATRIX_METADATA_KEY)
    if cached_internal_matrix is None:
        internal_matrix = kinetic_matrix[candidate.vertices, :][:, candidate.vertices]
    else:
        internal_matrix = cached_internal_matrix

    cached_boundary_overlap = candidate.metadata.get(BOUNDARY_OVERLAP_MATRIX_METADATA_KEY)
    if cached_boundary_overlap is not None:
        return internal_matrix, as_dense_array(cached_boundary_overlap)

    column_block = kinetic_matrix[:, candidate.vertices]

    # Avoid constructing the explicit outside-complement block. Since
    # K[:,S]^†K[:,S] = K[S,S]^†K[S,S] + K[out,S]^†K[out,S], the boundary
    # overlap can be obtained by subtracting the internal contribution.
    full_overlap = column_block.conj().T @ column_block
    internal_overlap = internal_matrix.conj().T @ internal_matrix
    boundary_overlap = as_dense_array(full_overlap - internal_overlap)

    return internal_matrix, boundary_overlap


def _is_hermitian(matrix: np.ndarray, *, tolerance: float) -> bool:
    """Check whether a dense matrix is numerically Hermitian."""
    return bool(np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0.0))


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
    dense_boundary_matrix: np.ndarray | None,
    config: CageSolverConfig,
    metadata: dict[str, object],
    dense_boundary_overlap_matrix: np.ndarray | None = None,
    boundary_column_block: object | None = None,
    hamiltonian_column_block: object | None = None,
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
    if dense_boundary_matrix is not None:
        boundary_residual = float(scipy_linalg.norm(dense_boundary_matrix @ local_state_on_parent))
    elif boundary_column_block is not None:
        boundary_residual = _boundary_residual_from_column_block(
            boundary_column_block,
            parent_vertices,
            local_state_on_parent,
            dense_internal_matrix,
        )
    elif dense_boundary_overlap_matrix is not None:
        boundary_residual = _boundary_residual_from_overlap(
            dense_boundary_overlap_matrix,
            local_state_on_parent,
        )
    else:
        raise ValueError(
            "Either dense_boundary_matrix, boundary_column_block, or "
            "dense_boundary_overlap_matrix is required."
        )

    eigen_residual = float(
        scipy_linalg.norm(
            dense_internal_matrix @ local_state_on_parent - energy_value * local_state_on_parent
        )
    )

    full_residual: float | None = None
    if config.validate_full_residual:
        if hamiltonian_column_block is None:
            hamiltonian_column_block = hamiltonian[:, parent_vertices]

        full_residual = _full_residual_from_column_block(
            hamiltonian_column_block,
            parent_vertices,
            local_state_on_parent,
            energy_value,
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

    internal_kinetic_matrix, dense_boundary_overlap = _time_solver_stage(
        config.timing_collector,
        "solver.candidate_blocks",
        lambda: _candidate_kinetic_blocks_for_fixed_kappa(
            kinetic_matrix,
            candidate,
        ),
    )

    support_size = candidate.size
    dense_identity_matrix = np.eye(support_size, dtype=np.complex128)

    dense_kinetic_internal = as_dense_array(internal_kinetic_matrix)
    dense_hamiltonian_internal = dense_kinetic_internal + z_value * dense_identity_matrix
    hamiltonian_column_block = hamiltonian[:, candidate.vertices]

    cage_states: list[CageState] = []

    for target_kappa in target_kappas:
        shifted_matrix = dense_kinetic_internal - target_kappa * dense_identity_matrix
        gram_matrix = dense_boundary_overlap + shifted_matrix.conj().T @ shifted_matrix
        local_basis = _time_solver_stage(
            config.timing_collector,
            "solver.fixed_kappa_nullspace",
            lambda gram_matrix=gram_matrix: nullspace_from_gram(
                gram_matrix,
                tolerance=config.tolerance,
            ),
        )

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
                rank_completion_patience=config.ipr_rank_completion_patience,
                batch_size=config.ipr_batch_size,
                amplitude_tolerance=(config.ipr_support_tolerance_factor * config.tolerance),
                rank_tolerance=(config.ipr_rank_tolerance_factor * config.tolerance),
                random_seed=config.ipr_random_seed,
            )
            local_basis = _time_solver_stage(
                config.timing_collector,
                "solver.ipr_localization",
                lambda local_basis=local_basis, ipr_config=ipr_config: (
                    localized_basis_by_many_start_ipr(
                        local_basis.astype(np.complex128, copy=False),
                        config=ipr_config,
                    )
                ),
            )
            used_ipr = True

        energy_value = complex(target_kappa) + z_value

        for localized_index in range(local_basis.shape[1]):
            local_state = local_basis[:, localized_index]
            # fmt: off
            cage_state = _time_solver_stage(
                config.timing_collector,
                "solver.validation",
                lambda local_state=local_state,
                energy_value=energy_value,
                target_kappa=target_kappa,
                fixed_kappa_subspace_dim=fixed_kappa_subspace_dim,
                used_ipr=used_ipr: _build_validated_cage_state(
                    hamiltonian,
                    candidate.vertices,
                    local_state,
                    energy_value,
                    candidate=candidate,
                    dense_internal_matrix=dense_hamiltonian_internal,
                    dense_boundary_matrix=None,
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
                    dense_boundary_overlap_matrix=dense_boundary_overlap,
                    boundary_column_block=hamiltonian_column_block,
                    hamiltonian_column_block=hamiltonian_column_block,
                ),
            )
            # fmt: on

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

    internal_matrix, boundary_matrix, _outside_indices = _time_solver_stage(
        config.timing_collector,
        "solver.extract_subblocks",
        lambda: extract_subblocks(
            hamiltonian,
            candidate.vertices,
        ),
    )

    subspace_basis = _time_solver_stage(
        config.timing_collector,
        "solver.invariant_boundary_nullspace",
        lambda: invariant_boundary_nullspace(
            internal_matrix,
            boundary_matrix,
            tolerance=config.tolerance,
            max_power=config.max_power,
            stabilization_rounds=config.stabilization_rounds,
        ),
    )

    if subspace_basis.shape[1] == 0:
        return []

    dense_internal_matrix = as_dense_array(internal_matrix)
    dense_boundary_matrix = as_dense_array(boundary_matrix)
    hamiltonian_column_block = (
        hamiltonian[:, candidate.vertices] if config.validate_full_residual else None
    )

    projected_matrix = subspace_basis.conj().T @ dense_internal_matrix @ subspace_basis

    if _is_hermitian(projected_matrix, tolerance=config.tolerance):
        energy_values, projected_states = _time_solver_stage(
            config.timing_collector,
            "solver.projected_eigensolve",
            lambda: scipy_linalg.eigh(projected_matrix),
        )
    else:
        energy_values, projected_states = _time_solver_stage(
            config.timing_collector,
            "solver.projected_eigensolve",
            lambda: scipy_linalg.eig(projected_matrix),
        )

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
                rank_completion_patience=config.ipr_rank_completion_patience,
                batch_size=config.ipr_batch_size,
                amplitude_tolerance=(config.ipr_support_tolerance_factor * config.tolerance),
                rank_tolerance=(config.ipr_rank_tolerance_factor * config.tolerance),
                random_seed=config.ipr_random_seed,
            )

            # fmt: off
            local_basis = _time_solver_stage(
                config.timing_collector,
                "solver.ipr_localization",
                lambda local_basis=local_basis, ipr_config=ipr_config: (
                    localized_basis_by_many_start_ipr(
                        local_basis.astype(np.complex128, copy=False),
                        config=ipr_config,
                    )
                ),
            )
            used_ipr = True
            # fmt: on

        for localized_index in range(local_basis.shape[1]):
            local_state = local_basis[:, localized_index]

            # fmt: off
            cage_state = _time_solver_stage(
                config.timing_collector,
                "solver.validation",
                lambda local_state=local_state,
                representative_energy=representative_energy,
                group_index=group_index,
                state_indices=state_indices,
                used_ipr=used_ipr: _build_validated_cage_state(
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
                    hamiltonian_column_block=hamiltonian_column_block,
                ),
            )
            # fmt: on

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
