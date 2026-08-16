from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import (
    EnvironmentReductionConfig,
    EnvironmentRemovalProbeReport,
)
from qlinks.caging.analysis.environment.operator import (
    _apply_reduced_local_operator,
    _common_mask,
    _complement_support_indices,
    _get_reduced_local_operator_application_context,
    _group_local_transitions_by_source,
    _local_transitions_for_zero,
    _q_sector_weight,
    _ReducedLocalOperatorApplicationContext,
)
from qlinks.caging.analysis.environment.support import support_key_from_mask


def _active_frontier_zero_indices(
    kinetic_matrix: sp.csr_array,
    *,
    support_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
    active_state_indices: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return zero-amplitude domain vertices adjacent to active support.

    The reduced-IZ search only needs vertices ``h`` that receive at least one
    kinetic contribution from a finite-amplitude source ``u``.  In matrix
    language this means ``K[h, u] != 0`` for an active source column ``u``.
    Building this frontier from CSC columns avoids scanning every zero row in
    the Hilbert space for each environment-reduction analysis.
    """
    if active_state_indices.size == 0:
        return np.array([], dtype=np.int64)

    frontier_mask = np.zeros(support_mask.shape, dtype=np.bool_)
    kinetic_csc = kinetic_matrix.tocsc()

    for source_index_raw in active_state_indices:
        source_index = int(source_index_raw)
        col_start = kinetic_csc.indptr[source_index]
        col_end = kinetic_csc.indptr[source_index + 1]
        frontier_mask[kinetic_csc.indices[col_start:col_end]] = True

    frontier_mask &= domain_mask
    frontier_mask &= ~support_mask
    return np.flatnonzero(frontier_mask).astype(np.int64, copy=False)


def _find_trivial_zero_indices(
    full_state: NDArray[np.complex128],
    kinetic_matrix: sp.csr_array,
    *,
    support_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
    active_frontier_zero_indices: NDArray[np.int64] | None = None,
) -> set[int]:
    """Return zero-amplitude vertices with no active kinetic neighbors.

    A trivial zero is a zero-amplitude basis vertex that receives no
    direct contribution from the cage support under the parent kinetic
    Hamiltonian. Nontrivial IZs are handled separately.
    """
    if active_frontier_zero_indices is not None:
        trivial_mask = domain_mask & ~support_mask
        trivial_mask = np.array(trivial_mask, dtype=np.bool_, copy=True)
        trivial_mask[active_frontier_zero_indices] = False
        return {int(index) for index in np.flatnonzero(trivial_mask)}

    trivial_zero_indices: set[int] = set()

    for zero_index in np.flatnonzero(domain_mask):
        if support_mask[zero_index]:
            continue

        row_start = kinetic_matrix.indptr[zero_index]
        row_end = kinetic_matrix.indptr[zero_index + 1]

        neighbors = kinetic_matrix.indices[row_start:row_end]
        has_active_neighbor = bool(np.any(support_mask[neighbors]))

        if not has_active_neighbor:
            trivial_zero_indices.add(int(zero_index))

    return trivial_zero_indices


def _find_nontrivial_interference_zeros(
    full_state: NDArray[np.complex128],
    kinetic_matrix: sp.csr_array,
    *,
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    support_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
    active_frontier_zero_indices: NDArray[np.int64],
    active_state_indices: NDArray[np.int64],
    active_domain_indices: NDArray[np.int64],
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
) -> list[EnvironmentRemovalProbeReport]:
    """Find zero vertices with nontrivial cancellation from active neighbors."""
    reports: list[EnvironmentRemovalProbeReport] = []

    for zero_index_raw in active_frontier_zero_indices:
        zero_index = int(zero_index_raw)

        row_start = kinetic_matrix.indptr[zero_index]
        row_end = kinetic_matrix.indptr[zero_index + 1]

        neighbors = kinetic_matrix.indices[row_start:row_end]
        matrix_elements = kinetic_matrix.data[row_start:row_end]

        active_mask = support_mask[neighbors] & domain_mask[neighbors]
        if not np.any(active_mask):
            continue

        active_neighbors = neighbors[active_mask].astype(np.int64, copy=False)
        active_elements = matrix_elements[active_mask].astype(
            np.complex128,
            copy=False,
        )
        active_amplitudes = full_state[active_neighbors]

        cancellation = np.dot(active_elements, active_amplitudes)
        cancellation_residual = float(abs(cancellation))

        if cancellation_residual > config.cancellation_tolerance:
            continue

        report = _build_zero_report(
            zero_index,
            active_neighbors=active_neighbors,
            active_matrix_elements=active_elements,
            active_amplitudes=active_amplitudes,
            cancellation_residual=cancellation_residual,
            full_state=full_state,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            domain_mask=domain_mask,
            active_state_indices=active_state_indices,
            active_domain_indices=active_domain_indices,
            local_operator_contexts=local_operator_contexts,
            config=config,
        )
        reports.append(report)

    return reports


def _build_zero_report(
    zero_index: int,
    *,
    active_neighbors: NDArray[np.int64],
    active_matrix_elements: NDArray[np.complex128],
    active_amplitudes: NDArray[np.complex128],
    cancellation_residual: float,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    active_state_indices: NDArray[np.int64],
    active_domain_indices: NDArray[np.int64],
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
) -> EnvironmentRemovalProbeReport:
    """Build one interference-zero diagnostic report."""
    involved_indices = np.concatenate(
        [
            np.array([zero_index], dtype=np.int64),
            active_neighbors,
        ]
    )

    common_mask = _common_mask(basis_configs[involved_indices])
    local_mask = ~common_mask

    q_sector_weight = _q_sector_weight(
        full_state,
        basis_configs=basis_configs,
        reference_config=basis_configs[zero_index],
        common_mask=common_mask,
        active_indices=active_state_indices,
        config=config,
    )

    local_transitions = _local_transitions_for_zero(
        zero_index,
        active_neighbors=active_neighbors,
        active_matrix_elements=active_matrix_elements,
        basis_configs=basis_configs,
        local_mask=local_mask,
    )
    local_transition_lookup = _group_local_transitions_by_source(local_transitions)
    application_context = _get_reduced_local_operator_application_context(
        local_operator_contexts,
        basis_configs=basis_configs,
        domain_mask=domain_mask,
        local_mask=local_mask,
    )

    reduced_action, _reduced_targets, _reduced_inputs = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        domain_mask=domain_mask,
        common_mask=None,
        reference_config=None,
        local_mask=local_mask,
        local_transitions=local_transitions,
        local_transition_lookup=local_transition_lookup,
        application_context=application_context,
        source_indices=active_domain_indices,
        amplitude_tolerance=config.amplitude_tolerance,
    )

    complement_action, complement_target_indices, complement_contributing_input_indices = (
        _apply_reduced_local_operator(
            full_state,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            domain_mask=domain_mask,
            common_mask=common_mask,
            reference_config=basis_configs[zero_index],
            local_mask=local_mask,
            local_transitions=local_transitions,
            local_transition_lookup=local_transition_lookup,
            application_context=application_context,
            source_indices=active_domain_indices,
            use_complement_common_sector=True,
            amplitude_tolerance=config.amplitude_tolerance,
        )
    )

    complement_support_indices = _complement_support_indices(
        full_state,
        basis_configs=basis_configs,
        reference_config=basis_configs[zero_index],
        domain_mask=domain_mask,
        common_mask=common_mask,
        active_domain_indices=active_domain_indices,
        amplitude_tolerance=config.amplitude_tolerance,
    )

    projector_like_annihilated_input_indices = np.setdiff1d(
        complement_support_indices,
        complement_contributing_input_indices,
        assume_unique=False,
    ).astype(np.int64, copy=False)

    nonzero_complement_action_target_indices = np.array(
        [
            int(index)
            for index in complement_target_indices
            if abs(complement_action[int(index)]) > config.action_tolerance
        ],
        dtype=np.int64,
    )

    has_nonzero_complement_action = nonzero_complement_action_target_indices.size > 0
    complement_action_norm = float(np.linalg.norm(complement_action))
    complement_action_is_zero = complement_action_norm <= config.action_tolerance

    source_projector_like = (
        q_sector_weight > config.action_tolerance
        and complement_action_is_zero
        and projector_like_annihilated_input_indices.size > 0
    )

    return EnvironmentRemovalProbeReport(
        zero_index=int(zero_index),
        active_neighbors=active_neighbors,
        active_matrix_elements=active_matrix_elements,
        active_amplitudes=active_amplitudes,
        cancellation_residual=cancellation_residual,
        common_mask=common_mask,
        local_mask=local_mask,
        q_sector_weight=q_sector_weight,
        reduced_action_norm=float(np.linalg.norm(reduced_action)),
        complement_action_norm=complement_action_norm,
        complement_target_indices=complement_target_indices,
        explained_complement_target_indices=np.array([], dtype=np.int64),
        unexplained_complement_target_indices=complement_target_indices,
        complement_targets_are_known_zeros=False,
        has_unexpected_targets=False,
        has_nonzero_complement_action=has_nonzero_complement_action,
        unexpected_target_probe_failure_indices=np.array([], dtype=np.int64),
        nonzero_complement_action_target_indices=(nonzero_complement_action_target_indices),
        complement_support_indices=complement_support_indices,
        complement_contributing_input_indices=complement_contributing_input_indices,
        projector_like_annihilated_input_indices=(projector_like_annihilated_input_indices),
        source_projector_like=source_projector_like,
        trivial_target_indices=np.array([], dtype=np.int64),
        same_pattern_iz_target_indices=np.array([], dtype=np.int64),
        projector_like_iz_target_indices=np.array([], dtype=np.int64),
        unexpected_target_indices=np.array([], dtype=np.int64),
        probe_mechanism_label="unexplained_leakage",
        local_transitions=tuple(local_transitions),
        reduced_action_vector=reduced_action.astype(np.complex128, copy=True),
        local_variable_indices=support_key_from_mask(local_mask),
    )


def _resolve_environment_domain_mask(
    kinetic_matrix: sp.csr_array,
    *,
    support_mask: NDArray[np.bool_],
    sector_mask: NDArray[np.bool_] | None,
    config: EnvironmentReductionConfig,
) -> NDArray[np.bool_]:
    """Return the basis-domain mask used by the classifier.

    The environment domain is normally one topological sector or one
    connected Fock-space component. Reduced IZ probes are only allowed to
    see targets inside this domain.
    """
    n_basis = support_mask.size

    if sector_mask is not None:
        domain_mask = np.asarray(sector_mask, dtype=np.bool_)

        if domain_mask.shape != (n_basis,):
            raise ValueError("sector_mask must have shape (hilbert_size,).")

        if np.any(support_mask & ~domain_mask):
            raise ValueError("The cage support is not contained in the provided sector_mask.")

        return domain_mask

    if config.sector_policy == "ignore":
        return np.ones(n_basis, dtype=np.bool_)

    graph = kinetic_matrix.copy()
    graph.data = np.ones_like(graph.data, dtype=np.int8)
    graph = graph.maximum(graph.T)

    n_components, component_labels = sp.csgraph.connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    if n_components == 1:
        return np.ones(n_basis, dtype=np.bool_)

    support_components = np.unique(component_labels[support_mask])

    if support_components.size == 0:
        raise ValueError("Cannot infer a sector/component from empty support.")

    if support_components.size > 1:
        raise ValueError(
            "The cage support spans multiple disconnected Fock-space "
            "components. Provide sector_mask explicitly."
        )

    if config.sector_policy == "raise_if_disconnected":
        raise ValueError(
            "The kinetic/Fock-space graph is disconnected, but no sector_mask "
            "was provided. Either pass sector_mask for the intended "
            "topological sector, build the model directly in one sector, or "
            "set config.sector_policy='infer_support_component'."
        )

    if config.sector_policy == "infer_support_component":
        component = int(support_components[0])
        return component_labels == component

    raise ValueError(f"Unknown sector_policy: {config.sector_policy!r}")
