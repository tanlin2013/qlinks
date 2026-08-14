from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import EnvironmentReductionConfig
from qlinks.caging.analysis.environment.support import support_key_from_mask
from qlinks.caging.analysis.transitions import LocalTransitionPattern


@dataclass(frozen=True, slots=True)
class _ReducedLocalOperatorApplicationContext:
    """Cached constrained-basis lookups for one reduced local support.

    For a fixed local mask, applying a reduced local operator only changes the
    local coordinates and preserves the environment coordinates.  This context
    maps ``(environment_key, local_key)`` directly to the constrained-basis
    index, so repeated reduced-IZ probes avoid rebuilding full target
    configurations and hashing full configuration tuples.
    """

    local_variable_indices: tuple[int, ...]
    environment_variable_indices: tuple[int, ...]
    local_key_by_basis_index: dict[int, tuple[int, ...]]
    environment_key_by_basis_index: dict[int, tuple[int, ...]]
    index_by_environment_and_local: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        int,
    ]


def _common_mask(
    configs: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """
    Return positions where all configurations agree.

    This is the numerical version of Lambda_h.
    """
    reference = configs[0]
    return np.all(configs == reference[None, :], axis=0)


def _q_sector_weight(
    full_state: NDArray[np.complex128],
    *,
    basis_configs: NDArray[np.integer],
    reference_config: NDArray[np.integer],
    common_mask: NDArray[np.bool_],
    active_indices: NDArray[np.int64] | None = None,
    config: EnvironmentReductionConfig,
) -> float:
    """
    Weight outside the common product-state sector.

    This estimates || Q_beta |psi> ||^2.  Only finite-amplitude entries can
    contribute, so callers that already have those indices can avoid scanning
    the full constrained basis for every zero report.
    """
    if np.count_nonzero(common_mask) == 0:
        return 1.0

    if active_indices is None:
        active_indices = np.flatnonzero(np.abs(full_state) > config.amplitude_tolerance).astype(
            np.int64,
            copy=False,
        )

    if active_indices.size == 0:
        return 0.0

    active_configs = basis_configs[active_indices]
    same_common_sector = np.all(
        active_configs[:, common_mask] == reference_config[common_mask][None, :],
        axis=1,
    )
    complement_indices = active_indices[~same_common_sector]
    amplitudes = full_state[complement_indices]
    return float(np.sum(np.abs(amplitudes) ** 2))


def _complement_support_indices(
    full_state: NDArray[np.complex128],
    *,
    basis_configs: NDArray[np.integer],
    reference_config: NDArray[np.integer],
    common_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
    active_domain_indices: NDArray[np.int64] | None = None,
    amplitude_tolerance: float,
) -> NDArray[np.int64]:
    """Return finite-amplitude basis indices outside the beta common sector."""
    if active_domain_indices is None:
        active_mask = np.abs(full_state) > amplitude_tolerance
        active_domain_indices = np.flatnonzero(active_mask & domain_mask).astype(
            np.int64,
            copy=False,
        )

    if active_domain_indices.size == 0:
        return np.array([], dtype=np.int64)

    if np.count_nonzero(common_mask) == 0:
        return active_domain_indices.astype(np.int64, copy=False)

    active_configs = basis_configs[active_domain_indices]
    same_common_sector = np.all(
        active_configs[:, common_mask] == reference_config[common_mask][None, :],
        axis=1,
    )

    return active_domain_indices[~same_common_sector].astype(np.int64, copy=False)


def _local_transitions_for_zero(
    zero_index: int,
    *,
    active_neighbors: NDArray[np.int64],
    active_matrix_elements: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    local_mask: NDArray[np.bool_],
) -> list[LocalTransitionPattern]:
    """
    Construct local transitions defining Z_h.

    For each active edge u -> h, include the local transition
    u_local -> h_local with the matrix element H0[h, u].
    """
    target_local = _config_key(basis_configs[zero_index, local_mask])
    transitions: list[LocalTransitionPattern] = []

    for neighbor, matrix_element in zip(
        active_neighbors,
        active_matrix_elements,
        strict=True,
    ):
        source_local = _config_key(basis_configs[neighbor, local_mask])

        transitions.append(
            LocalTransitionPattern(
                source_local=source_local,
                target_local=target_local,
                matrix_element=complex(matrix_element),
            )
        )

        # Hermitian reverse. This is useful when testing the full reduced
        # operator Z_h^(R), not only the one-way leakage into |h>.
        transitions.append(
            LocalTransitionPattern(
                source_local=target_local,
                target_local=source_local,
                matrix_element=complex(np.conjugate(matrix_element)),
            )
        )

    return transitions


def _build_reduced_local_operator_application_context(
    *,
    basis_configs: NDArray[np.integer],
    domain_mask: NDArray[np.bool_],
    local_mask: NDArray[np.bool_],
) -> _ReducedLocalOperatorApplicationContext:
    """Build cached local/environment pattern lookups for one local mask."""
    local_variable_indices = support_key_from_mask(local_mask)
    environment_variable_indices = tuple(
        int(index) for index in np.flatnonzero(~np.asarray(local_mask, dtype=np.bool_))
    )

    local_columns = np.asarray(local_variable_indices, dtype=np.int64)
    environment_columns = np.asarray(environment_variable_indices, dtype=np.int64)

    local_key_by_basis_index: dict[int, tuple[int, ...]] = {}
    environment_key_by_basis_index: dict[int, tuple[int, ...]] = {}
    index_by_environment_and_local: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        int,
    ] = {}

    for basis_index_raw in np.flatnonzero(domain_mask):
        basis_index = int(basis_index_raw)
        config = basis_configs[basis_index]
        local_key = _indexed_config_key(config, local_columns)
        environment_key = _indexed_config_key(config, environment_columns)

        local_key_by_basis_index[basis_index] = local_key
        environment_key_by_basis_index[basis_index] = environment_key
        index_by_environment_and_local[(environment_key, local_key)] = basis_index

    return _ReducedLocalOperatorApplicationContext(
        local_variable_indices=local_variable_indices,
        environment_variable_indices=environment_variable_indices,
        local_key_by_basis_index=local_key_by_basis_index,
        environment_key_by_basis_index=environment_key_by_basis_index,
        index_by_environment_and_local=index_by_environment_and_local,
    )


def _get_reduced_local_operator_application_context(
    cache: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    *,
    basis_configs: NDArray[np.integer],
    domain_mask: NDArray[np.bool_],
    local_mask: NDArray[np.bool_],
) -> _ReducedLocalOperatorApplicationContext | None:
    """Return a cached local-operator application context when a cache is supplied."""
    if cache is None:
        return None

    support_key = support_key_from_mask(local_mask)
    context = cache.get(support_key)
    if context is None:
        context = _build_reduced_local_operator_application_context(
            basis_configs=basis_configs,
            domain_mask=domain_mask,
            local_mask=local_mask,
        )
        cache[support_key] = context

    return context


def _apply_reduced_local_operator(
    full_state: NDArray[np.complex128],
    *,
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    local_mask: NDArray[np.bool_],
    local_transitions: tuple[LocalTransitionPattern, ...] | list[LocalTransitionPattern],
    domain_mask: NDArray[np.bool_],
    local_transition_lookup: (
        dict[tuple[int, ...], tuple[LocalTransitionPattern, ...]] | None
    ) = None,
    application_context: _ReducedLocalOperatorApplicationContext | None = None,
    source_indices: NDArray[np.int64] | None = None,
    common_mask: NDArray[np.bool_] | None = None,
    reference_config: NDArray[np.integer] | None = None,
    use_complement_common_sector: bool = False,
    amplitude_tolerance: float = 0.0,
) -> tuple[NDArray[np.complex128], NDArray[np.int64], NDArray[np.int64]]:
    """
    Apply local Z_h pattern to the full state.

    Returns
    -------
    output:
        The final vector after summing all contributions.
    target_indices:
        Vertices that received at least one raw contribution before
        destructive cancellation.

    This distinction matters: if the complement action detects another
    interference zero, the final output amplitude can vanish, but the target
    vertex should still be recorded.
    """
    output = np.zeros_like(full_state)
    target_indices: set[int] = set()
    contributing_input_indices: set[int] = set()
    transitions_by_source = (
        _group_local_transitions_by_source(local_transitions)
        if local_transition_lookup is None
        else local_transition_lookup
    )

    if source_indices is None:
        active_mask = domain_mask & (np.abs(full_state) > amplitude_tolerance)
        source_indices = np.flatnonzero(active_mask).astype(np.int64, copy=False)

    if application_context is not None:
        expected_support = support_key_from_mask(local_mask)
        if application_context.local_variable_indices != expected_support:
            raise ValueError(
                "application_context was built for a different local support: "
                f"{application_context.local_variable_indices!r} != {expected_support!r}."
            )

    for source_index_raw in source_indices:
        source_index = int(source_index_raw)
        if not domain_mask[source_index]:
            continue

        source_amplitude = full_state[source_index]

        if abs(source_amplitude) <= amplitude_tolerance:
            continue

        source_config = basis_configs[source_index]

        if common_mask is not None:
            if reference_config is None:
                raise ValueError("reference_config is required when common_mask is used.")

            in_common_sector = np.all(source_config[common_mask] == reference_config[common_mask])

            if use_complement_common_sector and in_common_sector:
                continue

            if not use_complement_common_sector and not in_common_sector:
                continue

        if application_context is None:
            source_local = _config_key(source_config[local_mask])
            environment_key: tuple[int, ...] | None = None
        else:
            source_local = application_context.local_key_by_basis_index.get(source_index)
            environment_key = application_context.environment_key_by_basis_index.get(source_index)
            if source_local is None or environment_key is None:
                continue

        matching_transitions = transitions_by_source.get(source_local)

        if matching_transitions is None:
            continue

        for transition in matching_transitions:
            if application_context is None:
                target_config = np.array(source_config, copy=True)
                target_config[local_mask] = np.array(
                    transition.target_local,
                    dtype=target_config.dtype,
                )

                target_index = config_to_index.get(_config_key(target_config))
            else:
                target_index = application_context.index_by_environment_and_local.get(
                    (environment_key, transition.target_local)
                )

            if target_index is None:
                continue

            if not domain_mask[target_index]:
                continue

            contribution = transition.matrix_element * source_amplitude

            if abs(contribution) <= amplitude_tolerance:
                continue

            output[target_index] += contribution
            target_indices.add(int(target_index))
            contributing_input_indices.add(int(source_index))

    return (
        output,
        np.array(sorted(target_indices), dtype=np.int64),
        np.array(sorted(contributing_input_indices), dtype=np.int64),
    )


def _group_local_transitions_by_source(
    local_transitions: tuple[LocalTransitionPattern, ...] | list[LocalTransitionPattern],
) -> dict[tuple[int, ...], tuple[LocalTransitionPattern, ...]]:
    """Group local transitions by source pattern for fast local-operator application."""
    transition_groups: dict[tuple[int, ...], list[LocalTransitionPattern]] = {}

    for transition in local_transitions:
        transition_groups.setdefault(transition.source_local, []).append(transition)

    return {
        source_local: tuple(transitions) for source_local, transitions in transition_groups.items()
    }


def _build_config_to_index(
    basis_configs: NDArray[np.integer],
) -> dict[tuple[int, ...], int]:
    """Map each full basis configuration to its basis index."""
    return {_config_key(config): int(index) for index, config in enumerate(basis_configs)}


def _config_key(config: NDArray[np.integer]) -> tuple[int, ...]:
    """Hashable representation of one basis configuration."""
    return tuple(int(value) for value in np.asarray(config).ravel())


def _indexed_config_key(
    config: NDArray[np.integer],
    indices: NDArray[np.int64],
) -> tuple[int, ...]:
    """Hashable key for a selected subset of one basis configuration."""
    if indices.size == 0:
        return ()
    return tuple(int(value) for value in np.asarray(config)[indices])
