from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import (
    CollectiveCancellationReport,
    EnvironmentProbeDetailLabel,
    EnvironmentReductionConfig,
    EnvironmentRemovalProbeReport,
)
from qlinks.caging.analysis.environment.operator import (
    _apply_reduced_local_operator,
    _get_reduced_local_operator_application_context,
    _ReducedLocalOperatorApplicationContext,
)
from qlinks.caging.analysis.environment.support import support_key_from_mask
from qlinks.caging.analysis.transitions import (
    local_transition_pattern_signature,
)


def _annotate_probe_mechanisms(
    zero_reports: list[EnvironmentRemovalProbeReport],
    *,
    trivial_zero_indices: set[int],
    config: EnvironmentReductionConfig,
) -> list[EnvironmentRemovalProbeReport]:
    """Determine how each exterior probe is removed.

    The supported physical routes are deliberately narrow:

    * no environment weight (``q_empty``);
    * projective annihilation (``domain_blocked`` or ``projector_like``); and
    * reuse of the same support-aware local cancellation pattern
      (``closed_by_same_pattern_zeros``).

    A known interference-zero target with a different local transition
    signature is *not* sufficient for environment removal. Such a target is
    reported as unexpected unless projective annihilation already removes the
    source probe.
    """
    report_by_zero = {int(report.zero_index): report for report in zero_reports}
    known_zero_indices = set(report_by_zero)
    signature_by_zero = {
        zero_index: local_transition_pattern_signature(
            report.local_variable_indices or support_key_from_mask(report.local_mask),
            report.local_transitions,
        )
        for zero_index, report in report_by_zero.items()
    }

    trivial_targets_by_zero: dict[int, set[int]] = {}
    iz_targets_by_zero: dict[int, set[int]] = {}
    unknown_targets_by_zero: dict[int, set[int]] = {}
    for report in zero_reports:
        source = int(report.zero_index)
        trivial_targets: set[int] = set()
        iz_targets: set[int] = set()
        unknown_targets: set[int] = set()
        for target_index in report.complement_target_indices:
            target = int(target_index)
            if target in trivial_zero_indices:
                trivial_targets.add(target)
            elif target in known_zero_indices:
                iz_targets.add(target)
            else:
                unknown_targets.add(target)
        trivial_targets_by_zero[source] = trivial_targets
        iz_targets_by_zero[source] = iz_targets
        unknown_targets_by_zero[source] = unknown_targets

    source_projector_dependent_sources = {
        int(report.zero_index) for report in zero_reports if report.source_projector_like
    }
    domain_blocked_sources = {
        int(report.zero_index)
        for report in zero_reports
        if report.source_projector_like and report.complement_contributing_input_indices.size == 0
    }
    projector_dependent_sources = set(source_projector_dependent_sources)

    # Projective dependence is allowed to propagate through the known-zero
    # network because the closure is then supplied by the projection itself,
    # not by equality of cancellation patterns.
    changed = True
    while changed:
        changed = False
        for source, iz_targets in iz_targets_by_zero.items():
            if source in projector_dependent_sources:
                continue
            if any(target in projector_dependent_sources for target in iz_targets):
                projector_dependent_sources.add(source)
                changed = True

    annotated_reports: list[EnvironmentRemovalProbeReport] = []
    for report in zero_reports:
        source = int(report.zero_index)
        trivial_targets = trivial_targets_by_zero[source]
        iz_targets = iz_targets_by_zero[source]
        unknown_targets = unknown_targets_by_zero[source]

        projector_like_iz_targets = {
            target for target in iz_targets if target in projector_dependent_sources
        }
        ordinary_iz_targets = iz_targets - projector_like_iz_targets
        same_pattern_iz_targets = {
            target
            for target in ordinary_iz_targets
            if signature_by_zero[target] == signature_by_zero[source]
        }
        mismatched_pattern_targets = ordinary_iz_targets - same_pattern_iz_targets

        # A pattern mismatch is unsafe only when projective annihilation does
        # not already provide the removal mechanism.
        source_is_projective = source in projector_dependent_sources
        unexpected_targets = set(unknown_targets)
        if not source_is_projective:
            unexpected_targets.update(mismatched_pattern_targets)

        has_unexpected_targets = bool(unexpected_targets)
        has_nonzero_complement_action = report.has_nonzero_complement_action
        q_empty = report.q_sector_weight <= config.action_tolerance

        if has_unexpected_targets or has_nonzero_complement_action:
            probe_mechanism_label: EnvironmentProbeDetailLabel = "unexplained_leakage"
        elif source in domain_blocked_sources:
            probe_mechanism_label = "domain_blocked"
        elif source_is_projective:
            probe_mechanism_label = "projector_like"
        elif q_empty:
            probe_mechanism_label = "q_empty"
        else:
            probe_mechanism_label = "closed_by_same_pattern_zeros"

        explained_targets = sorted(
            trivial_targets | same_pattern_iz_targets | projector_like_iz_targets
        )
        annotated_reports.append(
            _replace_environment_probe_report(
                report,
                probe_mechanism_label=probe_mechanism_label,
                trivial_target_indices=np.array(sorted(trivial_targets), dtype=np.int64),
                same_pattern_iz_target_indices=np.array(
                    sorted(same_pattern_iz_targets), dtype=np.int64
                ),
                projector_like_iz_target_indices=np.array(
                    sorted(projector_like_iz_targets), dtype=np.int64
                ),
                unexpected_target_indices=np.array(sorted(unexpected_targets), dtype=np.int64),
                has_unexpected_targets=has_unexpected_targets,
                has_nonzero_complement_action=has_nonzero_complement_action,
                unexpected_target_probe_failure_indices=np.array(
                    sorted(unexpected_targets), dtype=np.int64
                ),
                nonzero_complement_action_target_indices=(
                    report.nonzero_complement_action_target_indices
                ),
                explained_complement_target_indices=np.array(explained_targets, dtype=np.int64),
                unexplained_complement_target_indices=np.array(
                    sorted(unexpected_targets), dtype=np.int64
                ),
                complement_targets_are_known_zeros=(
                    len(report.complement_target_indices) > 0 and len(unknown_targets) == 0
                ),
            )
        )

    return annotated_reports


def _annotate_collective_cancellations(
    zero_reports: list[EnvironmentRemovalProbeReport],
    *,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    active_domain_indices: NDArray[np.int64] | None,
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
) -> tuple[list[EnvironmentRemovalProbeReport], tuple[CollectiveCancellationReport, ...]]:
    if config.collective_cancellation_mode == "disabled":
        return zero_reports, ()

    candidates = [
        report
        for report in zero_reports
        if (
            report.probe_mechanism_label == "unexplained_leakage"
            and report.has_nonzero_complement_action
            and not report.has_unexpected_targets
        )
    ]

    if len(candidates) < config.collective_min_group_size:
        return zero_reports, ()

    if config.collective_cancellation_mode in {
        "same_local_pattern_sum",
        "same_local_pattern_nullspace",
    }:
        groups = _group_reports_by_local_pattern(candidates)
    else:
        raise ValueError(
            f"Unknown collective_cancellation_mode: {config.collective_cancellation_mode!r}"
        )

    collective_reports: list[CollectiveCancellationReport] = []
    replacement_by_zero: dict[int, EnvironmentRemovalProbeReport] = {}
    next_group_id = 0

    for grouped_reports in groups:
        if len(grouped_reports) < config.collective_min_group_size:
            continue

        grouping_kind = "same_local_pattern"

        if config.collective_cancellation_mode == "same_local_pattern_sum":
            collective = _find_unit_sum_collective_cancellation(
                grouped_reports,
                group_id=next_group_id,
                full_state=full_state,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                domain_mask=domain_mask,
                active_domain_indices=active_domain_indices,
                local_operator_contexts=local_operator_contexts,
                config=config,
                grouping_kind=grouping_kind,
            )
        elif config.collective_cancellation_mode == "same_local_pattern_nullspace":
            collective = _find_nullspace_collective_cancellation(
                grouped_reports,
                group_id=next_group_id,
                full_state=full_state,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                domain_mask=domain_mask,
                active_domain_indices=active_domain_indices,
                local_operator_contexts=local_operator_contexts,
                config=config,
                grouping_kind=grouping_kind,
            )
        else:
            raise ValueError(
                f"Unknown collective_cancellation_mode: {config.collective_cancellation_mode!r}"
            )

        if collective is None:
            continue

        collective_reports.append(collective)

        partners = collective.source_zero_indices
        for source_zero, coefficient in zip(
            collective.source_zero_indices,
            collective.coefficients,
            strict=True,
        ):
            original = next(
                report for report in grouped_reports if int(report.zero_index) == int(source_zero)
            )

            replacement_by_zero[int(source_zero)] = _replace_environment_probe_report(
                original,
                probe_mechanism_label="collective_cancellation",
                collective_cancellation_group_id=collective.group_id,
                collective_cancellation_partner_zero_indices=partners,
                collective_cancellation_coefficient=complex(coefficient),
                collective_cancellation_norm=collective.collective_action_norm,
                has_nonzero_complement_action=False,
                nonzero_complement_action_target_indices=np.array([], dtype=np.int64),
                explained_complement_target_indices=(original.complement_target_indices),
                unexplained_complement_target_indices=np.array([], dtype=np.int64),
            )

        next_group_id += 1

    if not replacement_by_zero:
        return zero_reports, ()

    annotated_reports = [
        replacement_by_zero.get(int(report.zero_index), report) for report in zero_reports
    ]

    return annotated_reports, tuple(collective_reports)


def _replace_environment_probe_report(
    report: EnvironmentRemovalProbeReport,
    **updates: object,
) -> EnvironmentRemovalProbeReport:
    """Return a copy of an EnvironmentRemovalProbeReport with updated fields."""
    values = {
        "zero_index": report.zero_index,
        "active_neighbors": report.active_neighbors,
        "active_matrix_elements": report.active_matrix_elements,
        "active_amplitudes": report.active_amplitudes,
        "cancellation_residual": report.cancellation_residual,
        "common_mask": report.common_mask,
        "local_mask": report.local_mask,
        "local_transitions": report.local_transitions,
        "q_sector_weight": report.q_sector_weight,
        "reduced_action_norm": report.reduced_action_norm,
        "complement_action_norm": report.complement_action_norm,
        "complement_target_indices": report.complement_target_indices,
        "explained_complement_target_indices": (report.explained_complement_target_indices),
        "unexplained_complement_target_indices": (report.unexplained_complement_target_indices),
        "complement_targets_are_known_zeros": (report.complement_targets_are_known_zeros),
        "trivial_target_indices": report.trivial_target_indices,
        "same_pattern_iz_target_indices": (report.same_pattern_iz_target_indices),
        "projector_like_iz_target_indices": (report.projector_like_iz_target_indices),
        "unexpected_target_indices": report.unexpected_target_indices,
        "complement_support_indices": report.complement_support_indices,
        "complement_contributing_input_indices": (report.complement_contributing_input_indices),
        "projector_like_annihilated_input_indices": (
            report.projector_like_annihilated_input_indices
        ),
        "source_projector_like": report.source_projector_like,
        "has_unexpected_targets": report.has_unexpected_targets,
        "has_nonzero_complement_action": (report.has_nonzero_complement_action),
        "unexpected_target_probe_failure_indices": (report.unexpected_target_probe_failure_indices),
        "nonzero_complement_action_target_indices": (
            report.nonzero_complement_action_target_indices
        ),
        "probe_mechanism_label": report.probe_mechanism_label,
        "collective_cancellation_group_id": report.collective_cancellation_group_id,
        "collective_cancellation_partner_zero_indices": (
            report.collective_cancellation_partner_zero_indices
        ),
        "collective_cancellation_coefficient": (report.collective_cancellation_coefficient),
        "collective_cancellation_norm": report.collective_cancellation_norm,
        "reduced_action_vector": report.reduced_action_vector,
        "local_variable_indices": report.local_variable_indices,
    }

    values.update(updates)

    return EnvironmentRemovalProbeReport(**values)


def _complement_action_for_report(
    report: EnvironmentRemovalProbeReport,
    *,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    active_domain_indices: NDArray[np.int64] | None,
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    application_context = _get_reduced_local_operator_application_context(
        local_operator_contexts,
        basis_configs=basis_configs,
        domain_mask=domain_mask,
        local_mask=report.local_mask,
    )

    action, target_indices, _input_indices = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        domain_mask=domain_mask,
        common_mask=report.common_mask,
        reference_config=basis_configs[int(report.zero_index)],
        local_mask=report.local_mask,
        local_transitions=report.local_transitions,
        application_context=application_context,
        source_indices=active_domain_indices,
        use_complement_common_sector=True,
        amplitude_tolerance=config.amplitude_tolerance,
    )

    return action, target_indices


def _find_unit_sum_collective_cancellation_from_actions(
    reports: list[EnvironmentRemovalProbeReport],
    actions: list[NDArray[np.complex128]],
    target_indices: list[NDArray[np.int64]],
    *,
    group_id: int,
    config: EnvironmentReductionConfig,
    grouping_kind: Literal["same_local_pattern"],
) -> CollectiveCancellationReport | None:
    if len(reports) < config.collective_min_group_size:
        return None

    if len(reports) != len(actions) or len(reports) != len(target_indices):
        raise ValueError("reports, actions, and target_indices must have the same length.")

    collective_action = np.sum(actions, axis=0)
    collective_norm = float(np.linalg.norm(collective_action))

    if collective_norm > _collective_tolerance(config):
        return None

    source_zero_indices = np.array(
        [int(report.zero_index) for report in reports],
        dtype=np.int64,
    )
    individual_norms = np.array(
        [float(np.linalg.norm(action)) for action in actions],
        dtype=np.float64,
    )
    collective_targets = np.array(
        sorted({int(target) for targets in target_indices for target in targets}),
        dtype=np.int64,
    )

    local_mask = _union_local_mask(reports)

    return CollectiveCancellationReport(
        group_id=group_id,
        source_zero_indices=source_zero_indices,
        coefficients=np.ones(len(reports), dtype=np.complex128),
        individual_complement_action_norms=individual_norms,
        collective_action_norm=collective_norm,
        collective_target_indices=collective_targets,
        local_mask=local_mask,
        local_region_size=int(np.count_nonzero(local_mask)),
        relation_kind="unit_sum",
        grouping_kind=grouping_kind,
    )


def _find_nullspace_collective_cancellation_from_actions(
    reports: list[EnvironmentRemovalProbeReport],
    actions: list[NDArray[np.complex128]],
    target_indices: list[NDArray[np.int64]],
    *,
    group_id: int,
    config: EnvironmentReductionConfig,
    grouping_kind: Literal["same_local_pattern"],
) -> CollectiveCancellationReport | None:
    """Find a nontrivial linear relation among complement leakage vectors.

    The input actions are columns l_h = Zbar_h |psi>.  This helper checks
    whether there is a nonzero coefficient vector c such that L c ~= 0.
    """
    if len(reports) < config.collective_min_group_size:
        return None

    if len(reports) != len(actions) or len(reports) != len(target_indices):
        raise ValueError("reports, actions, and target_indices must have the same length.")

    if len(actions) == 0:
        return None

    leakage_matrix = np.column_stack(actions)
    n_columns = leakage_matrix.shape[1]

    if n_columns < config.collective_min_group_size:
        return None

    _u, singular_values, vh = np.linalg.svd(
        leakage_matrix,
        full_matrices=True,
    )

    tolerance = _collective_tolerance(config)
    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank >= n_columns:
        return None

    coefficients = np.conjugate(vh[rank, :]).astype(
        np.complex128,
        copy=False,
    )

    collective_action = leakage_matrix @ coefficients
    collective_norm = float(np.linalg.norm(collective_action))

    if collective_norm > tolerance:
        return None

    local_mask = _union_local_mask(reports)

    collective_targets = np.array(
        sorted({int(target) for targets in target_indices for target in targets}),
        dtype=np.int64,
    )

    return CollectiveCancellationReport(
        group_id=int(group_id),
        source_zero_indices=np.array(
            [int(report.zero_index) for report in reports],
            dtype=np.int64,
        ),
        coefficients=coefficients,
        individual_complement_action_norms=np.array(
            [float(np.linalg.norm(action)) for action in actions],
            dtype=np.float64,
        ),
        collective_action_norm=collective_norm,
        collective_target_indices=collective_targets,
        local_mask=local_mask,
        local_region_size=int(np.count_nonzero(local_mask)),
        relation_kind="nullspace",
        grouping_kind=grouping_kind,
    )


def _find_unit_sum_collective_cancellation(
    reports: list[EnvironmentRemovalProbeReport],
    *,
    group_id: int,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    active_domain_indices: NDArray[np.int64] | None,
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
    grouping_kind: Literal["same_local_pattern"],
) -> CollectiveCancellationReport | None:
    actions: list[NDArray[np.complex128]] = []
    target_indices: list[NDArray[np.int64]] = []

    for report in reports:
        action, targets = _complement_action_for_report(
            report,
            full_state=full_state,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            domain_mask=domain_mask,
            active_domain_indices=active_domain_indices,
            local_operator_contexts=local_operator_contexts,
            config=config,
        )
        actions.append(action)
        target_indices.append(targets)

    return _find_unit_sum_collective_cancellation_from_actions(
        reports,
        actions,
        target_indices,
        group_id=group_id,
        config=config,
        grouping_kind=grouping_kind,
    )


def _find_nullspace_collective_cancellation(
    reports: list[EnvironmentRemovalProbeReport],
    *,
    group_id: int,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    active_domain_indices: NDArray[np.int64] | None,
    local_operator_contexts: dict[tuple[int, ...], _ReducedLocalOperatorApplicationContext] | None,
    config: EnvironmentReductionConfig,
    grouping_kind: Literal["same_local_pattern"],
) -> CollectiveCancellationReport | None:
    actions: list[NDArray[np.complex128]] = []
    target_indices: list[NDArray[np.int64]] = []

    for report in reports:
        action, targets = _complement_action_for_report(
            report,
            full_state=full_state,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            domain_mask=domain_mask,
            active_domain_indices=active_domain_indices,
            local_operator_contexts=local_operator_contexts,
            config=config,
        )
        actions.append(action)
        target_indices.append(targets)

    return _find_nullspace_collective_cancellation_from_actions(
        reports,
        actions,
        target_indices,
        group_id=group_id,
        config=config,
        grouping_kind=grouping_kind,
    )


def _union_local_mask(
    reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.bool_]:
    if not reports:
        return np.array([], dtype=np.bool_)

    local_mask = np.zeros_like(reports[0].local_mask, dtype=np.bool_)

    for report in reports:
        local_mask |= report.local_mask

    return local_mask


def _group_reports_by_local_pattern(
    reports: list[EnvironmentRemovalProbeReport],
) -> list[list[EnvironmentRemovalProbeReport]]:
    """Group probes only when their local support and transition pattern match."""
    grouped: dict[object, list[EnvironmentRemovalProbeReport]] = {}
    for report in reports:
        key = local_transition_pattern_signature(
            report.local_variable_indices or support_key_from_mask(report.local_mask),
            report.local_transitions,
        )
        grouped.setdefault(key, []).append(report)
    return list(grouped.values())


def _collective_tolerance(config: EnvironmentReductionConfig) -> float:
    if config.collective_relation_tolerance is not None:
        return float(config.collective_relation_tolerance)
    return float(config.action_tolerance)
