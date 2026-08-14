from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import (
    EnvironmentRemovalProbeReport,
    ReducedIZMonitorComponentGroup,
    ReducedIZMonitorDecomposition,
    ReducedIZProbeSupport,
)
from qlinks.caging.analysis.environment.support import (
    support_key_for_zero_report,
)


class _EnvironmentReductionReportLike(Protocol):
    """Minimal report surface needed by reduced-IZ monitor selection."""

    zero_reports: tuple[EnvironmentRemovalProbeReport, ...]


def reduced_iz_probe_support_from_report(
    zero_report: EnvironmentRemovalProbeReport,
) -> ReducedIZProbeSupport:
    """Return cached public support metadata for a reduced IZ probe."""
    variable_indices = support_key_for_zero_report(zero_report)

    return ReducedIZProbeSupport(
        zero_index=int(zero_report.zero_index),
        mechanism_label=zero_report.probe_mechanism_label,
        variable_indices=variable_indices,
        local_region_size=len(variable_indices),
        complement_action_norm=float(zero_report.complement_action_norm),
        reduced_action_norm=float(zero_report.reduced_action_norm),
        n_local_transitions=len(zero_report.local_transitions),
        n_complement_targets=int(zero_report.n_complement_targets),
        n_unexplained_complement_targets=int(zero_report.n_unexplained_complement_targets),
    )


def _reduced_iz_region_variables_from_supports(
    probe_supports: tuple[ReducedIZProbeSupport, ...],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                variable_index
                for probe_support in probe_supports
                if probe_support.is_valid_for_region_union
                for variable_index in probe_support.variable_indices
            }
        )
    )


def select_reduced_iz_monitor_reports(
    report: _EnvironmentReductionReportLike,
    *,
    include_q_empty: bool = True,
    include_same_pattern_cancellation: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
) -> tuple[EnvironmentRemovalProbeReport, ...]:
    """Select reduced-IZ reports from a environment-reduction report for monitor use."""
    return select_reduced_iz_monitor_reports_from_zero_reports(
        report.zero_reports,
        include_q_empty=include_q_empty,
        include_same_pattern_cancellation=include_same_pattern_cancellation,
        include_projector_like=include_projector_like,
        include_collective_cancellation=include_collective_cancellation,
    )


def select_reduced_iz_monitor_reports_from_zero_reports(
    zero_reports: tuple[EnvironmentRemovalProbeReport, ...] | list[EnvironmentRemovalProbeReport],
    *,
    include_q_empty: bool = True,
    include_same_pattern_cancellation: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
) -> tuple[EnvironmentRemovalProbeReport, ...]:
    """Select non-invalid reduced-IZ zero reports for monitor assembly."""
    selected: list[EnvironmentRemovalProbeReport] = []

    for zero_report in zero_reports:
        label = zero_report.probe_mechanism_label

        if label == "q_empty" and include_q_empty:
            selected.append(zero_report)
        elif label == "closed_by_same_pattern_zeros" and include_same_pattern_cancellation:
            selected.append(zero_report)
        elif label in {"domain_blocked", "projector_like"} and include_projector_like:
            selected.append(zero_report)
        elif label == "collective_cancellation" and include_collective_cancellation:
            selected.append(zero_report)
        elif label == "unexplained_leakage":
            continue

    return tuple(selected)


def group_reduced_iz_reports_by_exact_support(
    reports: tuple[EnvironmentRemovalProbeReport, ...],
) -> tuple[tuple[EnvironmentRemovalProbeReport, ...], ...]:
    """Group reduced-IZ reports with identical support variables."""
    grouped: dict[tuple[int, ...], list[EnvironmentRemovalProbeReport]] = {}

    for zero_report in reports:
        key = support_key_for_zero_report(zero_report)
        grouped.setdefault(key, []).append(zero_report)

    return tuple(tuple(group) for _key, group in sorted(grouped.items(), key=lambda item: item[0]))


def group_reduced_iz_reports_by_connected_support(
    reports: tuple[EnvironmentRemovalProbeReport, ...],
) -> tuple[tuple[EnvironmentRemovalProbeReport, ...], ...]:
    """Group reduced-IZ reports whose supports overlap transitively."""
    if len(reports) == 0:
        return ()

    supports = [set(support_key_for_zero_report(zero_report)) for zero_report in reports]

    visited: set[int] = set()
    groups: list[tuple[EnvironmentRemovalProbeReport, ...]] = []

    for start_index in range(len(reports)):
        if start_index in visited:
            continue

        stack = [start_index]
        component_indices: list[int] = []
        visited.add(start_index)

        while stack:
            current_index = stack.pop()
            component_indices.append(current_index)
            current_support = supports[current_index]

            for candidate_index, candidate_support in enumerate(supports):
                if candidate_index in visited:
                    continue

                if not current_support.isdisjoint(candidate_support):
                    visited.add(candidate_index)
                    stack.append(candidate_index)

        component_indices.sort()
        groups.append(tuple(reports[index] for index in component_indices))

    return tuple(groups)


def group_reduced_iz_monitor_reports(
    reports: tuple[EnvironmentRemovalProbeReport, ...],
    *,
    decomposition: ReducedIZMonitorDecomposition,
) -> tuple[tuple[EnvironmentRemovalProbeReport, ...], ...]:
    """Group reports according to a reduced-IZ monitor decomposition."""
    if decomposition == "single_sum":
        return (reports,) if reports else ()

    if decomposition == "exact_support":
        return group_reduced_iz_reports_by_exact_support(reports)

    if decomposition == "connected_support":
        return group_reduced_iz_reports_by_connected_support(reports)

    raise ValueError(f"Unknown reduced-IZ monitor decomposition: {decomposition!r}")


def reduced_iz_component_groups_from_reports(
    reports: tuple[EnvironmentRemovalProbeReport, ...],
    *,
    decomposition: ReducedIZMonitorDecomposition,
    use_collective_coefficients: bool = True,
) -> tuple[ReducedIZMonitorComponentGroup, ...]:
    """Return cached report-side metadata for reduced-IZ monitor components."""
    groups = group_reduced_iz_monitor_reports(
        reports,
        decomposition=decomposition,
    )
    component_groups: list[ReducedIZMonitorComponentGroup] = []

    for component_id, report_group in enumerate(groups):
        support_variables = tuple(
            sorted(
                {
                    variable_index
                    for zero_report in report_group
                    for variable_index in support_key_for_zero_report(zero_report)
                }
            )
        )
        component_groups.append(
            ReducedIZMonitorComponentGroup(
                component_id=component_id,
                decomposition=decomposition,
                zero_indices=tuple(int(report.zero_index) for report in report_group),
                support_variables=support_variables,
                state_action_vector=_reduced_iz_component_state_action_from_reports(
                    report_group,
                    use_collective_coefficients=use_collective_coefficients,
                ),
            )
        )

    return tuple(component_groups)


def _reduced_iz_component_state_action_from_reports(
    reports: tuple[EnvironmentRemovalProbeReport, ...],
    *,
    use_collective_coefficients: bool,
) -> NDArray[np.complex128]:
    """Return cached ``sum_h c_h Z_h^(R)|psi>`` for one component group.

    Empty arrays are returned when reports do not contain compatible cached
    reduced-action vectors, which preserves compatibility with hand-built or
    older serialized reports.
    """
    if len(reports) == 0:
        return np.array([], dtype=np.complex128)

    first_action = np.asarray(reports[0].reduced_action_vector, dtype=np.complex128)
    if first_action.ndim != 1 or first_action.size == 0:
        return np.array([], dtype=np.complex128)

    result = np.zeros_like(first_action, dtype=np.complex128)
    for zero_report in reports:
        action = np.asarray(zero_report.reduced_action_vector, dtype=np.complex128)
        if action.shape != first_action.shape:
            return np.array([], dtype=np.complex128)

        try:
            coefficient = _monitor_coefficient_for_zero_report(
                zero_report,
                use_collective_coefficients=use_collective_coefficients,
            )
        except ValueError:
            return np.array([], dtype=np.complex128)

        result = result + coefficient * action

    return result


def _monitor_coefficient_for_zero_report(
    zero_report: EnvironmentRemovalProbeReport,
    *,
    use_collective_coefficients: bool,
) -> complex:
    if (
        use_collective_coefficients
        and zero_report.probe_mechanism_label == "collective_cancellation"
    ):
        coefficient = complex(zero_report.collective_cancellation_coefficient)
        if coefficient == 0:
            raise ValueError(
                "Collective-cancellation zero report has zero stored coefficient. "
                "Cannot cache the reduced-IZ component action with collective coefficients."
            )
        return coefficient

    return 1.0 + 0.0j
