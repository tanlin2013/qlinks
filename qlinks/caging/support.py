from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qlinks.caging.classification import (
    CageClassificationReport,
    InterferenceZeroReport,
    IZProbeMechanismLabel,
)

CageSupportExtractionPolicy = Literal[
    "raise_on_unexplained",
    "ignore_unexplained",
]


@dataclass(frozen=True, slots=True)
class ReducedIZProbeSupport:
    """Support data for one reduced IZ probe Z_h^(R)."""

    zero_index: int
    mechanism_label: IZProbeMechanismLabel
    variable_indices: tuple[int, ...]
    local_region_size: int
    complement_action_norm: float
    reduced_action_norm: float
    n_local_transitions: int
    n_complement_targets: int
    n_unexplained_complement_targets: int

    @property
    def is_valid_for_region_union(self) -> bool:
        return self.mechanism_label != "unexplained_leakage"


@dataclass(frozen=True, slots=True)
class CageRegionSupport:
    """Union support R extracted from reduced IZ probes."""

    variable_indices: tuple[int, ...]
    probe_supports: tuple[ReducedIZProbeSupport, ...]
    ignored_probe_supports: tuple[ReducedIZProbeSupport, ...]
    n_total_probes: int
    n_used_probes: int
    n_ignored_probes: int
    n_unexplained_leakage_probes: int
    max_complement_action_norm: float
    max_reduced_action_norm: float
    metadata: dict[str, object]

    @property
    def variable_index_set(self) -> frozenset[int]:
        return frozenset(self.variable_indices)

    @property
    def region_size(self) -> int:
        return len(self.variable_indices)

    @property
    def has_unexplained_leakage(self) -> bool:
        return self.n_unexplained_leakage_probes > 0

    def contains_variable(self, variable_index: int) -> bool:
        return int(variable_index) in self.variable_index_set


def extract_cage_region_support(
    report: CageClassificationReport,
    *,
    policy: CageSupportExtractionPolicy = "raise_on_unexplained",
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    complement_action_tolerance: float | None = None,
) -> CageRegionSupport:
    """Extract the union support R from trusted reduced IZ probes.

    This function does not interpret the report label as a scaling statement.
    It only asks whether each reduced probe has explained complement behavior.

    By default, q_empty, closed_by_known_zeros, and projector_like probes are
    all allowed.  Only unexplained_leakage is rejected.
    """
    probe_supports: list[ReducedIZProbeSupport] = []
    ignored_probe_supports: list[ReducedIZProbeSupport] = []
    union_variables: set[int] = set()

    for zero_report in report.zero_reports:
        probe_support = reduced_iz_probe_support_from_report(zero_report)

        use_probe = _should_use_probe_support(
            probe_support,
            include_q_empty=include_q_empty,
            include_closed_by_known_zeros=include_closed_by_known_zeros,
            include_projector_like=include_projector_like,
            include_collective_cancellation=include_collective_cancellation,
            complement_action_tolerance=complement_action_tolerance,
        )

        if probe_support.mechanism_label == "unexplained_leakage":
            ignored_probe_supports.append(probe_support)
            continue

        if use_probe:
            probe_supports.append(probe_support)
            union_variables.update(probe_support.variable_indices)
        else:
            ignored_probe_supports.append(probe_support)

    n_unexplained = sum(
        support.mechanism_label == "unexplained_leakage" for support in ignored_probe_supports
    )

    if n_unexplained > 0 and policy == "raise_on_unexplained":
        raise ValueError(
            "Cannot safely extract cage region support because at least one "
            "reduced IZ probe has unexplained leakage. "
            f"n_unexplained_leakage_probes={n_unexplained}"
        )

    complement_norms = [support.complement_action_norm for support in probe_supports]
    reduced_norms = [support.reduced_action_norm for support in probe_supports]

    return CageRegionSupport(
        variable_indices=tuple(sorted(union_variables)),
        probe_supports=tuple(probe_supports),
        ignored_probe_supports=tuple(ignored_probe_supports),
        n_total_probes=len(report.zero_reports),
        n_used_probes=len(probe_supports),
        n_ignored_probes=len(ignored_probe_supports),
        n_unexplained_leakage_probes=int(n_unexplained),
        max_complement_action_norm=(max(complement_norms) if complement_norms else 0.0),
        max_reduced_action_norm=(max(reduced_norms) if reduced_norms else 0.0),
        metadata={
            "report_label": report.label,
            "support_size": report.support_size,
            "hilbert_size": report.hilbert_size,
            "support_fraction": report.support_fraction,
            "n_nontrivial_zeros": report.n_nontrivial_zeros,
            "n_distinct_local_patterns": report.n_distinct_local_patterns,
        },
    )


def reduced_iz_probe_support_from_report(
    zero_report: InterferenceZeroReport,
) -> ReducedIZProbeSupport:
    variable_indices = tuple(int(index) for index in np.flatnonzero(zero_report.local_mask))

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


def _should_use_probe_support(
    probe_support: ReducedIZProbeSupport,
    *,
    include_q_empty: bool,
    include_closed_by_known_zeros: bool,
    include_projector_like: bool,
    include_collective_cancellation: bool = True,
    complement_action_tolerance: float | None,
) -> bool:
    if complement_action_tolerance is not None:
        if probe_support.complement_action_norm > complement_action_tolerance:
            return False

    if probe_support.mechanism_label == "q_empty":
        return include_q_empty

    if probe_support.mechanism_label == "closed_by_known_zeros":
        return include_closed_by_known_zeros

    if probe_support.mechanism_label == "projector_like":
        return include_projector_like

    if probe_support.mechanism_label == "collective_cancellation":
        return include_collective_cancellation

    if probe_support.mechanism_label == "unexplained_leakage":
        return False

    raise ValueError(f"Unknown IZ probe mechanism label: {probe_support.mechanism_label!r}")


@dataclass(frozen=True, slots=True)
class ReducedIZPatternSupport:
    """One distinct reduced IZ local pattern and the variables it uses."""

    pattern_key: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...]
    variable_indices: tuple[int, ...]
    source_zero_indices: tuple[int, ...]
    mechanism_labels: tuple[IZProbeMechanismLabel, ...]


def distinct_reduced_iz_pattern_supports(
    report: CageClassificationReport,
    *,
    include_projector_like: bool = True,
) -> tuple[ReducedIZPatternSupport, ...]:
    """Group reduced IZ probes by local transition pattern and support."""
    grouped: dict[
        tuple[
            tuple[int, ...],
            tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...],
        ],
        list[InterferenceZeroReport],
    ] = {}

    for zero_report in report.zero_reports:
        if zero_report.probe_mechanism_label == "unexplained_leakage":
            continue
        if zero_report.probe_mechanism_label == "projector_like" and not include_projector_like:
            continue

        variables = tuple(int(i) for i in np.flatnonzero(zero_report.local_mask))
        key = (variables, _public_local_pattern_key(zero_report))
        grouped.setdefault(key, []).append(zero_report)

    pattern_supports: list[ReducedIZPatternSupport] = []

    for (variables, pattern_key), zero_reports in grouped.items():
        pattern_supports.append(
            ReducedIZPatternSupport(
                pattern_key=pattern_key,
                variable_indices=variables,
                source_zero_indices=tuple(int(report.zero_index) for report in zero_reports),
                mechanism_labels=tuple(report.probe_mechanism_label for report in zero_reports),
            )
        )

    return tuple(pattern_supports)


def _public_local_pattern_key(
    report: InterferenceZeroReport,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...]:
    return tuple(
        sorted(
            (
                transition.source_local,
                transition.target_local,
                _complex_key(transition.matrix_element),
            )
            for transition in report.local_transitions
        )
    )


def _complex_key(value: complex, *, digits: int = 12) -> tuple[float, float]:
    return (
        round(float(np.real(value)), digits),
        round(float(np.imag(value)), digits),
    )
