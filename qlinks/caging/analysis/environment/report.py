from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qlinks.caging.analysis.environment.contracts import (
    CollectiveCancellationReport,
    EnvironmentProbeDetailLabel,
    EnvironmentRemovalMechanismLabel,
    EnvironmentRemovalProbeReport,
    EnvironmentRemovalSummary,
    ReducedIZMonitorComponentGroup,
    ReducedIZMonitorDecomposition,
    ReducedIZProbeSupport,
)
from qlinks.caging.analysis.environment.monitor import (
    group_reduced_iz_monitor_reports,
    reduced_iz_component_groups_from_reports,
    select_reduced_iz_monitor_reports,
)


@dataclass(frozen=True, slots=True)
class EnvironmentReductionReport:
    """Exterior-environment reduction diagnostic for one caged state.

    The report does not classify the eigenstate. It records whether exterior
    environment degrees of freedom can be removed while preserving a bounded
    local caging operator, and the local mechanism used by each boundary probe.
    """

    # State support is retained only as diagnostic context.
    support_size: int
    hilbert_size: int
    support_fraction: float

    # IZ inventory.
    n_nontrivial_zeros: int
    n_distinct_local_patterns: int

    # Complement closure summary.
    n_complement_targets: int
    n_unexplained_complement_targets: int
    fraction_probes_safely_removable: float

    # Source-probe mechanism counts.
    n_q_empty_source_probes: int
    n_same_pattern_zero_closure_probes: int
    n_projector_like_source_probes: int
    n_invalid_source_probes: int
    n_collective_cancellation_source_probes: int
    collective_cancellation_source_zero_indices: NDArray[np.int64]

    # Source-zero index groups.
    q_empty_source_zero_indices: NDArray[np.int64]
    same_pattern_zero_closure_indices: NDArray[np.int64]
    projector_like_source_zero_indices: NDArray[np.int64]
    invalid_source_zero_indices: NDArray[np.int64]

    # Complement target explanation counts.
    n_trivial_targets: int
    n_same_pattern_iz_targets: int
    n_projector_like_iz_targets: int
    n_unexpected_targets: int

    # Invalid-probe reason counts and source indices.
    n_unexpected_target_probe_failures: int
    n_nonzero_complement_action_probe_failures: int
    unexpected_target_probe_failure_indices: NDArray[np.int64]
    nonzero_complement_action_probe_failure_indices: NDArray[np.int64]

    # Projector-like diagnostics.
    n_source_projector_like_probes: int
    n_indirect_projector_like_probes: int
    n_projector_like_annihilated_inputs: int
    source_projector_like_probe_indices: NDArray[np.int64]
    indirect_projector_like_probe_indices: NDArray[np.int64]
    projector_like_annihilated_input_indices: NDArray[np.int64]

    # Norm diagnostics.
    mean_q_sector_weight: float
    max_q_sector_weight: float
    mean_reduced_action_norm: float
    max_reduced_action_norm: float
    mean_complement_action_norm: float
    max_complement_action_norm: float

    # Details.
    zero_reports: tuple[EnvironmentRemovalProbeReport, ...]
    collective_cancellation_reports: tuple[CollectiveCancellationReport, ...]

    # Reduced-IZ monitor preparation cached at environment-reduction time.
    reduced_iz_probe_supports: tuple[ReducedIZProbeSupport, ...] = field(default_factory=tuple)
    reduced_iz_region_variable_indices: tuple[int, ...] = ()
    reduced_iz_monitor_component_groups: dict[
        ReducedIZMonitorDecomposition,
        tuple[ReducedIZMonitorComponentGroup, ...],
    ] = field(default_factory=dict)

    removal_summary: EnvironmentRemovalSummary = field(default_factory=EnvironmentRemovalSummary)

    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "EnvironmentReductionReport("
            f"is_safely_removable={self.is_safely_removable!r}, "
            f"support_size={self.support_size}, "
            f"hilbert_size={self.hilbert_size}, "
            f"n_nontrivial_zeros={self.n_nontrivial_zeros}, "
            f"n_projector_like_source_probes={self.n_projector_like_source_probes}, "
            f"n_unexpected_target_probe_failures="
            f"{self.n_unexpected_target_probe_failures}, "
            f"n_nonzero_complement_action_probe_failures="
            f"{self.n_nonzero_complement_action_probe_failures}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_text()

    def __rich__(self):
        return self.to_rich()

    @property
    def is_safely_removable(self) -> bool:
        return self.removal_summary.is_safely_removable

    @property
    def removal_mechanisms(self) -> tuple[EnvironmentRemovalMechanismLabel, ...]:
        return self.removal_summary.mechanisms_present

    @property
    def n_reduced_iz_probe_supports(self) -> int:
        return len(self.reduced_iz_probe_supports)

    @property
    def n_reduced_iz_region_variables(self) -> int:
        return len(self.reduced_iz_region_variable_indices)

    @property
    def domain_blocked_source_zero_indices(self) -> NDArray[np.int64]:
        """Source-zero indices whose probe is domain-blocked.

        Domain-blocked probes have finite complement support, but that
        complement support contains no input configuration on which the
        transplanted reduced-IZ operator can fire.  They are a regional
        closure mechanism in the state-level environment reduction.
        """
        return _zero_indices_with_mechanism(
            self.zero_reports,
            "domain_blocked",
        )

    @property
    def n_domain_blocked_source_probes(self) -> int:
        return int(self.domain_blocked_source_zero_indices.size)

    def selected_reduced_iz_reports(
        self,
        *,
        include_q_empty: bool = True,
        include_same_pattern_cancellation: bool = True,
        include_projector_like: bool = True,
        include_collective_cancellation: bool = True,
    ) -> tuple[EnvironmentRemovalProbeReport, ...]:
        """Return reduced-IZ reports selected for monitor assembly."""
        return select_reduced_iz_monitor_reports(
            self,
            include_q_empty=include_q_empty,
            include_same_pattern_cancellation=include_same_pattern_cancellation,
            include_projector_like=include_projector_like,
            include_collective_cancellation=include_collective_cancellation,
        )

    def reduced_iz_report_groups(
        self,
        *,
        decomposition: ReducedIZMonitorDecomposition,
        include_q_empty: bool = True,
        include_same_pattern_cancellation: bool = True,
        include_projector_like: bool = True,
        include_collective_cancellation: bool = True,
    ) -> tuple[tuple[EnvironmentRemovalProbeReport, ...], ...]:
        """Return report groups for a reduced-IZ monitor decomposition."""
        reports = self.selected_reduced_iz_reports(
            include_q_empty=include_q_empty,
            include_same_pattern_cancellation=include_same_pattern_cancellation,
            include_projector_like=include_projector_like,
            include_collective_cancellation=include_collective_cancellation,
        )
        return group_reduced_iz_monitor_reports(
            reports,
            decomposition=decomposition,
        )

    def reduced_iz_component_groups(
        self,
        *,
        decomposition: ReducedIZMonitorDecomposition,
        include_q_empty: bool = True,
        include_same_pattern_cancellation: bool = True,
        include_projector_like: bool = True,
        include_collective_cancellation: bool = True,
        use_collective_coefficients: bool = True,
    ) -> tuple[ReducedIZMonitorComponentGroup, ...]:
        """Return cached/recomputed reduced-IZ component-group metadata."""
        if (
            include_q_empty
            and include_same_pattern_cancellation
            and include_projector_like
            and include_collective_cancellation
            and use_collective_coefficients
            and decomposition in self.reduced_iz_monitor_component_groups
        ):
            return self.reduced_iz_monitor_component_groups[decomposition]

        return reduced_iz_component_groups_from_reports(
            self.selected_reduced_iz_reports(
                include_q_empty=include_q_empty,
                include_same_pattern_cancellation=include_same_pattern_cancellation,
                include_projector_like=include_projector_like,
                include_collective_cancellation=include_collective_cancellation,
            ),
            decomposition=decomposition,
            use_collective_coefficients=use_collective_coefficients,
        )

    def to_rich(
        self,
        *,
        verbose: bool = False,
        max_zero_reports: int = 10,
    ) -> Group:
        """Return a Rich renderable focused on exterior-environment removal."""
        status = "safe" if self.is_safely_removable else "unsafe"
        header = Panel(
            Group(
                Text("Exterior-environment reduction", style="bold"),
                Text(f"removal status: {status}"),
                Text(f"mechanisms: {', '.join(self.removal_mechanisms) or 'none'}"),
            ),
            expand=False,
        )

        overview = Table(title="Environment-reduction overview")
        overview.add_column("quantity", style="bold")
        overview.add_column("value", justify="right")
        overview.add_row("support size", str(self.support_size))
        overview.add_row("Hilbert size", str(self.hilbert_size))
        overview.add_row("nontrivial boundary zeros", str(self.n_nontrivial_zeros))
        overview.add_row("distinct local patterns", str(self.n_distinct_local_patterns))
        overview.add_row(
            "environment domain", str(self.metadata.get("environment_domain_size", "n/a"))
        )

        mechanisms = Table(title="Exterior-removal mechanisms")
        mechanisms.add_column("mechanism", style="bold")
        mechanisms.add_column("count", justify="right")
        mechanisms.add_row(
            "no environment weight",
            str(self.removal_summary.n_no_environment_weight_probes),
        )
        mechanisms.add_row(
            "projective annihilation",
            str(self.removal_summary.n_projective_annihilation_probes),
        )
        mechanisms.add_row(
            "same local cancellation pattern",
            str(self.removal_summary.n_same_local_cancellation_pattern_probes),
        )
        mechanisms.add_row("unsafe / unexplained", str(self.removal_summary.n_unsafe_probes))

        diagnostics = Table(title="Residual diagnostics")
        diagnostics.add_column("quantity", style="bold")
        diagnostics.add_column("value", justify="right")
        diagnostics.add_row("complement targets", str(self.n_complement_targets))
        diagnostics.add_row(
            "unexplained complement targets",
            str(self.n_unexplained_complement_targets),
        )
        diagnostics.add_row(
            "max complement action norm",
            _format_float(self.max_complement_action_norm),
        )
        diagnostics.add_row(
            "max reduced action norm",
            _format_float(self.max_reduced_action_norm),
        )

        renderables: list[object] = [header, overview, mechanisms, diagnostics]
        if self.collective_cancellation_reports:
            collective = Table(title="Same-pattern collective cancellations")
            collective.add_column("group", justify="right")
            collective.add_column("size", justify="right")
            collective.add_column("zeros")
            collective.add_column("norm", justify="right")
            for report in self.collective_cancellation_reports:
                collective.add_row(
                    str(report.group_id),
                    str(report.group_size),
                    _format_index_preview(report.source_zero_indices),
                    _format_float(report.collective_action_norm),
                )
            renderables.append(collective)

        if self.metadata:
            renderables.append(
                _rich_key_value_section(
                    "Metadata",
                    [(key, value) for key, value in sorted(self.metadata.items())],
                )
            )

        if verbose:
            renderables.append(
                _rich_zero_reports_section(
                    self.zero_reports[:max_zero_reports],
                    n_hidden=max(0, len(self.zero_reports) - max_zero_reports),
                )
            )

        return Group(*renderables)

    def to_text(
        self,
        *,
        verbose: bool = False,
        max_zero_reports: int = 10,
        width: int = 120,
    ) -> str:
        """Return a plain-text Rich rendering of the environment report."""
        console = Console(record=True, width=width, force_terminal=False, color_system=None)
        console.print(self.to_rich(verbose=verbose, max_zero_reports=max_zero_reports))
        return console.export_text(styles=False).rstrip()

    def to_summary_dict(self) -> dict[str, dict[str, object]]:
        """Return a compact structured environment-reduction summary."""
        return {
            "Environment reduction": {
                "is safely removable": self.is_safely_removable,
                "mechanisms": self.removal_mechanisms,
                "nontrivial boundary zeros": self.n_nontrivial_zeros,
                "distinct local patterns": self.n_distinct_local_patterns,
            },
            "Mechanism counts": {
                "no environment weight": (self.removal_summary.n_no_environment_weight_probes),
                "projective annihilation": (self.removal_summary.n_projective_annihilation_probes),
                "same local cancellation pattern": (
                    self.removal_summary.n_same_local_cancellation_pattern_probes
                ),
                "unsafe": self.removal_summary.n_unsafe_probes,
            },
            "Residual diagnostics": {
                "complement targets": self.n_complement_targets,
                "unexplained complement targets": self.n_unexplained_complement_targets,
                "max complement action norm": self.max_complement_action_norm,
                "max reduced action norm": self.max_reduced_action_norm,
            },
        }


def _format_float(value: float) -> str:
    """Compact float formatting for human-readable reports."""
    if value == 0.0:
        return "0"

    abs_value = abs(value)
    if abs_value < 1e-4 or abs_value >= 1e4:
        return f"{value:.3e}"

    return f"{value:.6g}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_float(float(value))


def _format_index_preview(
    indices: NDArray[np.int64],
    *,
    max_items: int = 20,
) -> str:
    values = [int(value) for value in indices[:max_items]]
    suffix = "" if len(indices) <= max_items else f", ... +{len(indices) - max_items}"
    return f"{values}{suffix}"


def _format_index_tuple(
    indices: tuple[int, ...] | NDArray[np.int64],
    *,
    max_items: int = 20,
) -> str:
    array = np.asarray(indices, dtype=np.int64)
    return _format_index_preview(array, max_items=max_items)


def _rich_key_value_section(
    title: str,
    rows: list[tuple[str, object]],
) -> Table:
    table = Table.grid(padding=(0, 2))
    table.title = title
    table.add_column("field", style="bold")
    table.add_column("value")

    for key, value in rows:
        table.add_row(str(key), str(value))

    return table


def _rich_zero_reports_section(
    zero_reports: tuple[EnvironmentRemovalProbeReport, ...] | list[EnvironmentRemovalProbeReport],
    *,
    n_hidden: int,
) -> Group:
    renderables = [Text("Zero reports", style="bold")]

    for report_index, zero_report in enumerate(zero_reports):
        table = Table.grid(padding=(0, 2))
        table.title = f"[{report_index}] source zero {zero_report.zero_index}"
        table.add_column("field", style="bold")
        table.add_column("value")

        rows = [
            ("active neighbors", zero_report.active_neighbors.tolist()),
            ("probe mechanism", zero_report.probe_mechanism_label),
            ("q-empty probe", zero_report.is_q_empty),
            ("same-pattern zero closure", zero_report.is_closed_by_same_pattern_zeros),
            ("domain-blocked probe", zero_report.is_domain_blocked),
            ("projector-like probe", zero_report.is_projector_like),
            ("invalid/leakage probe", zero_report.is_invalid_probe),
            (
                "cancellation residual",
                _format_float(zero_report.cancellation_residual),
            ),
            ("local region size", zero_report.local_region_size),
            ("Q-sector weight", _format_float(zero_report.q_sector_weight)),
            (
                "complement action norm",
                _format_float(zero_report.complement_action_norm),
            ),
            ("complement targets", zero_report.complement_target_indices.tolist()),
            (
                "unexplained targets",
                zero_report.unexplained_complement_target_indices.tolist(),
            ),
            (
                "complement targets are known zeros",
                zero_report.complement_targets_are_known_zeros,
            ),
            ("source projector-like", zero_report.source_projector_like),
            ("trivial targets", zero_report.trivial_target_indices.tolist()),
            (
                "same-pattern IZ targets",
                zero_report.same_pattern_iz_target_indices.tolist(),
            ),
            (
                "projector-like IZ targets",
                zero_report.projector_like_iz_target_indices.tolist(),
            ),
            ("unexpected targets", zero_report.unexpected_target_indices.tolist()),
            ("has unexpected targets", zero_report.has_unexpected_targets),
            (
                "has nonzero complement action",
                zero_report.has_nonzero_complement_action,
            ),
            (
                "nonzero complement-action targets",
                zero_report.nonzero_complement_action_target_indices.tolist(),
            ),
            (
                "complement support inputs",
                zero_report.complement_support_indices.tolist(),
            ),
            (
                "complement contributing inputs",
                zero_report.complement_contributing_input_indices.tolist(),
            ),
            (
                "projector-annihilated inputs",
                zero_report.projector_like_annihilated_input_indices.tolist(),
            ),
            ("collective group id", zero_report.collective_cancellation_group_id),
            (
                "collective partners",
                zero_report.collective_cancellation_partner_zero_indices.tolist(),
            ),
            (
                "collective coefficient",
                zero_report.collective_cancellation_coefficient,
            ),
            (
                "collective cancellation norm",
                _format_float(zero_report.collective_cancellation_norm),
            ),
        ]

        for key, value in rows:
            table.add_row(str(key), str(value))

        renderables.append(table)

    if n_hidden > 0:
        renderables.append(Text(f"... {n_hidden} more zero reports omitted"))

    return Group(*renderables)


def _zero_indices_with_mechanism(
    zero_reports: list[EnvironmentRemovalProbeReport],
    mechanism: EnvironmentProbeDetailLabel,
) -> NDArray[np.int64]:
    return np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == mechanism
        ],
        dtype=np.int64,
    )


def _zero_indices_with_unexpected_target_failure(
    zero_reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.has_unexpected_targets],
        dtype=np.int64,
    )


def _zero_indices_with_nonzero_complement_action_failure(
    zero_reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.has_nonzero_complement_action],
        dtype=np.int64,
    )


def _zero_indices_with_source_projector_like(
    zero_reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.source_projector_like],
        dtype=np.int64,
    )


def _zero_indices_with_indirect_projector_like(
    zero_reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.int64]:
    return np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if (
                report.probe_mechanism_label == "projector_like"
                and not report.source_projector_like
            )
        ],
        dtype=np.int64,
    )


def _union_projector_like_annihilated_inputs(
    zero_reports: list[EnvironmentRemovalProbeReport],
) -> NDArray[np.int64]:
    arrays = [
        report.projector_like_annihilated_input_indices
        for report in zero_reports
        if report.source_projector_like
    ]

    if len(arrays) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)
