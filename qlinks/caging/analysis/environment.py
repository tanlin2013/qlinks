from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qlinks.caging.analysis.transitions import (
    LocalTransitionPattern,
    local_transition_pattern_signature,
    transition_pattern_key,
)
from qlinks.caging.results import CageState, cage_state_to_full_vector

EnvironmentRemovalMechanismLabel: TypeAlias = Literal[
    "no_environment_weight",
    "projective_annihilation",
    "same_local_cancellation_pattern",
    "unsafe",
]

# Mechanism label for the reduced IZ probe associated with a source zero.
#
# The label is attached to the probe Z_h^(R), not intrinsically to the
# source zero vertex or to the target vertices reached by the complement
# action.
EnvironmentProbeDetailLabel: TypeAlias = Literal[
    "q_empty",
    "closed_by_same_pattern_zeros",
    "domain_blocked",
    "projector_like",
    "collective_cancellation",
    "unexplained_leakage",
]
CollectiveCancellationMode: TypeAlias = Literal[
    "disabled",
    "same_local_pattern_sum",
    "same_local_pattern_nullspace",
]
ReducedIZMonitorDecomposition: TypeAlias = Literal[
    "single_sum",
    "exact_support",
    "connected_support",
]
SectorPolicy = Literal[
    "raise_if_disconnected",
    "infer_support_component",
    "ignore",
]


@dataclass(frozen=True, slots=True)
class EnvironmentReductionConfig:
    """Numerical parameters for exterior-environment reduction diagnostics."""

    amplitude_tolerance: float = 1e-10
    cancellation_tolerance: float = 1e-9
    action_tolerance: float = 1e-9
    sector_policy: SectorPolicy = "raise_if_disconnected"

    collective_cancellation_mode: CollectiveCancellationMode = "same_local_pattern_nullspace"
    collective_min_group_size: int = 2
    collective_relation_tolerance: float | None = None


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


@dataclass(frozen=True, slots=True)
class EnvironmentRemovalProbeReport:
    """Diagnostics for one source nontrivial interference zero.

    The field ``zero_index`` is the source zero h used to construct the
    reduced IZ probe Z_h^(R). The mechanism label describes the behavior of
    this source probe on the cage state.
    """

    # Source zero and parent-Hamiltonian cancellation data.
    zero_index: int
    active_neighbors: NDArray[np.int64]
    active_matrix_elements: NDArray[np.complex128]
    active_amplitudes: NDArray[np.complex128]
    cancellation_residual: float

    # Local reduced-operator geometry.
    common_mask: NDArray[np.bool_]
    local_mask: NDArray[np.bool_]
    local_transitions: tuple[LocalTransitionPattern, ...]

    # Operator-action diagnostics.
    q_sector_weight: float
    reduced_action_norm: float
    complement_action_norm: float

    # Complement target structure.
    complement_target_indices: NDArray[np.int64]
    explained_complement_target_indices: NDArray[np.int64]
    unexplained_complement_target_indices: NDArray[np.int64]
    complement_targets_are_known_zeros: bool

    # Complement target explanations.
    trivial_target_indices: NDArray[np.int64]
    same_pattern_iz_target_indices: NDArray[np.int64]
    projector_like_iz_target_indices: NDArray[np.int64]
    unexpected_target_indices: NDArray[np.int64]

    # Projector-like input diagnostics.
    complement_support_indices: NDArray[np.int64]
    complement_contributing_input_indices: NDArray[np.int64]
    projector_like_annihilated_input_indices: NDArray[np.int64]
    source_projector_like: bool

    # Invalid-probe diagnostics.
    has_unexpected_targets: bool
    has_nonzero_complement_action: bool
    unexpected_target_probe_failure_indices: NDArray[np.int64]
    nonzero_complement_action_target_indices: NDArray[np.int64]

    # Final source-probe label.
    probe_mechanism_label: EnvironmentProbeDetailLabel

    # Collective-cancellation diagnostics.
    collective_cancellation_group_id: int | None = None
    collective_cancellation_partner_zero_indices: NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    collective_cancellation_coefficient: complex = 0.0 + 0.0j
    collective_cancellation_norm: float = np.inf

    # Cached action of the reduced local operator on the analyzed cage state.
    # This is intentionally optional: older tests and hand-built reports can
    # leave it empty, in which case downstream code falls back to sparse
    # operator materialization.
    reduced_action_vector: NDArray[np.complex128] = field(
        default_factory=lambda: np.array([], dtype=np.complex128)
    )

    # Cached tuple form of ``np.flatnonzero(local_mask)``.  This keeps report
    # grouping/decomposition from repeatedly scanning the same boolean masks.
    local_variable_indices: tuple[int, ...] = ()

    @property
    def local_region_size(self) -> int:
        if self.local_variable_indices:
            return len(self.local_variable_indices)
        return int(np.count_nonzero(self.local_mask))

    @property
    def n_active_neighbors(self) -> int:
        return int(self.active_neighbors.size)

    @property
    def n_complement_targets(self) -> int:
        return int(self.complement_target_indices.size)

    @property
    def n_unexplained_complement_targets(self) -> int:
        return int(self.unexplained_complement_target_indices.size)

    @property
    def removal_mechanism(self) -> EnvironmentRemovalMechanismLabel:
        """Coarse physical mechanism by which this exterior probe is removed."""
        if self.probe_mechanism_label == "q_empty":
            return "no_environment_weight"
        if self.probe_mechanism_label in {"domain_blocked", "projector_like"}:
            return "projective_annihilation"
        if self.probe_mechanism_label in {
            "closed_by_same_pattern_zeros",
            "collective_cancellation",
        }:
            return "same_local_cancellation_pattern"
        return "unsafe"

    @property
    def is_safely_removable(self) -> bool:
        return self.removal_mechanism != "unsafe"

    @property
    def is_q_empty(self) -> bool:
        return self.probe_mechanism_label == "q_empty"

    @property
    def is_closed_by_same_pattern_zeros(self) -> bool:
        return self.probe_mechanism_label == "closed_by_same_pattern_zeros"

    @property
    def is_domain_blocked(self) -> bool:
        return self.probe_mechanism_label == "domain_blocked"

    @property
    def is_projector_like(self) -> bool:
        return self.probe_mechanism_label == "projector_like"

    @property
    def is_projector_blocked_family(self) -> bool:
        return self.probe_mechanism_label in {"domain_blocked", "projector_like"}

    @property
    def is_collective_cancellation(self) -> bool:
        return self.probe_mechanism_label == "collective_cancellation"

    @property
    def is_invalid_probe(self) -> bool:
        return self.probe_mechanism_label == "unexplained_leakage"

    @property
    def n_trivial_targets(self) -> int:
        return int(self.trivial_target_indices.size)

    @property
    def n_same_pattern_iz_targets(self) -> int:
        return int(self.same_pattern_iz_target_indices.size)

    @property
    def n_projector_like_iz_targets(self) -> int:
        return int(self.projector_like_iz_target_indices.size)

    @property
    def n_unexpected_targets(self) -> int:
        return int(self.unexpected_target_indices.size)

    @property
    def n_unexpected_target_probe_failures(self) -> int:
        return int(self.unexpected_target_probe_failure_indices.size)

    @property
    def n_nonzero_complement_action_targets(self) -> int:
        return int(self.nonzero_complement_action_target_indices.size)

    @property
    def n_complement_support_inputs(self) -> int:
        return int(self.complement_support_indices.size)

    @property
    def n_complement_contributing_inputs(self) -> int:
        return int(self.complement_contributing_input_indices.size)

    @property
    def n_projector_like_annihilated_inputs(self) -> int:
        return int(self.projector_like_annihilated_input_indices.size)


@dataclass(frozen=True, slots=True)
class CollectiveCancellationReport:
    """A group of reduced IZ probes whose complement leakages cancel together."""

    group_id: int
    source_zero_indices: NDArray[np.int64]
    coefficients: NDArray[np.complex128]
    individual_complement_action_norms: NDArray[np.float64]
    collective_action_norm: float
    collective_target_indices: NDArray[np.int64]
    local_mask: NDArray[np.bool_]
    local_region_size: int
    relation_kind: Literal["unit_sum", "nullspace"]
    grouping_kind: Literal["same_local_pattern"]

    @property
    def group_size(self) -> int:
        return int(self.source_zero_indices.size)


@dataclass(frozen=True, slots=True)
class EnvironmentRemovalSummary:
    """Summary of whether and how the exterior environment is removable.

    This summary is deliberately not a classification of the caged eigenstate.
    It only records the local mechanisms that justify deleting exterior
    environment degrees of freedom when constructing a bounded local caging
    operator.
    """

    n_no_environment_weight_probes: int = 0
    n_projective_annihilation_probes: int = 0
    n_same_local_cancellation_pattern_probes: int = 0
    n_unsafe_probes: int = 0
    n_projector_like_iz_targets: int = 0
    n_unexpected_targets: int = 0
    n_nonzero_complement_action_failures: int = 0

    @property
    def n_total_probes(self) -> int:
        return (
            self.n_no_environment_weight_probes
            + self.n_projective_annihilation_probes
            + self.n_same_local_cancellation_pattern_probes
            + self.n_unsafe_probes
        )

    @property
    def is_safely_removable(self) -> bool:
        return self.n_total_probes > 0 and self.n_unsafe_probes == 0

    @property
    def mechanisms_present(self) -> tuple[EnvironmentRemovalMechanismLabel, ...]:
        mechanisms: list[EnvironmentRemovalMechanismLabel] = []
        if self.n_no_environment_weight_probes:
            mechanisms.append("no_environment_weight")
        if self.n_projective_annihilation_probes:
            mechanisms.append("projective_annihilation")
        if self.n_same_local_cancellation_pattern_probes:
            mechanisms.append("same_local_cancellation_pattern")
        if self.n_unsafe_probes:
            mechanisms.append("unsafe")
        return tuple(mechanisms)


@dataclass(frozen=True, slots=True)
class ReducedIZProbeSupport:
    """Cached support data for one reduced IZ probe ``Z_h^(R)``."""

    zero_index: int
    mechanism_label: EnvironmentProbeDetailLabel
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
class ReducedIZMonitorComponentGroup:
    """Cached report-side plan for one reduced-IZ monitor component.

    The construction layer can consume these groups directly instead of
    rediscovering reduced-IZ supports and frustration-free decompositions.
    When available, ``state_action_vector`` stores the cached action of this
    component monitor on the analyzed cage state.
    """

    component_id: int
    decomposition: ReducedIZMonitorDecomposition
    zero_indices: tuple[int, ...]
    support_variables: tuple[int, ...]
    state_action_vector: NDArray[np.complex128] = field(
        default_factory=lambda: np.array([], dtype=np.complex128)
    )

    @property
    def n_terms(self) -> int:
        return len(self.zero_indices)

    @property
    def support_size(self) -> int:
        return len(self.support_variables)

    @property
    def has_state_action_vector(self) -> bool:
        return self.state_action_vector.size > 0


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


def diagnose_cage_environment_reduction(
    cage_state: CageState,
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    hilbert_size: int | None = None,
    sector_mask: NDArray[np.bool_] | None = None,
    config: EnvironmentReductionConfig | None = None,
) -> EnvironmentReductionReport:
    """Diagnose exterior-environment reduction for one compact cage state.

    Args:
        cage_state: Compact cage state returned by the caging solver.
        kinetic_matrix: Off-diagonal Hamiltonian or kinetic matrix used to
            identify interference zeros and local ``Z_h`` patterns.
        basis_configs: Integer array with shape ``(n_basis, n_variables)``.
            Rows are product-state configurations in the global constrained
            basis.
        hilbert_size: Full Hilbert-space dimension.  Defaults to
            ``basis_configs.shape[0]``.
        sector_mask: Optional mask selecting the sector used for local
            diagnostics.
        config: Numerical environment-reduction parameters.

    Returns:
        Environment-reduction report describing whether exterior degrees of
        freedom are safely removable and the mechanism used by each probe.
    """
    if config is None:
        config = EnvironmentReductionConfig()

    if hilbert_size is None:
        hilbert_size = int(basis_configs.shape[0])

    full_state = cage_state_to_full_vector(
        cage_state,
        hilbert_size=hilbert_size,
    )

    return diagnose_environment_reduction(
        full_state,
        kinetic_matrix=kinetic_matrix,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=config,
        metadata={
            "energy": cage_state.energy,
            "support_size": cage_state.support_size,
            "boundary_residual": cage_state.boundary_residual,
            "eigen_residual": cage_state.eigen_residual,
            "full_residual": cage_state.full_residual,
        },
    )


def diagnose_environment_reduction(
    full_state: NDArray[np.complex128],
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    sector_mask: NDArray[np.bool_] | None = None,
    config: EnvironmentReductionConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> EnvironmentReductionReport:
    """Diagnose exterior-environment reduction for a full Hilbert-space vector."""
    if config is None:
        config = EnvironmentReductionConfig()

    full_state = np.asarray(full_state, dtype=np.complex128)
    basis_configs = np.asarray(basis_configs)

    if basis_configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    hilbert_size = int(full_state.size)
    if basis_configs.shape[0] != hilbert_size:
        raise ValueError("basis_configs.shape[0] must match full_state.size.")

    kinetic_csr = sp.csr_array(kinetic_matrix)

    support_mask = np.abs(full_state) > config.amplitude_tolerance
    support_size = int(np.count_nonzero(support_mask))
    support_fraction = support_size / float(hilbert_size)

    active_state_indices = np.flatnonzero(support_mask).astype(np.int64, copy=False)

    domain_mask = _resolve_environment_domain_mask(
        kinetic_csr,
        support_mask=support_mask,
        sector_mask=sector_mask,
        config=config,
    )

    active_domain_indices = active_state_indices[domain_mask[active_state_indices]].astype(
        np.int64,
        copy=False,
    )

    config_to_index = _build_config_to_index(basis_configs)
    local_operator_contexts: dict[
        tuple[int, ...],
        _ReducedLocalOperatorApplicationContext,
    ] = {}

    active_frontier_zero_indices = _active_frontier_zero_indices(
        kinetic_csr,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_state_indices=active_domain_indices,
    )

    zero_reports = _find_nontrivial_interference_zeros(
        full_state,
        kinetic_csr,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_frontier_zero_indices=active_frontier_zero_indices,
        active_state_indices=active_state_indices,
        active_domain_indices=active_domain_indices,
        local_operator_contexts=local_operator_contexts,
        config=config,
    )

    trivial_zero_indices = _find_trivial_zero_indices(
        full_state,
        kinetic_csr,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_frontier_zero_indices=active_frontier_zero_indices,
    )

    zero_reports = _annotate_probe_mechanisms(
        zero_reports,
        trivial_zero_indices=trivial_zero_indices,
        config=config,
    )

    zero_reports, collective_cancellation_reports = _annotate_collective_cancellations(
        zero_reports,
        full_state=full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        domain_mask=domain_mask,
        active_domain_indices=active_domain_indices,
        local_operator_contexts=local_operator_contexts,
        config=config,
    )

    pattern_keys = {transition_pattern_key(report.local_transitions) for report in zero_reports}

    q_weights = np.array(
        [report.q_sector_weight for report in zero_reports],
        dtype=float,
    )
    reduced_norms = np.array(
        [report.reduced_action_norm for report in zero_reports],
        dtype=float,
    )
    complement_norms = np.array(
        [report.complement_action_norm for report in zero_reports],
        dtype=float,
    )
    n_complement_targets = sum(report.n_complement_targets for report in zero_reports)
    n_unexplained_complement_targets = sum(
        report.n_unexplained_complement_targets for report in zero_reports
    )
    n_trivial_targets = sum(report.n_trivial_targets for report in zero_reports)
    n_same_pattern_iz_targets = sum(report.n_same_pattern_iz_targets for report in zero_reports)
    n_projector_like_iz_targets = sum(report.n_projector_like_iz_targets for report in zero_reports)
    n_unexpected_targets = sum(report.n_unexpected_targets for report in zero_reports)
    unexpected_target_probe_failure_indices = _zero_indices_with_unexpected_target_failure(
        zero_reports
    )
    nonzero_complement_action_probe_failure_indices = (
        _zero_indices_with_nonzero_complement_action_failure(zero_reports)
    )
    q_empty_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "q_empty",
    )
    same_pattern_zero_closure_indices = _zero_indices_with_mechanism(
        zero_reports,
        "closed_by_same_pattern_zeros",
    )
    source_projector_like_probe_indices = _zero_indices_with_source_projector_like(zero_reports)
    indirect_projector_like_probe_indices = _zero_indices_with_indirect_projector_like(zero_reports)
    projector_like_annihilated_input_indices = _union_projector_like_annihilated_inputs(
        zero_reports
    )
    # domain_blocked_source_zero_indices = _zero_indices_with_mechanism(
    #     zero_reports,
    #     "domain_blocked",
    # )  # noqa: F841
    projector_like_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "projector_like",
    )
    collective_cancellation_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "collective_cancellation",
    )
    invalid_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "unexplained_leakage",
    )
    if len(zero_reports) == 0:
        fraction_removable = 0.0
    else:
        fraction_removable = float(np.mean([report.is_safely_removable for report in zero_reports]))

    reduced_iz_probe_supports = tuple(
        reduced_iz_probe_support_from_report(report) for report in zero_reports
    )
    reduced_iz_region_variable_indices = _reduced_iz_region_variables_from_supports(
        reduced_iz_probe_supports
    )
    default_reduced_iz_reports = select_reduced_iz_monitor_reports_from_zero_reports(
        tuple(zero_reports)
    )
    reduced_iz_monitor_component_groups = {
        decomposition: reduced_iz_component_groups_from_reports(
            default_reduced_iz_reports,
            decomposition=decomposition,
        )
        for decomposition in ("single_sum", "exact_support", "connected_support")
    }

    removal_summary = _environment_removal_summary(tuple(zero_reports))

    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault(
        "environment_domain_size",
        int(np.count_nonzero(domain_mask)),
    )
    metadata.setdefault(
        "environment_domain_fraction",
        float(np.count_nonzero(domain_mask)) / float(hilbert_size),
    )
    metadata.setdefault("sector_policy", config.sector_policy)

    return EnvironmentReductionReport(
        support_size=support_size,
        hilbert_size=hilbert_size,
        support_fraction=support_fraction,
        n_nontrivial_zeros=len(zero_reports),
        n_distinct_local_patterns=len(pattern_keys),
        n_complement_targets=n_complement_targets,
        n_unexplained_complement_targets=n_unexplained_complement_targets,
        fraction_probes_safely_removable=fraction_removable,
        n_q_empty_source_probes=int(q_empty_source_zero_indices.size),
        n_same_pattern_zero_closure_probes=int(same_pattern_zero_closure_indices.size),
        n_projector_like_source_probes=int(projector_like_source_zero_indices.size),
        n_invalid_source_probes=int(invalid_source_zero_indices.size),
        q_empty_source_zero_indices=q_empty_source_zero_indices,
        same_pattern_zero_closure_indices=(same_pattern_zero_closure_indices),
        projector_like_source_zero_indices=projector_like_source_zero_indices,
        n_collective_cancellation_source_probes=int(
            collective_cancellation_source_zero_indices.size
        ),
        collective_cancellation_source_zero_indices=(collective_cancellation_source_zero_indices),
        collective_cancellation_reports=collective_cancellation_reports,
        invalid_source_zero_indices=invalid_source_zero_indices,
        n_trivial_targets=n_trivial_targets,
        n_same_pattern_iz_targets=n_same_pattern_iz_targets,
        n_projector_like_iz_targets=n_projector_like_iz_targets,
        n_unexpected_targets=n_unexpected_targets,
        n_unexpected_target_probe_failures=int(unexpected_target_probe_failure_indices.size),
        n_nonzero_complement_action_probe_failures=int(
            nonzero_complement_action_probe_failure_indices.size
        ),
        unexpected_target_probe_failure_indices=(unexpected_target_probe_failure_indices),
        nonzero_complement_action_probe_failure_indices=(
            nonzero_complement_action_probe_failure_indices
        ),
        n_source_projector_like_probes=int(source_projector_like_probe_indices.size),
        n_indirect_projector_like_probes=int(indirect_projector_like_probe_indices.size),
        n_projector_like_annihilated_inputs=int(projector_like_annihilated_input_indices.size),
        source_projector_like_probe_indices=source_projector_like_probe_indices,
        indirect_projector_like_probe_indices=(indirect_projector_like_probe_indices),
        projector_like_annihilated_input_indices=(projector_like_annihilated_input_indices),
        mean_q_sector_weight=_safe_mean(q_weights),
        max_q_sector_weight=_safe_max(q_weights),
        mean_reduced_action_norm=_safe_mean(reduced_norms),
        max_reduced_action_norm=_safe_max(reduced_norms),
        mean_complement_action_norm=_safe_mean(complement_norms),
        max_complement_action_norm=_safe_max(complement_norms),
        zero_reports=tuple(zero_reports),
        reduced_iz_probe_supports=reduced_iz_probe_supports,
        reduced_iz_region_variable_indices=reduced_iz_region_variable_indices,
        reduced_iz_monitor_component_groups=reduced_iz_monitor_component_groups,
        removal_summary=removal_summary,
        metadata=metadata,
    )


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
    report: EnvironmentReductionReport,
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


def support_key_from_mask(local_mask: NDArray[np.bool_]) -> tuple[int, ...]:
    """Return the variable-index support key for a local reduced-IZ mask."""
    return tuple(int(index) for index in np.flatnonzero(local_mask))


def support_key_for_zero_report(
    zero_report: EnvironmentRemovalProbeReport,
) -> tuple[int, ...]:
    """Return the variable-index support key for one reduced-IZ report."""
    if zero_report.local_variable_indices:
        return zero_report.local_variable_indices
    return support_key_from_mask(zero_report.local_mask)


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
            "Unknown collective_cancellation_mode: " f"{config.collective_cancellation_mode!r}"
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
                "Unknown collective_cancellation_mode: " f"{config.collective_cancellation_mode!r}"
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


def _common_mask(
    configs: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """
    Return positions where all configurations agree.

    This is the numerical version of Lambda_h.
    """
    reference = configs[0]
    return np.all(configs == reference[None, :], axis=0)


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


def _collective_tolerance(config: EnvironmentReductionConfig) -> float:
    if config.collective_relation_tolerance is not None:
        return float(config.collective_relation_tolerance)
    return float(config.action_tolerance)


def _environment_removal_summary(
    zero_reports: tuple[EnvironmentRemovalProbeReport, ...] | list[EnvironmentRemovalProbeReport],
) -> EnvironmentRemovalSummary:
    """Summarize the three supported exterior-removal mechanisms."""
    counts = {
        "no_environment_weight": 0,
        "projective_annihilation": 0,
        "same_local_cancellation_pattern": 0,
        "unsafe": 0,
    }
    for report in zero_reports:
        counts[report.removal_mechanism] += 1

    return EnvironmentRemovalSummary(
        n_no_environment_weight_probes=counts["no_environment_weight"],
        n_projective_annihilation_probes=counts["projective_annihilation"],
        n_same_local_cancellation_pattern_probes=(counts["same_local_cancellation_pattern"]),
        n_unsafe_probes=counts["unsafe"],
        n_projector_like_iz_targets=sum(
            report.n_projector_like_iz_targets for report in zero_reports
        ),
        n_unexpected_targets=sum(report.n_unexpected_targets for report in zero_reports),
        n_nonzero_complement_action_failures=sum(
            report.has_nonzero_complement_action for report in zero_reports
        ),
    )


def _safe_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


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
            raise ValueError("The cage support is not contained in the provided " "sector_mask.")

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
