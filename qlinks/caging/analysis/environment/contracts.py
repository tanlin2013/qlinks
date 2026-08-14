from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.transitions import LocalTransitionPattern

EnvironmentRemovalMechanismLabel: TypeAlias = Literal[
    "no_environment_weight",
    "projective_annihilation",
    "same_local_cancellation_pattern",
    "unsafe",
]
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
