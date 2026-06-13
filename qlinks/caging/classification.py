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

from qlinks.caging.results import CageState, cage_state_to_full_vector

CageSpatialLabel: TypeAlias = Literal[
    "regional_candidate",
    "extended_candidate",
    "invalid_or_inconsistent",
]
# Mechanism label for the reduced IZ probe associated with a source zero.
#
# The label is attached to the probe Z_h^(R), not intrinsically to the
# source zero vertex or to the target vertices reached by the complement
# action.
IZProbeMechanismLabel: TypeAlias = Literal[
    "q_empty",
    "closed_by_known_zeros",
    "projector_like",
    "collective_cancellation",
    "unexplained_leakage",
]
IZTargetExplanationLabel: TypeAlias = Literal[
    "trivial_zero",
    "destructive_iz",
    "projector_like_iz",
    "unexpected",
]
CollectiveCancellationMode: TypeAlias = Literal[
    "disabled",
    "same_local_support_sum",
    "same_local_support_nullspace",
    "all_problematic_sum",
    "all_problematic_nullspace",
]
SectorPolicy = Literal[
    "raise_if_disconnected",
    "infer_support_component",
    "ignore",
]


@dataclass(frozen=True, slots=True)
class CageClassificationConfig:
    """Numerical parameters for regional/extended cage diagnostics."""

    amplitude_tolerance: float = 1e-10
    cancellation_tolerance: float = 1e-9
    action_tolerance: float = 1e-9
    sector_policy: SectorPolicy = "raise_if_disconnected"

    collective_cancellation_mode: CollectiveCancellationMode = "same_local_support_nullspace"
    collective_min_group_size: int = 2
    collective_relation_tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class LocalTransitionPattern:
    """
    Local transition induced by one active edge u -> h.

    The local mask represents Omega - Lambda_h.
    """

    source_local: tuple[int, ...]
    target_local: tuple[int, ...]
    matrix_element: complex


@dataclass(frozen=True, slots=True)
class InterferenceZeroReport:
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
    known_nonprojector_iz_target_indices: NDArray[np.int64]
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
    probe_mechanism_label: IZProbeMechanismLabel

    # Collective-cancellation diagnostics.
    collective_cancellation_group_id: int | None = None
    collective_cancellation_partner_zero_indices: NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    collective_cancellation_coefficient: complex = 0.0 + 0.0j
    collective_cancellation_norm: float = np.inf

    # Cached action of the reduced local operator on the classified cage state.
    # This is intentionally optional: older tests and hand-built reports can
    # leave it empty, in which case downstream code falls back to sparse
    # operator materialization.
    reduced_action_vector: NDArray[np.complex128] = field(
        default_factory=lambda: np.array([], dtype=np.complex128)
    )

    @property
    def local_region_size(self) -> int:
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
    def is_q_empty(self) -> bool:
        return self.probe_mechanism_label == "q_empty"

    @property
    def is_closed_by_known_zeros(self) -> bool:
        return self.probe_mechanism_label == "closed_by_known_zeros"

    @property
    def is_projector_like(self) -> bool:
        return self.probe_mechanism_label == "projector_like"

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
    def n_known_nonprojector_iz_targets(self) -> int:
        return int(self.known_nonprojector_iz_target_indices.size)

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
    grouping_kind: Literal["same_local_support", "all_problematic"]

    @property
    def group_size(self) -> int:
        return int(self.source_zero_indices.size)


@dataclass(frozen=True, slots=True)
class CageClassificationReport:
    """Regional/extended diagnostic report for one cage state."""

    # State-level label and support.
    label: CageSpatialLabel
    support_size: int
    hilbert_size: int
    support_fraction: float

    # IZ inventory.
    n_nontrivial_zeros: int
    n_distinct_local_patterns: int

    # Complement closure summary.
    n_complement_targets: int
    n_unexplained_complement_targets: int
    fraction_zeros_with_closed_complement_targets: float

    # Source-probe mechanism counts.
    n_q_empty_source_probes: int
    n_closed_by_known_zero_network_source_probes: int
    n_projector_like_source_probes: int
    n_invalid_source_probes: int
    n_regional_source_probes: int
    n_collective_cancellation_source_probes: int
    collective_cancellation_source_zero_indices: NDArray[np.int64]

    # Source-zero index groups.
    q_empty_source_zero_indices: NDArray[np.int64]
    closed_by_known_zero_network_source_zero_indices: NDArray[np.int64]
    projector_like_source_zero_indices: NDArray[np.int64]
    invalid_source_zero_indices: NDArray[np.int64]
    regional_source_zero_indices: NDArray[np.int64]

    # Complement target explanation counts.
    n_trivial_targets: int
    n_known_nonprojector_iz_targets: int
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
    zero_reports: tuple[InterferenceZeroReport, ...]
    collective_cancellation_reports: tuple[CollectiveCancellationReport, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "CageClassificationReport("
            f"label={self.label!r}, "
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

    def to_rich(
        self,
        *,
        verbose: bool = False,
        max_zero_reports: int = 10,
    ) -> Group:
        """Return a Rich renderable for this report."""
        renderables = [
            Panel(
                Group(
                    Text("Cage classification report", style="bold"),
                    Text(f"label: {self.label}"),
                ),
                expand=False,
            ),
            _rich_key_value_section(
                "Support",
                [
                    ("support size", self.support_size),
                    ("Hilbert size", self.hilbert_size),
                    ("support fraction", _format_float(self.support_fraction)),
                ],
            ),
            _rich_key_value_section(
                "Interference zeros",
                [
                    ("nontrivial zeros", self.n_nontrivial_zeros),
                    ("distinct local patterns", self.n_distinct_local_patterns),
                ],
            ),
            _rich_key_value_section(
                "Complement closure",
                [
                    ("complement targets", self.n_complement_targets),
                    (
                        "unexplained complement targets",
                        self.n_unexplained_complement_targets,
                    ),
                    (
                        "fraction zeros with closed complement targets",
                        _format_float(self.fraction_zeros_with_closed_complement_targets),
                    ),
                ],
            ),
            _rich_key_value_section(
                "Operator diagnostics",
                [
                    ("mean Q-sector weight", _format_float(self.mean_q_sector_weight)),
                    ("max Q-sector weight", _format_float(self.max_q_sector_weight)),
                    (
                        "mean reduced action norm",
                        _format_float(self.mean_reduced_action_norm),
                    ),
                    (
                        "max reduced action norm",
                        _format_float(self.max_reduced_action_norm),
                    ),
                    (
                        "mean complement action norm",
                        _format_float(self.mean_complement_action_norm),
                    ),
                    (
                        "max complement action norm",
                        _format_float(self.max_complement_action_norm),
                    ),
                ],
            ),
            _rich_key_value_section(
                "Reduced IZ probe mechanisms",
                [
                    ("q-empty source probes", self.n_q_empty_source_probes),
                    (
                        "closed-by-known-zero-network source probes",
                        self.n_closed_by_known_zero_network_source_probes,
                    ),
                    ("projector-like source probes", self.n_projector_like_source_probes),
                    (
                        "unexplained-leakage source probes",
                        self.n_invalid_source_probes,
                    ),
                    (
                        "collective-cancellation source probes",
                        self.n_collective_cancellation_source_probes,
                    ),
                ],
            ),
            _rich_key_value_section(
                "Collective cancellation",
                [
                    ("collective groups", len(self.collective_cancellation_reports)),
                    (
                        "collectively cancelled source zeros",
                        _format_index_preview(self.collective_cancellation_source_zero_indices),
                    ),
                ],
            ),
            _rich_key_value_section(
                "Invalid probe reasons",
                [
                    (
                        "unexpected-target source probes",
                        self.n_unexpected_target_probe_failures,
                    ),
                    (
                        "nonzero-complement-action source probes",
                        self.n_nonzero_complement_action_probe_failures,
                    ),
                ],
            ),
            _rich_key_value_section(
                "Complement target explanations",
                [
                    ("trivial zero targets", self.n_trivial_targets),
                    (
                        "known non-projector IZ targets",
                        self.n_known_nonprojector_iz_targets,
                    ),
                    (
                        "projector-like IZ targets",
                        self.n_projector_like_iz_targets,
                    ),
                    ("unexpected targets", self.n_unexpected_targets),
                ],
            ),
            _rich_key_value_section(
                "State-level interpretation",
                [
                    (
                        "has only regional mechanisms",
                        self.n_projector_like_source_probes == 0
                        and self.n_invalid_source_probes == 0,
                    ),
                    (
                        "contains projector-like extended mechanisms",
                        self.n_projector_like_source_probes > 0,
                    ),
                    (
                        "has invalid probe failures",
                        self.n_invalid_source_probes > 0,
                    ),
                ],
            ),
            _rich_key_value_section(
                "Mechanism zero indices",
                [
                    ("q-empty", _format_index_preview(self.q_empty_source_zero_indices)),
                    (
                        "closed-by-known-zero",
                        _format_index_preview(
                            self.closed_by_known_zero_network_source_zero_indices
                        ),
                    ),
                    (
                        "projector-like",
                        _format_index_preview(self.projector_like_source_zero_indices),
                    ),
                    (
                        "unexplained-leakage",
                        _format_index_preview(self.invalid_source_zero_indices),
                    ),
                    (
                        "unexpected-target failures",
                        _format_index_preview(self.unexpected_target_probe_failure_indices),
                    ),
                    (
                        "nonzero-complement-action failures",
                        _format_index_preview(self.nonzero_complement_action_probe_failure_indices),
                    ),
                ],
            ),
        ]

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
                    n_hidden=len(self.zero_reports)
                    - min(
                        len(self.zero_reports),
                        max_zero_reports,
                    ),
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
        """Return a plain-text Rich rendering of the classification report."""
        console = Console(
            record=True,
            width=width,
            force_terminal=False,
            color_system=None,
        )
        console.print(
            self.to_rich(
                verbose=verbose,
                max_zero_reports=max_zero_reports,
            )
        )
        return console.export_text(styles=False).rstrip()

    def to_summary_dict(self) -> dict[str, dict[str, object]]:
        """Structured summary used by text rendering and tests."""
        return {
            "Support": {
                "support size": self.support_size,
                "Hilbert size": self.hilbert_size,
                "support fraction": self.support_fraction,
            },
            "Interference zeros": {
                "nontrivial zeros": self.n_nontrivial_zeros,
                "distinct local patterns": self.n_distinct_local_patterns,
            },
            "Reduced IZ probe mechanisms": {
                "q-empty source probes": self.n_q_empty_source_probes,
                "closed-by-known-zero-network source probes": (
                    self.n_closed_by_known_zero_network_source_probes
                ),
                "projector-like source probes": self.n_projector_like_source_probes,
                "collective-cancellation source probes": (
                    self.n_collective_cancellation_source_probes
                ),
                "unexplained-leakage source probes": (self.n_invalid_source_probes),
            },
            "Invalid probe reasons": {
                "unexpected-target source probes": (self.n_unexpected_target_probe_failures),
                "nonzero-complement-action source probes": (
                    self.n_nonzero_complement_action_probe_failures
                ),
            },
            "Complement target explanations": {
                "trivial zero targets": self.n_trivial_targets,
                "known non-projector IZ targets": self.n_known_nonprojector_iz_targets,
                "projector-like IZ targets": self.n_projector_like_iz_targets,
                "unexpected targets": self.n_unexpected_targets,
            },
        }


def classify_cage_state(
    cage_state: CageState,
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    hilbert_size: int | None = None,
    sector_mask: NDArray[np.bool_] | None = None,
    config: CageClassificationConfig | None = None,
) -> CageClassificationReport:
    """
    Classify one cage state as regional-like, extended-like, or ambiguous.

    Parameters
    ----------
    cage_state:
        Compact cage state returned by the caging solver.
    kinetic_matrix:
        Off-diagonal Hamiltonian H0/K. This is used to identify
        interference zeros and define local Z_h patterns.
    basis_configs:
        Integer array of shape (n_basis, n_variables). Each row is the
        product-state/basis configuration. For QDM/QLM, the variables can
        be link occupations/fluxes. For spin chains, they can be site spins.
    hilbert_size:
        Full Hilbert-space dimension. Defaults to basis_configs.shape[0].
    config:
        Numerical classification parameters.
    """
    if config is None:
        config = CageClassificationConfig()

    if hilbert_size is None:
        hilbert_size = int(basis_configs.shape[0])

    full_state = cage_state_to_full_vector(
        cage_state,
        hilbert_size=hilbert_size,
    )

    return classify_full_state(
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


def classify_full_state(
    full_state: NDArray[np.complex128],
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    sector_mask: NDArray[np.bool_] | None = None,
    config: CageClassificationConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> CageClassificationReport:
    """Classify a full Hilbert-space vector."""
    if config is None:
        config = CageClassificationConfig()

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

    domain_mask = _resolve_classification_domain_mask(
        kinetic_csr,
        support_mask=support_mask,
        sector_mask=sector_mask,
        config=config,
    )

    config_to_index = _build_config_to_index(basis_configs)

    zero_reports = _find_nontrivial_interference_zeros(
        full_state,
        kinetic_csr,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        support_mask=support_mask,
        domain_mask=domain_mask,
        config=config,
    )

    trivial_zero_indices = _find_trivial_zero_indices(
        full_state,
        kinetic_csr,
        support_mask=support_mask,
        domain_mask=domain_mask,
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
        config=config,
    )

    label = _classify_from_zero_reports(
        zero_reports=zero_reports,
        config=config,
    )

    pattern_keys = {_local_pattern_key(report) for report in zero_reports}

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
    n_known_nonprojector_iz_targets = sum(
        report.n_known_nonprojector_iz_targets for report in zero_reports
    )
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
    closed_by_known_zero_network_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "closed_by_known_zeros",
    )
    source_projector_like_probe_indices = _zero_indices_with_source_projector_like(zero_reports)
    indirect_projector_like_probe_indices = _zero_indices_with_indirect_projector_like(zero_reports)
    projector_like_annihilated_input_indices = _union_projector_like_annihilated_inputs(
        zero_reports
    )
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
    regional_source_zero_indices = np.sort(
        np.concatenate(
            [
                q_empty_source_zero_indices,
                closed_by_known_zero_network_source_zero_indices,
            ]
        )
    ).astype(np.int64, copy=False)

    if len(zero_reports) == 0:
        fraction_closed = 0.0
    else:
        fraction_closed = float(
            np.mean([report.complement_targets_are_known_zeros for report in zero_reports])
        )

    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault(
        "classification_domain_size",
        int(np.count_nonzero(domain_mask)),
    )
    metadata.setdefault(
        "classification_domain_fraction",
        float(np.count_nonzero(domain_mask)) / float(hilbert_size),
    )
    metadata.setdefault("sector_policy", config.sector_policy)

    return CageClassificationReport(
        label=label,
        support_size=support_size,
        hilbert_size=hilbert_size,
        support_fraction=support_fraction,
        n_nontrivial_zeros=len(zero_reports),
        n_distinct_local_patterns=len(pattern_keys),
        n_complement_targets=n_complement_targets,
        n_unexplained_complement_targets=n_unexplained_complement_targets,
        fraction_zeros_with_closed_complement_targets=fraction_closed,
        n_q_empty_source_probes=int(q_empty_source_zero_indices.size),
        n_closed_by_known_zero_network_source_probes=int(
            closed_by_known_zero_network_source_zero_indices.size
        ),
        n_projector_like_source_probes=int(projector_like_source_zero_indices.size),
        n_invalid_source_probes=int(invalid_source_zero_indices.size),
        n_regional_source_probes=int(regional_source_zero_indices.size),
        q_empty_source_zero_indices=q_empty_source_zero_indices,
        closed_by_known_zero_network_source_zero_indices=(
            closed_by_known_zero_network_source_zero_indices
        ),
        projector_like_source_zero_indices=projector_like_source_zero_indices,
        n_collective_cancellation_source_probes=int(
            collective_cancellation_source_zero_indices.size
        ),
        collective_cancellation_source_zero_indices=(collective_cancellation_source_zero_indices),
        collective_cancellation_reports=collective_cancellation_reports,
        invalid_source_zero_indices=invalid_source_zero_indices,
        regional_source_zero_indices=regional_source_zero_indices,
        n_trivial_targets=n_trivial_targets,
        n_known_nonprojector_iz_targets=n_known_nonprojector_iz_targets,
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
        metadata=metadata,
    )


def _find_trivial_zero_indices(
    full_state: NDArray[np.complex128],
    kinetic_matrix: sp.csr_array,
    *,
    support_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
) -> set[int]:
    """Return zero-amplitude vertices with no active kinetic neighbors.

    A trivial zero is a zero-amplitude basis vertex that receives no
    direct contribution from the cage support under the parent kinetic
    Hamiltonian. Nontrivial IZs are handled separately.
    """
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
    config: CageClassificationConfig,
) -> list[InterferenceZeroReport]:
    """Find zero vertices with nontrivial cancellation from active neighbors."""
    reports: list[InterferenceZeroReport] = []

    for zero_index in range(full_state.size):
        if not domain_mask[zero_index]:
            continue

        if support_mask[zero_index]:
            continue

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
    config: CageClassificationConfig,
) -> InterferenceZeroReport:
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

    return InterferenceZeroReport(
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
        known_nonprojector_iz_target_indices=np.array([], dtype=np.int64),
        projector_like_iz_target_indices=np.array([], dtype=np.int64),
        unexpected_target_indices=np.array([], dtype=np.int64),
        probe_mechanism_label="unexplained_leakage",
        local_transitions=tuple(local_transitions),
        reduced_action_vector=reduced_action.astype(np.complex128, copy=True),
    )


def _annotate_probe_mechanisms(
    zero_reports: list[InterferenceZeroReport],
    *,
    trivial_zero_indices: set[int],
    config: CageClassificationConfig,
) -> list[InterferenceZeroReport]:
    """Annotate reduced IZ source probes with target explanations.

    The label belongs to the source probe Z_h^(R), not intrinsically to
    the target vertices.

    A source probe is projector_like if either:
      1. it directly has finite Q-sector weight but no complement targets;
      2. it closes onto a target IZ whose own source probe is
         projector-dependent.

    A source probe is closed_by_known_zeros only if all of its complement
    targets are either trivial zeros or non-projector-dependent known IZs.
    """
    known_zero_indices = {int(report.zero_index) for report in zero_reports}

    # First split each source probe's raw targets.
    trivial_targets_by_zero: dict[int, set[int]] = {}
    iz_targets_by_zero: dict[int, set[int]] = {}
    unexpected_targets_by_zero: dict[int, set[int]] = {}

    for report in zero_reports:
        source = int(report.zero_index)

        trivial_targets: set[int] = set()
        iz_targets: set[int] = set()
        unexpected_targets: set[int] = set()

        for target_index in report.complement_target_indices:
            target = int(target_index)

            if target in trivial_zero_indices:
                trivial_targets.add(target)
            elif target in known_zero_indices:
                iz_targets.add(target)
            else:
                unexpected_targets.add(target)

        trivial_targets_by_zero[source] = trivial_targets
        iz_targets_by_zero[source] = iz_targets
        unexpected_targets_by_zero[source] = unexpected_targets

    # Seed invalid and projector-dependent source probes.
    unexpected_target_failure_sources = {
        source for source, targets in unexpected_targets_by_zero.items() if len(targets) > 0
    }

    nonzero_complement_action_failure_sources = {
        int(report.zero_index) for report in zero_reports if report.has_nonzero_complement_action
    }

    invalid_sources = unexpected_target_failure_sources | nonzero_complement_action_failure_sources

    projector_dependent_sources = {
        int(report.zero_index) for report in zero_reports if report.source_projector_like
    }

    # Propagate projector-dependence backward:
    # if h closes onto h' and h' is projector-dependent, then h is also
    # projector-dependent.
    changed = True
    while changed:
        changed = False

        for source, iz_targets in iz_targets_by_zero.items():
            if source in projector_dependent_sources:
                continue

            if any(target in projector_dependent_sources for target in iz_targets):
                projector_dependent_sources.add(source)
                changed = True

    annotated_reports: list[InterferenceZeroReport] = []

    for report in zero_reports:
        source = int(report.zero_index)

        trivial_targets = trivial_targets_by_zero[source]
        iz_targets = iz_targets_by_zero[source]
        unexpected_targets = unexpected_targets_by_zero[source]
        has_unexpected_targets = source in unexpected_target_failure_sources
        has_nonzero_complement_action = source in nonzero_complement_action_failure_sources

        projector_like_iz_targets = {
            target for target in iz_targets if target in projector_dependent_sources
        }
        nonprojector_iz_targets = iz_targets - projector_like_iz_targets

        q_empty = report.q_sector_weight <= config.action_tolerance

        if source in invalid_sources:
            probe_mechanism_label: IZProbeMechanismLabel = "unexplained_leakage"
        elif source in projector_dependent_sources:
            probe_mechanism_label = "projector_like"
        elif q_empty:
            probe_mechanism_label = "q_empty"
        else:
            # At this point all targets are trivial zeros or
            # non-projector-dependent known IZs.
            probe_mechanism_label = "closed_by_known_zeros"

        explained_targets = sorted(
            trivial_targets | nonprojector_iz_targets | projector_like_iz_targets
        )

        annotated_reports.append(
            _replace_interference_zero_report(
                report,
                probe_mechanism_label=probe_mechanism_label,
                trivial_target_indices=np.array(
                    sorted(trivial_targets),
                    dtype=np.int64,
                ),
                known_nonprojector_iz_target_indices=np.array(
                    sorted(nonprojector_iz_targets),
                    dtype=np.int64,
                ),
                projector_like_iz_target_indices=np.array(
                    sorted(projector_like_iz_targets),
                    dtype=np.int64,
                ),
                unexpected_target_indices=np.array(
                    sorted(unexpected_targets),
                    dtype=np.int64,
                ),
                has_unexpected_targets=has_unexpected_targets,
                has_nonzero_complement_action=has_nonzero_complement_action,
                unexpected_target_probe_failure_indices=np.array(
                    sorted(unexpected_targets),
                    dtype=np.int64,
                ),
                nonzero_complement_action_target_indices=(
                    report.nonzero_complement_action_target_indices
                ),
                explained_complement_target_indices=np.array(
                    explained_targets,
                    dtype=np.int64,
                ),
                unexplained_complement_target_indices=np.array(
                    sorted(unexpected_targets),
                    dtype=np.int64,
                ),
                complement_targets_are_known_zeros=(
                    len(report.complement_target_indices) > 0 and len(unexpected_targets) == 0
                ),
            )
        )

    return annotated_reports


def _annotate_collective_cancellations(
    zero_reports: list[InterferenceZeroReport],
    *,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
) -> tuple[list[InterferenceZeroReport], tuple[CollectiveCancellationReport, ...]]:
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
        "same_local_support_sum",
        "same_local_support_nullspace",
    }:
        groups = _group_reports_by_local_support(candidates)
    elif config.collective_cancellation_mode in {
        "all_problematic_sum",
        "all_problematic_nullspace",
    }:
        groups = _group_all_problematic_reports(candidates)
    else:
        raise ValueError(
            "Unknown collective_cancellation_mode: " f"{config.collective_cancellation_mode!r}"
        )

    collective_reports: list[CollectiveCancellationReport] = []
    replacement_by_zero: dict[int, InterferenceZeroReport] = {}
    next_group_id = 0

    for grouped_reports in groups:
        if len(grouped_reports) < config.collective_min_group_size:
            continue

        grouping_kind = (
            "same_local_support"
            if config.collective_cancellation_mode == "same_local_support_nullspace"
            else "all_problematic"
        )

        if config.collective_cancellation_mode in {
            "same_local_support_sum",
            "all_problematic_sum",
        }:
            collective = _find_unit_sum_collective_cancellation(
                grouped_reports,
                group_id=next_group_id,
                full_state=full_state,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                domain_mask=domain_mask,
                config=config,
                grouping_kind=grouping_kind,
            )
        elif config.collective_cancellation_mode in {
            "same_local_support_nullspace",
            "all_problematic_nullspace",
        }:
            collective = _find_nullspace_collective_cancellation(
                grouped_reports,
                group_id=next_group_id,
                full_state=full_state,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                domain_mask=domain_mask,
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

            replacement_by_zero[int(source_zero)] = _replace_interference_zero_report(
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


def _replace_interference_zero_report(
    report: InterferenceZeroReport,
    **updates: object,
) -> InterferenceZeroReport:
    """Return a copy of an InterferenceZeroReport with updated fields."""
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
        "known_nonprojector_iz_target_indices": (report.known_nonprojector_iz_target_indices),
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
    }

    values.update(updates)

    return InterferenceZeroReport(**values)


def _complement_action_for_report(
    report: InterferenceZeroReport,
    *,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    action, target_indices, _input_indices = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        domain_mask=domain_mask,
        common_mask=report.common_mask,
        reference_config=basis_configs[int(report.zero_index)],
        local_mask=report.local_mask,
        local_transitions=report.local_transitions,
        use_complement_common_sector=True,
        amplitude_tolerance=config.amplitude_tolerance,
    )

    return action, target_indices


def _find_unit_sum_collective_cancellation_from_actions(
    reports: list[InterferenceZeroReport],
    actions: list[NDArray[np.complex128]],
    target_indices: list[NDArray[np.int64]],
    *,
    group_id: int,
    config: CageClassificationConfig,
    grouping_kind: Literal["same_local_support", "all_problematic"],
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
    reports: list[InterferenceZeroReport],
    actions: list[NDArray[np.complex128]],
    target_indices: list[NDArray[np.int64]],
    *,
    group_id: int,
    config: CageClassificationConfig,
    grouping_kind: Literal[
        "same_local_support",
        "all_problematic",
    ],
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
    reports: list[InterferenceZeroReport],
    *,
    group_id: int,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
    grouping_kind: Literal["same_local_support", "all_problematic"],
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
    reports: list[InterferenceZeroReport],
    *,
    group_id: int,
    full_state: NDArray[np.complex128],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    domain_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
    grouping_kind: Literal["same_local_support", "all_problematic"],
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
    reports: list[InterferenceZeroReport],
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


def _group_reports_by_local_support(
    reports: list[InterferenceZeroReport],
) -> list[list[InterferenceZeroReport]]:
    grouped: dict[tuple[int, ...], list[InterferenceZeroReport]] = {}

    for report in reports:
        key = tuple(int(index) for index in np.flatnonzero(report.local_mask))
        grouped.setdefault(key, []).append(report)

    return list(grouped.values())


def _group_all_problematic_reports(
    reports: list[InterferenceZeroReport],
) -> list[list[InterferenceZeroReport]]:
    if len(reports) == 0:
        return []
    return [reports]


def _q_sector_weight(
    full_state: NDArray[np.complex128],
    *,
    basis_configs: NDArray[np.integer],
    reference_config: NDArray[np.integer],
    common_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
) -> float:
    """
    Weight outside the common product-state sector.

    This estimates || Q_beta |psi> ||^2.
    """
    if np.count_nonzero(common_mask) == 0:
        return 1.0

    same_common_sector = np.all(
        basis_configs[:, common_mask] == reference_config[common_mask][None, :],
        axis=1,
    )
    complement_mask = ~same_common_sector

    amplitudes = full_state[complement_mask]
    active = np.abs(amplitudes) > config.amplitude_tolerance
    return float(np.sum(np.abs(amplitudes[active]) ** 2))


def _complement_support_indices(
    full_state: NDArray[np.complex128],
    *,
    basis_configs: NDArray[np.integer],
    reference_config: NDArray[np.integer],
    common_mask: NDArray[np.bool_],
    domain_mask: NDArray[np.bool_],
    amplitude_tolerance: float,
) -> NDArray[np.int64]:
    """Return finite-amplitude basis indices outside the beta common sector."""
    if np.count_nonzero(common_mask) == 0:
        complement_mask = np.ones(full_state.size, dtype=np.bool_)
    else:
        same_common_sector = np.all(
            basis_configs[:, common_mask] == reference_config[common_mask][None, :],
            axis=1,
        )
        complement_mask = ~same_common_sector

    active_mask = np.abs(full_state) > amplitude_tolerance

    return np.flatnonzero(complement_mask & active_mask & domain_mask).astype(
        np.int64,
        copy=False,
    )


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

    for source_index, source_config in enumerate(basis_configs):
        if not domain_mask[source_index]:
            continue

        source_amplitude = full_state[source_index]

        if abs(source_amplitude) <= amplitude_tolerance:
            continue

        if common_mask is not None:
            if reference_config is None:
                raise ValueError("reference_config is required when common_mask is used.")

            in_common_sector = np.all(source_config[common_mask] == reference_config[common_mask])

            if use_complement_common_sector and in_common_sector:
                continue

            if not use_complement_common_sector and not in_common_sector:
                continue

        source_local = _config_key(source_config[local_mask])
        matching_transitions = transitions_by_source.get(source_local)

        if matching_transitions is None:
            continue

        for transition in matching_transitions:
            target_config = np.array(source_config, copy=True)
            target_config[local_mask] = np.array(
                transition.target_local,
                dtype=target_config.dtype,
            )

            target_index = config_to_index.get(_config_key(target_config))
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


def _local_pattern_key(
    report: InterferenceZeroReport,
) -> tuple[tuple[int, ...], ...]:
    """Crude equivalence key for local Z_h patterns."""
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
    return round(float(np.real(value)), digits), round(float(np.imag(value)), digits)


def _collective_tolerance(config: CageClassificationConfig) -> float:
    if config.collective_relation_tolerance is not None:
        return float(config.collective_relation_tolerance)
    return float(config.action_tolerance)


def _classify_from_zero_reports(
    *,
    zero_reports: list[InterferenceZeroReport],
    config: CageClassificationConfig,
) -> CageSpatialLabel:
    """
    First-layer regional/extended classification.

    State-level rule
    ----------------
    invalid_or_inconsistent:
        At least one zero has unexplained leakage.

    extended_candidate:
        No unexplained leakage, but at least one zero is projector_like.

    regional_candidate:
        All zeros are regional mechanisms: q_empty or closed_by_known_zeros.
    """
    del config

    if len(zero_reports) == 0:
        return "invalid_or_inconsistent"

    if any(report.is_invalid_probe for report in zero_reports):
        return "invalid_or_inconsistent"

    if any(
        report.is_projector_like or report.is_collective_cancellation for report in zero_reports
    ):
        return "extended_candidate"

    if all(
        report.probe_mechanism_label in {"q_empty", "closed_by_known_zeros"}
        for report in zero_reports
    ):
        return "regional_candidate"

    return "invalid_or_inconsistent"


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


def _format_index_preview(
    indices: NDArray[np.int64],
    *,
    max_items: int = 20,
) -> str:
    values = [int(value) for value in indices[:max_items]]
    suffix = "" if len(indices) <= max_items else f", ... +{len(indices) - max_items}"
    return f"{values}{suffix}"


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
    zero_reports: tuple[InterferenceZeroReport, ...] | list[InterferenceZeroReport],
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
            ("closed known-zero-network probe", zero_report.is_closed_by_known_zeros),
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
                "known non-projector IZ targets",
                zero_report.known_nonprojector_iz_target_indices.tolist(),
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
    zero_reports: list[InterferenceZeroReport],
    mechanism: IZProbeMechanismLabel,
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
    zero_reports: list[InterferenceZeroReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.has_unexpected_targets],
        dtype=np.int64,
    )


def _zero_indices_with_nonzero_complement_action_failure(
    zero_reports: list[InterferenceZeroReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.has_nonzero_complement_action],
        dtype=np.int64,
    )


def _zero_indices_with_source_projector_like(
    zero_reports: list[InterferenceZeroReport],
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.source_projector_like],
        dtype=np.int64,
    )


def _zero_indices_with_indirect_projector_like(
    zero_reports: list[InterferenceZeroReport],
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
    zero_reports: list[InterferenceZeroReport],
) -> NDArray[np.int64]:
    arrays = [
        report.projector_like_annihilated_input_indices
        for report in zero_reports
        if report.source_projector_like
    ]

    if len(arrays) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _resolve_classification_domain_mask(
    kinetic_matrix: sp.csr_array,
    *,
    support_mask: NDArray[np.bool_],
    sector_mask: NDArray[np.bool_] | None,
    config: CageClassificationConfig,
) -> NDArray[np.bool_]:
    """Return the basis-domain mask used by the classifier.

    The classification domain is normally one topological sector or one
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
