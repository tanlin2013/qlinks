from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray

from qlinks.caging.results import CageState, cage_state_to_full_vector

CageSpatialLabel = Literal[
    "regional_candidate",
    "extended_candidate",
    "invalid_or_inconsistent",
]
ZeroMechanismLabel = Literal[
    "q_empty",
    "closed_by_known_zeros",
    "projector_like",
    "unexplained_leakage",
]
IZTargetExplanationLabel = Literal[
    "trivial_zero",
    "destructive_iz",
    "projector_like_iz",
    "unexpected",
]


@dataclass(frozen=True, slots=True)
class CageClassificationConfig:
    """Numerical parameters for regional/extended cage diagnostics."""

    amplitude_tolerance: float = 1e-10
    cancellation_tolerance: float = 1e-9
    action_tolerance: float = 1e-9


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
    """Diagnostics for one nontrivial interference zero."""

    zero_index: int
    active_neighbors: NDArray[np.int64]
    active_matrix_elements: NDArray[np.complex128]
    active_amplitudes: NDArray[np.complex128]

    cancellation_residual: float
    common_mask: NDArray[np.bool_]
    local_mask: NDArray[np.bool_]

    q_sector_weight: float
    reduced_action_norm: float
    complement_action_norm: float

    complement_target_indices: NDArray[np.int64]
    explained_complement_target_indices: NDArray[np.int64]
    unexplained_complement_target_indices: NDArray[np.int64]
    complement_targets_are_known_zeros: bool

    trivial_target_indices: NDArray[np.int64]
    destructive_iz_target_indices: NDArray[np.int64]
    projector_like_iz_target_indices: NDArray[np.int64]
    unexpected_target_indices: NDArray[np.int64]

    complement_support_indices: NDArray[np.int64]
    complement_contributing_input_indices: NDArray[np.int64]
    projector_like_annihilated_input_indices: NDArray[np.int64]

    has_unexpected_targets: bool
    has_nonzero_complement_action: bool
    unexpected_target_probe_failure_indices: NDArray[np.int64]
    nonzero_complement_action_target_indices: NDArray[np.int64]

    source_projector_like: bool
    mechanism_label: ZeroMechanismLabel
    local_transitions: tuple[LocalTransitionPattern, ...]

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
        return self.mechanism_label == "q_empty"

    @property
    def is_closed_by_known_zeros(self) -> bool:
        return self.mechanism_label == "closed_by_known_zeros"

    @property
    def is_projector_like(self) -> bool:
        return self.mechanism_label == "projector_like"

    @property
    def is_unexplained_leakage(self) -> bool:
        return self.mechanism_label == "unexplained_leakage"

    @property
    def n_trivial_targets(self) -> int:
        return int(self.trivial_target_indices.size)

    @property
    def n_destructive_iz_targets(self) -> int:
        return int(self.destructive_iz_target_indices.size)

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
class CageClassificationReport:
    """Regional/extended diagnostic report for one cage state."""

    label: CageSpatialLabel
    support_size: int
    hilbert_size: int
    support_fraction: float

    n_nontrivial_zeros: int
    n_distinct_local_patterns: int

    n_complement_targets: int
    n_unexplained_complement_targets: int
    fraction_zeros_with_closed_complement_targets: float

    n_q_empty_zeros: int
    n_closed_by_known_zero_zeros: int
    n_projector_like_zeros: int
    n_unexplained_leakage_zeros: int

    n_trivial_targets: int
    n_destructive_iz_targets: int
    n_projector_like_iz_targets: int
    n_unexpected_targets: int
    n_unexpected_target_probe_failures: int
    n_nonzero_complement_action_probe_failures: int

    unexpected_target_probe_failure_indices: NDArray[np.int64]
    nonzero_complement_action_probe_failure_indices: NDArray[np.int64]

    n_regional_mechanism_zeros: int
    n_extended_mechanism_zeros: int
    n_failure_mechanism_zeros: int

    q_empty_zero_indices: NDArray[np.int64]
    closed_by_known_zero_indices: NDArray[np.int64]
    projector_like_zero_indices: NDArray[np.int64]
    unexplained_leakage_zero_indices: NDArray[np.int64]

    n_source_projector_like_probes: int
    n_indirect_projector_like_probes: int
    n_projector_like_annihilated_inputs: int

    source_projector_like_probe_indices: NDArray[np.int64]
    indirect_projector_like_probe_indices: NDArray[np.int64]
    projector_like_annihilated_input_indices: NDArray[np.int64]

    regional_mechanism_zero_indices: NDArray[np.int64]
    extended_mechanism_zero_indices: NDArray[np.int64]
    failure_mechanism_zero_indices: NDArray[np.int64]

    mean_q_sector_weight: float
    max_q_sector_weight: float
    mean_reduced_action_norm: float
    max_reduced_action_norm: float
    mean_complement_action_norm: float
    max_complement_action_norm: float

    zero_reports: tuple[InterferenceZeroReport, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "CageClassificationReport("
            f"label={self.label!r}, "
            f"support_size={self.support_size}, "
            f"hilbert_size={self.hilbert_size}, "
            f"n_nontrivial_zeros={self.n_nontrivial_zeros}, "
            f"n_projector_like_zeros={self.n_projector_like_zeros}, "
            f"n_unexpected_target_probe_failures="
            f"{self.n_unexpected_target_probe_failures}, "
            f"n_nonzero_complement_action_probe_failures="
            f"{self.n_nonzero_complement_action_probe_failures}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_text()

    def to_text(
        self,
        *,
        verbose: bool = False,
        max_zero_reports: int = 10,
    ) -> str:
        """Return a human-readable multiline classification report."""
        lines = [
            "Cage classification report",
            "==========================",
            f"label: {self.label}",
            "",
            "Support",
            "-------",
            f"support size: {self.support_size}",
            f"Hilbert size: {self.hilbert_size}",
            f"support fraction: {_format_float(self.support_fraction)}",
            "",
            "Interference zeros",
            "------------------",
            f"nontrivial zeros: {self.n_nontrivial_zeros}",
            f"distinct local patterns: {self.n_distinct_local_patterns}",
            "",
            "Complement closure",
            "------------------",
            f"complement targets: {self.n_complement_targets}",
            ("unexplained complement targets: " f"{self.n_unexplained_complement_targets}"),
            (
                "fraction zeros with closed complement targets: "
                f"{_format_float(self.fraction_zeros_with_closed_complement_targets)}"
            ),
            "",
            "Operator diagnostics",
            "--------------------",
            f"mean Q-sector weight: {_format_float(self.mean_q_sector_weight)}",
            f"max Q-sector weight: {_format_float(self.max_q_sector_weight)}",
            ("mean reduced action norm: " f"{_format_float(self.mean_reduced_action_norm)}"),
            ("max reduced action norm: " f"{_format_float(self.max_reduced_action_norm)}"),
            ("mean complement action norm: " f"{_format_float(self.mean_complement_action_norm)}"),
            ("max complement action norm: " f"{_format_float(self.max_complement_action_norm)}"),
        ]
        lines.extend(
            [
                "",
                "Reduced IZ probe mechanisms",
                "---------------------------",
                f"q-empty source probes: {self.n_q_empty_zeros}",
                (
                    "closed-by-known-zero-network source probes: "
                    f"{self.n_closed_by_known_zero_zeros}"
                ),
                f"projector-like source probes: {self.n_projector_like_zeros}",
                ("unexplained-leakage source probes: " f"{self.n_unexplained_leakage_zeros}"),
                "",
                "Invalid probe reasons",
                "---------------------",
                ("unexpected-target source probes: " f"{self.n_unexpected_target_probe_failures}"),
                (
                    "nonzero-complement-action source probes: "
                    f"{self.n_nonzero_complement_action_probe_failures}"
                ),
                "",
                "Complement target explanations",
                "------------------------------",
                f"trivial zero targets: {self.n_trivial_targets}",
                f"destructive IZ targets: {self.n_destructive_iz_targets}",
                f"projector-like IZ targets: {self.n_projector_like_iz_targets}",
                f"unexpected targets: {self.n_unexpected_targets}",
            ]
        )
        lines.extend(
            [
                "",
                "Projector-like diagnostics",
                "--------------------------",
                ("direct projector-like source probes: " f"{self.n_source_projector_like_probes}"),
                (
                    "indirect projector-like source probes: "
                    f"{self.n_indirect_projector_like_probes}"
                ),
                (
                    "projector-annihilated complement inputs: "
                    f"{self.n_projector_like_annihilated_inputs}"
                ),
            ]
        )

        has_only_regional_mechanisms = (
            self.n_extended_mechanism_zeros == 0 and self.n_failure_mechanism_zeros == 0
        )

        lines.extend(
            [
                "",
                "State-level interpretation",
                "--------------------------",
                ("has only regional mechanisms: " f"{has_only_regional_mechanisms}"),
                (
                    "contains projector-like extended mechanisms: "
                    f"{self.n_extended_mechanism_zeros > 0}"
                ),
                ("has unexplained leakage: " f"{self.n_failure_mechanism_zeros > 0}"),
            ]
        )
        lines.extend(
            [
                "",
                "Mechanism zero indices",
                "----------------------",
                f"q-empty: {_format_index_preview(self.q_empty_zero_indices)}",
                (
                    "closed-by-known-zero: "
                    f"{_format_index_preview(self.closed_by_known_zero_indices)}"
                ),
                f"projector-like: {_format_index_preview(self.projector_like_zero_indices)}",
                (
                    "unexplained-leakage: "
                    f"{_format_index_preview(self.unexplained_leakage_zero_indices)}"
                ),
                (
                    "unexpected-target failures: "
                    f"{_format_index_preview(self.unexpected_target_probe_failure_indices)}"
                ),
                (
                    "nonzero-complement-action failures: "
                    f"{_format_index_preview(self.nonzero_complement_action_probe_failure_indices)}"
                ),
                (
                    "direct projector-like source probes: "
                    f"{_format_index_preview(self.source_projector_like_probe_indices)}"
                ),
                (
                    "indirect projector-like source probes: "
                    f"{_format_index_preview(self.indirect_projector_like_probe_indices)}"
                ),
                (
                    "projector-annihilated complement inputs: "
                    f"{_format_index_preview(self.projector_like_annihilated_input_indices)}"
                ),
            ]
        )

        if self.metadata:
            lines.extend(
                [
                    "",
                    "Metadata",
                    "--------",
                ]
            )
            for key, value in sorted(self.metadata.items()):
                lines.append(f"{key}: {value}")

        if verbose:
            lines.extend(
                [
                    "",
                    "Zero reports",
                    "------------",
                ]
            )

            zero_reports = self.zero_reports[:max_zero_reports]
            for report_index, zero_report in enumerate(zero_reports):
                lines.extend(
                    [
                        f"[{report_index}] zero index: {zero_report.zero_index}",
                        ("    active neighbors: " f"{zero_report.active_neighbors.tolist()}"),
                        ("    mechanism: " f"{zero_report.mechanism_label}"),
                        ("    regional mechanism: " f"{zero_report.is_q_empty}"),
                        ("    extended mechanism: " f"{zero_report.is_projector_like}"),
                        ("    closed known zeros: " f"{zero_report.is_closed_by_known_zeros}"),
                        ("    failure mechanism: " f"{zero_report.is_unexplained_leakage}"),
                        (
                            "    cancellation residual: "
                            f"{_format_float(zero_report.cancellation_residual)}"
                        ),
                        ("    local region size: " f"{zero_report.local_region_size}"),
                        ("    Q-sector weight: " f"{_format_float(zero_report.q_sector_weight)}"),
                        (
                            "    complement action norm: "
                            f"{_format_float(zero_report.complement_action_norm)}"
                        ),
                        (
                            "    complement targets: "
                            f"{zero_report.complement_target_indices.tolist()}"
                        ),
                        (
                            "    unexplained targets: "
                            f"{zero_report.unexplained_complement_target_indices.tolist()}"
                        ),
                        (
                            "    complement targets are known zeros: "
                            f"{zero_report.complement_targets_are_known_zeros}"
                        ),
                    ]
                )
                lines.extend(
                    [
                        f" mechanism: {zero_report.mechanism_label}",
                        (" source projector-like: " f"{zero_report.source_projector_like}"),
                        (" trivial targets: " f"{zero_report.trivial_target_indices.tolist()}"),
                        (
                            " destructive IZ targets: "
                            f"{zero_report.destructive_iz_target_indices.tolist()}"
                        ),
                        (
                            " projector-like IZ targets: "
                            f"{zero_report.projector_like_iz_target_indices.tolist()}"
                        ),
                        (
                            " unexpected targets: "
                            f"{zero_report.unexpected_target_indices.tolist()}"
                        ),
                        ("    has unexpected targets: " f"{zero_report.has_unexpected_targets}"),
                        (
                            "    has nonzero complement action: "
                            f"{zero_report.has_nonzero_complement_action}"
                        ),
                        (
                            "    nonzero complement-action targets: "
                            f"{zero_report.nonzero_complement_action_target_indices.tolist()}"
                        ),
                    ]
                )
                lines.extend(
                    [
                        ("    source projector-like: " f"{zero_report.source_projector_like}"),
                        (
                            "    complement support inputs: "
                            f"{zero_report.complement_support_indices.tolist()}"
                        ),
                        (
                            "    complement contributing inputs: "
                            f"{zero_report.complement_contributing_input_indices.tolist()}"
                        ),
                        (
                            "    projector-annihilated inputs: "
                            f"{zero_report.projector_like_annihilated_input_indices.tolist()}"
                        ),
                        (
                            "    projector-like IZ targets: "
                            f"{zero_report.projector_like_iz_target_indices.tolist()}"
                        ),
                    ]
                )

            n_hidden = len(self.zero_reports) - len(zero_reports)
            if n_hidden > 0:
                lines.append(f"... {n_hidden} more zero reports omitted")

        return "\n".join(lines)


def classify_cage_state(
    cage_state: CageState,
    *,
    kinetic_matrix: scipy_sparse.spmatrix | scipy_sparse.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    hilbert_size: int | None = None,
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
    kinetic_matrix: scipy_sparse.spmatrix | scipy_sparse.sparray | NDArray,
    basis_configs: NDArray[np.integer],
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

    kinetic_csr = scipy_sparse.csr_array(kinetic_matrix)

    support_mask = np.abs(full_state) > config.amplitude_tolerance
    support_size = int(np.count_nonzero(support_mask))
    support_fraction = support_size / float(hilbert_size)

    config_to_index = _build_config_to_index(basis_configs)

    zero_reports = _find_nontrivial_interference_zeros(
        full_state,
        kinetic_csr,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        support_mask=support_mask,
        config=config,
    )

    trivial_zero_indices = _find_trivial_zero_indices(
        full_state,
        kinetic_csr,
        support_mask=support_mask,
    )

    zero_reports = _annotate_probe_mechanisms(
        zero_reports,
        trivial_zero_indices=trivial_zero_indices,
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
    n_destructive_iz_targets = sum(report.n_destructive_iz_targets for report in zero_reports)
    n_projector_like_iz_targets = sum(report.n_projector_like_iz_targets for report in zero_reports)
    n_unexpected_targets = sum(report.n_unexpected_targets for report in zero_reports)
    unexpected_target_probe_failure_indices = _zero_indices_with_unexpected_target_failure(
        zero_reports
    )
    nonzero_complement_action_probe_failure_indices = (
        _zero_indices_with_nonzero_complement_action_failure(zero_reports)
    )
    q_empty_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "q_empty",
    )
    closed_by_known_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "closed_by_known_zeros",
    )
    source_projector_like_probe_indices = _zero_indices_with_source_projector_like(zero_reports)
    indirect_projector_like_probe_indices = _zero_indices_with_indirect_projector_like(zero_reports)
    projector_like_annihilated_input_indices = _union_projector_like_annihilated_inputs(
        zero_reports
    )
    projector_like_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "projector_like",
    )
    unexplained_leakage_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "unexplained_leakage",
    )

    regional_mechanism_zero_indices = _zero_indices_with_flag(
        zero_reports,
        flag="regional",
    )
    extended_mechanism_zero_indices = _zero_indices_with_flag(
        zero_reports,
        flag="extended",
    )
    failure_mechanism_zero_indices = _zero_indices_with_flag(
        zero_reports,
        flag="failure",
    )

    if len(zero_reports) == 0:
        fraction_closed = 0.0
    else:
        fraction_closed = float(
            np.mean([report.complement_targets_are_known_zeros for report in zero_reports])
        )

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
        n_q_empty_zeros=int(q_empty_zero_indices.size),
        n_closed_by_known_zero_zeros=int(closed_by_known_zero_indices.size),
        n_projector_like_zeros=int(projector_like_zero_indices.size),
        n_unexplained_leakage_zeros=int(unexplained_leakage_zero_indices.size),
        n_unexpected_target_probe_failures=int(unexpected_target_probe_failure_indices.size),
        n_nonzero_complement_action_probe_failures=int(
            nonzero_complement_action_probe_failure_indices.size
        ),
        unexpected_target_probe_failure_indices=unexpected_target_probe_failure_indices,
        nonzero_complement_action_probe_failure_indices=(
            nonzero_complement_action_probe_failure_indices
        ),
        n_regional_mechanism_zeros=int(regional_mechanism_zero_indices.size),
        n_extended_mechanism_zeros=int(extended_mechanism_zero_indices.size),
        n_failure_mechanism_zeros=int(failure_mechanism_zero_indices.size),
        n_trivial_targets=n_trivial_targets,
        n_destructive_iz_targets=n_destructive_iz_targets,
        n_projector_like_iz_targets=n_projector_like_iz_targets,
        n_unexpected_targets=n_unexpected_targets,
        q_empty_zero_indices=q_empty_zero_indices,
        closed_by_known_zero_indices=closed_by_known_zero_indices,
        n_source_projector_like_probes=int(source_projector_like_probe_indices.size),
        n_indirect_projector_like_probes=int(indirect_projector_like_probe_indices.size),
        n_projector_like_annihilated_inputs=int(projector_like_annihilated_input_indices.size),
        source_projector_like_probe_indices=source_projector_like_probe_indices,
        indirect_projector_like_probe_indices=(indirect_projector_like_probe_indices),
        projector_like_annihilated_input_indices=(projector_like_annihilated_input_indices),
        projector_like_zero_indices=projector_like_zero_indices,
        unexplained_leakage_zero_indices=unexplained_leakage_zero_indices,
        regional_mechanism_zero_indices=regional_mechanism_zero_indices,
        extended_mechanism_zero_indices=extended_mechanism_zero_indices,
        failure_mechanism_zero_indices=failure_mechanism_zero_indices,
        mean_q_sector_weight=_safe_mean(q_weights),
        max_q_sector_weight=_safe_max(q_weights),
        mean_reduced_action_norm=_safe_mean(reduced_norms),
        max_reduced_action_norm=_safe_max(reduced_norms),
        mean_complement_action_norm=_safe_mean(complement_norms),
        max_complement_action_norm=_safe_max(complement_norms),
        zero_reports=tuple(zero_reports),
        metadata={} if metadata is None else dict(metadata),
    )


def _find_trivial_zero_indices(
    full_state: NDArray[np.complex128],
    kinetic_matrix: scipy_sparse.csr_array,
    *,
    support_mask: NDArray[np.bool_],
) -> set[int]:
    """Return zero-amplitude vertices with no active kinetic neighbors.

    A trivial zero is a zero-amplitude basis vertex that receives no
    direct contribution from the cage support under the parent kinetic
    Hamiltonian. Nontrivial IZs are handled separately.
    """
    trivial_zero_indices: set[int] = set()

    for zero_index in range(full_state.size):
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
    kinetic_matrix: scipy_sparse.csr_array,
    *,
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    support_mask: NDArray[np.bool_],
    config: CageClassificationConfig,
) -> list[InterferenceZeroReport]:
    """Find zero vertices with nontrivial cancellation from active neighbors."""
    reports: list[InterferenceZeroReport] = []

    for zero_index in range(full_state.size):
        if support_mask[zero_index]:
            continue

        row_start = kinetic_matrix.indptr[zero_index]
        row_end = kinetic_matrix.indptr[zero_index + 1]

        neighbors = kinetic_matrix.indices[row_start:row_end]
        matrix_elements = kinetic_matrix.data[row_start:row_end]

        active_mask = support_mask[neighbors]
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

    reduced_action, _reduced_targets, _reduced_inputs = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        common_mask=None,
        reference_config=None,
        local_mask=local_mask,
        local_transitions=local_transitions,
        amplitude_tolerance=config.amplitude_tolerance,
    )

    complement_action, complement_target_indices, complement_contributing_input_indices = (
        _apply_reduced_local_operator(
            full_state,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            common_mask=common_mask,
            reference_config=basis_configs[zero_index],
            local_mask=local_mask,
            local_transitions=local_transitions,
            use_complement_common_sector=True,
            amplitude_tolerance=config.amplitude_tolerance,
        )
    )

    complement_support_indices = _complement_support_indices(
        full_state,
        basis_configs=basis_configs,
        reference_config=basis_configs[zero_index],
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

    source_projector_like = (
        q_sector_weight > config.action_tolerance
        and complement_target_indices.size == 0
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
        complement_action_norm=float(np.linalg.norm(complement_action)),
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
        destructive_iz_target_indices=np.array([], dtype=np.int64),
        projector_like_iz_target_indices=np.array([], dtype=np.int64),
        unexpected_target_indices=np.array([], dtype=np.int64),
        mechanism_label="unexplained_leakage",
        local_transitions=tuple(local_transitions),
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
        destructive_iz_targets = iz_targets - projector_like_iz_targets

        q_empty = report.q_sector_weight <= config.action_tolerance

        if source in invalid_sources:
            mechanism_label: ZeroMechanismLabel = "unexplained_leakage"
        elif source in projector_dependent_sources:
            mechanism_label = "projector_like"
        elif q_empty:
            mechanism_label = "q_empty"
        else:
            # At this point all targets are trivial zeros or
            # non-projector-dependent known IZs.
            mechanism_label = "closed_by_known_zeros"

        explained_targets = sorted(
            trivial_targets | destructive_iz_targets | projector_like_iz_targets
        )

        annotated_reports.append(
            _replace_interference_zero_report(
                report,
                mechanism_label=mechanism_label,
                trivial_target_indices=np.array(
                    sorted(trivial_targets),
                    dtype=np.int64,
                ),
                destructive_iz_target_indices=np.array(
                    sorted(destructive_iz_targets),
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
        "q_sector_weight": report.q_sector_weight,
        "reduced_action_norm": report.reduced_action_norm,
        "complement_action_norm": report.complement_action_norm,
        "complement_target_indices": report.complement_target_indices,
        "explained_complement_target_indices": (report.explained_complement_target_indices),
        "unexplained_complement_target_indices": (report.unexplained_complement_target_indices),
        "complement_targets_are_known_zeros": (report.complement_targets_are_known_zeros),
        "mechanism_label": report.mechanism_label,
        "has_unexpected_targets": report.has_unexpected_targets,
        "has_nonzero_complement_action": report.has_nonzero_complement_action,
        "unexpected_target_probe_failure_indices": (report.unexpected_target_probe_failure_indices),
        "nonzero_complement_action_target_indices": (
            report.nonzero_complement_action_target_indices
        ),
        "complement_support_indices": report.complement_support_indices,
        "complement_contributing_input_indices": (report.complement_contributing_input_indices),
        "projector_like_annihilated_input_indices": (
            report.projector_like_annihilated_input_indices
        ),
        "source_projector_like": report.source_projector_like,
        "trivial_target_indices": report.trivial_target_indices,
        "destructive_iz_target_indices": (report.destructive_iz_target_indices),
        "projector_like_iz_target_indices": (report.projector_like_iz_target_indices),
        "unexpected_target_indices": report.unexpected_target_indices,
        "local_transitions": report.local_transitions,
    }

    values.update(updates)

    return InterferenceZeroReport(**values)


def _annotate_zero_mechanisms(
    zero_reports: list[InterferenceZeroReport],
    *,
    config: CageClassificationConfig,
) -> list[InterferenceZeroReport]:
    """
    Annotate each nontrivial zero with its zero-level mechanism.

    Mechanism meanings
    ------------------
    q_empty:
        There is no Q-sector amplitude to transfer.

    closed_by_known_zeros:
        There is Q-sector amplitude, and the complement operator targets
        only known nontrivial interference zeros.

    projector_like:
        There is Q-sector amplitude, but the complement operator has no
        raw targets. This is the extended/projector-like mechanism.

    unexplained_leakage:
        There is Q-sector amplitude, and at least one complement target is
        not a known nontrivial zero. For a cage found by the caging solver,
        this should be treated as a failure/inconsistency, not as a valid
        extended mechanism.
    """
    known_zero_indices = {int(report.zero_index) for report in zero_reports}

    annotated_reports: list[InterferenceZeroReport] = []

    for report in zero_reports:
        explained: list[int] = []
        unexplained: list[int] = []

        for target_index in report.complement_target_indices:
            target = int(target_index)

            if target in known_zero_indices:
                explained.append(target)
            else:
                unexplained.append(target)

        n_targets = int(report.complement_target_indices.size)
        n_unexplained = len(unexplained)

        complement_targets_are_known_zeros = n_targets > 0 and n_unexplained == 0

        q_empty = report.q_sector_weight <= config.action_tolerance

        if q_empty:
            mechanism_label: ZeroMechanismLabel = "q_empty"
        elif complement_targets_are_known_zeros:
            mechanism_label = "closed_by_known_zeros"
        elif n_targets == 0:
            mechanism_label = "projector_like"
        else:
            mechanism_label = "unexplained_leakage"

        annotated_reports.append(
            InterferenceZeroReport(
                zero_index=report.zero_index,
                active_neighbors=report.active_neighbors,
                active_matrix_elements=report.active_matrix_elements,
                active_amplitudes=report.active_amplitudes,
                cancellation_residual=report.cancellation_residual,
                common_mask=report.common_mask,
                local_mask=report.local_mask,
                q_sector_weight=report.q_sector_weight,
                reduced_action_norm=report.reduced_action_norm,
                complement_action_norm=report.complement_action_norm,
                complement_target_indices=report.complement_target_indices,
                explained_complement_target_indices=np.array(
                    explained,
                    dtype=np.int64,
                ),
                unexplained_complement_target_indices=np.array(
                    unexplained,
                    dtype=np.int64,
                ),
                complement_targets_are_known_zeros=(complement_targets_are_known_zeros),
                mechanism_label=mechanism_label,
                local_transitions=report.local_transitions,
            )
        )

    return annotated_reports


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

    return np.flatnonzero(complement_mask & active_mask).astype(
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

    for source_index, source_config in enumerate(basis_configs):
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

        for transition in local_transitions:
            if source_local != transition.source_local:
                continue

            target_config = np.array(source_config, copy=True)
            target_config[local_mask] = np.array(
                transition.target_local,
                dtype=target_config.dtype,
            )

            target_index = config_to_index.get(_config_key(target_config))
            if target_index is None:
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

    if any(report.is_unexplained_leakage for report in zero_reports):
        return "invalid_or_inconsistent"

    if any(report.is_projector_like for report in zero_reports):
        return "extended_candidate"

    if all(
        report.mechanism_label in {"q_empty", "closed_by_known_zeros"} for report in zero_reports
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


def _zero_indices_with_mechanism(
    zero_reports: list[InterferenceZeroReport],
    mechanism: ZeroMechanismLabel,
) -> NDArray[np.int64]:
    return np.array(
        [int(report.zero_index) for report in zero_reports if report.mechanism_label == mechanism],
        dtype=np.int64,
    )


def _zero_indices_with_flag(
    zero_reports: list[InterferenceZeroReport],
    *,
    flag: Literal["regional", "extended", "failure"],
) -> NDArray[np.int64]:
    if flag == "regional":
        return np.array(
            [
                int(report.zero_index)
                for report in zero_reports
                if report.mechanism_label in {"q_empty", "closed_by_known_zeros"}
            ],
            dtype=np.int64,
        )

    if flag == "extended":
        return np.array(
            [
                int(report.zero_index)
                for report in zero_reports
                if report.mechanism_label == "projector_like"
            ],
            dtype=np.int64,
        )

    if flag == "failure":
        return np.array(
            [
                int(report.zero_index)
                for report in zero_reports
                if report.mechanism_label == "unexplained_leakage"
            ],
            dtype=np.int64,
        )

    raise ValueError(f"Unknown mechanism flag: {flag!r}")


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
            if (report.mechanism_label == "projector_like" and not report.source_projector_like)
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
