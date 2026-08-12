from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.local_structure.embedding import (
    _embed_local_pattern_operator_from_context,
    _embedding_context_from_basis_context,
    _LocalPatternEmbeddingContext,
)
from qlinks.local_structure.matrix_units import (
    LocalMatrixUnitTerm,
)
from qlinks.local_structure.reduced_density import (
    LocalReducedDensityMatrix,
    _local_pattern_basis_context_from_basis,
    _local_reduced_density_matrix_from_basis_context,
    _local_reduced_density_matrix_from_basis_context_and_states,
    _normalize_state_matrix_columns,
)

RecyclingJumpSource = Literal[
    "none",
    "local_rdm_rank_one",
    "local_rdm_two_pattern",
    "local_rdm_null_basis",
    "local_rdm_block_reset",
]


@dataclass(frozen=True, slots=True)
class TwoPatternRecyclingStructure:
    """Detected local two-pattern recycling structure.

    This represents a local jump of the form |minus><plus|, up to phase
    and convention.
    """

    variable_indices: tuple[int, ...]
    pattern_a: tuple[int, ...]
    pattern_b: tuple[int, ...]
    alpha_index: int
    beta_index: int
    phase: complex
    residual: float
    matrix_unit_terms: tuple[LocalMatrixUnitTerm, ...]

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)


@dataclass(frozen=True, slots=True)
class LocalRecyclingCandidate:
    """One embedded local RDM recycling jump candidate.

    Most candidates are rank-one maps ``|alpha><beta|`` between the local
    support and null spaces of the target reduced density matrix.  A compressed
    block-reset candidate can instead carry a higher-rank ``local_operator``
    that maps several null directions into the local target support with one
    Lindblad channel.  The ``local_alpha_vector`` and ``local_beta_vector``
    fields are retained for rank-one readouts and hold representative support
    and null vectors for block-reset candidates.
    """

    variable_indices: tuple[int, ...]
    alpha_index: int
    beta_index: int
    jump: sp.csr_array
    target_residual: float
    inflow_norm: float
    outflow_norm: float
    projector_commutator_norm: float
    local_alpha_vector: npt.NDArray[np.complex128]
    local_beta_vector: npt.NDArray[np.complex128]
    local_operator: npt.NDArray[np.complex128] | None = None

    @property
    def is_dark(self) -> bool:
        return self.target_residual < 1e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1e-10


@dataclass(frozen=True, slots=True)
class LocalRecyclingScanResult:
    """Candidate jumps from one local region."""

    reduced_density_matrix: LocalReducedDensityMatrix
    candidates: tuple[LocalRecyclingCandidate, ...]

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def best_candidates(self) -> tuple[LocalRecyclingCandidate, ...]:
        return tuple(
            sorted(
                self.candidates,
                key=lambda candidate: (
                    -float(candidate.inflow_norm),
                    float(candidate.target_residual),
                    int(candidate.jump.nnz),
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class LocalRecyclingSelection:
    """Selected recycling candidate plus optional detected structure."""

    candidate: LocalRecyclingCandidate
    two_pattern_structure: TwoPatternRecyclingStructure | None
    score: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LocalRecyclingBuildResult:
    """Selected recycling jumps from several local regions."""

    scan_results: tuple[LocalRecyclingScanResult, ...]
    selections: tuple[LocalRecyclingSelection, ...]

    @property
    def jumps(self) -> tuple[sp.csr_array, ...]:
        return tuple(selection.candidate.jump for selection in self.selections)

    @property
    def n_jumps(self) -> int:
        return len(self.selections)

    @property
    def variable_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(selection.candidate.variable_indices for selection in self.selections)

    @property
    def alpha_beta_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                int(selection.candidate.alpha_index),
                int(selection.candidate.beta_index),
            )
            for selection in self.selections
        )

    def to_subspace_support_report(self) -> "LocalSubspaceSupportReport":
        """Return a local-support/nullity report for the scanned regions."""
        return local_subspace_support_report_from_recycling_build_result(self)


@dataclass(frozen=True, slots=True)
class LocalSubspaceSupportReportEntry:
    """Local support/nullity diagnostics for one candidate region.

    The local RDM is the reduced density matrix of the normalized projector onto
    a target subspace.  ``nullity`` is the number of local source directions
    annihilated by the target manifold.  If it is zero, no strictly local
    right-detector ``D_R`` on this region can satisfy ``D_R P_M = 0``.
    """

    variable_indices: tuple[int, ...]
    local_dim: int
    support_rank: int
    nullity: int
    support_trace: float
    min_nonzero_eigenvalue: float | None
    max_eigenvalue: float | None
    n_candidate_jumps: int
    n_selected_jumps: int

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)

    @property
    def has_local_null_detector(self) -> bool:
        return self.nullity > 0

    @property
    def has_selected_recycler(self) -> bool:
        return self.n_selected_jumps > 0

    @property
    def parent_detector_directions(self) -> int:
        # Operators annihilating the local support have arbitrary output from
        # every local null-source direction.
        return int(self.local_dim * self.nullity)

    @property
    def minimum_block_reset_channels(self) -> int:
        if self.support_rank <= 0 or self.nullity <= 0:
            return 0
        return int((self.nullity + self.support_rank - 1) // self.support_rank)

    @property
    def status(self) -> str:
        if self.nullity == 0:
            return "full_local_support"
        if self.n_selected_jumps > 0:
            return "selected_recyclers"
        if self.n_candidate_jumps > 0:
            return "candidates_not_selected"
        return "null_detectors_no_inflow"

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "variable_indices": self.variable_indices,
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "support_rank": self.support_rank,
            "nullity": self.nullity,
            "support_trace": self.support_trace,
            "min_nonzero_eigenvalue": self.min_nonzero_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "parent_detector_directions": self.parent_detector_directions,
            "minimum_block_reset_channels": self.minimum_block_reset_channels,
            "n_candidate_jumps": self.n_candidate_jumps,
            "n_selected_jumps": self.n_selected_jumps,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class LocalSubspaceSupportReport:
    """Local-support report explaining manifold recycler availability."""

    entries: tuple[LocalSubspaceSupportReportEntry, ...]

    @property
    def n_regions(self) -> int:
        return len(self.entries)

    @property
    def n_regions_with_nullity(self) -> int:
        return sum(1 for entry in self.entries if entry.nullity > 0)

    @property
    def n_regions_with_selected_recyclers(self) -> int:
        return sum(1 for entry in self.entries if entry.n_selected_jumps > 0)

    @property
    def total_candidate_jumps(self) -> int:
        return sum(entry.n_candidate_jumps for entry in self.entries)

    @property
    def total_selected_jumps(self) -> int:
        return sum(entry.n_selected_jumps for entry in self.entries)

    @property
    def all_regions_have_full_local_support(self) -> bool:
        return self.n_regions > 0 and self.n_regions_with_nullity == 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_regions": self.n_regions,
            "n_regions_with_nullity": self.n_regions_with_nullity,
            "n_regions_with_selected_recyclers": self.n_regions_with_selected_recyclers,
            "total_candidate_jumps": self.total_candidate_jumps,
            "total_selected_jumps": self.total_selected_jumps,
            "all_regions_have_full_local_support": self.all_regions_have_full_local_support,
            "entries": [entry.to_summary_dict() for entry in self.entries],
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_regions: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "LocalSubspaceSupportReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("regions", str(self.n_regions))
        overview.add_row("regions with local nullity", str(self.n_regions_with_nullity))
        overview.add_row(
            "regions with selected recyclers",
            str(self.n_regions_with_selected_recyclers),
        )
        overview.add_row("candidate jumps", str(self.total_candidate_jumps))
        overview.add_row("selected jumps", str(self.total_selected_jumps))
        overview.add_row(
            "all full local support",
            str(self.all_regions_have_full_local_support),
        )

        table = Table(title="Local manifold support by region")
        table.add_column("region", style="bold")
        table.add_column("dim", justify="right")
        table.add_column("rank", justify="right")
        table.add_column("null", justify="right")
        table.add_column("parent dirs", justify="right")
        table.add_column("block min", justify="right")
        table.add_column("cand", justify="right")
        table.add_column("sel", justify="right")
        table.add_column("status")

        shown = self.entries[: max(int(max_regions), 0)]
        for entry in shown:
            table.add_row(
                str(entry.variable_indices),
                str(entry.local_dim),
                str(entry.support_rank),
                str(entry.nullity),
                str(entry.parent_detector_directions),
                str(entry.minimum_block_reset_channels),
                str(entry.n_candidate_jumps),
                str(entry.n_selected_jumps),
                _rich_status_for_local_subspace_entry(entry),
            )

        if len(self.entries) > len(shown):
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.entries) - len(shown)} more regions",
            )

        return Panel(
            Group(overview, table),
            title=Text("Local manifold-support report", style="bold cyan"),
            border_style="cyan",
        )


def local_subspace_support_report_from_recycling_build_result(
    build_result: LocalRecyclingBuildResult,
) -> LocalSubspaceSupportReport:
    """Summarize local RDM support/nullity and selected recyclers by region."""
    selected_counts: dict[tuple[int, ...], int] = {}
    for selection in build_result.selections:
        key = tuple(int(index) for index in selection.candidate.variable_indices)
        selected_counts[key] = selected_counts.get(key, 0) + 1

    entries: list[LocalSubspaceSupportReportEntry] = []
    for scan_result in build_result.scan_results:
        rdm = scan_result.reduced_density_matrix
        support_eigenvalues = tuple(float(value) for value in rdm.eigenvalues[-rdm.support_rank :])
        key = tuple(int(index) for index in rdm.variable_indices)
        entries.append(
            LocalSubspaceSupportReportEntry(
                variable_indices=key,
                local_dim=rdm.local_dim,
                support_rank=rdm.support_rank,
                nullity=rdm.nullity,
                support_trace=float(np.trace(rdm.density_matrix).real),
                min_nonzero_eigenvalue=(min(support_eigenvalues) if support_eigenvalues else None),
                max_eigenvalue=(max(support_eigenvalues) if support_eigenvalues else None),
                n_candidate_jumps=scan_result.n_candidates,
                n_selected_jumps=selected_counts.get(key, 0),
            )
        )

    return LocalSubspaceSupportReport(entries=tuple(entries))


def _rich_status_for_local_subspace_entry(entry: LocalSubspaceSupportReportEntry) -> str:
    if entry.status == "full_local_support":
        return "[yellow]full local support[/yellow]"
    if entry.status == "selected_recyclers":
        return "[green]selected[/green]"
    if entry.status == "candidates_not_selected":
        return "[yellow]not selected[/yellow]"
    return "[red]no inflow candidates[/red]"


def local_subspace_support_report_for_subspace(
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    regions: tuple[tuple[int, ...], ...],
    source: RecyclingJumpSource = "local_rdm_block_reset",
    deduplicate_regions: bool = False,
    max_jumps_per_region: int = 1,
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-12,
    max_candidates_per_region: int | None = None,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> LocalSubspaceSupportReport:
    """Build only the local manifold-support report for candidate regions."""
    return local_subspace_support_report_from_recycling_build_result(
        build_local_recycling_jumps_from_subspace_regions(
            basis_configs=basis_configs,
            states=states,
            regions=regions,
            source=source,
            deduplicate_regions=deduplicate_regions,
            max_jumps_per_region=max_jumps_per_region,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_candidates_per_region=max_candidates_per_region,
            prefer_sparse=prefer_sparse,
            two_pattern_tolerance=two_pattern_tolerance,
        )
    )


def score_recycling_jump_for_subspace(
    *,
    jump: Any,
    states: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> tuple[float, float, float, float]:
    """Return residual/inflow/outflow diagnostics for a target subspace.

    The target subspace is represented by an orthonormal basis ``Q``.  The
    returned values are Frobenius norms of ``J Q`` and ``J^† Q``; when ``J Q=0``
    the second norm equals the direct inflow block ``Q_perp J^† Q``.
    """
    jump_matrix = jump.tocsr() if hasattr(jump, "tocsr") else sp.csr_array(jump)
    dim = int(jump_matrix.shape[1])
    state_basis = _normalize_state_matrix_columns(states, dim=dim, tolerance=tolerance)

    target_action = np.asarray(jump_matrix @ state_basis, dtype=np.complex128)
    adjoint_target_action = np.asarray(jump_matrix.conj().T @ state_basis, dtype=np.complex128)
    overlap = state_basis.conj().T @ target_action

    target_norm_sq = float(np.linalg.norm(target_action) ** 2)
    adjoint_norm_sq = float(np.linalg.norm(adjoint_target_action) ** 2)
    overlap_norm_sq = float(np.linalg.norm(overlap) ** 2)

    target_residual = float(np.sqrt(max(target_norm_sq, 0.0)))
    inflow_norm = float(np.sqrt(max(adjoint_norm_sq - overlap_norm_sq, 0.0)))
    outflow_norm = float(np.sqrt(max(target_norm_sq - overlap_norm_sq, 0.0)))
    projector_commutator_norm = float(
        np.sqrt(max(target_norm_sq + adjoint_norm_sq - 2.0 * overlap_norm_sq, 0.0))
    )

    return (
        target_residual,
        inflow_norm,
        outflow_norm,
        projector_commutator_norm,
    )


def score_recycling_jump(
    *,
    jump: Any,
    target_state: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Return target residual, inflow, outflow, and projector commutator.

    The diagnostics are Frobenius norms of the corresponding projected
    operators.  They can be evaluated from ``J|psi>`` and ``J^dagger|psi>``
    without materializing the dense projectors ``|psi><psi|`` and
    ``I-|psi><psi|``.
    """
    state = np.asarray(target_state, dtype=np.complex128)
    norm = np.linalg.norm(state)

    if norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    state = state / norm

    if sp.issparse(jump):
        target_vector = np.asarray(jump @ state, dtype=np.complex128)
        adjoint_target_vector = np.asarray(jump.conj().T @ state, dtype=np.complex128)
    else:
        jump_array = np.asarray(jump, dtype=np.complex128)
        target_vector = np.asarray(jump_array @ state, dtype=np.complex128)
        adjoint_target_vector = np.asarray(jump_array.conj().T @ state, dtype=np.complex128)

    expectation = complex(np.vdot(state, target_vector))
    expectation_norm_sq = abs(expectation) ** 2
    target_norm_sq = float(np.vdot(target_vector, target_vector).real)
    adjoint_target_norm_sq = float(np.vdot(adjoint_target_vector, adjoint_target_vector).real)

    target_residual = float(np.sqrt(max(target_norm_sq, 0.0)))
    inflow_norm = float(np.sqrt(max(adjoint_target_norm_sq - expectation_norm_sq, 0.0)))
    outflow_norm = float(np.sqrt(max(target_norm_sq - expectation_norm_sq, 0.0)))
    projector_commutator_norm = float(
        np.sqrt(max(target_norm_sq + adjoint_target_norm_sq - 2.0 * expectation_norm_sq, 0.0))
    )

    return (
        target_residual,
        inflow_norm,
        outflow_norm,
        projector_commutator_norm,
    )


def scan_local_recycling_candidates(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
    max_candidates: int | None = None,
) -> LocalRecyclingScanResult:
    """Scan local rank-one recycling jumps from rho_Omega."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context(
        context=basis_context,
        state=target_state,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_eigenvalues = reduced_density_matrix.eigenvalues[
        reduced_density_matrix.eigenvalues > rdm_tolerance
    ]

    candidates: list[LocalRecyclingCandidate] = []

    if support_basis.shape[1] == 0 or null_basis.shape[1] == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    embedding_context = _embedding_context_from_basis_context(basis_context)

    for alpha_index in range(support_basis.shape[1]):
        alpha_vector = support_basis[:, alpha_index]

        for beta_index in range(null_basis.shape[1]):
            beta_vector = null_basis[:, beta_index]
            local_operator = np.outer(alpha_vector, beta_vector.conj())

            jump = _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            )

            inflow_norm = float(np.sqrt(max(float(support_eigenvalues[alpha_index]), 0.0)))
            target_residual = 0.0
            outflow_norm = 0.0
            projector_commutator_norm = inflow_norm

            if target_residual > dark_tolerance:
                continue

            if inflow_norm <= inflow_tolerance:
                continue

            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=int(alpha_index),
                    beta_index=int(beta_index),
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector.astype(np.complex128),
                    local_beta_vector=beta_vector.astype(np.complex128),
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def _detect_two_pattern_recycling_structure_from_vectors(
    *,
    variable_indices: tuple[int, ...],
    alpha_index: int,
    beta_index: int,
    local_patterns: tuple[tuple[int, ...], ...],
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    tolerance: float = 1e-8,
) -> TwoPatternRecyclingStructure | None:
    alpha_array = np.asarray(alpha, dtype=np.complex128)
    beta_array = np.asarray(beta, dtype=np.complex128)

    if alpha_array.ndim != 1 or beta_array.ndim != 1:
        raise ValueError("alpha and beta must be one-dimensional.")

    if alpha_array.shape != beta_array.shape:
        raise ValueError("alpha and beta must have the same shape.")

    if alpha_array.size != len(local_patterns):
        raise ValueError("alpha/beta size must match the number of local patterns.")

    alpha_support = np.flatnonzero(np.abs(alpha_array) > tolerance)
    beta_support = np.flatnonzero(np.abs(beta_array) > tolerance)

    if alpha_support.size != 2 or beta_support.size != 2:
        return None

    if set(int(index) for index in alpha_support) != set(int(index) for index in beta_support):
        return None

    pattern_indices = tuple(
        sorted((int(index) for index in alpha_support), key=lambda index: local_patterns[index])
    )
    pattern_a = local_patterns[pattern_indices[0]]
    pattern_b = local_patterns[pattern_indices[1]]

    coefficients = np.asarray(
        [
            alpha_array[pattern_indices[0]] * beta_array[pattern_indices[0]].conj(),
            alpha_array[pattern_indices[0]] * beta_array[pattern_indices[1]].conj(),
            alpha_array[pattern_indices[1]] * beta_array[pattern_indices[0]].conj(),
            alpha_array[pattern_indices[1]] * beta_array[pattern_indices[1]].conj(),
        ],
        dtype=np.complex128,
    )

    templates = (
        np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.complex128) / 2.0,
        np.asarray([-1.0, -1.0, 1.0, 1.0], dtype=np.complex128) / 2.0,
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.complex128) / 2.0,
        np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.complex128) / 2.0,
    )

    best_phase = 0.0 + 0.0j
    best_residual = np.inf

    for template in templates:
        overlap = np.vdot(template, coefficients)

        if abs(overlap) <= tolerance:
            continue

        phase = overlap / abs(overlap)
        residual = float(np.linalg.norm(coefficients - phase * template))

        if residual < best_residual:
            best_residual = residual
            best_phase = complex(phase)

    if best_residual > tolerance:
        return None

    terms = tuple(
        LocalMatrixUnitTerm(
            coefficient=complex(alpha_array[target_index] * beta_array[source_index].conj()),
            target_pattern=tuple(int(value) for value in local_patterns[target_index]),
            source_pattern=tuple(int(value) for value in local_patterns[source_index]),
        )
        for target_index in pattern_indices
        for source_index in pattern_indices
    )

    return TwoPatternRecyclingStructure(
        variable_indices=tuple(int(index) for index in variable_indices),
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        alpha_index=int(alpha_index),
        beta_index=int(beta_index),
        phase=best_phase,
        residual=best_residual,
        matrix_unit_terms=terms,
    )


def detect_two_pattern_recycling_structure(
    *,
    candidate: LocalRecyclingCandidate,
    local_patterns: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-8,
) -> TwoPatternRecyclingStructure | None:
    """Detect whether a candidate is a two-pattern |minus><plus| jump."""
    return _detect_two_pattern_recycling_structure_from_vectors(
        variable_indices=candidate.variable_indices,
        alpha_index=candidate.alpha_index,
        beta_index=candidate.beta_index,
        local_patterns=local_patterns,
        alpha=candidate.local_alpha_vector,
        beta=candidate.local_beta_vector,
        tolerance=tolerance,
    )


def select_local_recycling_candidates(
    *,
    scan_result: LocalRecyclingScanResult,
    source: RecyclingJumpSource = "local_rdm_two_pattern",
    max_candidates: int = 1,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> tuple[LocalRecyclingSelection, ...]:
    """Select recycling candidates from one scan result.

    ``local_rdm_rank_one`` and ``local_rdm_two_pattern`` keep the historical
    behavior: they choose the best few rank-one reset maps ``|alpha><beta|``.

    ``local_rdm_null_basis`` is designed for monitor-recycler jumps
    ``L=V P``.  A single rank-one recycler can make ``V P`` much more singular
    than the monitor ``P`` itself, because it only tests one local ``beta``
    direction.  This source instead selects one good target-support vector
    ``alpha`` for every local-RDM null vector ``beta``.

    ``local_rdm_block_reset`` is the compressed version: it groups up to
    ``rank(rho_R)`` null vectors into each reset channel.  The
    ``max_candidates`` argument is intentionally ignored for both null-basis and
    block-reset sources, because their counts are determined by the local RDM
    ranks.
    """
    if source == "none":
        return ()

    if source == "local_rdm_block_reset":
        selections = []
        for candidate in scan_result.candidates:
            nnz = candidate.jump.nnz if hasattr(candidate.jump, "nnz") else np.inf
            selections.append(
                LocalRecyclingSelection(
                    candidate=candidate,
                    two_pattern_structure=None,
                    score=(
                        int(candidate.beta_index),
                        float(candidate.target_residual),
                        float(nnz) if prefer_sparse else 0.0,
                    ),
                )
            )
        return tuple(sorted(selections, key=lambda selection: selection.score))

    if source == "local_rdm_null_basis":
        best_by_beta: dict[int, LocalRecyclingSelection] = {}

        for candidate in scan_result.candidates:
            structure = detect_two_pattern_recycling_structure(
                candidate=candidate,
                local_patterns=scan_result.reduced_density_matrix.local_patterns,
                tolerance=two_pattern_tolerance,
            )
            nnz = candidate.jump.nnz if hasattr(candidate.jump, "nnz") else np.inf
            score = (
                -float(candidate.inflow_norm),
                float(candidate.target_residual),
                float(nnz) if prefer_sparse else 0.0,
                int(candidate.alpha_index),
            )
            selection = LocalRecyclingSelection(
                candidate=candidate,
                two_pattern_structure=structure,
                score=score,
            )
            previous = best_by_beta.get(int(candidate.beta_index))
            if previous is None or selection.score < previous.score:
                best_by_beta[int(candidate.beta_index)] = selection

        return tuple(best_by_beta[index] for index in sorted(best_by_beta))

    selections: list[LocalRecyclingSelection] = []

    for candidate in scan_result.candidates:
        structure = detect_two_pattern_recycling_structure(
            candidate=candidate,
            local_patterns=scan_result.reduced_density_matrix.local_patterns,
            tolerance=two_pattern_tolerance,
        )

        if source == "local_rdm_two_pattern" and structure is None:
            continue

        nnz = candidate.jump.nnz if hasattr(candidate.jump, "nnz") else np.inf

        score = (
            0.0 if structure is not None else 1.0,
            float(nnz) if prefer_sparse else 0.0,
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
        )

        selections.append(
            LocalRecyclingSelection(
                candidate=candidate,
                two_pattern_structure=structure,
                score=score,
            )
        )

    selections = sorted(selections, key=lambda selection: selection.score)

    return tuple(selections[:max_candidates])


def _two_pattern_support_indices(
    vector: npt.ArrayLike,
    *,
    tolerance: float,
) -> tuple[int, int] | None:
    support = np.flatnonzero(np.abs(np.asarray(vector, dtype=np.complex128)) > tolerance)

    if support.size != 2:
        return None

    return int(support[0]), int(support[1])


def _scan_local_two_pattern_recycling_candidates(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
    two_pattern_tolerance: float = 1e-8,
) -> LocalRecyclingScanResult:
    """Scan only two-pattern local RDM recycling candidates.

    This avoids embedding/scoring every rank-one support-null pair when the
    caller will discard all non-two-pattern candidates anyway.
    """
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context(
        context=basis_context,
        state=target_state,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_eigenvalues = reduced_density_matrix.eigenvalues[
        reduced_density_matrix.eigenvalues > rdm_tolerance
    ]

    candidates: list[LocalRecyclingCandidate] = []

    if support_basis.shape[1] == 0 or null_basis.shape[1] == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    alpha_two_pattern_supports = tuple(
        _two_pattern_support_indices(
            support_basis[:, alpha_index],
            tolerance=two_pattern_tolerance,
        )
        for alpha_index in range(support_basis.shape[1])
    )
    beta_two_pattern_supports = tuple(
        _two_pattern_support_indices(
            null_basis[:, beta_index],
            tolerance=two_pattern_tolerance,
        )
        for beta_index in range(null_basis.shape[1])
    )

    embedding_context: _LocalPatternEmbeddingContext | None = None

    for alpha_index in range(support_basis.shape[1]):
        alpha_support = alpha_two_pattern_supports[alpha_index]

        if alpha_support is None:
            continue

        alpha_vector = support_basis[:, alpha_index]
        inflow_norm = float(np.sqrt(max(float(support_eigenvalues[alpha_index]), 0.0)))
        target_residual = 0.0
        outflow_norm = 0.0
        projector_commutator_norm = inflow_norm

        if target_residual > dark_tolerance or inflow_norm <= inflow_tolerance:
            continue

        for beta_index in range(null_basis.shape[1]):
            if beta_two_pattern_supports[beta_index] != alpha_support:
                continue

            beta_vector = null_basis[:, beta_index]
            structure = _detect_two_pattern_recycling_structure_from_vectors(
                variable_indices=reduced_density_matrix.variable_indices,
                alpha_index=int(alpha_index),
                beta_index=int(beta_index),
                local_patterns=reduced_density_matrix.local_patterns,
                alpha=alpha_vector,
                beta=beta_vector,
                tolerance=two_pattern_tolerance,
            )

            if structure is None:
                continue

            if embedding_context is None:
                embedding_context = _embedding_context_from_basis_context(basis_context)

            local_operator = np.outer(alpha_vector, beta_vector.conj())
            jump = _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            )
            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=int(alpha_index),
                    beta_index=int(beta_index),
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector.astype(np.complex128),
                    local_beta_vector=beta_vector.astype(np.complex128),
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def _scan_local_block_reset_recycling_candidates(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
) -> LocalRecyclingScanResult:
    """Scan compressed block-reset local RDM recycling candidates.

    For a local target support ``S`` and null space ``N``, rank-one null-basis
    recycling uses one jump per null vector.  A block reset packs up to
    ``dim(S)`` null vectors into one jump,

        J_block = sum_a |s_a><n_{b+a}|,

    so the number of jumps is ``ceil(dim(N) / dim(S))`` while preserving the
    exact dark condition ``J_block |psi> = 0``.
    """
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context(
        context=basis_context,
        state=target_state,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_rank = int(support_basis.shape[1])
    nullity = int(null_basis.shape[1])

    if support_rank == 0 or nullity == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    embedding_context = _embedding_context_from_basis_context(basis_context)
    candidates: list[LocalRecyclingCandidate] = []

    for start in range(0, nullity, support_rank):
        block = null_basis[:, start : start + support_rank]
        block_rank = int(block.shape[1])
        local_operator = support_basis[:, :block_rank] @ block.conj().T
        jump = _embed_local_pattern_operator_from_context(
            context=embedding_context,
            local_operator=local_operator.astype(np.complex128, copy=False),
        )
        (
            target_residual,
            inflow_norm,
            outflow_norm,
            projector_commutator_norm,
        ) = score_recycling_jump(
            jump=jump,
            target_state=target_state,
        )

        if target_residual > dark_tolerance:
            continue
        if inflow_norm <= inflow_tolerance:
            continue

        alpha_vector = np.zeros(reduced_density_matrix.local_dim, dtype=np.complex128)
        beta_vector = np.zeros(reduced_density_matrix.local_dim, dtype=np.complex128)
        alpha_vector[:] = support_basis[:, 0]
        beta_vector[:] = block[:, 0]

        candidates.append(
            LocalRecyclingCandidate(
                variable_indices=reduced_density_matrix.variable_indices,
                alpha_index=0,
                beta_index=int(start),
                jump=jump,
                target_residual=target_residual,
                inflow_norm=inflow_norm,
                outflow_norm=outflow_norm,
                projector_commutator_norm=projector_commutator_norm,
                local_alpha_vector=alpha_vector,
                local_beta_vector=beta_vector,
                local_operator=local_operator.astype(np.complex128, copy=False),
            )
        )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.beta_index),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def _scan_local_block_reset_recycling_candidates_for_subspace(
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
) -> LocalRecyclingScanResult:
    """Scan compressed block-reset recyclers for a target subspace."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    state_basis = _normalize_state_matrix_columns(
        states,
        dim=basis_context.dim,
        tolerance=rdm_tolerance,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context_and_states(
        context=basis_context,
        states=state_basis,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_rank = int(support_basis.shape[1])
    nullity = int(null_basis.shape[1])

    if support_rank == 0 or nullity == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    embedding_context = _embedding_context_from_basis_context(basis_context)
    candidates: list[LocalRecyclingCandidate] = []

    for start in range(0, nullity, support_rank):
        block = null_basis[:, start : start + support_rank]
        block_rank = int(block.shape[1])
        local_operator = support_basis[:, :block_rank] @ block.conj().T
        jump = _embed_local_pattern_operator_from_context(
            context=embedding_context,
            local_operator=local_operator.astype(np.complex128, copy=False),
        )
        (
            target_residual,
            inflow_norm,
            outflow_norm,
            projector_commutator_norm,
        ) = score_recycling_jump_for_subspace(
            jump=jump,
            states=state_basis,
            tolerance=rdm_tolerance,
        )

        if target_residual > dark_tolerance:
            continue
        if inflow_norm <= inflow_tolerance:
            continue

        candidates.append(
            LocalRecyclingCandidate(
                variable_indices=reduced_density_matrix.variable_indices,
                alpha_index=0,
                beta_index=int(start),
                jump=jump,
                target_residual=target_residual,
                inflow_norm=inflow_norm,
                outflow_norm=outflow_norm,
                projector_commutator_norm=projector_commutator_norm,
                local_alpha_vector=support_basis[:, 0].astype(np.complex128),
                local_beta_vector=block[:, 0].astype(np.complex128),
                local_operator=local_operator.astype(np.complex128, copy=False),
            )
        )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.beta_index),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def scan_local_recycling_candidates_for_subspace(
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
    max_candidates: int | None = None,
) -> LocalRecyclingScanResult:
    """Scan local rank-one recycling jumps from a target subspace RDM."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    state_basis = _normalize_state_matrix_columns(
        states,
        dim=basis_context.dim,
        tolerance=rdm_tolerance,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context_and_states(
        context=basis_context,
        states=state_basis,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_rank = int(support_basis.shape[1])
    nullity = int(null_basis.shape[1])
    candidates: list[LocalRecyclingCandidate] = []

    if support_rank == 0 or nullity == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    embedding_context = _embedding_context_from_basis_context(basis_context)

    for alpha_index in range(support_rank):
        alpha_vector = support_basis[:, alpha_index]
        for beta_index in range(nullity):
            beta_vector = null_basis[:, beta_index]
            local_operator = np.outer(alpha_vector, beta_vector.conj())
            jump = _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            )
            (
                target_residual,
                inflow_norm,
                outflow_norm,
                projector_commutator_norm,
            ) = score_recycling_jump_for_subspace(
                jump=jump,
                states=state_basis,
                tolerance=rdm_tolerance,
            )

            if target_residual > dark_tolerance:
                continue
            if inflow_norm <= inflow_tolerance:
                continue

            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=int(alpha_index),
                    beta_index=int(beta_index),
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector.astype(np.complex128),
                    local_beta_vector=beta_vector.astype(np.complex128),
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def build_local_recycling_jumps_from_subspace_regions(
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    regions: tuple[tuple[int, ...], ...],
    source: RecyclingJumpSource = "local_rdm_block_reset",
    deduplicate_regions: bool = False,
    max_jumps_per_region: int = 1,
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-12,
    max_candidates_per_region: int | None = None,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> LocalRecyclingBuildResult:
    """Scan local recycling jumps that annihilate a target state subspace."""
    if source == "none":
        return LocalRecyclingBuildResult(scan_results=(), selections=())

    if source == "local_rdm_two_pattern":
        raise NotImplementedError(
            "local_rdm_two_pattern is only implemented for a single target state. "
            "Use local_rdm_rank_one, local_rdm_null_basis, or local_rdm_block_reset "
            "for a target subspace."
        )

    scan_results: list[LocalRecyclingScanResult] = []
    selections: list[LocalRecyclingSelection] = []
    scan_result_cache: dict[tuple[int, ...], LocalRecyclingScanResult] = {}
    selected_region_keys: set[tuple[int, ...]] = set()

    for region in regions:
        region_key = tuple(int(index) for index in region)
        if deduplicate_regions and region_key in selected_region_keys:
            continue
        selected_region_keys.add(region_key)

        scan_result = scan_result_cache.get(region_key)
        if scan_result is None:
            if source == "local_rdm_block_reset":
                scan_result = _scan_local_block_reset_recycling_candidates_for_subspace(
                    basis_configs=basis_configs,
                    states=states,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                )
            else:
                scan_result = scan_local_recycling_candidates_for_subspace(
                    basis_configs=basis_configs,
                    states=states,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    max_candidates=max_candidates_per_region,
                )
            scan_result_cache[region_key] = scan_result

        scan_results.append(scan_result)
        selections.extend(
            select_local_recycling_candidates(
                scan_result=scan_result,
                source=source,
                max_candidates=max_jumps_per_region,
                prefer_sparse=prefer_sparse,
                two_pattern_tolerance=two_pattern_tolerance,
            )
        )

    return LocalRecyclingBuildResult(
        scan_results=tuple(scan_results),
        selections=tuple(selections),
    )


def build_local_recycling_jumps_from_regions(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    regions: tuple[tuple[int, ...], ...],
    source: RecyclingJumpSource = "local_rdm_two_pattern",
    deduplicate_regions: bool = False,
    max_jumps_per_region: int = 1,
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-12,
    max_candidates_per_region: int | None = None,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> LocalRecyclingBuildResult:
    """Scan several regions and return selected local recycling jumps.

    Set ``deduplicate_regions=True`` when repeated monitor components share the
    same local support and should use one recycler family rather than one copy
    per component.  The default keeps the historical behavior.
    """
    scan_results: list[LocalRecyclingScanResult] = []
    selections: list[LocalRecyclingSelection] = []
    scan_result_cache: dict[tuple[int, ...], LocalRecyclingScanResult] = {}
    selected_region_keys: set[tuple[int, ...]] = set()

    if source == "none":
        return LocalRecyclingBuildResult(scan_results=(), selections=())

    for region in regions:
        region_key = tuple(int(index) for index in region)
        if deduplicate_regions and region_key in selected_region_keys:
            continue
        selected_region_keys.add(region_key)

        scan_result = scan_result_cache.get(region_key)

        if scan_result is None:
            if source == "local_rdm_two_pattern":
                scan_result = _scan_local_two_pattern_recycling_candidates(
                    basis_configs=basis_configs,
                    target_state=target_state,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    two_pattern_tolerance=two_pattern_tolerance,
                )
            elif source == "local_rdm_block_reset":
                scan_result = _scan_local_block_reset_recycling_candidates(
                    basis_configs=basis_configs,
                    target_state=target_state,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                )
            else:
                scan_result = scan_local_recycling_candidates(
                    basis_configs=basis_configs,
                    target_state=target_state,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    max_candidates=max_candidates_per_region,
                )
            scan_result_cache[region_key] = scan_result

        scan_results.append(scan_result)

        selections.extend(
            select_local_recycling_candidates(
                scan_result=scan_result,
                source=source,
                max_candidates=max_jumps_per_region,
                prefer_sparse=prefer_sparse,
                two_pattern_tolerance=two_pattern_tolerance,
            )
        )

    return LocalRecyclingBuildResult(
        scan_results=tuple(scan_results),
        selections=tuple(selections),
    )
