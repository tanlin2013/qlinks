"""Residual-kernel diagnostics and targeted jump selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.open_system.manifold_dark import (
    _as_csr,
    _combined_operator,
    _multi_jump_projected_inflow_norm,
    _normalize_state_columns,
    _orthogonal_complement_basis,
    _projected_inflow_norm,
    _right_kernel_basis,
)
from qlinks.open_system.manifold_detector_types import (
    LocalOperatorMatrixReadout,
    ManifoldDarkOperatorBasisReport,
    RecycledManifoldCandidateFamilyKernelReport,
    RecycledManifoldDarkDetectorCandidate,
    RecycledManifoldDarkDetectorReport,
    ResidualKernelLocalSupportEntry,
    ResidualKernelOperatorActionEntry,
    ResidualKernelOperatorActionReport,
    TargetedResidualKernelJumpSelectionStep,
    TargetedResidualKernelLinearCandidate,
    TargetedResidualKernelLinearTerm,
)
from qlinks.open_system.manifold_recycling import (
    _detector_coefficients_from_report,
    _local_operator_from_matrix_unit_terms,
    _local_patterns_from_basis_configs,
    _local_recycler_specs,
    _normalize_local_regions,
    _state_ipr,
    diagnose_recycled_manifold_candidate_family_kernel,
    diagnose_recycled_manifold_dark_detectors,
)


def _local_operator_from_targeted_candidate(
    *,
    candidate: TargetedResidualKernelLinearCandidate,
    basis_configs: npt.NDArray[np.integer],
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in candidate.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(candidate.local_dim):
        raise ValueError(
            "basis_configs/local_patterns are incompatible with the selected candidate."
        )
    if candidate.operator_source != "matrix_units":
        raise ValueError(
            "targeted matrix readouts currently support operator_source='matrix_units'."
        )
    local_operator = _local_operator_from_matrix_unit_terms(
        local_patterns=local_patterns,
        terms=candidate.terms,
    )
    return LocalOperatorMatrixReadout(
        label=f"targeted_{candidate.candidate_index}",
        source="targeted_operator",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=local_operator,
        metadata=(
            ("candidate_index", int(candidate.candidate_index)),
            ("region_index", int(candidate.region_index)),
            ("operator_source", candidate.operator_source),
            ("residual_objective", candidate.residual_objective),
            ("residual_score_norm", float(candidate.residual_score_norm)),
            ("total_inflow_norm", float(candidate.total_inflow_norm)),
            ("jump_nnz", int(candidate.jump_nnz)),
        ),
    )


@dataclass(frozen=True, slots=True)
class RecycledManifoldResidualKernelReport:
    """Diagnostics for the residual complement kernel left by a recycled family.

    The report focuses on the bad subspace

        B = (cap_mu ker J_mu) cap M^perp,

    where ``J_mu`` ranges over the chosen recycled-detector family.  It is meant
    to distinguish a mere subset-selection failure from a structural residual
    sector that the current local operator family cannot see.
    """

    manifold_dimension: int
    hilbert_dimension: int
    family_report: RecycledManifoldCandidateFamilyKernelReport
    residual_basis: npt.NDArray[np.complex128]
    hamiltonian_target_coupling_norm: float
    hamiltonian_residual_block_norm: float
    hamiltonian_outside_residual_norm: float
    hamiltonian_residual_eigenvalues: tuple[complex, ...]
    operator_action_reports: tuple[ResidualKernelOperatorActionReport, ...]
    local_support_entries: tuple[ResidualKernelLocalSupportEntry, ...]
    kernel_tolerance: float

    @property
    def residual_dimension(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def residual_iprs(self) -> tuple[float, ...]:
        return tuple(
            _state_ipr(self.residual_basis[:, index]) for index in range(self.residual_dimension)
        )

    @property
    def residual_ipr_min(self) -> float | None:
        values = self.residual_iprs
        if len(values) == 0:
            return None
        return float(min(values))

    @property
    def residual_ipr_max(self) -> float | None:
        values = self.residual_iprs
        if len(values) == 0:
            return None
        return float(max(values))

    @property
    def hamiltonian_keeps_residual_sector(self) -> bool:
        return (
            self.hamiltonian_target_coupling_norm <= self.kernel_tolerance
            and self.hamiltonian_outside_residual_norm <= self.kernel_tolerance
        )

    @property
    def n_local_support_entries(self) -> int:
        return len(self.local_support_entries)

    @property
    def all_local_residual_support_inside_target(self) -> bool:
        return all(entry.residual_support_inside_target for entry in self.local_support_entries)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "residual_ipr_min": self.residual_ipr_min,
            "residual_ipr_max": self.residual_ipr_max,
            "hamiltonian_target_coupling_norm": self.hamiltonian_target_coupling_norm,
            "hamiltonian_residual_block_norm": self.hamiltonian_residual_block_norm,
            "hamiltonian_outside_residual_norm": self.hamiltonian_outside_residual_norm,
            "hamiltonian_keeps_residual_sector": self.hamiltonian_keeps_residual_sector,
            "hamiltonian_residual_eigenvalues": tuple(
                complex(value) for value in self.hamiltonian_residual_eigenvalues
            ),
            "operator_action_reports": tuple(
                report.to_summary_dict() for report in self.operator_action_reports
            ),
            "n_local_support_entries": self.n_local_support_entries,
            "all_local_residual_support_inside_target": (
                self.all_local_residual_support_inside_target
            ),
            "local_support_entries": tuple(
                entry.to_summary_dict() for entry in self.local_support_entries
            ),
            "family_report": self.family_report.to_summary_dict(),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_operator_rows: int = 12, max_local_rows: int = 16):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldResidualKernelReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual bad-kernel dimension", str(self.residual_dimension))
        overview.add_row(
            "residual IPR range",
            (
                "none"
                if self.residual_ipr_min is None
                else f"{self.residual_ipr_min:.3e} .. {self.residual_ipr_max:.3e}"
            ),
        )
        overview.add_row("H target coupling", f"{self.hamiltonian_target_coupling_norm:.3e}")
        overview.add_row("H outside residual", f"{self.hamiltonian_outside_residual_norm:.3e}")
        overview.add_row("H keeps residual sector", str(self.hamiltonian_keeps_residual_sector))
        overview.add_row(
            "local residual support inside target",
            str(self.all_local_residual_support_inside_target),
        )

        operator_table = Table(title="Probe-operator action on residual kernel")
        operator_table.add_column("group")
        operator_table.add_column("operators", justify="right")
        operator_table.add_column("||O B||", justify="right")
        operator_table.add_column("target", justify="right")
        operator_table.add_column("residual", justify="right")
        operator_table.add_column("outside", justify="right")
        for report in self.operator_action_reports[: max(int(max_operator_rows), 0)]:
            operator_table.add_row(
                report.group_name,
                str(report.n_operators),
                f"{report.total_action_norm:.3e}",
                f"{report.total_target_component_norm:.3e}",
                f"{report.total_residual_component_norm:.3e}",
                f"{report.total_outside_component_norm:.3e}",
            )

        local_table = Table(title="Local support comparison")
        local_table.add_column("region")
        local_table.add_column("dim", justify="right")
        local_table.add_column("target rank", justify="right")
        local_table.add_column("residual rank", justify="right")
        local_table.add_column("combined rank", justify="right")
        local_table.add_column("resid outside target", justify="right")
        for entry in self.local_support_entries[: max(int(max_local_rows), 0)]:
            local_table.add_row(
                str(entry.variable_indices),
                str(entry.local_dim),
                str(entry.target_support_rank),
                str(entry.residual_support_rank),
                str(entry.combined_support_rank),
                f"{entry.residual_support_outside_target_norm:.3e}",
            )
        if len(self.local_support_entries) > max_local_rows:
            local_table.add_row(
                "…",
                "",
                "",
                "",
                "",
                f"{len(self.local_support_entries) - max_local_rows} more regions",
            )

        return Panel(
            Group(overview, operator_table, local_table),
            title=Text("Recycled residual-kernel report", style="bold red"),
            border_style="red" if self.residual_dimension else "green",
        )


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelLinearSearchReport:
    """Constrained local search targeting a recycled-family residual kernel.

    For each local region with operator basis ``O_a``, the search solves

        sum_a c_a O_a P_M = 0

    and, inside that dark nullspace, maximizes either

        ||P_M (sum_a c_a O_a) B||_F

    or

        ||(sum_a c_a O_a) B||_F,

    where ``B`` is the residual bad common-kernel basis left by a recycled
    detector family.  The first objective emphasizes direct inflow to the
    target manifold; the second directly attacks a remaining dark kernel sector
    even when it does not couple to the target in one jump.
    """

    manifold_dimension: int
    hilbert_dimension: int
    residual_basis: npt.NDArray[np.complex128]
    region_variable_indices: tuple[tuple[int, ...], ...]
    operator_source: str
    family_report: RecycledManifoldCandidateFamilyKernelReport | None
    candidates: tuple[TargetedResidualKernelLinearCandidate, ...]
    candidate_jumps: tuple[sp.csr_array, ...]
    tolerance: float
    dark_tolerance: float
    inflow_tolerance: float
    residual_objective: str = "target_inflow"
    n_regions_evaluated: int = 0
    n_regions_skipped_by_local_dim: int = 0
    n_regions_with_no_recycler_specs: int = 0
    n_regions_with_no_nonzero_local_operators: int = 0
    n_regions_with_zero_dark_nullity: int = 0
    n_regions_with_dark_nullity_detected: int = -1
    n_regions_with_zero_residual_inflow: int = 0
    n_candidate_modes_generated: int = -1
    max_encountered_local_dim: int = 0

    @property
    def residual_dimension(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def n_regions(self) -> int:
        return len(self.region_variable_indices)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def max_region_size(self) -> int:
        return max((len(region) for region in self.region_variable_indices), default=0)

    @property
    def max_local_dim(self) -> int:
        return max(
            self.max_encountered_local_dim,
            max((candidate.local_dim for candidate in self.candidates), default=0),
        )

    @property
    def n_regions_with_dark_nullity(self) -> int:
        if self.n_regions_with_dark_nullity_detected >= 0:
            return self.n_regions_with_dark_nullity_detected
        return len(
            {candidate.region_index for candidate in self.candidates if candidate.dark_nullity > 0}
        )

    @property
    def n_reported_candidate_modes(self) -> int:
        return len(self.candidates)

    @property
    def n_generated_candidate_modes(self) -> int:
        if self.n_candidate_modes_generated >= 0:
            return self.n_candidate_modes_generated
        return len(self.candidates)

    @property
    def targeted_search_failure_counts(self) -> dict[str, int]:
        return {
            "n_regions_evaluated": self.n_regions_evaluated,
            "n_regions_skipped_by_local_dim": self.n_regions_skipped_by_local_dim,
            "n_regions_with_no_recycler_specs": self.n_regions_with_no_recycler_specs,
            "n_regions_with_no_nonzero_local_operators": (
                self.n_regions_with_no_nonzero_local_operators
            ),
            "n_regions_with_zero_dark_nullity": self.n_regions_with_zero_dark_nullity,
            "n_regions_with_dark_nullity": self.n_regions_with_dark_nullity,
            "n_regions_with_zero_residual_inflow": self.n_regions_with_zero_residual_inflow,
        }

    @property
    def n_candidates_hitting_residual(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance
            and max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
            > self.inflow_tolerance
            for candidate in self.candidates
        )

    @property
    def best_residual_target_inflow_norm(self) -> float:
        return max(
            (candidate.residual_target_inflow_norm for candidate in self.candidates), default=0.0
        )

    @property
    def best_residual_action_norm(self) -> float:
        return max((candidate.residual_action_norm for candidate in self.candidates), default=0.0)

    @property
    def best_residual_score_norm(self) -> float:
        return max(
            (
                max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
                for candidate in self.candidates
            ),
            default=0.0,
        )

    @property
    def best_total_inflow_norm(self) -> float:
        return max((candidate.total_inflow_norm for candidate in self.candidates), default=0.0)

    @property
    def has_targeted_solution(self) -> bool:
        return self.n_candidates_hitting_residual > 0

    def residual_kernel_dimension_after_candidate_prefix(
        self,
        n_candidates: int | None = None,
        *,
        tolerance: float | None = None,
    ) -> int:
        """Return residual-kernel dimension after the first reported jumps."""
        residual_dimension = int(self.residual_basis.shape[1])
        if residual_dimension == 0:
            return 0
        n_used = len(self.candidate_jumps) if n_candidates is None else max(int(n_candidates), 0)
        if n_used == 0:
            return residual_dimension
        gram = np.zeros((residual_dimension, residual_dimension), dtype=np.complex128)
        for jump in self.candidate_jumps[:n_used]:
            image = np.asarray(jump @ self.residual_basis, dtype=np.complex128)
            gram += image.conj().T @ image
        gram = 0.5 * (gram + gram.conj().T)
        eigenvalues = np.linalg.eigvalsh(gram).real
        largest = float(np.max(np.maximum(eigenvalues, 0.0))) if eigenvalues.size else 0.0
        cutoff = self.tolerance if tolerance is None else float(tolerance)
        cutoff = max(cutoff, cutoff * max(largest, 1.0))
        return int(np.count_nonzero(eigenvalues <= cutoff))

    @property
    def reported_candidate_residual_kernel_dimension(self) -> int:
        """Residual-family kernel dimension after all reported targeted candidates.

        This is the dimension of the residual bad kernel supplied to this
        targeted search after applying the reported candidate jumps. In the
        end-to-end workflow this is the residual left by the full recycled
        detector family, not necessarily the bad common kernel left by a
        compact selected recycled subset.
        """
        return self.residual_kernel_dimension_after_candidate_prefix(None)

    @property
    def reported_candidate_family_residual_kernel_dimension(self) -> int:
        """Alias with explicit workflow terminology."""
        return self.reported_candidate_residual_kernel_dimension

    @property
    def reported_candidates_remove_residual_kernel(self) -> bool:
        return self.reported_candidate_residual_kernel_dimension == 0

    @property
    def reported_candidates_remove_family_residual_kernel(self) -> bool:
        """Whether reported candidates remove the full-family residual kernel."""
        return self.reported_candidates_remove_residual_kernel

    def candidate_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        max_readouts: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for reported targeted candidates."""
        candidates = self.candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_targeted_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
            )
            for candidate in candidates
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "n_regions": self.n_regions,
            "region_variable_indices": self.region_variable_indices,
            "max_region_size": self.max_region_size,
            "max_local_dim": self.max_local_dim,
            "operator_source": self.operator_source,
            "residual_objective": self.residual_objective,
            "n_candidates": self.n_candidates,
            "n_generated_candidate_modes": self.n_generated_candidate_modes,
            "n_reported_candidate_modes": self.n_reported_candidate_modes,
            "n_regions_evaluated": self.n_regions_evaluated,
            "n_regions_skipped_by_local_dim": self.n_regions_skipped_by_local_dim,
            "n_regions_with_no_recycler_specs": self.n_regions_with_no_recycler_specs,
            "n_regions_with_no_nonzero_local_operators": (
                self.n_regions_with_no_nonzero_local_operators
            ),
            "n_regions_with_zero_dark_nullity": self.n_regions_with_zero_dark_nullity,
            "n_regions_with_dark_nullity": self.n_regions_with_dark_nullity,
            "n_regions_with_zero_residual_inflow": self.n_regions_with_zero_residual_inflow,
            "targeted_search_failure_counts": self.targeted_search_failure_counts,
            "n_candidates_hitting_residual": self.n_candidates_hitting_residual,
            "has_targeted_solution": self.has_targeted_solution,
            "reported_candidate_residual_kernel_dimension": (
                self.reported_candidate_residual_kernel_dimension
            ),
            "reported_candidate_family_residual_kernel_dimension": (
                self.reported_candidate_family_residual_kernel_dimension
            ),
            "reported_candidates_remove_residual_kernel": (
                self.reported_candidates_remove_residual_kernel
            ),
            "reported_candidates_remove_family_residual_kernel": (
                self.reported_candidates_remove_family_residual_kernel
            ),
            "best_residual_target_inflow_norm": self.best_residual_target_inflow_norm,
            "best_residual_action_norm": self.best_residual_action_norm,
            "best_residual_score_norm": self.best_residual_score_norm,
            "best_total_inflow_norm": self.best_total_inflow_norm,
            "family_bad_common_jump_kernel_dimension": (
                None
                if self.family_report is None
                else self.family_report.family_bad_common_jump_kernel_dimension
            ),
            "tolerance": self.tolerance,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "candidates": tuple(candidate.to_summary_dict() for candidate in self.candidates),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 16, max_terms: int = 6):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "TargetedResidualKernelLinearSearchReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual dimension", str(self.residual_dimension))
        overview.add_row("regions", str(self.n_regions))
        overview.add_row("max region size", str(self.max_region_size))
        overview.add_row("operator source", self.operator_source)
        overview.add_row("residual objective", self.residual_objective)
        overview.add_row("candidates", str(self.n_candidates))
        overview.add_row("hits residual", str(self.n_candidates_hitting_residual))
        overview.add_row(
            "reported family residual kernel",
            str(self.reported_candidate_family_residual_kernel_dimension),
        )
        overview.add_row("best target inflow", f"{self.best_residual_target_inflow_norm:.3e}")
        overview.add_row("best residual action", f"{self.best_residual_action_norm:.3e}")
        overview.add_row("best residual score", f"{self.best_residual_score_norm:.3e}")
        overview.add_row("best total inflow", f"{self.best_total_inflow_norm:.3e}")

        table = Table(title="Targeted residual-kernel local dark jumps")
        table.add_column("#", justify="right")
        table.add_column("region")
        table.add_column("dim", justify="right")
        table.add_column("dark null", justify="right")
        table.add_column("resid score", justify="right")
        table.add_column("target inflow", justify="right")
        table.add_column("dark resid", justify="right")
        table.add_column("||J||", justify="right")
        table.add_column("terms")

        for candidate in self.candidates[: max(int(max_candidates), 0)]:
            term_text = ", ".join(
                f"{term.coefficient:.3g}·{term.operator_name}"
                for term in candidate.terms[: max(int(max_terms), 0)]
            )
            if candidate.n_terms > max_terms:
                term_text += f", … {candidate.n_terms - max_terms} more"
            style = "green" if candidate.hits_residual_kernel else ""
            table.add_row(
                str(candidate.candidate_index),
                str(candidate.variable_indices),
                str(candidate.local_dim),
                str(candidate.dark_nullity),
                f"{max(candidate.residual_score_norm, candidate.residual_target_inflow_norm):.3e}",
                f"{candidate.residual_target_inflow_norm:.3e}",
                f"{candidate.relative_dark_residual:.3e}",
                f"{candidate.jump_frobenius_norm:.3e}",
                term_text,
                style=style,
            )

        if len(self.candidates) > max_candidates:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.candidates) - max_candidates} more candidates",
            )

        return Panel(
            Group(overview, table),
            title=Text("Targeted residual-kernel linear search", style="bold blue"),
            border_style="green" if self.has_targeted_solution else "red",
        )


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelJumpSelectionReport:
    """Greedy subset of targeted local jumps that removes a residual kernel."""

    manifold_dimension: int
    hilbert_dimension: int
    residual_dimension: int
    max_selected_jumps: int
    target_residual_kernel_dimension: int
    targeted_report: TargetedResidualKernelLinearSearchReport
    base_jumps: tuple[sp.csr_array, ...]
    jumps: tuple[sp.csr_array, ...]
    steps: tuple[TargetedResidualKernelJumpSelectionStep, ...]
    final_diagnostics: Any | None
    selection_target: Literal["reported_residual_kernel", "combined_common_kernel"]
    initial_selection_kernel_dimension: int
    target_selection_kernel_dimension: int
    kernel_tolerance: float
    dark_tolerance: float
    inflow_tolerance: float
    selected_inflow_norm: float | None = None

    @property
    def n_selected_jumps(self) -> int:
        return len(self.jumps)

    @property
    def n_base_jumps(self) -> int:
        return len(self.base_jumps)

    @property
    def n_combined_jumps(self) -> int:
        return self.n_base_jumps + self.n_selected_jumps

    @property
    def all_jumps(self) -> tuple[sp.csr_array, ...]:
        return self.base_jumps + self.jumps

    @property
    def selected_candidates(self) -> tuple[TargetedResidualKernelLinearCandidate, ...]:
        return tuple(step.candidate for step in self.steps)

    @property
    def selected_region_indices(self) -> tuple[int, ...]:
        return tuple(step.candidate.region_index for step in self.steps)

    def selected_operator_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        max_readouts: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for selected targeted completion operators."""
        candidates = self.selected_candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_targeted_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
            )
            for candidate in candidates
        )

    @property
    def final_residual_kernel_dimension(self) -> int:
        current_basis = np.asarray(
            self.targeted_report.residual_basis,
            dtype=np.complex128,
        )
        for jump in self.jumps:
            image = np.asarray(jump @ current_basis, dtype=np.complex128)
            kernel_basis = _right_kernel_basis(image, tolerance=self.kernel_tolerance)
            current_basis = current_basis @ kernel_basis
        return int(current_basis.shape[1])

    @property
    def residual_kernel_removed(self) -> bool:
        return self.final_residual_kernel_dimension <= self.target_residual_kernel_dimension

    @property
    def final_selection_kernel_dimension(self) -> int:
        if len(self.steps) == 0:
            return self.initial_selection_kernel_dimension
        return int(self.steps[-1].residual_kernel_dimension)

    @property
    def selection_kernel_removed(self) -> bool:
        return self.final_selection_kernel_dimension <= self.target_selection_kernel_dimension

    @property
    def total_selected_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.jumps))

    @property
    def total_combined_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.all_jumps))

    @property
    def max_selected_jump_nnz(self) -> int:
        return int(max((jump.nnz for jump in self.jumps), default=0))

    @property
    def combined_bad_common_jump_kernel_dimension(self) -> int | None:
        if self.final_diagnostics is None:
            return None
        return int(self.final_diagnostics.bad_common_jump_kernel_dimension)

    @property
    def combined_inflow_norm(self) -> float | None:
        if self.final_diagnostics is not None:
            return float(self.final_diagnostics.inflow_norm)
        if self.selected_inflow_norm is None:
            return None
        return float(self.selected_inflow_norm)

    @property
    def combined_complement_common_kernel_removed(self) -> bool | None:
        value = self.combined_bad_common_jump_kernel_dimension
        if value is None:
            return None
        return value == 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "target_residual_kernel_dimension": self.target_residual_kernel_dimension,
            "selection_target": self.selection_target,
            "initial_selection_kernel_dimension": self.initial_selection_kernel_dimension,
            "target_selection_kernel_dimension": self.target_selection_kernel_dimension,
            "max_selected_jumps": self.max_selected_jumps,
            "n_base_jumps": self.n_base_jumps,
            "n_selected_jumps": self.n_selected_jumps,
            "n_combined_jumps": self.n_combined_jumps,
            "selected_region_indices": self.selected_region_indices,
            "total_selected_jump_nnz": self.total_selected_jump_nnz,
            "total_combined_jump_nnz": self.total_combined_jump_nnz,
            "max_selected_jump_nnz": self.max_selected_jump_nnz,
            "final_residual_kernel_dimension": self.final_residual_kernel_dimension,
            "residual_kernel_removed": self.residual_kernel_removed,
            "final_selection_kernel_dimension": self.final_selection_kernel_dimension,
            "selection_kernel_removed": self.selection_kernel_removed,
            "combined_bad_common_jump_kernel_dimension": (
                self.combined_bad_common_jump_kernel_dimension
            ),
            "selected_inflow_norm": self.selected_inflow_norm,
            "combined_inflow_norm": self.combined_inflow_norm,
            "combined_complement_common_kernel_removed": (
                self.combined_complement_common_kernel_removed
            ),
            "kernel_tolerance": self.kernel_tolerance,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "steps": tuple(step.to_summary_dict() for step in self.steps),
            "final_diagnostics": (
                None if self.final_diagnostics is None else self.final_diagnostics.to_summary_dict()
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_steps: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "TargetedResidualKernelJumpSelectionReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual dimension", str(self.residual_dimension))
        overview.add_row("selection target", self.selection_target)
        overview.add_row("initial selection kernel", str(self.initial_selection_kernel_dimension))
        overview.add_row("base jumps", str(self.n_base_jumps))
        overview.add_row("selected targeted jumps", str(self.n_selected_jumps))
        overview.add_row("combined jumps", str(self.n_combined_jumps))
        overview.add_row("final residual kernel", str(self.final_residual_kernel_dimension))
        overview.add_row("selection kernel removed", str(self.selection_kernel_removed))
        overview.add_row(
            "combined bad kernel",
            (
                "not checked"
                if self.combined_bad_common_jump_kernel_dimension is None
                else str(self.combined_bad_common_jump_kernel_dimension)
            ),
        )
        overview.add_row(
            "combined inflow",
            (
                "not checked"
                if self.combined_inflow_norm is None
                else f"{self.combined_inflow_norm:.3e}"
            ),
        )

        table = Table(title="Greedy selected targeted residual-kernel jumps")
        table.add_column("step", justify="right")
        table.add_column("region")
        table.add_column("selection kernel", justify="right")
        table.add_column("resid inflow", justify="right")
        table.add_column("total inflow", justify="right")
        table.add_column("jump nnz", justify="right")
        table.add_column("terms", justify="right")
        for step in self.steps[: max(int(max_steps), 0)]:
            candidate = step.candidate
            table.add_row(
                str(step.step_index),
                str(candidate.variable_indices),
                str(step.residual_kernel_dimension),
                f"{candidate.residual_target_inflow_norm:.3e}",
                f"{candidate.total_inflow_norm:.3e}",
                str(candidate.jump_nnz),
                str(candidate.n_terms),
                style=(
                    "green"
                    if step.residual_kernel_dimension <= self.target_residual_kernel_dimension
                    else ""
                ),
            )
        if len(self.steps) > max_steps:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.steps) - max_steps} more steps",
            )
        return Panel(
            Group(overview, table),
            title=Text("Targeted residual-kernel jump-selection report", style="bold blue"),
            border_style="green" if self.selection_kernel_removed else "red",
        )


def _bad_kernel_basis_from_recycled_family(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    kernel_tolerance: float = 1.0e-10,
    max_detectors: int | None = None,
) -> npt.NDArray[np.complex128]:
    """Return the bad complement common kernel for a recycled family."""
    from qlinks.local_structure.embedding import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
    )
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    coefficients = _detector_coefficients_from_report(
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        n_operators=len(detector_matrices),
        max_detectors=max_detectors,
    )
    detectors = tuple(
        _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        for detector_index in range(coefficients.shape[1])
    )

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

    regions = _normalize_local_regions(local_regions)
    contexts = tuple(
        _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        for region in regions
    )
    embedding_contexts = tuple(
        _embedding_context_from_basis_context(context) for context in contexts
    )
    rdms = tuple(
        _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        for context in contexts
    )
    recycler_specs_by_region = tuple(
        _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        for rdm in rdms
    )
    recycler_cache: dict[tuple[int, int], sp.csr_array] = {}

    complement_basis = _orthogonal_complement_basis(state_basis, tolerance=kernel_tolerance)
    complement_dimension = int(complement_basis.shape[1])
    if complement_dimension == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    family_gram = np.zeros(
        (complement_dimension, complement_dimension),
        dtype=np.complex128,
    )

    for candidate in candidates:
        if candidate.detector_index < 0 or candidate.detector_index >= len(detectors):
            raise ValueError("candidate.detector_index is out of range for detector coefficients.")
        if candidate.region_index < 0 or candidate.region_index >= len(regions):
            raise ValueError("candidate.region_index is out of range for local_regions.")
        recycler_specs = recycler_specs_by_region[candidate.region_index]
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")

        cache_key = (int(candidate.region_index), int(candidate.recycler_index))
        recycler = recycler_cache.get(cache_key)
        if recycler is None:
            _, local_operator = recycler_specs[candidate.recycler_index]
            recycler = _embed_local_pattern_operator_from_context(
                context=embedding_contexts[candidate.region_index],
                local_operator=local_operator,
            ).tocsr()
            recycler_cache[cache_key] = recycler

        jump = (recycler @ detectors[candidate.detector_index]).tocsr()
        if jump.nnz == 0:
            continue
        image = np.asarray(jump @ complement_basis, dtype=np.complex128)
        family_gram += image.conj().T @ image

    family_gram = 0.5 * (family_gram + family_gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(family_gram)
    largest = float(np.max(np.maximum(eigenvalues.real, 0.0))) if eigenvalues.size else 0.0
    cutoff = max(float(kernel_tolerance), float(kernel_tolerance) * max(largest, 1.0))
    kernel_mask = np.asarray(eigenvalues.real <= cutoff, dtype=bool)
    return (complement_basis @ eigenvectors[:, kernel_mask]).astype(np.complex128, copy=False)


def _operator_action_report_on_residual_kernel(
    *,
    group_name: str,
    operators: tuple[Any, ...] | list[Any],
    operator_names: tuple[str, ...] | list[str] | None,
    target_basis: npt.NDArray[np.complex128],
    residual_basis: npt.NDArray[np.complex128],
    max_entries: int | None,
) -> ResidualKernelOperatorActionReport:
    matrices = tuple(_as_csr(operator) for operator in operators)
    if operator_names is None:
        names = tuple(f"O_{index}" for index in range(len(matrices)))
    else:
        names = tuple(str(name) for name in operator_names)
        if len(names) != len(matrices):
            raise ValueError("operator_names length must match operators.")

    dim = int(target_basis.shape[0])
    if residual_basis.shape[0] != dim:
        raise ValueError("residual_basis has incompatible dimension.")
    for matrix in matrices:
        if matrix.shape != (dim, dim):
            raise ValueError("operator has incompatible shape for residual-kernel report.")

    entries: list[ResidualKernelOperatorActionEntry] = []
    for operator_index, matrix in enumerate(matrices):
        action = np.asarray(matrix @ residual_basis, dtype=np.complex128)
        target_component = target_basis.conj().T @ action
        residual_component = residual_basis.conj().T @ action
        projected = target_basis @ target_component + residual_basis @ residual_component
        outside = action - projected
        entries.append(
            ResidualKernelOperatorActionEntry(
                operator_index=operator_index,
                operator_name=names[operator_index],
                action_norm=float(np.linalg.norm(action)),
                target_component_norm=float(np.linalg.norm(target_component)),
                residual_component_norm=float(np.linalg.norm(residual_component)),
                outside_component_norm=float(np.linalg.norm(outside)),
            )
        )

    sorted_entries = tuple(
        sorted(
            entries,
            key=lambda entry: (
                -entry.action_norm,
                -entry.target_component_norm,
                entry.operator_index,
            ),
        )
    )
    if max_entries is not None:
        sorted_entries = sorted_entries[: max(int(max_entries), 0)]
    return ResidualKernelOperatorActionReport(
        group_name=str(group_name),
        n_operators=len(matrices),
        entries=sorted_entries,
    )


def _residual_local_support_entries(
    *,
    target_basis: npt.NDArray[np.complex128],
    residual_basis: npt.NDArray[np.complex128],
    basis_configs: npt.NDArray[np.integer],
    local_regions: tuple[tuple[int, ...], ...],
    tolerance: float,
) -> tuple[ResidualKernelLocalSupportEntry, ...]:
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    if residual_basis.shape[1] == 0:
        return ()

    basis_array = np.asarray(basis_configs)
    entries: list[ResidualKernelLocalSupportEntry] = []
    for region in local_regions:
        context = _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        target_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=target_basis,
            tolerance=tolerance,
        )
        residual_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=residual_basis,
            tolerance=tolerance,
        )
        combined_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=np.column_stack([target_basis, residual_basis]),
            tolerance=tolerance,
        )

        target_support = target_rdm.support_basis
        residual_support = residual_rdm.support_basis
        if residual_support.shape[1] == 0:
            outside_norm = 0.0
        elif target_support.shape[1] == 0:
            outside_norm = float(np.linalg.norm(residual_support))
        else:
            projected = target_support @ (target_support.conj().T @ residual_support)
            outside_norm = float(np.linalg.norm(residual_support - projected))

        entries.append(
            ResidualKernelLocalSupportEntry(
                variable_indices=tuple(int(index) for index in region),
                local_dim=context.local_dim,
                target_support_rank=target_rdm.support_rank,
                target_nullity=target_rdm.nullity,
                residual_support_rank=residual_rdm.support_rank,
                residual_nullity=residual_rdm.nullity,
                combined_support_rank=combined_rdm.support_rank,
                combined_nullity=combined_rdm.nullity,
                residual_support_outside_target_norm=outside_norm,
            )
        )
    return tuple(entries)


def diagnose_recycled_manifold_residual_kernel(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    family_report: RecycledManifoldCandidateFamilyKernelReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    operator_groups: (
        tuple[
            tuple[str, tuple[Any, ...] | list[Any], tuple[str, ...] | list[str] | None],
            ...,
        ]
        | None
    ) = None,
    local_support_regions: (
        tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]] | None
    ) = None,
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    expand_candidate_report: bool = True,
    max_operator_entries: int | None = 64,
) -> RecycledManifoldResidualKernelReport:
    """Diagnose the residual bad kernel left by a recycled-detector family.

    This function first computes the full-family common jump kernel for the
    specified recycled-detector candidates.  It then extracts the bad complement
    subspace and reports whether the Hamiltonian and optional probe-operator
    groups couple that subspace to the target manifold, leave it invariant, or
    push it outside the target-plus-residual sector.
    """
    regions = _normalize_local_regions(local_regions)
    target_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(target_basis.shape[0])
    manifold_dimension = int(target_basis.shape[1])

    if family_report is None:
        family_report = diagnose_recycled_manifold_candidate_family_kernel(
            hamiltonian=hamiltonian,
            states=target_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            candidate_report=candidate_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            max_detectors=max_detectors,
            expand_candidate_report=expand_candidate_report,
            kernel_method="streamed",
            store_candidate_jumps=False,
        )

    if family_report.candidate_report_is_truncated and expand_candidate_report:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=target_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
    else:
        candidate_report = family_report.candidate_report

    eligible_candidates = tuple(
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    )
    residual_basis = _bad_kernel_basis_from_recycled_family(
        states=target_basis,
        basis_configs=basis_configs,
        detector_operators=detector_operators,
        local_regions=regions,
        candidates=eligible_candidates,
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        recycler_source=recycler_source,
        tolerance=tolerance,
        rdm_tolerance=rdm_tolerance,
        kernel_tolerance=kernel_tolerance,
        max_detectors=max_detectors,
    )

    hamiltonian_matrix = _as_csr(hamiltonian)
    if hamiltonian_matrix.shape != (dim, dim):
        raise ValueError("hamiltonian must have shape (hilbert_dimension, hilbert_dimension).")

    if residual_basis.shape[1] == 0:
        hamiltonian_target_coupling_norm = 0.0
        hamiltonian_residual_block_norm = 0.0
        hamiltonian_outside_residual_norm = 0.0
        hamiltonian_residual_eigenvalues: tuple[complex, ...] = ()
    else:
        h_residual = np.asarray(hamiltonian_matrix @ residual_basis, dtype=np.complex128)
        h_target_block = target_basis.conj().T @ h_residual
        h_residual_block = residual_basis.conj().T @ h_residual
        projected = target_basis @ h_target_block + residual_basis @ h_residual_block
        h_outside = h_residual - projected
        hamiltonian_target_coupling_norm = float(np.linalg.norm(h_target_block))
        hamiltonian_residual_block_norm = float(np.linalg.norm(h_residual_block))
        hamiltonian_outside_residual_norm = float(np.linalg.norm(h_outside))
        h_residual_block = 0.5 * (h_residual_block + h_residual_block.conj().T)
        hamiltonian_residual_eigenvalues = tuple(
            complex(value) for value in np.linalg.eigvalsh(h_residual_block)
        )

    action_reports: list[ResidualKernelOperatorActionReport] = []
    if operator_groups is not None:
        for group_name, operators, operator_names in operator_groups:
            action_reports.append(
                _operator_action_report_on_residual_kernel(
                    group_name=group_name,
                    operators=operators,
                    operator_names=operator_names,
                    target_basis=target_basis,
                    residual_basis=residual_basis,
                    max_entries=max_operator_entries,
                )
            )

    support_regions = (
        regions
        if local_support_regions is None
        else _normalize_local_regions(local_support_regions)
    )
    local_entries = _residual_local_support_entries(
        target_basis=target_basis,
        residual_basis=residual_basis,
        basis_configs=basis_configs,
        local_regions=support_regions,
        tolerance=rdm_tolerance,
    )

    return RecycledManifoldResidualKernelReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        family_report=family_report,
        residual_basis=residual_basis,
        hamiltonian_target_coupling_norm=hamiltonian_target_coupling_norm,
        hamiltonian_residual_block_norm=hamiltonian_residual_block_norm,
        hamiltonian_outside_residual_norm=hamiltonian_outside_residual_norm,
        hamiltonian_residual_eigenvalues=hamiltonian_residual_eigenvalues,
        operator_action_reports=tuple(action_reports),
        local_support_entries=local_entries,
        kernel_tolerance=float(kernel_tolerance),
    )


def _right_kernel_basis_from_gram_matrix(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return a right-kernel basis via the smaller column Gram matrix.

    This is useful when ``matrix`` is tall and has at most a few hundred
    columns.  It avoids the much more expensive full SVD used by the generic
    helper while preserving the same relative singular-value cutoff semantics.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_columns = int(matrix.shape[1])
    if n_columns == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    if matrix.shape[0] == 0:
        return np.eye(n_columns, dtype=np.complex128)

    gram = matrix.conj().T @ matrix
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues.real, 0.0)
    largest_singular = float(np.sqrt(np.max(eigenvalues))) if eigenvalues.size else 0.0
    cutoff = max(float(tolerance), float(tolerance) * max(largest_singular, 1.0))
    kernel_mask = eigenvalues <= cutoff * cutoff
    return eigenvectors[:, kernel_mask].astype(np.complex128, copy=False)


def diagnose_targeted_residual_kernel_linear_search(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    residual_basis: npt.ArrayLike | None = None,
    residual_report: RecycledManifoldResidualKernelReport | None = None,
    detector_operators: tuple[Any, ...] | list[Any] | None = None,
    residual_family_local_regions: (
        tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]] | None
    ) = None,
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    family_report: RecycledManifoldCandidateFamilyKernelReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    operator_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "matrix_units",
    residual_objective: Literal["target_inflow", "action_norm"] = "target_inflow",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    max_detectors: int | None = None,
    max_modes_per_region: int = 1,
    max_report_candidates: int | None = 32,
    max_local_dim: int | None = None,
    coefficient_tolerance: float = 1.0e-8,
) -> TargetedResidualKernelLinearSearchReport:
    """Search local dark jumps that directly target a residual bad kernel.

    This is stronger than optimizing linear combinations of the factorized
    recycled-detector family ``R D``.  For each supplied local region, it builds
    a local operator basis ``O_a`` and solves the constrained problem

        sum_a c_a O_a P_M = 0,

    then ranks dark combinations by their action

        ||P_M (sum_a c_a O_a) B||_F,

    where ``B`` is the residual complement common kernel left by a recycled
    detector family.
    """
    from qlinks.local_structure.embedding import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
    )
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    target_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(target_basis.shape[0])
    manifold_dimension = int(target_basis.shape[1])
    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")
    if residual_objective not in {"target_inflow", "action_norm"}:
        raise ValueError('residual_objective must be "target_inflow" or "action_norm".')

    report_family = family_report
    if residual_report is not None:
        bad_basis = np.asarray(residual_report.residual_basis, dtype=np.complex128)
        report_family = residual_report.family_report
    elif residual_basis is not None:
        bad_basis, _ = _normalize_state_columns(residual_basis, tolerance=tolerance)
    else:
        if detector_operators is None:
            raise ValueError(
                "Pass residual_basis, residual_report, or detector_operators to infer "
                "the recycled-family residual kernel."
            )
        family_regions = _normalize_local_regions(
            local_regions
            if residual_family_local_regions is None
            else residual_family_local_regions
        )
        if family_report is not None:
            candidate_report = family_report.candidate_report
        if candidate_report is None:
            candidate_report = diagnose_recycled_manifold_dark_detectors(
                states=target_basis,
                basis_configs=basis_array,
                detector_operators=detector_operators,
                local_regions=family_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                max_detectors=max_detectors,
                max_report_candidates=None,
                sort_by_inflow=True,
            )
        eligible_candidates = tuple(
            candidate
            for candidate in candidate_report.candidates
            if candidate.relative_dark_residual <= dark_tolerance
            and candidate.inflow_norm > inflow_tolerance
        )
        bad_basis = _bad_kernel_basis_from_recycled_family(
            states=target_basis,
            basis_configs=basis_array,
            detector_operators=detector_operators,
            local_regions=family_regions,
            candidates=eligible_candidates,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            kernel_tolerance=kernel_tolerance,
            max_detectors=max_detectors,
        )

    if bad_basis.ndim == 1:
        bad_basis = bad_basis.reshape(dim, 1)
    if bad_basis.ndim != 2 or bad_basis.shape[0] != dim:
        raise ValueError("residual_basis has incompatible shape.")
    if bad_basis.shape[1] > 0:
        # Re-orthonormalize and explicitly remove any numerical target component.
        bad_basis = bad_basis - target_basis @ (target_basis.conj().T @ bad_basis)
        bad_basis, _ = _normalize_state_columns(bad_basis, tolerance=tolerance)
    else:
        bad_basis = np.zeros((dim, 0), dtype=np.complex128)

    regions = _normalize_local_regions(local_regions)
    candidates: list[TargetedResidualKernelLinearCandidate] = []
    candidate_jumps: list[sp.csr_array] = []
    n_regions_evaluated = 0
    n_regions_skipped_by_local_dim = 0
    n_regions_with_no_recycler_specs = 0
    n_regions_with_no_nonzero_local_operators = 0
    n_regions_with_zero_dark_nullity = 0
    n_regions_with_dark_nullity_detected = 0
    n_regions_with_zero_residual_inflow = 0
    max_encountered_local_dim = 0

    for region_index, region in enumerate(regions):
        n_regions_evaluated += 1
        context = _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=target_basis,
            tolerance=rdm_tolerance,
        )
        max_encountered_local_dim = max(max_encountered_local_dim, int(rdm.local_dim))
        if max_local_dim is not None and rdm.local_dim > max_local_dim:
            n_regions_skipped_by_local_dim += 1
            continue

        specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=operator_source,
        )
        if len(specs) == 0:
            n_regions_with_no_recycler_specs += 1
            continue

        embedding_context = _embedding_context_from_basis_context(context)
        local_ops = tuple(
            _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            ).tocsr()
            for _name, local_operator in specs
        )
        nonzero_indices = [index for index, operator in enumerate(local_ops) if operator.nnz > 0]
        if len(nonzero_indices) == 0:
            n_regions_with_no_nonzero_local_operators += 1
            continue
        local_ops = tuple(local_ops[index] for index in nonzero_indices)
        spec_names = tuple(specs[index][0] for index in nonzero_indices)

        dark_matrix = np.column_stack(
            [
                np.asarray(operator @ target_basis, dtype=np.complex128).reshape(-1)
                for operator in local_ops
            ]
        )
        dark_kernel = _right_kernel_basis_from_gram_matrix(dark_matrix, tolerance=dark_tolerance)
        dark_nullity = int(dark_kernel.shape[1])
        dark_rank = int(dark_kernel.shape[0] - dark_nullity)
        if dark_nullity == 0:
            n_regions_with_zero_dark_nullity += 1
            continue
        n_regions_with_dark_nullity_detected += 1
        region_has_residual_inflow = False

        if bad_basis.shape[1] == 0:
            score_matrix = np.zeros((0, len(local_ops)), dtype=np.complex128)
        elif residual_objective == "target_inflow":
            score_matrix = np.column_stack(
                [
                    (
                        target_basis.conj().T
                        @ np.asarray(operator @ bad_basis, dtype=np.complex128)
                    ).reshape(-1)
                    for operator in local_ops
                ]
            )
        else:
            score_matrix = np.column_stack(
                [
                    np.asarray(operator @ bad_basis, dtype=np.complex128).reshape(-1)
                    for operator in local_ops
                ]
            )
        restricted_inflow = score_matrix @ dark_kernel
        if restricted_inflow.size == 0:
            singular_values = np.zeros((0,), dtype=np.float64)
            right_vectors = np.zeros((dark_nullity, 0), dtype=np.complex128)
        else:
            _u, singular_values, vh = np.linalg.svd(restricted_inflow, full_matrices=False)
            right_vectors = vh.conj().T

        n_modes = min(max(int(max_modes_per_region), 0), int(right_vectors.shape[1]))
        for mode_index in range(n_modes):
            coefficients = np.asarray(
                dark_kernel @ right_vectors[:, mode_index], dtype=np.complex128
            )
            coefficient_norm = float(np.linalg.norm(coefficients))
            if coefficient_norm == 0.0:
                continue
            coefficients = coefficients / coefficient_norm
            jump = sp.csr_array((dim, dim), dtype=np.complex128)
            for coefficient, operator in zip(coefficients, local_ops, strict=True):
                if abs(coefficient) <= 0.0:
                    continue
                jump = jump + coefficient * operator
            jump = jump.tocsr()
            if jump.nnz == 0:
                continue

            dark_residual = float(np.linalg.norm(jump @ target_basis))
            jump_norm = float(sp.linalg.norm(jump))
            relative_dark_residual = dark_residual / max(jump_norm, 1.0)
            bad_action = np.asarray(jump @ bad_basis, dtype=np.complex128)
            residual_action_norm = float(np.linalg.norm(bad_action))
            residual_target_action = target_basis.conj().T @ bad_action
            residual_target_inflow_norm = float(np.linalg.norm(residual_target_action))
            residual_score_norm = (
                residual_target_inflow_norm
                if residual_objective == "target_inflow"
                else residual_action_norm
            )
            if residual_score_norm > inflow_tolerance:
                region_has_residual_inflow = True
            total_inflow_norm, target_block_norm = _projected_inflow_norm(
                jump=jump,
                state_basis=target_basis,
            )
            terms = tuple(
                TargetedResidualKernelLinearTerm(
                    operator_index=int(index),
                    operator_name=spec_names[index],
                    coefficient=complex(coefficient),
                    weight=float(abs(coefficient)),
                )
                for index, coefficient in sorted(
                    enumerate(coefficients),
                    key=lambda item: -abs(item[1]),
                )
                if abs(coefficient) > coefficient_tolerance
            )
            candidates.append(
                TargetedResidualKernelLinearCandidate(
                    candidate_index=len(candidates),
                    region_index=int(region_index),
                    variable_indices=tuple(int(value) for value in region),
                    local_dim=int(rdm.local_dim),
                    operator_source=operator_source,
                    dark_constraint_rank=dark_rank,
                    dark_nullity=dark_nullity,
                    singular_value=float(singular_values[mode_index]),
                    residual_target_inflow_norm=residual_target_inflow_norm,
                    residual_action_norm=residual_action_norm,
                    residual_score_norm=residual_score_norm,
                    residual_objective=residual_objective,
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    total_inflow_norm=total_inflow_norm,
                    target_block_norm=target_block_norm,
                    jump_frobenius_norm=jump_norm,
                    jump_nnz=int(jump.nnz),
                    coefficients=coefficients,
                    terms=terms,
                )
            )
            candidate_jumps.append(jump)

        if not region_has_residual_inflow:
            n_regions_with_zero_residual_inflow += 1

    n_candidate_modes_generated = len(candidates)

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].relative_dark_residual > dark_tolerance,
            -max(
                candidates[index].residual_score_norm,
                candidates[index].residual_target_inflow_norm,
            ),
            -candidates[index].residual_target_inflow_norm,
            -candidates[index].total_inflow_norm,
            candidates[index].jump_nnz,
            candidates[index].region_index,
        ),
    )
    if max_report_candidates is not None:
        order = order[: max(int(max_report_candidates), 0)]

    sorted_candidates: list[TargetedResidualKernelLinearCandidate] = []
    sorted_jumps: list[sp.csr_array] = []
    for new_index, old_index in enumerate(order):
        candidate = candidates[old_index]
        sorted_candidates.append(
            TargetedResidualKernelLinearCandidate(
                candidate_index=new_index,
                region_index=candidate.region_index,
                variable_indices=candidate.variable_indices,
                local_dim=candidate.local_dim,
                operator_source=candidate.operator_source,
                dark_constraint_rank=candidate.dark_constraint_rank,
                dark_nullity=candidate.dark_nullity,
                singular_value=candidate.singular_value,
                residual_target_inflow_norm=candidate.residual_target_inflow_norm,
                residual_action_norm=candidate.residual_action_norm,
                residual_score_norm=candidate.residual_score_norm,
                residual_objective=candidate.residual_objective,
                dark_residual=candidate.dark_residual,
                relative_dark_residual=candidate.relative_dark_residual,
                total_inflow_norm=candidate.total_inflow_norm,
                target_block_norm=candidate.target_block_norm,
                jump_frobenius_norm=candidate.jump_frobenius_norm,
                jump_nnz=candidate.jump_nnz,
                coefficients=candidate.coefficients,
                terms=candidate.terms,
            )
        )
        sorted_jumps.append(candidate_jumps[old_index])

    return TargetedResidualKernelLinearSearchReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        residual_basis=bad_basis,
        region_variable_indices=regions,
        operator_source=operator_source,
        residual_objective=residual_objective,
        family_report=report_family,
        candidates=tuple(sorted_candidates),
        candidate_jumps=tuple(sorted_jumps),
        tolerance=float(tolerance),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        n_regions_evaluated=int(n_regions_evaluated),
        n_regions_skipped_by_local_dim=int(n_regions_skipped_by_local_dim),
        n_regions_with_no_recycler_specs=int(n_regions_with_no_recycler_specs),
        n_regions_with_no_nonzero_local_operators=int(n_regions_with_no_nonzero_local_operators),
        n_regions_with_zero_dark_nullity=int(n_regions_with_zero_dark_nullity),
        n_regions_with_dark_nullity_detected=int(n_regions_with_dark_nullity_detected),
        n_regions_with_zero_residual_inflow=int(n_regions_with_zero_residual_inflow),
        n_candidate_modes_generated=int(n_candidate_modes_generated),
        max_encountered_local_dim=int(max_encountered_local_dim),
    )


def _bad_common_kernel_basis_for_jumps(
    *,
    jumps: tuple[sp.csr_array, ...],
    target_basis: npt.NDArray[np.complex128],
    dim: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return the complement common kernel for a concrete jump list."""
    from qlinks.open_system._subspace import (
        _common_kernel_basis_from_sparse_operators,
        _kernel_basis_orthogonal_to_manifold,
    )

    if len(jumps) == 0:
        common_kernel_basis = np.eye(int(dim), dtype=np.complex128)
    else:
        common_kernel_basis = _common_kernel_basis_from_sparse_operators(
            operators=jumps,
            dim=int(dim),
            tolerance=float(tolerance),
        )
    return _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=target_basis,
        tolerance=float(tolerance),
    )


def select_targeted_residual_kernel_jumps(
    *,
    targeted_report: TargetedResidualKernelLinearSearchReport,
    hamiltonian: Any | None = None,
    states: npt.ArrayLike | None = None,
    base_jumps: tuple[Any, ...] | list[Any] = (),
    max_selected_jumps: int = 16,
    target_residual_kernel_dimension: int = 0,
    selection_target: Literal[
        "reported_residual_kernel",
        "combined_common_kernel",
    ] = "reported_residual_kernel",
    allow_non_improving: bool = False,
    kernel_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    liouvillian_zero_tolerance: float = 1.0e-9,
    check_manifold_diagnostics: bool = True,
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "none",
    sparse_liouvillian_eigenvalue_count: int = 32,
) -> TargetedResidualKernelJumpSelectionReport:
    """Greedily select targeted local jumps removing a residual bad kernel.

    The input report is produced by
    :func:`diagnose_targeted_residual_kernel_linear_search`.

    With ``selection_target="reported_residual_kernel"`` this preserves the
    original behavior: each step minimizes the remaining kernel inside the
    residual basis stored in ``targeted_report``.  With
    ``selection_target="combined_common_kernel"`` the selector instead starts
    from the complement common jump-kernel of the supplied ``base_jumps`` and
    greedily minimizes the combined bad kernel of ``base_jumps`` plus the
    targeted candidates.  The latter is the right mode after a compressed
    recycled-detector subset leaves a small complement kernel that is not
    identical to the full-family residual basis used to create the targeted
    report.
    """
    from qlinks.open_system.diagnostics.dark import diagnose_dark_manifold

    residual_basis = np.asarray(targeted_report.residual_basis, dtype=np.complex128)
    if residual_basis.ndim != 2:
        raise ValueError("targeted_report.residual_basis must be two-dimensional.")
    residual_dimension = int(residual_basis.shape[1])
    dim = int(targeted_report.hilbert_dimension)
    if residual_basis.shape[0] != dim:
        raise ValueError("targeted_report has inconsistent residual-basis dimension.")

    if selection_target not in {"reported_residual_kernel", "combined_common_kernel"}:
        raise ValueError(
            'selection_target must be "reported_residual_kernel" or "combined_common_kernel".'
        )

    base_jump_tuple = tuple(_as_csr(jump) for jump in base_jumps)
    for jump in base_jump_tuple:
        if jump.shape != (dim, dim):
            raise ValueError("base_jumps must have shape (hilbert_dimension, hilbert_dimension).")

    if selection_target == "combined_common_kernel":
        if states is None:
            raise ValueError(
                "states must be supplied when selection_target='combined_common_kernel'."
            )
        target_basis, _ = _normalize_state_columns(states, tolerance=kernel_tolerance)
        if target_basis.shape[0] != dim:
            raise ValueError("states has incompatible Hilbert-space dimension.")
        current_basis = _bad_common_kernel_basis_for_jumps(
            jumps=base_jump_tuple,
            target_basis=target_basis,
            dim=dim,
            tolerance=kernel_tolerance,
        )
        eligible = [
            (candidate, _as_csr(jump))
            for candidate, jump in zip(
                targeted_report.candidates,
                targeted_report.candidate_jumps,
                strict=True,
            )
            if candidate.relative_dark_residual <= dark_tolerance
            and max(
                candidate.residual_score_norm,
                candidate.residual_target_inflow_norm,
                candidate.total_inflow_norm,
            )
            > inflow_tolerance
        ]
    else:
        current_basis = residual_basis
        eligible = [
            (candidate, _as_csr(jump))
            for candidate, jump in zip(
                targeted_report.candidates,
                targeted_report.candidate_jumps,
                strict=True,
            )
            if candidate.relative_dark_residual <= dark_tolerance
            and max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
            > inflow_tolerance
        ]

    selected_candidates: list[TargetedResidualKernelLinearCandidate] = []
    selected_jumps: list[sp.csr_array] = []
    selected_ids: set[int] = set()
    steps: list[TargetedResidualKernelJumpSelectionStep] = []
    initial_selection_kernel_dimension = int(current_basis.shape[1])
    current_dimension = initial_selection_kernel_dimension
    target_selection_kernel_dimension = int(target_residual_kernel_dimension)

    for _step_index in range(max(int(max_selected_jumps), 0)):
        if current_dimension <= target_selection_kernel_dimension:
            break
        best_entry = None
        for candidate, jump in eligible:
            candidate_id = id(candidate)
            if candidate_id in selected_ids:
                continue
            image = np.asarray(jump @ current_basis, dtype=np.complex128)
            kernel_basis = _right_kernel_basis(image, tolerance=kernel_tolerance)
            next_dimension = int(kernel_basis.shape[1])
            score = (
                next_dimension,
                -max(candidate.residual_score_norm, candidate.residual_target_inflow_norm),
                -candidate.residual_target_inflow_norm,
                -candidate.total_inflow_norm,
                candidate.jump_nnz,
                candidate.n_terms,
                candidate.region_index,
                candidate.candidate_index,
            )
            if best_entry is None or score < best_entry[0]:
                best_entry = (score, candidate, jump, kernel_basis)

        if best_entry is None:
            break

        _score, candidate, jump, kernel_basis = best_entry
        next_dimension = int(kernel_basis.shape[1])
        if not allow_non_improving and next_dimension >= current_dimension:
            break

        selected_candidates.append(candidate)
        selected_jumps.append(jump)
        selected_ids.add(id(candidate))
        current_basis = current_basis @ kernel_basis
        current_dimension = next_dimension
        steps.append(
            TargetedResidualKernelJumpSelectionStep(
                step_index=len(steps),
                candidate=candidate,
                residual_kernel_dimension=current_dimension,
                n_selected_jumps=len(selected_jumps),
            )
        )

    selected_inflow_norm = None
    if states is not None:
        inflow_basis, _ = _normalize_state_columns(states, tolerance=kernel_tolerance)
        if inflow_basis.shape[0] != dim:
            raise ValueError("states has incompatible Hilbert-space dimension.")
        selected_inflow_norm = _multi_jump_projected_inflow_norm(
            jumps=base_jump_tuple + tuple(selected_jumps),
            state_basis=inflow_basis,
        )

    final_diagnostics = None
    if check_manifold_diagnostics and hamiltonian is not None and states is not None:
        final_diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=base_jump_tuple + tuple(selected_jumps),
            target_states=states,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=liouvillian_spectrum_method != "none",
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    return TargetedResidualKernelJumpSelectionReport(
        manifold_dimension=targeted_report.manifold_dimension,
        hilbert_dimension=targeted_report.hilbert_dimension,
        residual_dimension=residual_dimension,
        max_selected_jumps=max(int(max_selected_jumps), 0),
        target_residual_kernel_dimension=int(target_residual_kernel_dimension),
        targeted_report=targeted_report,
        selection_target=selection_target,
        initial_selection_kernel_dimension=initial_selection_kernel_dimension,
        target_selection_kernel_dimension=target_selection_kernel_dimension,
        base_jumps=base_jump_tuple,
        jumps=tuple(selected_jumps),
        steps=tuple(steps),
        final_diagnostics=final_diagnostics,
        selected_inflow_norm=selected_inflow_norm,
        kernel_tolerance=float(kernel_tolerance),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
    )
