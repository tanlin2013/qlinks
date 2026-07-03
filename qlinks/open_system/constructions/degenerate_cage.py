from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging.search import CageRecord
from qlinks.models.base import ModelBuildResult
from qlinks.models.local_terms import LocalTermKind
from qlinks.open_system.backend import OpenSystemBackendName
from qlinks.open_system.constructions.cage import _local_terms_by_operator_kind
from qlinks.open_system.diagnostics import (
    CommonKernelHamiltonianInvariantSectorReport,
    DarkManifoldDiagnostics,
    diagnose_common_kernel_h_invariant_sector,
    diagnose_dark_manifold,
)
from qlinks.open_system.local_recycling import (
    LocalRecyclingBuildResult,
    LocalSubspaceSupportReport,
    RecyclingJumpSource,
    build_local_recycling_jumps_from_subspace_regions,
    local_subspace_support_report_from_recycling_build_result,
)
from qlinks.open_system.manifold_detectors import (
    DressedManifoldDarkDetectorReport,
    ManifoldDarkOperatorBasisReport,
    RecycledManifoldCandidateFamilyKernelReport,
    RecycledManifoldDarkDetectorReport,
    RecycledManifoldJumpSelectionReport,
    RecycledManifoldResidualKernelReport,
    TargetedResidualKernelJumpSelectionReport,
    TargetedResidualKernelLinearSearchReport,
    diagnose_dressed_manifold_dark_detectors,
    diagnose_manifold_dark_operator_basis,
    diagnose_recycled_manifold_candidate_family_kernel,
    diagnose_recycled_manifold_dark_detectors,
    diagnose_recycled_manifold_residual_kernel,
    diagnose_targeted_residual_kernel_linear_search,
    expand_local_regions_to_cluster_unions,
    expand_local_regions_to_pair_unions,
    select_recycled_manifold_dark_detector_jumps,
    select_targeted_residual_kernel_jumps,
)
from qlinks.open_system.operators import lindblad_rhs_density_matrix
from qlinks.open_system.solvers import LindbladProblem

LocalRegionSource = Literal["kinetic", "potential", "all"]


def _as_csr(operator: Any) -> sp.csr_array:
    if hasattr(operator, "tocsr"):
        return operator.tocsr()
    return sp.csr_array(operator)


def _orthonormalize_state_matrix(
    states: NDArray[np.complex128],
    *,
    dim: int,
    tolerance: float,
) -> NDArray[np.complex128]:
    matrix = np.asarray(states, dtype=np.complex128)

    if matrix.ndim == 1:
        if matrix.size != dim:
            raise ValueError("state vector has incompatible dimension.")
        matrix = matrix.reshape(dim, 1)
    elif matrix.ndim == 2:
        if matrix.shape[0] == dim:
            pass
        elif matrix.shape[1] == dim:
            matrix = matrix.T
        else:
            raise ValueError("states must have shape (dim, n_states) or (n_states, dim).")
    else:
        raise ValueError("states must be one- or two-dimensional.")

    if matrix.shape[1] == 0:
        raise ValueError("states must contain at least one state.")

    q, r = np.linalg.qr(matrix)
    rank = int(np.count_nonzero(np.abs(np.diag(r)) > tolerance))
    if rank == 0:
        raise ValueError("states have numerical rank zero.")

    return q[:, :rank].astype(np.complex128, copy=False)


def _state_matrix_from_records(
    records: Sequence[CageRecord],
    *,
    hilbert_size: int,
) -> NDArray[np.complex128]:
    if len(records) == 0:
        raise ValueError("records must contain at least one CageRecord.")

    matrix = np.zeros((hilbert_size, len(records)), dtype=np.complex128)
    for column_index, record in enumerate(records):
        if record.full_state is not None:
            state = np.asarray(record.full_state, dtype=np.complex128)
            if state.shape != (hilbert_size,):
                raise ValueError(
                    "record.full_state has incompatible dimension: "
                    f"{state.shape} != {(hilbert_size,)}."
                )
            matrix[:, column_index] = state
        else:
            support = np.asarray(record.support, dtype=np.int64)
            local_state = np.asarray(record.local_state, dtype=np.complex128)
            if support.ndim != 1 or local_state.shape != support.shape:
                raise ValueError("record support and local_state have incompatible shapes.")
            if np.any(support < 0) or np.any(support >= hilbert_size):
                raise ValueError("record support contains out-of-range basis indices.")
            matrix[support, column_index] = local_state

    return matrix


def _validate_record_signatures(
    records: Sequence[CageRecord],
) -> tuple[int, int] | None:
    if len(records) == 0:
        return None

    signature = tuple(int(value) for value in records[0].signature)
    for record in records[1:]:
        if tuple(int(value) for value in record.signature) != signature:
            raise ValueError(
                "all cage records must have the same signature when "
                "validate_record_signature=True."
            )

    return signature


def _local_regions_from_model_terms(
    *,
    model: Any,
    local_term_kind: LocalTermKind | None,
    region_source: LocalRegionSource,
) -> tuple[tuple[int, ...], ...]:
    kinetic_terms, potential_terms, _ = _local_terms_by_operator_kind(
        model,
        term_kind=local_term_kind,
    )

    if region_source == "kinetic":
        terms = kinetic_terms
    elif region_source == "potential":
        terms = potential_terms
    elif region_source == "all":
        terms = kinetic_terms + potential_terms
    else:
        raise ValueError("region_source must be 'kinetic', 'potential', or 'all'.")

    regions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for term in terms:
        region = tuple(sorted(int(index) for index in term.support_variable_set))
        if len(region) == 0 or region in seen:
            continue
        seen.add(region)
        regions.append(region)

    if len(regions) == 0:
        raise ValueError(
            "Could not infer local regions from model local terms. "
            "Pass local_regions explicitly."
        )

    return tuple(regions)


def _normalize_local_regions(
    local_regions: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    regions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for region_like in local_regions:
        region = tuple(sorted(int(index) for index in region_like))
        if len(region) == 0:
            raise ValueError("local_regions must not contain empty regions.")
        if region in seen:
            continue
        seen.add(region)
        regions.append(region)

    if len(regions) == 0:
        raise ValueError("local_regions must contain at least one region.")

    return tuple(regions)


def _manifold_projector(
    manifold_basis: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return manifold_basis @ manifold_basis.conj().T


def _hamiltonian_closure_residual(
    *,
    hamiltonian: Any,
    manifold_basis: NDArray[np.complex128],
) -> float:
    hamiltonian_matrix = _as_csr(hamiltonian)
    action = np.asarray(hamiltonian_matrix @ manifold_basis, dtype=np.complex128)
    projected_action = manifold_basis @ (manifold_basis.conj().T @ action)
    return float(np.linalg.norm(action - projected_action))


def _max_jump_residual(
    *,
    jumps: tuple[Any, ...],
    manifold_basis: NDArray[np.complex128],
) -> tuple[float, tuple[float, ...]]:
    residuals = tuple(float(np.linalg.norm(_as_csr(jump) @ manifold_basis)) for jump in jumps)
    return (max(residuals) if residuals else 0.0), residuals


def _inflow_norm(
    *,
    jumps: tuple[Any, ...],
    manifold_basis: NDArray[np.complex128],
) -> float:
    total = 0.0
    for jump in jumps:
        jump_matrix = _as_csr(jump)
        adjoint_action = np.asarray(jump_matrix.conj().T @ manifold_basis, dtype=np.complex128)
        total += float(np.linalg.norm(adjoint_action) ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _manifold_density_matrix(
    manifold_basis: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    projector = _manifold_projector(manifold_basis)
    return projector / float(manifold_basis.shape[1])


def _format_float(value: float) -> str:
    return f"{value:.3e}"


@dataclass(frozen=True, slots=True)
class DegenerateCageJumpDesignWorkflowReport:
    """End-to-end jump-design workflow for a degenerate cage manifold.

    The report packages the currently successful cheap design loop:

    1. find collective dark detectors from a local operator basis,
    2. select a compact recycled-detector ``R D`` jump subset,
    3. diagnose the full recycled-detector family residual kernel,
    4. search direct local dark jumps that hit that residual kernel, and
    5. select a compact targeted subset against the combined common kernel.

    The final acceptance criterion is the common-kernel diagnostic stored in
    ``targeted_selection``; no Liouvillian spectrum is required unless requested.
    """

    dark_operator_report: ManifoldDarkOperatorBasisReport
    recycled_report: RecycledManifoldDarkDetectorReport
    recycled_selection: RecycledManifoldJumpSelectionReport
    family_report: RecycledManifoldCandidateFamilyKernelReport | None
    residual_report: RecycledManifoldResidualKernelReport | None
    targeted_report: TargetedResidualKernelLinearSearchReport | None
    targeted_selection: TargetedResidualKernelJumpSelectionReport | None
    h_invariant_report: CommonKernelHamiltonianInvariantSectorReport | None
    recycled_local_regions: tuple[tuple[int, ...], ...]
    targeted_local_regions: tuple[tuple[int, ...], ...]
    recycled_region_mode: str
    targeted_region_mode: str
    recycled_recycler_source: str
    targeted_operator_source: str
    design_mode: str = "full"
    early_stop_reason: str | None = None

    @property
    def manifold_dimension(self) -> int:
        return int(self.dark_operator_report.manifold_dimension)

    @property
    def hilbert_dimension(self) -> int:
        return int(self.dark_operator_report.hilbert_dimension)

    @property
    def jumps(self) -> tuple[sp.csr_array, ...]:
        """Return the final combined jump list.

        In ``design_mode="h_invariant_fast"`` the workflow may stop after
        recycled-jump selection, before residual and targeted stages are run.
        In that case the final jump list is simply the recycled subset.
        """
        if self.targeted_selection is None:
            return self.recycled_selection.jumps
        return self.targeted_selection.all_jumps

    @property
    def recycled_jumps(self) -> tuple[sp.csr_array, ...]:
        return self.recycled_selection.jumps

    @property
    def targeted_jumps(self) -> tuple[sp.csr_array, ...]:
        if self.targeted_selection is None:
            return ()
        return self.targeted_selection.jumps

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def final_diagnostics(self) -> DarkManifoldDiagnostics | None:
        if self.targeted_selection is None:
            return self.recycled_selection.final_diagnostics
        return self.targeted_selection.final_diagnostics

    @property
    def complement_common_kernel_removed(self) -> bool | None:
        if self.targeted_selection is None:
            return self.recycled_selection.complement_common_kernel_removed
        return self.targeted_selection.combined_complement_common_kernel_removed

    @property
    def likely_successful_common_kernel_design(self) -> bool:
        return self.complement_common_kernel_removed is True

    @property
    def likely_successful_h_invariant_design(self) -> bool | None:
        if self.h_invariant_report is None:
            return None
        return self.h_invariant_report.likely_attractive_by_h_invariant_kernel

    def to_lindblad_problem(
        self,
        *,
        hamiltonian: Any,
        backend: str | None = None,
    ) -> LindbladProblem:
        return LindbladProblem(
            hamiltonian=hamiltonian,
            jumps=self.jumps,
            backend="scipy" if backend is None else backend,
        )

    def to_summary_dict(self) -> dict[str, object]:
        final_diagnostics = self.final_diagnostics
        family_report = self.family_report
        residual_report = self.residual_report
        targeted_report = self.targeted_report
        targeted_selection = self.targeted_selection

        if targeted_selection is None:
            n_targeted_jumps = 0
            targeted_selected_family_residual_kernel_dimension = None
            targeted_selected_candidates_remove_family_residual = None
            targeted_selection_target = None
            targeted_initial_selection_kernel_dimension = None
            targeted_final_selection_kernel_dimension = None
            targeted_selection_kernel_removed = None
            targeted_selection_removes_combined_kernel = None
            combined_bad_common_jump_kernel_dimension = (
                self.recycled_selection.final_bad_common_jump_kernel_dimension
            )
            combined_complement_common_kernel_removed = (
                self.recycled_selection.complement_common_kernel_removed
            )
            combined_inflow_norm = self.recycled_selection.final_inflow_norm
        else:
            n_targeted_jumps = targeted_selection.n_selected_jumps
            targeted_selected_family_residual_kernel_dimension = (
                targeted_selection.final_residual_kernel_dimension
            )
            targeted_selected_candidates_remove_family_residual = (
                targeted_selection.residual_kernel_removed
            )
            targeted_selection_target = targeted_selection.selection_target
            targeted_initial_selection_kernel_dimension = (
                targeted_selection.initial_selection_kernel_dimension
            )
            targeted_final_selection_kernel_dimension = (
                targeted_selection.final_selection_kernel_dimension
            )
            targeted_selection_kernel_removed = targeted_selection.selection_kernel_removed
            targeted_selection_removes_combined_kernel = (
                targeted_selection.selection_kernel_removed
                if targeted_selection.selection_target == "combined_common_kernel"
                else None
            )
            combined_bad_common_jump_kernel_dimension = (
                targeted_selection.combined_bad_common_jump_kernel_dimension
            )
            combined_complement_common_kernel_removed = (
                targeted_selection.combined_complement_common_kernel_removed
            )
            combined_inflow_norm = targeted_selection.combined_inflow_norm

        targeted_failure_counts = (
            None if targeted_report is None else targeted_report.targeted_search_failure_counts
        )
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "manifold_dimension": self.manifold_dimension,
            "design_mode": self.design_mode,
            "early_stop_reason": self.early_stop_reason,
            "n_final_jumps": self.n_jumps,
            "n_recycled_jumps": self.recycled_selection.n_selected_jumps,
            "n_targeted_jumps": n_targeted_jumps,
            "recycled_region_mode": self.recycled_region_mode,
            "targeted_region_mode": self.targeted_region_mode,
            "n_recycled_regions": len(self.recycled_local_regions),
            "n_targeted_regions": len(self.targeted_local_regions),
            "max_recycled_region_size": max(
                (len(region) for region in self.recycled_local_regions),
                default=0,
            ),
            "max_targeted_region_size": max(
                (len(region) for region in self.targeted_local_regions),
                default=0,
            ),
            "recycled_recycler_source": self.recycled_recycler_source,
            "targeted_operator_source": self.targeted_operator_source,
            "dark_detector_nullity": self.dark_operator_report.detector_nullity,
            "recycled_candidate_pool_size": self.recycled_selection.candidate_pool_size,
            "recycled_bad_common_kernel_dimension": (
                self.recycled_selection.final_bad_common_jump_kernel_dimension
            ),
            "recycled_complement_kernel_removed": (
                self.recycled_selection.complement_common_kernel_removed
            ),
            "recycled_inflow_norm": self.recycled_selection.final_inflow_norm,
            "family_candidate_jumps": (
                None if family_report is None else family_report.n_candidate_jumps
            ),
            "family_bad_common_jump_kernel_dimension": (
                None
                if family_report is None
                else family_report.family_bad_common_jump_kernel_dimension
            ),
            "family_complement_kernel_removed": (
                None if family_report is None else family_report.complement_common_kernel_removed
            ),
            "residual_dimension": (
                None if residual_report is None else residual_report.residual_dimension
            ),
            "targeted_candidates": (
                None if targeted_report is None else targeted_report.n_candidates
            ),
            "targeted_generated_candidate_modes": (
                None if targeted_report is None else targeted_report.n_generated_candidate_modes
            ),
            "targeted_reported_candidate_modes": (
                None if targeted_report is None else targeted_report.n_reported_candidate_modes
            ),
            "targeted_regions_skipped_by_local_dim": (
                None if targeted_report is None else targeted_report.n_regions_skipped_by_local_dim
            ),
            "targeted_regions_with_no_recycler_specs": (
                None
                if targeted_report is None
                else targeted_report.n_regions_with_no_recycler_specs
            ),
            "targeted_regions_with_no_nonzero_local_operators": (
                None
                if targeted_report is None
                else targeted_report.n_regions_with_no_nonzero_local_operators
            ),
            "targeted_regions_with_zero_dark_nullity": (
                None
                if targeted_report is None
                else targeted_report.n_regions_with_zero_dark_nullity
            ),
            "targeted_regions_with_dark_nullity": (
                None if targeted_report is None else targeted_report.n_regions_with_dark_nullity
            ),
            "targeted_regions_with_zero_residual_inflow": (
                None
                if targeted_report is None
                else targeted_report.n_regions_with_zero_residual_inflow
            ),
            "targeted_search_failure_counts": targeted_failure_counts,
            "targeted_candidates_hitting_residual": (
                None if targeted_report is None else targeted_report.n_candidates_hitting_residual
            ),
            "targeted_reported_family_residual_kernel_dimension": (
                None
                if targeted_report is None
                else targeted_report.reported_candidate_family_residual_kernel_dimension
            ),
            "targeted_reported_candidates_remove_family_residual": (
                None
                if targeted_report is None
                else targeted_report.reported_candidates_remove_family_residual_kernel
            ),
            # Backward-compatible alias. Prefer the explicit
            # ``targeted_reported_candidates_remove_family_residual`` key in new code.
            "targeted_reported_candidates_remove_residual": (
                None
                if targeted_report is None
                else targeted_report.reported_candidates_remove_residual_kernel
            ),
            "targeted_selected_family_residual_kernel_dimension": (
                targeted_selected_family_residual_kernel_dimension
            ),
            "targeted_selected_candidates_remove_family_residual": (
                targeted_selected_candidates_remove_family_residual
            ),
            "targeted_selection_target": targeted_selection_target,
            "targeted_initial_selection_kernel_dimension": (
                targeted_initial_selection_kernel_dimension
            ),
            "targeted_final_selection_kernel_dimension": targeted_final_selection_kernel_dimension,
            "targeted_selection_kernel_removed": targeted_selection_kernel_removed,
            "targeted_selection_removes_combined_kernel": (
                targeted_selection_removes_combined_kernel
            ),
            "combined_bad_common_jump_kernel_dimension": combined_bad_common_jump_kernel_dimension,
            "combined_complement_common_kernel_removed": (
                combined_complement_common_kernel_removed
            ),
            "combined_inflow_norm": combined_inflow_norm,
            "h_invariant_bad_kernel_dimension": (
                None
                if self.h_invariant_report is None
                else self.h_invariant_report.bad_h_invariant_kernel_dimension
            ),
            "h_invariant_common_bad_kernel_dimension": (
                None
                if self.h_invariant_report is None
                else self.h_invariant_report.bad_common_jump_kernel_dimension
            ),
            "h_invariant_leakage_norm_from_bad_kernel": (
                None
                if self.h_invariant_report is None
                else self.h_invariant_report.h_leakage_norm_from_bad_kernel
            ),
            "likely_successful_h_invariant_design": (self.likely_successful_h_invariant_design),
            "max_target_jump_residual": (
                None if final_diagnostics is None else final_diagnostics.max_target_jump_residual
            ),
            "h_closure_residual": (
                None
                if final_diagnostics is None
                else final_diagnostics.hamiltonian_closure_residual
            ),
            "likely_successful_common_kernel_design": (self.likely_successful_common_kernel_design),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DegenerateCageJumpDesignWorkflowReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        summary = self.to_summary_dict()
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        for key in (
            "hilbert_dimension",
            "manifold_dimension",
            "n_final_jumps",
            "n_recycled_jumps",
            "n_targeted_jumps",
            "dark_detector_nullity",
            "combined_bad_common_jump_kernel_dimension",
            "combined_complement_common_kernel_removed",
            "combined_inflow_norm",
            "likely_successful_common_kernel_design",
            "likely_successful_h_invariant_design",
        ):
            overview.add_row(key.replace("_", " "), str(summary[key]))

        stages = Table(title="Workflow stages")
        stages.add_column("stage", style="bold")
        stages.add_column("key result")
        stages.add_row(
            "dark detectors",
            f"nullity={self.dark_operator_report.detector_nullity}",
        )
        stages.add_row(
            "recycled selection",
            (
                f"selected={self.recycled_selection.n_selected_jumps}, "
                f"bad={self.recycled_selection.final_bad_common_jump_kernel_dimension}"
            ),
        )
        if self.family_report is None:
            stages.add_row("full recycled family", "skipped")
        else:
            stages.add_row(
                "full recycled family",
                (
                    f"jumps={self.family_report.n_candidate_jumps}, "
                    f"bad={self.family_report.family_bad_common_jump_kernel_dimension}"
                ),
            )
        stages.add_row(
            "residual kernel",
            (
                "skipped"
                if self.residual_report is None
                else f"dim={self.residual_report.residual_dimension}"
            ),
        )
        if self.targeted_report is None:
            stages.add_row("targeted search", "skipped")
        else:
            stages.add_row(
                "targeted search",
                (
                    f"candidates={self.targeted_report.n_candidates}, "
                    f"hits family residual={self.targeted_report.n_candidates_hitting_residual}, "
                    f"family residual dim="
                    f"{self.targeted_report.reported_candidate_family_residual_kernel_dimension}"
                ),
            )
        if self.targeted_selection is None:
            stages.add_row("targeted selection", "skipped")
        else:
            stages.add_row(
                "targeted selection",
                (
                    f"selected={self.targeted_selection.n_selected_jumps}, "
                    f"target={self.targeted_selection.selection_target}, "
                    f"selection dim={self.targeted_selection.final_selection_kernel_dimension}, "
                    f"combined bad="
                    f"{self.targeted_selection.combined_bad_common_jump_kernel_dimension}"
                ),
            )
        if self.h_invariant_report is not None:
            stages.add_row(
                "H-invariant kernel",
                (
                    f"bad={self.h_invariant_report.bad_h_invariant_kernel_dimension}, "
                    f"success={self.h_invariant_report.likely_attractive_by_h_invariant_kernel}"
                ),
            )

        return Panel(
            Group(overview, stages),
            title=Text("Degenerate cage jump-design workflow", style="bold cyan"),
            border_style=(
                "green"
                if (
                    self.likely_successful_common_kernel_design
                    or self.likely_successful_h_invariant_design is True
                )
                else "yellow"
            ),
        )


@dataclass(frozen=True, slots=True)
class DegenerateCageLindbladConstruction:
    """Lindblad construction targeting a cage-state manifold in one sector.

    The target is the projector onto ``manifold_basis`` rather than a single
    vector.  Local reset jumps are built from the local RDM support of the
    normalized manifold projector, so every jump annihilates every vector in the
    target manifold.
    """

    manifold_basis: NDArray[np.complex128]
    jumps: tuple[Any, ...]
    local_regions: tuple[tuple[int, ...], ...]
    recycling_build_result: LocalRecyclingBuildResult
    open_system_backend: OpenSystemBackendName
    recycling_jump_source: RecyclingJumpSource
    record_signature: tuple[int, int] | None
    hamiltonian_closure_residual: float
    max_jump_residual: float
    jump_residuals: tuple[float, ...]
    inflow_norm: float
    liouvillian_residual: float | None = None

    @property
    def hilbert_dimension(self) -> int:
        return int(self.manifold_basis.shape[0])

    @property
    def manifold_dimension(self) -> int:
        return int(self.manifold_basis.shape[1])

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def target_density_matrix(self) -> NDArray[np.complex128]:
        return _manifold_density_matrix(self.manifold_basis)

    @property
    def local_subspace_support_report(self) -> LocalSubspaceSupportReport:
        """Explain local manifold support/nullity for each recycling region."""
        return local_subspace_support_report_from_recycling_build_result(
            self.recycling_build_result
        )

    def __rich__(self):
        return self.to_rich()

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "manifold_dimension": self.manifold_dimension,
            "record_signature": self.record_signature,
            "n_jumps": self.n_jumps,
            "n_regions": len(self.local_regions),
            "local_regions": self.local_regions,
            "recycling_jump_source": self.recycling_jump_source,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_jump_residual": self.max_jump_residual,
            "jump_residuals": self.jump_residuals,
            "inflow_norm": self.inflow_norm,
            "liouvillian_residual": self.liouvillian_residual,
            "local_subspace_support": self.local_subspace_support_report.to_summary_dict(),
            "recycling_variable_indices": self.recycling_build_result.variable_indices,
            "recycling_alpha_beta_indices": self.recycling_build_result.alpha_beta_indices,
        }

    def to_rich(self, *, max_regions: int = 24):
        """Return a rich renderable summary of the degenerate construction."""
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DegenerateCageLindbladConstruction.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("record signature", str(self.record_signature))
        overview.add_row("jumps", str(self.n_jumps))
        overview.add_row("regions", str(len(self.local_regions)))
        overview.add_row("recycling source", str(self.recycling_jump_source))

        checks = Table(title="Construction checks")
        checks.add_column("quantity", style="bold")
        checks.add_column("value", justify="right")
        checks.add_row("H closure residual", _format_float(self.hamiltonian_closure_residual))
        checks.add_row("max ||J_mu P_M||", _format_float(self.max_jump_residual))
        checks.add_row("inflow norm", _format_float(self.inflow_norm))
        checks.add_row(
            "||L(P_M/m)||",
            (
                "not checked"
                if self.liouvillian_residual is None
                else _format_float(float(self.liouvillian_residual))
            ),
        )

        return Panel(
            Group(
                overview,
                checks,
                self.local_subspace_support_report.to_rich(max_regions=max_regions),
            ),
            title=Text("Degenerate cage Lindblad construction", style="bold cyan"),
            border_style="cyan",
        )

    def to_lindblad_problem(
        self,
        *,
        hamiltonian: Any,
        backend: str | None = None,
    ) -> LindbladProblem:
        return LindbladProblem(
            hamiltonian=hamiltonian,
            jumps=self.jumps,
            backend=self.open_system_backend if backend is None else backend,
        )

    def build_liouvillian(
        self,
        hamiltonian: Any,
        *,
        sparse_format: str = "csc",
        backend: str | None = None,
    ) -> Any:
        problem = self.to_lindblad_problem(
            hamiltonian=hamiltonian,
            backend=backend,
        )
        return problem.build_liouvillian(sparse_format=sparse_format)

    def diagnose_manifold(
        self,
        *,
        hamiltonian: Any,
        backend: str | None = None,
        kernel_tolerance: float = 1e-10,
        liouvillian_zero_tolerance: float = 1e-9,
        check_liouvillian_spectrum: bool = True,
        max_liouvillian_dense_dimension: int = 4096,
        liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
        sparse_liouvillian_eigenvalue_count: int = 32,
    ) -> DarkManifoldDiagnostics:
        """Return manifold-aware dark/attractiveness diagnostics."""
        return diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=self.jumps,
            target_states=self.manifold_basis,
            backend=self.open_system_backend if backend is None else backend,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=check_liouvillian_spectrum,
            max_liouvillian_dense_dimension=max_liouvillian_dense_dimension,
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    def diagnose_dark_operator_basis(
        self,
        *,
        operators: tuple[Any, ...] | list[Any],
        operator_names: tuple[str, ...] | list[str] | None = None,
        tolerance: float = 1.0e-10,
        coefficient_tolerance: float = 1.0e-8,
        max_candidates: int | None = 16,
    ) -> ManifoldDarkOperatorBasisReport:
        """Find collective operator combinations ``D`` with ``D P_M = 0``.

        This is useful when every small region has full local RDM support, so no
        strictly local parent projector exists, but sums of local terms may still
        annihilate the manifold by interference/cancellation.
        """
        return diagnose_manifold_dark_operator_basis(
            states=self.manifold_basis,
            operators=operators,
            operator_names=operator_names,
            tolerance=tolerance,
            coefficient_tolerance=coefficient_tolerance,
            max_candidates=max_candidates,
        )

    def diagnose_dressed_dark_detectors(
        self,
        *,
        detector_operators: tuple[Any, ...] | list[Any],
        left_multipliers: tuple[Any, ...] | list[Any],
        detector_coefficients: NDArray[np.complex128] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        left_multiplier_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        max_detectors: int | None = None,
        sort_by_inflow: bool = True,
    ) -> DressedManifoldDarkDetectorReport:
        """Test dressed jumps ``J = V D`` built from manifold-dark detectors.

        A nonzero inflow norm is a necessary condition for attraction into the
        target manifold.  This diagnostic does not by itself prove absence of
        invariant complement sectors; use ``diagnose_manifold`` on selected
        jumps for that stronger check.
        """
        return diagnose_dressed_manifold_dark_detectors(
            states=self.manifold_basis,
            detector_operators=detector_operators,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            left_multipliers=left_multipliers,
            detector_operator_names=detector_operator_names,
            left_multiplier_names=left_multiplier_names,
            detector_names=detector_names,
            tolerance=tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            sort_by_inflow=sort_by_inflow,
        )

    def local_region_pair_unions(
        self,
        *,
        pair_mode: Literal["overlap", "all"] = "overlap",
        min_overlap: int = 1,
        max_region_size: int | None = None,
        include_single_regions: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        """Return two-region recycler supports derived from this construction.

        For QDM plaquette regions, ``pair_mode="overlap"`` and
        ``min_overlap=1`` generate adjacent two-plaquette unions.  These
        regions can be passed as ``local_regions`` to recycled-detector
        diagnostics/selectors.
        """
        return expand_local_regions_to_pair_unions(
            self.local_regions,
            pair_mode=pair_mode,
            min_overlap=min_overlap,
            max_region_size=max_region_size,
            include_single_regions=include_single_regions,
        )

    def local_region_cluster_unions(
        self,
        *,
        cluster_size: int = 3,
        cluster_mode: Literal["overlap_connected", "all"] = "overlap_connected",
        min_overlap: int = 1,
        max_region_size: int | None = None,
        include_single_regions: bool = False,
        include_smaller_clusters: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        """Return multi-region recycler supports derived from this construction.

        For QDM plaquette regions, ``cluster_mode="overlap_connected"`` and
        ``min_overlap=1`` generate connected multi-plaquette patches.  This is
        useful when pair-union direct targeted jumps cannot hit a residual
        complement kernel.
        """
        return expand_local_regions_to_cluster_unions(
            self.local_regions,
            cluster_size=cluster_size,
            cluster_mode=cluster_mode,
            min_overlap=min_overlap,
            max_region_size=max_region_size,
            include_single_regions=include_single_regions,
            include_smaller_clusters=include_smaller_clusters,
        )

    def diagnose_recycled_dark_detectors(
        self,
        *,
        basis_configs: NDArray[np.integer],
        detector_operators: tuple[Any, ...] | list[Any],
        local_regions: Sequence[Sequence[int]] | None = None,
        detector_coefficients: NDArray[np.complex128] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        recycler_source: Literal[
            "matrix_units",
            "rdm_support_matrix_units",
        ] = "rdm_support_matrix_units",
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        max_detectors: int | None = None,
        max_report_candidates: int | None = 256,
        sort_by_inflow: bool = True,
    ) -> RecycledManifoldDarkDetectorReport:
        """Test local recycler-dressed jumps ``J = R D``.

        This is the degenerate-manifold analogue of the successful single-cage
        recycler idea.  The right detector ``D`` guarantees
        ``J P_M = R D P_M = 0``.  The local recycler ``R`` is therefore allowed
        to be a general local matrix unit or RDM-support reset operator, and is
        scored by its direct inflow ``||P_M J (I-P_M)||_F``.
        """
        regions = (
            self.local_regions if local_regions is None else _normalize_local_regions(local_regions)
        )
        return diagnose_recycled_manifold_dark_detectors(
            states=self.manifold_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            local_regions=regions,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=max_report_candidates,
            sort_by_inflow=sort_by_inflow,
        )

    def diagnose_recycled_candidate_family_kernel(
        self,
        *,
        hamiltonian: Any,
        basis_configs: NDArray[np.integer],
        detector_operators: tuple[Any, ...] | list[Any],
        local_regions: Sequence[Sequence[int]] | None = None,
        detector_coefficients: NDArray[np.complex128] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        candidate_report: RecycledManifoldDarkDetectorReport | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        recycler_source: Literal[
            "matrix_units",
            "rdm_support_matrix_units",
        ] = "rdm_support_matrix_units",
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        kernel_tolerance: float = 1.0e-10,
        liouvillian_zero_tolerance: float = 1.0e-9,
        max_detectors: int | None = None,
        expand_candidate_report: bool = True,
        kernel_method: Literal["streamed", "diagnostics"] = "streamed",
        store_candidate_jumps: bool = False,
    ) -> RecycledManifoldCandidateFamilyKernelReport:
        """
        Check whether the full recycled-detector candidate family removes the complement kernel.
        """
        regions = (
            self.local_regions if local_regions is None else _normalize_local_regions(local_regions)
        )
        return diagnose_recycled_manifold_candidate_family_kernel(
            hamiltonian=hamiltonian,
            states=self.manifold_basis,
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
            kernel_method=kernel_method,
            store_candidate_jumps=store_candidate_jumps,
        )

    def diagnose_recycled_residual_kernel(
        self,
        *,
        hamiltonian: Any,
        basis_configs: NDArray[np.integer],
        detector_operators: tuple[Any, ...] | list[Any],
        local_regions: Sequence[Sequence[int]] | None = None,
        detector_coefficients: NDArray[np.complex128] | None = None,
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
        local_support_regions: Sequence[Sequence[int]] | None = None,
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
        """Diagnose the residual bad kernel left by a recycled-detector family."""
        regions = (
            self.local_regions if local_regions is None else _normalize_local_regions(local_regions)
        )
        support_regions = (
            None
            if local_support_regions is None
            else _normalize_local_regions(local_support_regions)
        )
        return diagnose_recycled_manifold_residual_kernel(
            hamiltonian=hamiltonian,
            states=self.manifold_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            candidate_report=candidate_report,
            family_report=family_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            operator_groups=operator_groups,
            local_support_regions=support_regions,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            max_detectors=max_detectors,
            expand_candidate_report=expand_candidate_report,
            max_operator_entries=max_operator_entries,
        )

    def diagnose_targeted_residual_kernel_linear_search(
        self,
        *,
        basis_configs: NDArray[np.integer],
        local_regions: Sequence[Sequence[int]] | None = None,
        residual_basis: NDArray[np.complex128] | None = None,
        residual_report: RecycledManifoldResidualKernelReport | None = None,
        detector_operators: tuple[Any, ...] | list[Any] | None = None,
        residual_family_local_regions: Sequence[Sequence[int]] | None = None,
        detector_coefficients: NDArray[np.complex128] | None = None,
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
        """Search local dark jumps that directly hit a residual bad kernel."""
        regions = (
            self.local_regions if local_regions is None else _normalize_local_regions(local_regions)
        )
        family_regions = (
            None
            if residual_family_local_regions is None
            else _normalize_local_regions(residual_family_local_regions)
        )
        return diagnose_targeted_residual_kernel_linear_search(
            states=self.manifold_basis,
            basis_configs=basis_configs,
            local_regions=regions,
            residual_basis=residual_basis,
            residual_report=residual_report,
            detector_operators=detector_operators,
            residual_family_local_regions=family_regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            candidate_report=candidate_report,
            family_report=family_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            operator_source=operator_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            kernel_tolerance=kernel_tolerance,
            max_detectors=max_detectors,
            max_modes_per_region=max_modes_per_region,
            max_report_candidates=max_report_candidates,
            max_local_dim=max_local_dim,
            coefficient_tolerance=coefficient_tolerance,
        )

    def select_targeted_residual_kernel_jumps(
        self,
        *,
        targeted_report: TargetedResidualKernelLinearSearchReport,
        hamiltonian: Any | None = None,
        base_jumps: Sequence[Any] = (),
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
        """Greedily select targeted residual-kernel jumps.

        Use ``selection_target="combined_common_kernel"`` to keep adding
        targeted jumps until the bad common jump-kernel of ``base_jumps`` plus
        selected targeted jumps is removed.  If ``hamiltonian`` is supplied, the
        returned report also contains a dark-manifold diagnostic for the
        combined jump list.
        """
        return select_targeted_residual_kernel_jumps(
            targeted_report=targeted_report,
            hamiltonian=hamiltonian,
            states=self.manifold_basis,
            base_jumps=tuple(base_jumps),
            max_selected_jumps=max_selected_jumps,
            target_residual_kernel_dimension=target_residual_kernel_dimension,
            selection_target=selection_target,
            allow_non_improving=allow_non_improving,
            kernel_tolerance=kernel_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_manifold_diagnostics=check_manifold_diagnostics,
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    def select_recycled_dark_detector_jumps(
        self,
        *,
        hamiltonian: Any,
        basis_configs: NDArray[np.integer],
        detector_operators: tuple[Any, ...] | list[Any],
        local_regions: Sequence[Sequence[int]] | None = None,
        detector_coefficients: NDArray[np.complex128] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        candidate_report: RecycledManifoldDarkDetectorReport | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        recycler_source: Literal[
            "matrix_units",
            "rdm_support_matrix_units",
        ] = "rdm_support_matrix_units",
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        kernel_tolerance: float = 1.0e-10,
        liouvillian_zero_tolerance: float = 1.0e-9,
        max_detectors: int | None = None,
        max_candidate_pool: int | None = 128,
        max_selected_jumps: int = 16,
        target_bad_kernel_dimension: int = 0,
        allow_non_improving: bool = False,
        expand_candidate_report: bool = False,
        selection_strategy: Literal["diagnostics", "kernel_projection"] = "diagnostics",
    ) -> RecycledManifoldJumpSelectionReport:
        """Greedily select a small local recycled-detector jump subset.

        This promotes the necessary-condition recycler scan into an actual jump
        set and checks, after each addition, whether the common jump kernel in
        the complement of the target manifold has been removed.
        """
        regions = (
            self.local_regions if local_regions is None else _normalize_local_regions(local_regions)
        )
        return select_recycled_manifold_dark_detector_jumps(
            hamiltonian=hamiltonian,
            states=self.manifold_basis,
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
            max_candidate_pool=max_candidate_pool,
            max_selected_jumps=max_selected_jumps,
            target_bad_kernel_dimension=target_bad_kernel_dimension,
            allow_non_improving=allow_non_improving,
            expand_candidate_report=expand_candidate_report,
            selection_strategy=selection_strategy,
        )

    def design_dark_manifold_jumps(
        self,
        *,
        hamiltonian: Any,
        basis_configs: NDArray[np.integer],
        detector_operators: tuple[Any, ...] | list[Any],
        detector_coefficients: NDArray[np.complex128] | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        recycled_report: RecycledManifoldDarkDetectorReport | None = None,
        recycled_selection: RecycledManifoldJumpSelectionReport | None = None,
        family_report: RecycledManifoldCandidateFamilyKernelReport | None = None,
        residual_report: RecycledManifoldResidualKernelReport | None = None,
        targeted_report: TargetedResidualKernelLinearSearchReport | None = None,
        recycled_local_regions: Sequence[Sequence[int]] | None = None,
        targeted_local_regions: Sequence[Sequence[int]] | None = None,
        local_region_mode: Literal[
            "construction",
            "pair_unions",
            "cluster_unions",
        ] = "pair_unions",
        recycled_region_mode: (
            Literal[
                "construction",
                "pair_unions",
                "cluster_unions",
            ]
            | None
        ) = None,
        targeted_region_mode: (
            Literal[
                "construction",
                "pair_unions",
                "cluster_unions",
            ]
            | None
        ) = None,
        pair_mode: Literal["overlap", "all"] = "overlap",
        min_pair_overlap: int = 1,
        max_pair_region_size: int | None = 7,
        include_single_regions_in_pairs: bool = False,
        cluster_size: int = 3,
        recycled_cluster_size: int | None = None,
        targeted_cluster_size: int | None = None,
        cluster_mode: Literal["overlap_connected", "all"] = "overlap_connected",
        min_cluster_overlap: int = 1,
        max_cluster_region_size: int | None = None,
        include_single_regions_in_clusters: bool = False,
        include_smaller_clusters: bool = False,
        recycled_recycler_source: Literal[
            "matrix_units",
            "rdm_support_matrix_units",
        ] = "matrix_units",
        targeted_operator_source: Literal[
            "matrix_units",
            "rdm_support_matrix_units",
        ] = "matrix_units",
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        kernel_tolerance: float = 1.0e-10,
        liouvillian_zero_tolerance: float = 1.0e-9,
        max_detectors: int | None = None,
        dark_operator_max_candidates: int | None = 16,
        max_recycled_report_candidates: int | None = None,
        max_recycled_candidate_pool: int | None = None,
        max_recycled_selected_jumps: int = 16,
        recycled_target_bad_kernel_dimension: int = 0,
        recycled_allow_non_improving: bool = False,
        recycled_selection_strategy: Literal[
            "diagnostics",
            "kernel_projection",
        ] = "kernel_projection",
        residual_operator_groups: (
            tuple[
                tuple[str, tuple[Any, ...] | list[Any], tuple[str, ...] | list[str] | None],
                ...,
            ]
            | None
        ) = None,
        residual_local_support_regions: Sequence[Sequence[int]] | None = None,
        max_residual_operator_entries: int | None = 64,
        max_targeted_modes_per_region: int = 3,
        max_targeted_report_candidates: int | None = 64,
        max_targeted_local_dim: int | None = 20,
        max_targeted_selected_jumps: int = 16,
        targeted_selection_target: Literal[
            "reported_residual_kernel",
            "combined_common_kernel",
        ] = "combined_common_kernel",
        targeted_target_residual_kernel_dimension: int = 0,
        targeted_allow_non_improving: bool = False,
        check_final_manifold_diagnostics: bool = True,
        check_h_invariant_sector: bool = True,
        liouvillian_spectrum_method: Literal[
            "auto",
            "dense",
            "sparse",
            "none",
        ] = "none",
        sparse_liouvillian_eigenvalue_count: int = 32,
        design_mode: Literal[
            "full",
            "h_invariant_fast",
        ] = "full",
    ) -> DegenerateCageJumpDesignWorkflowReport:
        """Run the reusable cheap jump-design workflow for a dark manifold.

        The default region/source choices are tuned for QDM-style plaquette
        cages: adjacent two-plaquette unions and matrix-unit recyclers.  The
        method avoids Liouvillian spectra by default and uses common-kernel
        diagnostics as the acceptance criterion.

        In ``design_mode="h_invariant_fast"``, the workflow first runs only
        the dark-detector and recycled-jump stages, then checks whether the
        remaining common-kernel complement has no Hamiltonian-invariant sector.
        If that cheaper physical criterion succeeds, it returns immediately and
        skips the full-family residual scan and targeted local-jump search.
        """

        def resolve_regions(
            explicit_regions: Sequence[Sequence[int]] | None,
            *,
            mode: Literal["construction", "pair_unions", "cluster_unions"],
            cluster_size_override: int | None,
        ) -> tuple[tuple[int, ...], ...]:
            if explicit_regions is not None:
                return _normalize_local_regions(explicit_regions)
            if mode == "construction":
                return self.local_regions
            if mode == "pair_unions":
                return self.local_region_pair_unions(
                    pair_mode=pair_mode,
                    min_overlap=min_pair_overlap,
                    max_region_size=max_pair_region_size,
                    include_single_regions=include_single_regions_in_pairs,
                )
            if mode == "cluster_unions":
                return self.local_region_cluster_unions(
                    cluster_size=cluster_size_override or cluster_size,
                    cluster_mode=cluster_mode,
                    min_overlap=min_cluster_overlap,
                    max_region_size=max_cluster_region_size,
                    include_single_regions=include_single_regions_in_clusters,
                    include_smaller_clusters=include_smaller_clusters,
                )
            raise ValueError(
                'region mode must be "construction", "pair_unions", or "cluster_unions".'
            )

        if design_mode not in {"full", "h_invariant_fast"}:
            raise ValueError('design_mode must be "full" or "h_invariant_fast".')

        resolved_recycled_region_mode = recycled_region_mode or local_region_mode
        resolved_targeted_region_mode = targeted_region_mode or local_region_mode
        recycled_regions = resolve_regions(
            recycled_local_regions,
            mode=resolved_recycled_region_mode,
            cluster_size_override=recycled_cluster_size,
        )
        targeted_regions: tuple[tuple[int, ...], ...] = ()
        if len(recycled_regions) == 0:
            raise ValueError("recycled local-region list is empty.")

        if dark_operator_report is None:
            dark_operator_report = self.diagnose_dark_operator_basis(
                operators=detector_operators,
                operator_names=detector_operator_names,
                tolerance=tolerance,
                max_candidates=dark_operator_max_candidates,
            )

        if recycled_report is None:
            recycled_report = self.diagnose_recycled_dark_detectors(
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=recycled_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycled_recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                max_detectors=max_detectors,
                max_report_candidates=max_recycled_report_candidates,
            )

        if recycled_selection is None:
            recycled_selection = self.select_recycled_dark_detector_jumps(
                hamiltonian=hamiltonian,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=recycled_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                candidate_report=recycled_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycled_recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                kernel_tolerance=kernel_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                max_detectors=max_detectors,
                max_candidate_pool=max_recycled_candidate_pool,
                max_selected_jumps=max_recycled_selected_jumps,
                target_bad_kernel_dimension=recycled_target_bad_kernel_dimension,
                allow_non_improving=recycled_allow_non_improving,
                expand_candidate_report=True,
                selection_strategy=recycled_selection_strategy,
            )

        if design_mode == "h_invariant_fast" and check_h_invariant_sector:
            recycled_h_invariant_report = diagnose_common_kernel_h_invariant_sector(
                hamiltonian=hamiltonian,
                jumps=recycled_selection.jumps,
                target_states=self.manifold_basis,
                kernel_tolerance=kernel_tolerance,
            )
            if recycled_h_invariant_report.likely_attractive_by_h_invariant_kernel:
                return DegenerateCageJumpDesignWorkflowReport(
                    dark_operator_report=dark_operator_report,
                    recycled_report=recycled_report,
                    recycled_selection=recycled_selection,
                    family_report=None,
                    residual_report=None,
                    targeted_report=None,
                    targeted_selection=None,
                    h_invariant_report=recycled_h_invariant_report,
                    recycled_local_regions=recycled_regions,
                    targeted_local_regions=(),
                    recycled_region_mode=(
                        resolved_recycled_region_mode
                        if recycled_local_regions is None
                        else "explicit"
                    ),
                    targeted_region_mode="skipped",
                    recycled_recycler_source=recycled_recycler_source,
                    targeted_operator_source=targeted_operator_source,
                    design_mode=design_mode,
                    early_stop_reason="recycled_h_invariant_success",
                )

        if family_report is None:
            family_report = self.diagnose_recycled_candidate_family_kernel(
                hamiltonian=hamiltonian,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=recycled_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                candidate_report=recycled_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycled_recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                kernel_tolerance=kernel_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                max_detectors=max_detectors,
                expand_candidate_report=True,
                kernel_method="streamed",
                store_candidate_jumps=False,
            )

        support_regions = (
            None
            if residual_local_support_regions is None
            else _normalize_local_regions(residual_local_support_regions)
        )
        if residual_report is None:
            residual_report = self.diagnose_recycled_residual_kernel(
                hamiltonian=hamiltonian,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=recycled_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                candidate_report=recycled_report,
                family_report=family_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycled_recycler_source,
                operator_groups=residual_operator_groups,
                local_support_regions=support_regions,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                kernel_tolerance=kernel_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                max_detectors=max_detectors,
                expand_candidate_report=True,
                max_operator_entries=max_residual_operator_entries,
            )

        targeted_regions = resolve_regions(
            targeted_local_regions,
            mode=resolved_targeted_region_mode,
            cluster_size_override=targeted_cluster_size,
        )
        if len(targeted_regions) == 0:
            raise ValueError("targeted local-region list is empty.")

        if targeted_report is None:
            targeted_report = self.diagnose_targeted_residual_kernel_linear_search(
                basis_configs=basis_configs,
                local_regions=targeted_regions,
                residual_report=residual_report,
                detector_operators=detector_operators,
                residual_family_local_regions=recycled_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                candidate_report=recycled_report,
                family_report=family_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycled_recycler_source,
                operator_source=targeted_operator_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                kernel_tolerance=kernel_tolerance,
                max_detectors=max_detectors,
                max_modes_per_region=max_targeted_modes_per_region,
                max_report_candidates=max_targeted_report_candidates,
                max_local_dim=max_targeted_local_dim,
            )

        targeted_selection = self.select_targeted_residual_kernel_jumps(
            targeted_report=targeted_report,
            hamiltonian=hamiltonian,
            base_jumps=recycled_selection.jumps,
            max_selected_jumps=max_targeted_selected_jumps,
            target_residual_kernel_dimension=targeted_target_residual_kernel_dimension,
            selection_target=targeted_selection_target,
            allow_non_improving=targeted_allow_non_improving,
            kernel_tolerance=kernel_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_manifold_diagnostics=check_final_manifold_diagnostics,
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

        h_invariant_report = None
        if check_h_invariant_sector:
            h_invariant_report = diagnose_common_kernel_h_invariant_sector(
                hamiltonian=hamiltonian,
                jumps=targeted_selection.all_jumps,
                target_states=self.manifold_basis,
                kernel_tolerance=kernel_tolerance,
            )

        return DegenerateCageJumpDesignWorkflowReport(
            dark_operator_report=dark_operator_report,
            recycled_report=recycled_report,
            recycled_selection=recycled_selection,
            family_report=family_report,
            residual_report=residual_report,
            targeted_report=targeted_report,
            targeted_selection=targeted_selection,
            h_invariant_report=h_invariant_report,
            recycled_local_regions=recycled_regions,
            targeted_local_regions=targeted_regions,
            recycled_region_mode=(
                resolved_recycled_region_mode if recycled_local_regions is None else "explicit"
            ),
            targeted_region_mode=(
                resolved_targeted_region_mode if targeted_local_regions is None else "explicit"
            ),
            recycled_recycler_source=recycled_recycler_source,
            targeted_operator_source=targeted_operator_source,
            design_mode=design_mode,
        )


def build_degenerate_cage_lindblad_construction(
    *,
    build_result: ModelBuildResult,
    records: Sequence[CageRecord] | None = None,
    states: NDArray[np.complex128] | None = None,
    model: Any | None = None,
    local_regions: Sequence[Sequence[int]] | None = None,
    local_term_kind: LocalTermKind | None = None,
    region_source: LocalRegionSource = "kinetic",
    recycling_jump_source: RecyclingJumpSource = "local_rdm_block_reset",
    max_jumps_per_region: int = 1,
    deduplicate_regions: bool = True,
    recycling_rdm_tolerance: float = 1e-10,
    recycling_dark_tolerance: float = 1e-10,
    recycling_inflow_tolerance: float = 1e-12,
    recycling_prefer_sparse: bool = True,
    recycling_two_pattern_tolerance: float = 1e-8,
    validate_record_signature: bool = True,
    open_system_backend: OpenSystemBackendName = "scipy",
    check_liouvillian: bool = True,
    residual_tolerance: float = 1e-10,
) -> DegenerateCageLindbladConstruction:
    """Build local reset jumps targeting a degenerate cage manifold.

    The supplied ``build_result`` is assumed to already represent the desired
    sector, for example one fixed QDM/QLM winding sector.  The target manifold
    can be supplied either as cage ``records`` or directly as a state matrix.
    If ``local_regions`` is omitted, they are inferred from the model's local
    kinetic-term supports.
    """
    dim = int(build_result.hamiltonian.shape[0])

    if records is None and states is None:
        raise ValueError("Either records or states must be provided.")
    if records is not None and states is not None:
        raise ValueError("Provide records or states, but not both.")

    record_signature = None
    if records is not None:
        if validate_record_signature:
            record_signature = _validate_record_signatures(records)
        states = _state_matrix_from_records(records, hilbert_size=dim)

    assert states is not None
    manifold_basis = _orthonormalize_state_matrix(
        np.asarray(states, dtype=np.complex128),
        dim=dim,
        tolerance=recycling_rdm_tolerance,
    )

    if local_regions is None:
        if model is None:
            raise ValueError(
                "model is required to infer local regions. Pass local_regions "
                "explicitly if no model is available."
            )
        regions = _local_regions_from_model_terms(
            model=model,
            local_term_kind=local_term_kind,
            region_source=region_source,
        )
    else:
        regions = _normalize_local_regions(local_regions)

    basis_configs = basis_configs_from_build_result(build_result)
    recycling_build_result = build_local_recycling_jumps_from_subspace_regions(
        basis_configs=basis_configs,
        states=manifold_basis,
        regions=regions,
        source=recycling_jump_source,
        max_jumps_per_region=max_jumps_per_region,
        deduplicate_regions=deduplicate_regions,
        rdm_tolerance=recycling_rdm_tolerance,
        dark_tolerance=recycling_dark_tolerance,
        inflow_tolerance=recycling_inflow_tolerance,
        prefer_sparse=recycling_prefer_sparse,
        two_pattern_tolerance=recycling_two_pattern_tolerance,
    )
    jumps = recycling_build_result.jumps

    h_closure_residual = _hamiltonian_closure_residual(
        hamiltonian=build_result.hamiltonian,
        manifold_basis=manifold_basis,
    )
    max_jump_residual, jump_residuals = _max_jump_residual(
        jumps=jumps,
        manifold_basis=manifold_basis,
    )
    inflow_norm = _inflow_norm(
        jumps=jumps,
        manifold_basis=manifold_basis,
    )

    if max_jump_residual > residual_tolerance:
        raise ValueError(
            "Degenerate cage jumps do not annihilate the target manifold: "
            f"max ||J P_M||_F={max_jump_residual:.3e}."
        )

    liouvillian_residual = None
    if check_liouvillian:
        rho = _manifold_density_matrix(manifold_basis)
        rhs = lindblad_rhs_density_matrix(
            rho,
            hamiltonian=build_result.hamiltonian,
            jumps=list(jumps),
            backend=open_system_backend,
        )
        liouvillian_residual = float(np.linalg.norm(rhs))

    return DegenerateCageLindbladConstruction(
        manifold_basis=manifold_basis,
        jumps=jumps,
        local_regions=regions,
        recycling_build_result=recycling_build_result,
        open_system_backend=open_system_backend,
        recycling_jump_source=recycling_jump_source,
        record_signature=record_signature,
        hamiltonian_closure_residual=h_closure_residual,
        max_jump_residual=max_jump_residual,
        jump_residuals=jump_residuals,
        inflow_norm=inflow_norm,
        liouvillian_residual=liouvillian_residual,
    )
