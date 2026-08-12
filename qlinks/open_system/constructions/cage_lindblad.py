from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.models.base import ModelBuildResult
from qlinks.models.local_terms import LocalTermDescriptor, LocalTermKind
from qlinks.open_system.backend import OpenSystemBackendName
from qlinks.open_system.diagnostics import (
    CommonKernelHamiltonianInvariantSectorReport,
    DarkManifoldDiagnostics,
    bad_h_invariant_common_kernel_basis,
    diagnose_common_kernel_h_invariant_sector,
    diagnose_dark_manifold,
    jump_activity_series,
    target_manifold_coherence_series,
    target_manifold_density_matrix_series,
    target_manifold_entropy_series,
    target_manifold_populations_series,
    target_manifold_projector,
    target_manifold_purity_series,
    target_manifold_weight_series,
)
from qlinks.open_system.local_recycling import (
    LocalRecyclingBuildResult,
    LocalSubspaceSupportReport,
    RecyclingJumpSource,
    build_local_recycling_jumps_from_subspace_regions,
    local_subspace_support_report_from_recycling_build_result,
)
from qlinks.open_system.manifold_detectors import (
    DarkDetectorMatrixReadout,
    DressedManifoldDarkDetectorReport,
    LocalOperatorMatrixReadout,
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
from qlinks.open_system.protocols import CageStateRecordLike
from qlinks.open_system.solvers import LindbladProblem

LocalRegionSource = Literal["kinetic", "potential", "all"]
CageLindbladRegionMode = Literal[
    "construction",
    "pair_unions",
    "cluster_unions",
    "regional_units",
    "regional_unit_clusters",
]


def _local_terms_by_operator_kind(
    model: Any,
    *,
    term_kind: LocalTermKind | None = None,
) -> tuple[
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
    dict[int, LocalTermDescriptor],
]:
    """Return local kinetic/potential descriptors from one model query.

    Calling ``local_term_descriptors`` once avoids rebuilding identical
    local-support tuples separately for kinetic and potential terms.
    ``term_kind`` may be used to restrict the construction to plaquettes,
    bonds, sites, or links.
    """
    terms = model.local_term_descriptors(term_kind=term_kind)
    kinetic_terms = tuple(term for term in terms if term.operator_kind == "kinetic")
    potential_terms = tuple(term for term in terms if term.operator_kind == "potential")
    potential_by_pid = {int(term.term_id): term for term in potential_terms}
    return kinetic_terms, potential_terms, potential_by_pid


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
    records: Sequence[CageStateRecordLike],
    *,
    hilbert_size: int,
) -> NDArray[np.complex128]:
    if len(records) == 0:
        raise ValueError("records must contain at least one cage-state record.")

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
    records: Sequence[CageStateRecordLike],
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


def _operator_kind_from_region_source(
    region_source: LocalRegionSource,
) -> Literal["kinetic", "potential", "hamiltonian"]:
    if region_source == "kinetic":
        return "kinetic"
    if region_source == "potential":
        return "potential"
    if region_source == "all":
        return "hamiltonian"
    raise ValueError("region_source must be 'kinetic', 'potential', or 'all'.")


def _call_local_term_descriptors(
    model: Any,
    *,
    operator_kind: Literal["kinetic", "potential", "hamiltonian"],
    term_kind: LocalTermKind | None,
) -> tuple[LocalTermDescriptor, ...]:
    """Call ``local_term_descriptors`` while supporting older test doubles."""
    try:
        return tuple(
            model.local_term_descriptors(
                operator_kind=operator_kind,
                term_kind=term_kind,
            )
        )
    except TypeError as exc:
        if "operator_kind" not in str(exc):
            raise
        terms = tuple(model.local_term_descriptors(term_kind=term_kind))
        if operator_kind == "hamiltonian":
            return terms
        return tuple(term for term in terms if term.operator_kind == operator_kind)


def _local_regions_from_model_terms(
    *,
    model: Any,
    local_term_kind: LocalTermKind | None,
    region_source: LocalRegionSource,
) -> tuple[tuple[int, ...], ...]:
    """Infer model-natural regional units from local-term metadata.

    For QDM/QLM this normally returns plaquette supports.  For XY-like
    nearest-neighbor hopping models it returns bond supports.  The helper uses
    the model's ``natural_region_units`` method when available, and otherwise
    falls back to deduplicating local-term descriptor supports.
    """
    operator_kind = _operator_kind_from_region_source(region_source)

    if hasattr(model, "natural_region_units"):
        regions = tuple(
            tuple(sorted(int(index) for index in region))
            for region in model.natural_region_units(
                operator_kind=operator_kind,
                term_kind=local_term_kind,
            )
        )
    else:
        terms = _call_local_term_descriptors(
            model,
            operator_kind=operator_kind,
            term_kind=local_term_kind,
        )
        regions = tuple(
            tuple(sorted(int(index) for index in term.support_variable_set)) for term in terms
        )

    deduplicated: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for region in regions:
        if len(region) == 0 or region in seen:
            continue
        seen.add(region)
        deduplicated.append(region)

    if len(deduplicated) == 0:
        raise ValueError(
            "Could not infer model-natural regional units from local terms. "
            "Pass local_regions explicitly."
        )

    return tuple(deduplicated)


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


def _jsonable(value: Any) -> Any:
    """Convert qlinks/numpy values to stable JSON-compatible objects."""
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if hasattr(value, "to_summary_dict"):
        return _jsonable(value.to_summary_dict())
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _sha256_array(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _sha256_sparse_matrix(matrix: Any) -> str:
    csr = _as_csr(matrix)
    digest = hashlib.sha256()
    digest.update(str(csr.shape).encode("utf-8"))
    digest.update(str(csr.dtype).encode("utf-8"))
    for array in (csr.indptr, csr.indices, csr.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _local_term_descriptor_to_dict(term: LocalTermDescriptor) -> dict[str, Any]:
    return {
        "term_id": int(term.term_id),
        "term_kind": term.term_kind,
        "operator_kind": term.operator_kind,
        "support_links": tuple(int(value) for value in term.support_links),
        "support_sites": tuple(int(value) for value in term.support_sites),
        "support_plaquettes": tuple(int(value) for value in term.support_plaquettes),
        "support_variables": tuple(int(value) for value in term.support_variable_set),
        "label": term.label,
    }


def _complex_entries_from_local_readout(
    readout: LocalOperatorMatrixReadout,
    *,
    tolerance: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for target_index, source_index, value in readout.nonzero_matrix_elements(
        tolerance=tolerance,
    ):
        entries.append(
            {
                "row": int(target_index),
                "col": int(source_index),
                "value": complex(value),
                "target_pattern": readout.local_patterns[int(target_index)],
                "source_pattern": readout.local_patterns[int(source_index)],
            }
        )
    return tuple(entries)


def _local_readout_to_export_dict(
    readout: LocalOperatorMatrixReadout,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    return {
        "kind": "local_matrix",
        "label": readout.label,
        "source": readout.source,
        "variable_indices": readout.variable_indices,
        "local_patterns": readout.local_patterns,
        "local_dim": readout.local_dim,
        "shape": readout.shape,
        "matrix_format": "coo",
        "nnz": readout.nnz,
        "entries": _complex_entries_from_local_readout(readout, tolerance=tolerance),
        "metadata": dict(readout.metadata),
    }


def _detector_readout_to_export_dict(readout: DarkDetectorMatrixReadout) -> dict[str, Any]:
    return {
        "detector_index": int(readout.detector_index),
        "label": readout.label,
        "form": "linear_combination",
        "coefficients": tuple(complex(value) for value in readout.coefficients),
        "operator_names": readout.operator_names,
        "n_terms": int(readout.n_terms),
        "terms": tuple(term.to_summary_dict() for term in readout.terms),
        "action_residual": float(readout.action_residual),
        "relative_action_residual": float(readout.relative_action_residual),
        "operator_frobenius_norm": float(readout.operator_frobenius_norm),
        "coefficient_ipr": float(readout.coefficient_ipr),
        "effective_operator_count": float(readout.effective_operator_count),
    }


def _ensure_export_directory(
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    output_path = Path(path)
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"export path exists and is not a directory: {output_path}")
        if not overwrite and any(output_path.iterdir()):
            raise FileExistsError(
                f"export directory is not empty: {output_path}. "
                "Pass overwrite=True to replace files."
            )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _detector_family_records(
    *,
    names: Sequence[str],
    terms: Sequence[LocalTermDescriptor],
    operators: Sequence[Any],
    matrix_dir: Path | None,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    has_terms = len(terms) == len(names)
    has_operators = len(operators) == len(names)
    for operator_index, name in enumerate(names):
        matrix_file: str | None = None
        operator_hash: str | None = None
        if has_operators:
            operator = operators[operator_index]
            operator_hash = _sha256_sparse_matrix(operator)
            if matrix_dir is not None:
                matrix_file = f"detector_{operator_index:04d}.npz"
                sp.save_npz(matrix_dir / matrix_file, _as_csr(operator))
        records.append(
            {
                "operator_index": int(operator_index),
                "operator_name": str(name),
                "term": (
                    None if not has_terms else _local_term_descriptor_to_dict(terms[operator_index])
                ),
                "sparse_matrix_file": matrix_file,
                "sparse_matrix_sha256": operator_hash,
            }
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class CageLindbladExportResult:
    """Paths written by :func:`export_cage_lindblad_design`."""

    path: Path
    manifest_path: Path

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "manifest_path": str(self.manifest_path),
        }


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

        In ``design_mode="h_invariant_fast"`` or ``"recycled_screening"`` the
        workflow may stop after recycled-jump selection, before residual and
        targeted stages are run.
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

    def detector_readouts(
        self,
        *,
        max_readouts: int | None = None,
    ) -> tuple[DarkDetectorMatrixReadout, ...]:
        """Return coefficient readouts for reported collective dark detectors."""
        return self.dark_operator_report.detector_readouts(max_readouts=max_readouts)

    def recycled_recycler_readouts(
        self,
        *,
        basis_configs: NDArray[np.integer],
        states: NDArray[np.complex128] | None = None,
        max_readouts: int | None = None,
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local readouts for selected recycled-stage recyclers.

        Pass ``states`` when the recycled source is ``rdm_support_matrix_units``
        so the local RDM support-basis recycler can be reconstructed.
        """
        return self.recycled_selection.selected_recycler_readouts(
            basis_configs=basis_configs,
            states=states,
            max_readouts=max_readouts,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
        )

    def targeted_operator_readouts(
        self,
        *,
        basis_configs: NDArray[np.integer],
        max_readouts: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local readouts for selected targeted-completion operators."""
        if self.targeted_selection is None:
            return ()
        return self.targeted_selection.selected_operator_readouts(
            basis_configs=basis_configs,
            max_readouts=max_readouts,
        )

    def local_operator_readouts(
        self,
        *,
        basis_configs: NDArray[np.integer],
        states: NDArray[np.complex128] | None = None,
        max_recycled: int | None = None,
        max_targeted: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return selected local recycler and targeted-operator readouts."""
        return self.recycled_recycler_readouts(
            basis_configs=basis_configs,
            states=states,
            max_readouts=max_recycled,
        ) + self.targeted_operator_readouts(
            basis_configs=basis_configs,
            max_readouts=max_targeted,
        )

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
            "recycled_unbundled_inflow_norm": self.recycled_selection.unbundled_inflow_norm,
            "recycled_collective_inflow_ratio": (self.recycled_selection.collective_inflow_ratio),
            "recycled_uses_collective_recyclers": (
                self.recycled_selection.uses_collective_recyclers
            ),
            "recycled_collective_jump_reduction": (
                self.recycled_selection.collective_jump_reduction
            ),
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
    regional_units: tuple[tuple[int, ...], ...]
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
            "n_regional_units": len(self.regional_units),
            "regional_units": self.regional_units,
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
        candidate_strategy: Literal["svd_basis", "coordinate_ipr"] = "svd_basis",
        candidate_overlap_tolerance: float = 1.0e-7,
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
            candidate_strategy=candidate_strategy,
            candidate_overlap_tolerance=candidate_overlap_tolerance,
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

    def regional_unit_cluster_unions(
        self,
        *,
        cluster_size: int = 2,
        cluster_mode: Literal["overlap_connected", "all"] = "overlap_connected",
        min_overlap: int = 1,
        max_region_size: int | None = None,
        include_single_units: bool = False,
        include_smaller_clusters: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        """Return supports built from model-natural regional units.

        The base units are model dependent: plaquettes for QDM/QLM-like
        plaquette Hamiltonians, bonds for nearest-neighbor XY-like hopping
        models, and any custom units exposed by ``model.natural_region_units``.
        ``cluster_size`` counts these units, not individual variables.
        ``cluster_size=1`` returns each model-natural unit directly; for QDM
        this is still a plaquette/rhombus/hexagon-local object rather than an
        onsite/link-local object.
        """
        return expand_local_regions_to_cluster_unions(
            self.regional_units,
            cluster_size=cluster_size,
            cluster_mode=cluster_mode,
            min_overlap=min_overlap,
            max_region_size=max_region_size,
            include_single_regions=include_single_units,
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
            residual_objective=residual_objective,
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
        selection_strategy: Literal[
            "diagnostics", "kernel_projection", "ranked_inflow"
        ] = "diagnostics",
        compression_strategy: Literal["none", "h_invariant"] = "none",
        max_compression_passes: int = 1,
        check_final_diagnostics: bool | None = None,
        collective_recycler_strategy: Literal["none", "bundle_by_region_detector"] = "none",
        collective_recycler_weighting: Literal["unit", "inflow", "normalized_inflow"] = "unit",
        normalize_collective_recyclers: bool = True,
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
            compression_strategy=compression_strategy,
            max_compression_passes=max_compression_passes,
            check_final_diagnostics=check_final_diagnostics,
            collective_recycler_strategy=collective_recycler_strategy,
            collective_recycler_weighting=collective_recycler_weighting,
            normalize_collective_recyclers=normalize_collective_recyclers,
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
        local_region_mode: CageLindbladRegionMode = "pair_unions",
        recycled_region_mode: CageLindbladRegionMode | None = None,
        targeted_region_mode: CageLindbladRegionMode | None = None,
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
        dark_operator_candidate_strategy: Literal["svd_basis", "coordinate_ipr"] = "svd_basis",
        dark_operator_candidate_overlap_tolerance: float = 1.0e-7,
        max_recycled_report_candidates: int | None = None,
        max_recycled_candidate_pool: int | None = None,
        max_recycled_selected_jumps: int = 16,
        recycled_target_bad_kernel_dimension: int = 0,
        recycled_allow_non_improving: bool = False,
        recycled_selection_strategy: Literal[
            "diagnostics",
            "kernel_projection",
            "ranked_inflow",
        ] = "ranked_inflow",
        recycled_compression_strategy: Literal["none", "h_invariant"] = "none",
        max_recycled_compression_passes: int = 1,
        check_recycled_selection_diagnostics: bool | None = None,
        recycled_collective_recycler_strategy: Literal[
            "none",
            "bundle_by_region_detector",
        ] = "none",
        recycled_collective_recycler_weighting: Literal[
            "unit",
            "inflow",
            "normalized_inflow",
        ] = "unit",
        normalize_recycled_collective_recyclers: bool = True,
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
        targeted_residual_objective: Literal["target_inflow", "action_norm"] = "target_inflow",
        max_h_invariant_completion_modes_per_region: int | None = None,
        max_h_invariant_completion_report_candidates: int | None = None,
        max_h_invariant_completion_selected_jumps: int | None = None,
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
            "h_invariant_completion",
            "recycled_screening",
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

        In ``design_mode="recycled_screening"``, the workflow deliberately stops
        after the recycled selector.  This is the production preselection mode
        for large triangular-style scans where constructing a good jump pool is
        useful, but a full common-kernel certificate should be run separately.
        """

        def resolve_regions(
            explicit_regions: Sequence[Sequence[int]] | None,
            *,
            mode: CageLindbladRegionMode,
            cluster_size_override: int | None,
        ) -> tuple[tuple[int, ...], ...]:
            if explicit_regions is not None:
                return _normalize_local_regions(explicit_regions)
            if mode == "construction":
                return self.local_regions
            if mode == "regional_units":
                return self.regional_units
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
            if mode == "regional_unit_clusters":
                return self.regional_unit_cluster_unions(
                    cluster_size=cluster_size_override or cluster_size,
                    cluster_mode=cluster_mode,
                    min_overlap=min_cluster_overlap,
                    max_region_size=max_cluster_region_size,
                    include_single_units=include_single_regions_in_clusters,
                    include_smaller_clusters=include_smaller_clusters,
                )
            raise ValueError(
                'region mode must be "construction", "regional_units", '
                '"pair_unions", "cluster_unions", or "regional_unit_clusters".'
            )

        if design_mode not in {
            "full",
            "h_invariant_fast",
            "h_invariant_completion",
            "recycled_screening",
        }:
            raise ValueError(
                'design_mode must be "full", "h_invariant_fast", '
                '"h_invariant_completion", or "recycled_screening".'
            )

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
                candidate_strategy=dark_operator_candidate_strategy,
                candidate_overlap_tolerance=dark_operator_candidate_overlap_tolerance,
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
                expand_candidate_report=recycled_selection_strategy != "ranked_inflow",
                selection_strategy=recycled_selection_strategy,
                compression_strategy=recycled_compression_strategy,
                max_compression_passes=max_recycled_compression_passes,
                check_final_diagnostics=check_recycled_selection_diagnostics,
                collective_recycler_strategy=recycled_collective_recycler_strategy,
                collective_recycler_weighting=recycled_collective_recycler_weighting,
                normalize_collective_recyclers=normalize_recycled_collective_recyclers,
            )

        if design_mode == "recycled_screening":
            return DegenerateCageJumpDesignWorkflowReport(
                dark_operator_report=dark_operator_report,
                recycled_report=recycled_report,
                recycled_selection=recycled_selection,
                family_report=None,
                residual_report=None,
                targeted_report=None,
                targeted_selection=None,
                h_invariant_report=None,
                recycled_local_regions=recycled_regions,
                targeted_local_regions=(),
                recycled_region_mode=(
                    resolved_recycled_region_mode if recycled_local_regions is None else "explicit"
                ),
                targeted_region_mode="skipped",
                recycled_recycler_source=recycled_recycler_source,
                targeted_operator_source=targeted_operator_source,
                design_mode=design_mode,
                early_stop_reason="recycled_screening",
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

        if design_mode == "h_invariant_completion" and check_h_invariant_sector:
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

            h_invariant_basis = bad_h_invariant_common_kernel_basis(
                hamiltonian=hamiltonian,
                jumps=recycled_selection.jumps,
                target_states=self.manifold_basis,
                kernel_tolerance=kernel_tolerance,
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
                    residual_basis=h_invariant_basis,
                    operator_source=targeted_operator_source,
                    residual_objective="action_norm",
                    tolerance=tolerance,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    kernel_tolerance=kernel_tolerance,
                    max_modes_per_region=(
                        max_targeted_modes_per_region
                        if max_h_invariant_completion_modes_per_region is None
                        else max_h_invariant_completion_modes_per_region
                    ),
                    max_report_candidates=(
                        max_targeted_report_candidates
                        if max_h_invariant_completion_report_candidates is None
                        else max_h_invariant_completion_report_candidates
                    ),
                    max_local_dim=max_targeted_local_dim,
                )

            targeted_selection = self.select_targeted_residual_kernel_jumps(
                targeted_report=targeted_report,
                hamiltonian=hamiltonian,
                base_jumps=recycled_selection.jumps,
                max_selected_jumps=(
                    max_targeted_selected_jumps
                    if max_h_invariant_completion_selected_jumps is None
                    else max_h_invariant_completion_selected_jumps
                ),
                target_residual_kernel_dimension=targeted_target_residual_kernel_dimension,
                selection_target="combined_common_kernel",
                allow_non_improving=targeted_allow_non_improving,
                kernel_tolerance=kernel_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                check_manifold_diagnostics=check_final_manifold_diagnostics,
                liouvillian_spectrum_method=liouvillian_spectrum_method,
                sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
            )
            completed_h_invariant_report = diagnose_common_kernel_h_invariant_sector(
                hamiltonian=hamiltonian,
                jumps=targeted_selection.all_jumps,
                target_states=self.manifold_basis,
                kernel_tolerance=kernel_tolerance,
            )

            return DegenerateCageJumpDesignWorkflowReport(
                dark_operator_report=dark_operator_report,
                recycled_report=recycled_report,
                recycled_selection=recycled_selection,
                family_report=None,
                residual_report=None,
                targeted_report=targeted_report,
                targeted_selection=targeted_selection,
                h_invariant_report=completed_h_invariant_report,
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
                early_stop_reason=(
                    "h_invariant_completion_success"
                    if completed_h_invariant_report.likely_attractive_by_h_invariant_kernel
                    else "h_invariant_completion_incomplete"
                ),
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
                residual_objective=targeted_residual_objective,
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
    records: Sequence[CageStateRecordLike] | None = None,
    states: NDArray[np.complex128] | None = None,
    model: Any | None = None,
    local_regions: Sequence[Sequence[int]] | None = None,
    regional_units: Sequence[Sequence[int]] | None = None,
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
        resolved_regional_units = regions
    else:
        regions = _normalize_local_regions(local_regions)
        if regional_units is not None:
            resolved_regional_units = _normalize_local_regions(regional_units)
        elif model is not None:
            resolved_regional_units = _local_regions_from_model_terms(
                model=model,
                local_term_kind=local_term_kind,
                region_source=region_source,
            )
        else:
            resolved_regional_units = regions

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
        regional_units=resolved_regional_units,
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


# ---------------------------------------------------------------------------
# Preferred public cage-Lindblad API
# ---------------------------------------------------------------------------

CageLindbladWorkflowReport = DegenerateCageJumpDesignWorkflowReport

DetectorOperatorKind = Literal["kinetic", "potential", "hamiltonian"]


@dataclass(frozen=True, slots=True)
class CageLindbladDetectorOperators:
    """Named local operators used to build dark detector combinations.

    ``operators`` are the matrices ``O_i`` supplied to the dark-detector solver,
    while ``names`` are the corresponding labels used in workflow/readout reports.
    """

    operators: tuple[Any, ...]
    names: tuple[str, ...]
    terms: tuple[LocalTermDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError("operators must contain at least one detector operator.")
        if len(self.operators) != len(self.names):
            raise ValueError("operators and names must have the same length.")
        if self.terms and len(self.terms) != len(self.operators):
            raise ValueError("terms and operators must have the same length when provided.")

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_operators": len(self.operators),
            "names": self.names,
            "n_terms": len(self.terms),
        }


def _resolve_target_states(
    *,
    target_state: NDArray[np.complex128] | None,
    target_states: NDArray[np.complex128] | None,
    states: NDArray[np.complex128] | None,
) -> NDArray[np.complex128] | None:
    supplied = [value is not None for value in (target_state, target_states, states)]
    if sum(supplied) > 1:
        raise ValueError("Provide only one of target_state, target_states, or states.")
    if target_state is not None:
        return np.asarray(target_state, dtype=np.complex128)
    if target_states is not None:
        return np.asarray(target_states, dtype=np.complex128)
    if states is not None:
        return np.asarray(states, dtype=np.complex128)
    return None


@dataclass(frozen=True, slots=True)
class CageLindbladDesignProblem:
    """Unified cage-state Lindblad design problem.

    The object stores only the target manifold, basis/configuration metadata, and
    local regions needed by the successful dark-detector workflow.  Use
    :func:`build_cage_lindblad_problem` to construct it from either one state,
    many states, or cage records.
    """

    build_result: ModelBuildResult
    construction: DegenerateCageLindbladConstruction

    @property
    def hamiltonian(self) -> Any:
        return self.build_result.hamiltonian

    @property
    def basis_configs(self) -> NDArray[np.integer]:
        return basis_configs_from_build_result(self.build_result)

    @property
    def manifold_basis(self) -> NDArray[np.complex128]:
        return self.construction.manifold_basis

    @property
    def target_basis(self) -> NDArray[np.complex128]:
        return self.construction.manifold_basis

    @property
    def target_density_matrix(self) -> NDArray[np.complex128]:
        return self.construction.target_density_matrix

    @property
    def target_manifold_projector(self) -> NDArray[np.complex128]:
        """Projector onto the target dark/cage manifold."""
        return target_manifold_projector(self.manifold_basis)

    def target_manifold_weight_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return ``Tr(P_target rho(t))`` for solver or MCWF output.

        The method forwards data-source keywords such as ``evolution_result``,
        ``density_matrices``, ``ensemble_result``, or ``state_snapshots`` to
        :func:`qlinks.open_system.diagnostics.target_manifold_weight_series`
        and automatically supplies this problem's target basis.
        """
        return target_manifold_weight_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    def target_manifold_density_matrix_series(self, **kwargs: Any) -> NDArray[np.complex128]:
        """Return the conditioned density matrix inside the target manifold."""
        return target_manifold_density_matrix_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    def target_manifold_populations_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return target-basis populations inside the target manifold."""
        return target_manifold_populations_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    def target_manifold_coherence_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return off-diagonal target-manifold coherence over time."""
        return target_manifold_coherence_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    def target_manifold_purity_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return purity of the conditioned target-manifold state over time."""
        return target_manifold_purity_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    def target_manifold_entropy_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return entropy of the conditioned target-manifold state over time."""
        return target_manifold_entropy_series(
            target_basis=self.manifold_basis,
            **kwargs,
        )

    @property
    def hilbert_dimension(self) -> int:
        return self.construction.hilbert_dimension

    @property
    def manifold_dimension(self) -> int:
        return self.construction.manifold_dimension

    @property
    def is_single_cage_target(self) -> bool:
        return self.manifold_dimension == 1

    @property
    def local_regions(self) -> tuple[tuple[int, ...], ...]:
        return self.construction.local_regions

    @property
    def regional_units(self) -> tuple[tuple[int, ...], ...]:
        """Model-natural region units used by regional-unit modes."""
        return self.construction.regional_units

    @property
    def record_signature(self) -> tuple[int, int] | None:
        return self.construction.record_signature

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "manifold_dimension": self.manifold_dimension,
            "is_single_cage_target": self.is_single_cage_target,
            "record_signature": self.record_signature,
            "n_local_regions": len(self.local_regions),
            "local_regions": self.local_regions,
            "n_regional_units": len(self.regional_units),
            "regional_units": self.regional_units,
            "h_closure_residual": self.construction.hamiltonian_closure_residual,
        }

    def to_lindblad_problem(
        self,
        *,
        jumps: Sequence[Any],
        hamiltonian: Any | None = None,
        backend: str | None = None,
    ) -> LindbladProblem:
        """Package a jump set as a solver-ready Lindblad problem.

        This method is retained as a low-level helper for custom jump designs.
        The preferred path is :meth:`design_jumps`, which returns both the
        workflow report and the packaged :class:`LindbladProblem` in one result.
        """
        return LindbladProblem(
            hamiltonian=self.hamiltonian if hamiltonian is None else hamiltonian,
            jumps=tuple(jumps),
            backend=self.construction.open_system_backend if backend is None else backend,
        )

    def _resolve_detector_inputs(
        self,
        *,
        detector_operators: CageLindbladDetectorOperators | Sequence[Any],
        detector_operator_names: Sequence[str] | None,
    ) -> tuple[tuple[Any, ...], tuple[str, ...] | None]:
        if isinstance(detector_operators, CageLindbladDetectorOperators):
            if detector_operator_names is not None:
                raise ValueError(
                    "detector_operator_names must be omitted when detector_operators "
                    "is a CageLindbladDetectorOperators bundle."
                )
            return detector_operators.operators, detector_operators.names

        return (
            tuple(detector_operators),
            None if detector_operator_names is None else tuple(detector_operator_names),
        )

    def design_workflow(
        self,
        *,
        detector_operators: CageLindbladDetectorOperators | Sequence[Any],
        detector_operator_names: Sequence[str] | None = None,
        hamiltonian: Any | None = None,
        basis_configs: NDArray[np.integer] | None = None,
        **workflow_kwargs: Any,
    ) -> DegenerateCageJumpDesignWorkflowReport:
        """Run the raw dark-detector/recycler workflow report.

        Most users should call :meth:`design_jumps`, which packages this report
        together with a solver-ready :class:`LindbladProblem`.  This lower-level
        method is useful when comparing alternative jump-design workflows.
        """
        operators, names = self._resolve_detector_inputs(
            detector_operators=detector_operators,
            detector_operator_names=detector_operator_names,
        )

        return self.construction.design_dark_manifold_jumps(
            hamiltonian=self.hamiltonian if hamiltonian is None else hamiltonian,
            basis_configs=(self.basis_configs if basis_configs is None else basis_configs),
            detector_operators=operators,
            detector_operator_names=names,
            **workflow_kwargs,
        )

    def design_jumps(
        self,
        *,
        detector_operators: CageLindbladDetectorOperators | Sequence[Any],
        detector_operator_names: Sequence[str] | None = None,
        hamiltonian: Any | None = None,
        basis_configs: NDArray[np.integer] | None = None,
        backend: str | None = None,
        **workflow_kwargs: Any,
    ) -> "CageLindbladDesignResult":
        """Design jumps and return a solver-ready result.

        The returned object exposes the workflow report through ``.workflow`` and
        delegates common report properties/methods, while ``.lindblad_problem``
        can be passed directly to solvers.  This avoids the older two-step
        pattern ``workflow = problem.design_jumps(...);
        problem = workflow.to_lindblad_problem(...)``.
        """
        workflow = self.design_workflow(
            detector_operators=detector_operators,
            detector_operator_names=detector_operator_names,
            hamiltonian=hamiltonian,
            basis_configs=basis_configs,
            **workflow_kwargs,
        )
        lindblad_problem = self.to_lindblad_problem(
            jumps=workflow.jumps,
            hamiltonian=hamiltonian,
            backend=backend,
        )
        operators, names = self._resolve_detector_inputs(
            detector_operators=detector_operators,
            detector_operator_names=detector_operator_names,
        )
        detector_terms = (
            detector_operators.terms
            if isinstance(detector_operators, CageLindbladDetectorOperators)
            else ()
        )
        return CageLindbladDesignResult(
            problem=self,
            workflow=workflow,
            lindblad_problem=lindblad_problem,
            detector_operators=operators,
            detector_operator_names=(
                tuple(names)
                if names is not None
                else tuple(workflow.dark_operator_report.operator_names)
            ),
            detector_terms=tuple(detector_terms),
        )

    def design_lindblad_problem(
        self,
        *,
        detector_operators: CageLindbladDetectorOperators | Sequence[Any],
        detector_operator_names: Sequence[str] | None = None,
        hamiltonian: Any | None = None,
        basis_configs: NDArray[np.integer] | None = None,
        backend: str | None = None,
        **workflow_kwargs: Any,
    ) -> LindbladProblem:
        """Design jumps and return only the solver-ready Lindblad problem."""
        return self.design_jumps(
            detector_operators=detector_operators,
            detector_operator_names=detector_operator_names,
            hamiltonian=hamiltonian,
            basis_configs=basis_configs,
            backend=backend,
            **workflow_kwargs,
        ).lindblad_problem


@dataclass(frozen=True, slots=True)
class CageLindbladDesignResult:
    """Result of the unified cage-Lindblad jump-design workflow.

    ``workflow`` contains all diagnostic/readout detail. ``lindblad_problem`` is
    the solver-ready object with the final jump set already packaged.  Common
    workflow methods and attributes are delegated for notebook convenience.
    Detector metadata is retained so the design can be exported as a
    reconstructable analytical ``J = R D`` data bundle.
    """

    problem: CageLindbladDesignProblem
    workflow: DegenerateCageJumpDesignWorkflowReport
    lindblad_problem: LindbladProblem
    detector_operators: tuple[Any, ...] = ()
    detector_operator_names: tuple[str, ...] = ()
    detector_terms: tuple[LocalTermDescriptor, ...] = ()

    @property
    def jumps(self) -> tuple[sp.csr_array, ...]:
        return self.workflow.jumps

    @property
    def recycled_jumps(self) -> tuple[sp.csr_array, ...]:
        return self.workflow.recycled_jumps

    @property
    def targeted_jumps(self) -> tuple[sp.csr_array, ...]:
        return self.workflow.targeted_jumps

    @property
    def n_jumps(self) -> int:
        return self.workflow.n_jumps

    @property
    def h_invariant_report(self) -> CommonKernelHamiltonianInvariantSectorReport | None:
        return self.workflow.h_invariant_report

    @property
    def likely_successful_h_invariant_design(self) -> bool | None:
        return self.workflow.likely_successful_h_invariant_design

    @property
    def solver_problem(self) -> LindbladProblem:
        """Alias for ``lindblad_problem`` used by some solver-oriented notebooks."""
        return self.lindblad_problem

    @property
    def target_manifold_projector(self) -> NDArray[np.complex128]:
        """Projector onto this design's target dark/cage manifold."""
        return self.problem.target_manifold_projector

    def target_manifold_weight_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return ``Tr(P_target rho(t))`` for evolution or MCWF output.

        Examples:
            ``design.target_manifold_weight_series(evolution_result=result)``
            for Lindblad density-matrix solvers, or
            ``design.target_manifold_weight_series(ensemble_result=mcwf)`` for
            MCWF results containing ``rho_t`` or ``state_snapshots``.
        """
        return self.problem.target_manifold_weight_series(**kwargs)

    def target_manifold_density_matrix_series(self, **kwargs: Any) -> NDArray[np.complex128]:
        """Return the conditioned density matrix inside the target manifold."""
        return self.problem.target_manifold_density_matrix_series(**kwargs)

    def target_manifold_populations_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return target-basis populations inside the target manifold."""
        return self.problem.target_manifold_populations_series(**kwargs)

    def target_manifold_coherence_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return off-diagonal target-manifold coherence over time."""
        return self.problem.target_manifold_coherence_series(**kwargs)

    def target_manifold_purity_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return purity of the conditioned target-manifold state over time."""
        return self.problem.target_manifold_purity_series(**kwargs)

    def target_manifold_entropy_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return entropy of the conditioned target-manifold state over time."""
        return self.problem.target_manifold_entropy_series(**kwargs)

    def jump_activity_series(self, **kwargs: Any) -> NDArray[np.float64]:
        """Return total jump activity ``sum_mu Tr(J_mu^dag J_mu rho(t))``."""
        return jump_activity_series(jumps=self.jumps, **kwargs)

    def evolve_with_target_weight(
        self,
        density_matrix_initial: Any,
        times: NDArray[np.float64],
        *,
        options: Any | None = None,
    ) -> tuple[Any, NDArray[np.float64]]:
        """Evolve this Lindblad problem and return target-manifold weights.

        This is a convenience wrapper around ``self.lindblad_problem.evolve``
        followed by ``Tr(P_target rho(t))``.
        """
        result = self.lindblad_problem.evolve(
            density_matrix_initial,
            times,
            options=options,
        )
        weights = self.target_manifold_weight_series(evolution_result=result)
        return result, weights

    def to_lindblad_problem(self) -> LindbladProblem:
        """Return the already packaged solver problem."""
        return self.lindblad_problem

    def to_summary_dict(self) -> dict[str, object]:
        summary = dict(self.workflow.to_summary_dict())
        summary["solver_backend"] = self.lindblad_problem.backend
        return summary

    def export(
        self,
        path: str | Path,
        *,
        include_basis: bool = True,
        include_global_matrices: bool = False,
        include_detector_matrices: bool = False,
        include_readouts: bool = True,
        matrix_element_tolerance: float = 0.0,
        overwrite: bool = False,
    ) -> CageLindbladExportResult:
        """Export this design as a versioned JSON/JSONL data bundle.

        The export stores the analytical jump structure by default: dark
        detectors as coefficient combinations and recycled/targeted local
        matrices in COO form.  Full sparse matrices are optional and stored as
        SciPy ``.npz`` files when requested.
        """
        return export_cage_lindblad_design(
            self,
            path=path,
            include_basis=include_basis,
            include_global_matrices=include_global_matrices,
            include_detector_matrices=include_detector_matrices,
            include_readouts=include_readouts,
            matrix_element_tolerance=matrix_element_tolerance,
            overwrite=overwrite,
        )

    def __rich__(self):
        return self.workflow.__rich__()

    def to_rich(self):
        return self.workflow.to_rich()

    def __getattr__(self, name: str) -> object:
        return getattr(self.workflow, name)


def _write_target_export_files(
    *,
    design: CageLindbladDesignResult,
    output_path: Path,
    include_basis: bool,
) -> dict[str, Any]:
    target_basis = np.asarray(design.problem.manifold_basis, dtype=np.complex128)
    basis_configs = np.asarray(design.problem.basis_configs)
    target_payload: dict[str, Any] = {
        "hilbert_dimension": int(design.problem.hilbert_dimension),
        "manifold_dimension": int(design.problem.manifold_dimension),
        "is_single_cage_target": bool(design.problem.is_single_cage_target),
        "target_basis_shape": tuple(int(value) for value in target_basis.shape),
        "target_basis_sha256": _sha256_array(target_basis),
        "basis_configs_shape": tuple(int(value) for value in basis_configs.shape),
        "basis_configs_sha256": _sha256_array(basis_configs),
        "record_signature": design.problem.record_signature,
    }
    if include_basis:
        np.save(output_path / "target_basis.npy", target_basis)
        np.save(output_path / "basis_configs.npy", basis_configs)
        target_payload["target_basis_file"] = "target_basis.npy"
        target_payload["basis_configs_file"] = "basis_configs.npy"
    return target_payload


def _recycled_jump_export_records(
    *,
    design: CageLindbladDesignResult,
    include_readouts: bool,
    matrix_element_tolerance: float,
) -> tuple[dict[str, Any], ...]:
    selection = design.workflow.recycled_selection
    basis_configs = design.problem.basis_configs
    records: list[dict[str, Any]] = []

    if selection.collective_groups:
        readouts = (
            selection.selected_recycler_readouts(basis_configs=basis_configs)
            if include_readouts
            else ()
        )
        for jump_index, group in enumerate(selection.collective_groups):
            record: dict[str, Any] = {
                "jump_index": int(jump_index),
                "stage": "recycled",
                "form": "collective_R_times_D",
                "detector_index": int(group.detector_index),
                "detector_label": group.detector_name,
                "region_index": int(group.region_index),
                "variable_indices": group.variable_indices,
                "candidate_indices": group.candidate_indices,
                "recycler_indices": group.recycler_indices,
                "recycler_names": group.recycler_names,
                "weights": tuple(complex(value) for value in group.weights),
                "n_bundled_recyclers": int(group.n_bundled_recyclers),
                "jump_frobenius_norm": float(group.jump_frobenius_norm),
                "jump_nnz": int(group.jump_nnz),
                "unbundled_inflow_norm": group.unbundled_inflow_norm,
                "bundled_inflow_norm": group.bundled_inflow_norm,
            }
            if include_readouts and jump_index < len(readouts):
                record["recycler"] = _local_readout_to_export_dict(
                    readouts[jump_index],
                    tolerance=matrix_element_tolerance,
                )
            records.append(record)
        return tuple(records)

    readouts = (
        selection.selected_recycler_readouts(
            basis_configs=basis_configs,
            states=design.problem.manifold_basis,
        )
        if include_readouts
        else ()
    )
    for jump_index, candidate in enumerate(selection.selected_candidates):
        record = {
            "jump_index": int(jump_index),
            "stage": "recycled",
            "form": "R_times_D",
            "detector_index": int(candidate.detector_index),
            "detector_label": candidate.detector_name,
            "region_index": int(candidate.region_index),
            "variable_indices": candidate.variable_indices,
            "local_dim": int(candidate.local_dim),
            "recycler_index": int(candidate.recycler_index),
            "recycler_name": candidate.recycler_name,
            "candidate": candidate.to_summary_dict(),
            "jump_frobenius_norm": float(candidate.jump_frobenius_norm),
            "jump_nnz": int(candidate.jump_nnz),
            "inflow_norm": float(candidate.inflow_norm),
        }
        if include_readouts and jump_index < len(readouts):
            record["recycler"] = _local_readout_to_export_dict(
                readouts[jump_index],
                tolerance=matrix_element_tolerance,
            )
        records.append(record)
    return tuple(records)


def _targeted_jump_export_records(
    *,
    design: CageLindbladDesignResult,
    include_readouts: bool,
    matrix_element_tolerance: float,
) -> tuple[dict[str, Any], ...]:
    selection = design.workflow.targeted_selection
    if selection is None:
        return ()

    basis_configs = design.problem.basis_configs
    readouts = (
        selection.selected_operator_readouts(basis_configs=basis_configs)
        if include_readouts
        else ()
    )
    offset = len(design.workflow.recycled_jumps)
    records: list[dict[str, Any]] = []
    for local_index, candidate in enumerate(selection.selected_candidates):
        record: dict[str, Any] = {
            "jump_index": int(offset + local_index),
            "targeted_jump_index": int(local_index),
            "stage": "targeted",
            "form": "local_dark_operator",
            "region_index": int(candidate.region_index),
            "variable_indices": candidate.variable_indices,
            "local_dim": int(candidate.local_dim),
            "operator_source": candidate.operator_source,
            "candidate": candidate.to_summary_dict(),
            "jump_frobenius_norm": float(candidate.jump_frobenius_norm),
            "jump_nnz": int(candidate.jump_nnz),
            "residual_score_norm": float(
                max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
            ),
        }
        if include_readouts and local_index < len(readouts):
            record["operator"] = _local_readout_to_export_dict(
                readouts[local_index],
                tolerance=matrix_element_tolerance,
            )
        records.append(record)
    return tuple(records)


def export_cage_lindblad_design(
    design: CageLindbladDesignResult,
    *,
    path: str | Path,
    include_basis: bool = True,
    include_global_matrices: bool = False,
    include_detector_matrices: bool = False,
    include_readouts: bool = True,
    include_certificates: bool = True,
    matrix_element_tolerance: float = 0.0,
    overwrite: bool = False,
) -> CageLindbladExportResult:
    """Export a cage-Lindblad design as a versioned JSON/JSONL bundle.

    The default export is intended for papers and arXiv data: it stores
    detectors as coefficient combinations, recycled jumps as ``R D`` records,
    targeted jumps as local dark-operator records, and certificates/summary
    metadata as JSON.  Full sparse matrices can be included with
    ``include_global_matrices=True`` and detector matrices with
    ``include_detector_matrices=True``.
    """
    output_path = _ensure_export_directory(path, overwrite=overwrite)

    matrix_dir: Path | None = None
    detector_matrix_dir: Path | None = None
    matrix_files: dict[str, Any] = {}
    if include_global_matrices:
        matrix_dir = output_path / "sparse_matrices"
        matrix_dir.mkdir(exist_ok=True)
        sp.save_npz(matrix_dir / "H.npz", _as_csr(design.problem.hamiltonian))
        jump_files: list[str] = []
        for jump_index, jump in enumerate(design.jumps):
            filename = f"jump_{jump_index:04d}.npz"
            sp.save_npz(matrix_dir / filename, _as_csr(jump))
            jump_files.append(filename)
        matrix_files = {
            "directory": "sparse_matrices",
            "hamiltonian": "H.npz",
            "jumps": tuple(jump_files),
        }
    if include_detector_matrices:
        detector_matrix_dir = output_path / "detector_matrices"
        detector_matrix_dir.mkdir(exist_ok=True)

    detector_names = (
        design.detector_operator_names
        if design.detector_operator_names
        else tuple(design.workflow.dark_operator_report.operator_names)
    )
    detector_family = _detector_family_records(
        names=detector_names,
        terms=design.detector_terms,
        operators=design.detector_operators,
        matrix_dir=detector_matrix_dir,
    )

    target_payload = _write_target_export_files(
        design=design,
        output_path=output_path,
        include_basis=include_basis,
    )
    workflow_summary = design.to_summary_dict()
    certificates = {
        "workflow": workflow_summary,
        "h_invariant_report": (
            None
            if design.workflow.h_invariant_report is None
            else design.workflow.h_invariant_report.to_summary_dict()
        ),
        "final_diagnostics": (
            None
            if design.workflow.final_diagnostics is None
            else design.workflow.final_diagnostics.to_summary_dict()
        ),
        "likely_successful_h_invariant_design": (
            design.workflow.likely_successful_h_invariant_design
        ),
        "likely_successful_common_kernel_design": (
            design.workflow.likely_successful_common_kernel_design
        ),
    }

    dark_detector_records = tuple(
        _detector_readout_to_export_dict(readout) for readout in design.detector_readouts()
    )
    recycled_records = _recycled_jump_export_records(
        design=design,
        include_readouts=include_readouts,
        matrix_element_tolerance=matrix_element_tolerance,
    )
    targeted_records = _targeted_jump_export_records(
        design=design,
        include_readouts=include_readouts,
        matrix_element_tolerance=matrix_element_tolerance,
    )

    _write_json(output_path / "target.json", target_payload)
    _write_json(
        output_path / "regional_units.json",
        {"regional_units": design.problem.regional_units},
    )
    _write_json(
        output_path / "local_regions.json",
        {"local_regions": design.problem.local_regions},
    )
    _write_json(output_path / "workflow_summary.json", workflow_summary)
    if include_certificates:
        _write_json(output_path / "certificates.json", certificates)
    _write_jsonl(output_path / "detector_family.jsonl", detector_family)
    _write_jsonl(output_path / "dark_detectors.jsonl", dark_detector_records)
    _write_jsonl(output_path / "recycled_jumps.jsonl", recycled_records)
    _write_jsonl(output_path / "targeted_jumps.jsonl", targeted_records)

    manifest = {
        "schema_name": "qlinks.cage_lindblad_design",
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hilbert_dimension": int(design.problem.hilbert_dimension),
        "manifold_dimension": int(design.problem.manifold_dimension),
        "n_jumps": int(design.n_jumps),
        "n_recycled_jumps": int(len(design.recycled_jumps)),
        "n_targeted_jumps": int(len(design.targeted_jumps)),
        "jump_forms": tuple(
            sorted({record["form"] for record in recycled_records + targeted_records})
        ),
        "basis_configs_sha256": target_payload["basis_configs_sha256"],
        "target_basis_sha256": target_payload["target_basis_sha256"],
        "hamiltonian_sha256": _sha256_sparse_matrix(design.problem.hamiltonian),
        "workflow_parameters": {
            "design_mode": design.workflow.design_mode,
            "recycled_region_mode": design.workflow.recycled_region_mode,
            "targeted_region_mode": design.workflow.targeted_region_mode,
            "recycled_recycler_source": design.workflow.recycled_recycler_source,
            "targeted_operator_source": design.workflow.targeted_operator_source,
        },
        "files": {
            "target": "target.json",
            "regional_units": "regional_units.json",
            "local_regions": "local_regions.json",
            "workflow_summary": "workflow_summary.json",
            "certificates": "certificates.json" if include_certificates else None,
            "detector_family": "detector_family.jsonl",
            "dark_detectors": "dark_detectors.jsonl",
            "recycled_jumps": "recycled_jumps.jsonl",
            "targeted_jumps": "targeted_jumps.jsonl",
            "basis_configs": "basis_configs.npy" if include_basis else None,
            "target_basis": "target_basis.npy" if include_basis else None,
            "sparse_matrices": matrix_files or None,
            "detector_matrices": "detector_matrices" if include_detector_matrices else None,
        },
    }
    manifest_path = output_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return CageLindbladExportResult(path=output_path, manifest_path=manifest_path)


def build_cage_lindblad_problem(
    *,
    build_result: ModelBuildResult,
    target_state: NDArray[np.complex128] | None = None,
    target_states: NDArray[np.complex128] | None = None,
    states: NDArray[np.complex128] | None = None,
    records: Sequence[CageStateRecordLike] | None = None,
    model: Any | None = None,
    local_regions: Sequence[Sequence[int]] | None = None,
    regional_units: Sequence[Sequence[int]] | None = None,
    local_term_kind: LocalTermKind | None = None,
    region_source: LocalRegionSource = "kinetic",
    validate_record_signature: bool = True,
    open_system_backend: OpenSystemBackendName = "scipy",
    residual_tolerance: float = 1e-10,
    target_tolerance: float = 1e-10,
) -> CageLindbladDesignProblem:
    """Create a unified cage Lindblad design problem.

    A single cage state is supplied with ``target_state``.  A degenerate cage
    manifold is supplied with ``target_states``/``states`` or ``records``.  The
    returned object uses the same ``design_jumps`` method in both cases.
    """
    resolved_states = _resolve_target_states(
        target_state=target_state,
        target_states=target_states,
        states=states,
    )
    if records is not None and resolved_states is not None:
        raise ValueError("Provide records or target states, but not both.")
    if records is None and resolved_states is None:
        raise ValueError("Provide target_state, target_states, states, or records.")

    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        records=records,
        states=resolved_states,
        model=model,
        local_regions=local_regions,
        regional_units=regional_units,
        local_term_kind=local_term_kind,
        region_source=region_source,
        validate_record_signature=validate_record_signature,
        open_system_backend=open_system_backend,
        check_liouvillian=False,
        residual_tolerance=residual_tolerance,
        recycling_rdm_tolerance=target_tolerance,
        recycling_dark_tolerance=target_tolerance,
    )
    return CageLindbladDesignProblem(
        build_result=build_result,
        construction=construction,
    )


def build_cage_lindblad_detector_operators(
    *,
    model: Any,
    build_result: ModelBuildResult,
    term_kind: LocalTermKind | None = "plaquette",
    operator_kind: DetectorOperatorKind = "potential",
    builder: str = "sparse",
    backend: str = "scipy",
    on_missing: str = "skip",
    name_prefix: str | None = None,
) -> CageLindbladDetectorOperators:
    """Build a named local detector-operator family from model local terms.

    ``operator_kind='hamiltonian'`` includes both kinetic and potential terms
    when the model exposes them.  The returned bundle can be passed directly to
    :meth:`CageLindbladDesignProblem.design_jumps`.
    """
    terms = tuple(
        model.local_term_descriptors(
            term_kind=term_kind,
            operator_kind=operator_kind,
        )
    )
    if len(terms) == 0:
        raise ValueError(
            "model.local_term_descriptors returned no detector terms for "
            f"operator_kind={operator_kind!r}."
        )

    matrices: list[Any] = []
    names: list[str] = []
    kept_terms: list[LocalTermDescriptor] = []
    for term in terms:
        try:
            matrix = model.build_local_term(
                term,
                build_result,
                builder=builder,
                backend=backend,
                on_missing=on_missing,
            )
        except TypeError:
            # Older model implementations do not expose every keyword.  Keep the
            # compatibility path local to this API wrapper.
            matrix = model.build_local_term(
                term,
                build_result,
                builder=builder,
                on_missing=on_missing,
            )
        if matrix is None:
            continue
        matrices.append(matrix)
        kept_terms.append(term)
        if term.label:
            label = str(term.label)
        else:
            label = f"{term.operator_kind}_{term.term_id}"
        names.append(label if name_prefix is None else f"{name_prefix}{label}")

    if len(matrices) == 0:
        raise ValueError("All detector local terms were skipped or missing.")

    return CageLindbladDetectorOperators(
        operators=tuple(matrices),
        names=tuple(names),
        terms=tuple(kept_terms),
    )
