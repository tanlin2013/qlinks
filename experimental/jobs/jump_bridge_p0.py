"""P0 scientific benchmark for the ICQMBS-to-Lindblad jump bridge.

The benchmark is intentionally finite-size and claim-oriented.  It compares
closed-system directed caging rows with the current caging-generated Lindblad
families using the theorem-based attractive-subspace diagnostic.  It does not
run long-time dynamics or make preparation-time scaling claims.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging import CageSearchConfig, CageSearcher
from qlinks.caging.analysis import (
    EnvironmentReductionConfig,
    diagnose_cage_environment_reduction,
    local_structure_report_from_environment_report,
)
from qlinks.caging.analysis.thermodynamic import (
    directed_transition_witness_template,
    hermitianize_local_witness_template,
)
from qlinks.local_structure import embed_local_pattern_operator
from qlinks.models import HoneycombQDMModel, SquareQDMModel
from qlinks.open_system import (
    AttractiveSubspaceDiagnostics,
    diagnose_attractive_subspace,
    diagnose_manifold_dark_operator_basis,
)
from qlinks.open_system.constructions import (
    CageLindbladDesignResult,
    build_cage_lindblad_detector_operators,
    build_cage_lindblad_problem,
)
from qlinks.open_system.constructions.deprecated import (
    build_type1_cage_lindblad_construction,
)
from qlinks.operators.plaquette import alternating_binary_patterns

TOLERANCE = 1.0e-9
SEARCH_SEED = 1234

MODERN_DESIGN_KWARGS: dict[str, object] = {
    "design_mode": "h_invariant_completion",
    "recycled_region_mode": "regional_unit_clusters",
    "targeted_region_mode": "regional_unit_clusters",
    "recycled_cluster_size": 1,
    "targeted_cluster_size": 1,
    "cluster_mode": "all",
    "max_cluster_region_size": 12,
    "max_detectors": 12,
    "dark_operator_candidate_strategy": "coordinate_ipr",
    "dark_operator_max_candidates": 16,
    "recycled_selection_strategy": "ranked_inflow",
    "check_h_invariant_sector": True,
}


@dataclass(frozen=True, slots=True)
class JumpBridgeCase:
    """One finite-size QDM target used by the P0 bridge benchmark."""

    name: str
    model_name: str
    record_count: int
    signature: tuple[int, int] = (0, 4)


@dataclass(frozen=True, slots=True)
class DirectedCagingRow:
    """One reconstructed directed local caging row ``A=|z><v|``."""

    state_index: int
    zero_index: int
    variable_indices: tuple[int, ...]
    target_pattern: tuple[int, ...]
    source_patterns: tuple[tuple[int, ...], ...]
    amplitudes: np.ndarray
    operator: sp.csr_array
    hermitian_operator: sp.csr_array
    positive_operator: sp.csr_array


@dataclass(frozen=True, slots=True)
class RetargetedDirectedJump:
    """One direct jump ``|tau><v_R|`` inherited from a directed caging row."""

    directed_row_index: int
    source_state_index: int
    zero_index: int
    variable_indices: tuple[int, ...]
    output_pattern: tuple[int, ...]
    operator: sp.csr_array


@dataclass(frozen=True, slots=True)
class ShiftedPotentialCagingOperator:
    """One local QDM shifted-potential candidate ``Y=(F_1-1)+(F_2-1)``.

    The companion square-QDM witness is defined on a reduced-IZ local singlet
    whose two contained plaquettes are simultaneously flippable.  The same
    structural probe is intentionally attempted on honeycomb targets, but a
    candidate is admitted as an ICQMBS ``Y_R`` only when it actually
    annihilates the originating caged eigenstate.
    """

    state_index: int
    component_index: int
    variable_indices: tuple[int, ...]
    plaquette_ids: tuple[int, int]
    operator: sp.csr_array
    state_darkness_residual: float


@dataclass(frozen=True, slots=True)
class TimedResult:
    """Small helper for recording scientific benchmark wall times."""

    value: Any
    seconds: float


def benchmark_cases() -> tuple[JumpBridgeCase, ...]:
    """Return the four claim-critical QDM targets from the provisioning cache."""
    return (
        JumpBridgeCase("square_qdm_4x4_single", "square", 1),
        JumpBridgeCase("square_qdm_4x4_multi8", "square", 8),
        JumpBridgeCase("honeycomb_qdm_4x4_single", "honeycomb", 1),
        JumpBridgeCase("honeycomb_qdm_4x4_multi4", "honeycomb", 4),
    )


def _model(model_name: str):
    if model_name == "square":
        return SquareQDMModel(
            lx=4,
            ly=4,
            boundary_condition="periodic",
            winding_x=0,
            winding_y=0,
            winding_convention="electric",
            coup_kin=-1.0,
            coup_pot=0.7,
        )
    if model_name == "honeycomb":
        return HoneycombQDMModel(
            lx=4,
            ly=4,
            boundary_condition="periodic",
            winding_x=-2,
            winding_y=0,
            coup_kin=-1.0,
            coup_pot=0.7,
        )
    raise ValueError(f"Unsupported model_name: {model_name!r}.")


def _timed(function) -> TimedResult:
    start = time.perf_counter()
    value = function()
    return TimedResult(value=value, seconds=time.perf_counter() - start)


def _search(build_result) -> TimedResult:
    config = CageSearchConfig(
        search_type="type1",
        tolerance=1.0e-10,
        degenerate_basis_strategy="ipr",
        ipr_n_restarts=64,
        ipr_candidate_count=64,
        ipr_random_seed=SEARCH_SEED,
        store_full_states=True,
    )
    searcher = CageSearcher.from_model_build_result(build_result, config=config)
    return _timed(searcher.run)


def _phase_fixed_dense(operator: Any) -> np.ndarray:
    dense = np.asarray(
        operator.toarray() if sp.issparse(operator) else operator,
        dtype=np.complex128,
    )
    norm = float(np.linalg.norm(dense))
    if norm <= 1.0e-14:
        return dense
    dense = dense / norm
    nonzero = np.flatnonzero(np.abs(dense.ravel()) > 1.0e-10)
    if nonzero.size:
        pivot = dense.ravel()[int(nonzero[0])]
        dense = dense * np.exp(-1.0j * np.angle(pivot))
    return dense


def _as_sparse_operator(operator: Any) -> sp.csr_array:
    """Materialize a benchmark operator as CSR, including legacy lazy wrappers."""
    if sp.issparse(operator):
        return sp.csr_array(operator)
    tocsr = getattr(operator, "tocsr", None)
    if callable(tocsr):
        return sp.csr_array(tocsr())
    return sp.csr_array(np.asarray(operator, dtype=np.complex128))


def _deduplicate_operators(
    operators: Iterable[Any],
    *,
    tolerance: float = 1.0e-8,
) -> tuple[sp.csr_array, ...]:
    unique: list[sp.csr_array] = []
    representatives: list[np.ndarray] = []
    for operator in operators:
        sparse_operator = sp.csr_array(operator)
        representative = _phase_fixed_dense(sparse_operator)
        if np.linalg.norm(representative) <= 1.0e-14:
            continue
        if any(
            np.linalg.norm(representative - existing) <= tolerance for existing in representatives
        ):
            continue
        unique.append(sparse_operator)
        representatives.append(representative)
    return tuple(unique)


def _full_target(records: Sequence[Any]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(record.full_state, dtype=np.complex128) for record in records]
    )


def _reconstruct_directed_rows(
    *,
    records: Sequence[Any],
    build_result: Any,
    search_result: Any,
) -> tuple[DirectedCagingRow, ...]:
    """Reconstruct the reduced-IZ directed rows used as ICQMBS ``A_R``.

    For one interference zero ``z`` with active cage-support neighbours ``s_j``,
    the row is ``A_R = |z> sum_j K[z,s_j] <s_j|`` on the reduced local support.
    The helper records the Hermitian ``Z_R=A_R+A_R^dagger`` and local
    ``Q_R=A_R^dagger A_R`` using the same bounded support.
    """
    basis_configs = basis_configs_from_build_result(build_result)
    rows: list[DirectedCagingRow] = []
    for state_index, record in enumerate(records):
        environment = diagnose_cage_environment_reduction(
            record.cage_state,
            kinetic_matrix=build_result.kinetic,
            basis_configs=basis_configs,
            hilbert_size=search_result.hilbert_size,
            config=EnvironmentReductionConfig(sector_policy="infer_support_component"),
        )
        for probe in environment.zero_reports:
            variable_indices = tuple(int(value) for value in probe.local_variable_indices)
            target_pattern = tuple(
                int(value) for value in basis_configs[int(probe.zero_index), variable_indices]
            )
            source_patterns = tuple(
                tuple(int(value) for value in basis_configs[int(index), variable_indices])
                for index in probe.active_neighbors
            )
            template = directed_transition_witness_template(
                target_pattern=target_pattern,
                source_patterns=source_patterns,
                amplitudes=probe.active_matrix_elements,
                source_zero_indices=(int(probe.zero_index),),
                metadata={
                    "name": "A_R",
                    "state_index": int(state_index),
                    "zero_index": int(probe.zero_index),
                    "mechanism": probe.probe_mechanism_label,
                },
                normalization="operator_norm",
            )
            witness = template.instantiate(variable_indices)
            directed = witness.embed(basis_configs)
            hermitian = (
                hermitianize_local_witness_template(
                    template,
                    normalization="operator_norm",
                    metadata={"name": "Z_R"},
                )
                .instantiate(variable_indices)
                .embed(basis_configs)
            )
            positive = embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=variable_indices,
                local_patterns=template.local_patterns,
                local_operator=template.q_operator,
            )
            rows.append(
                DirectedCagingRow(
                    state_index=int(state_index),
                    zero_index=int(probe.zero_index),
                    variable_indices=variable_indices,
                    target_pattern=target_pattern,
                    source_patterns=source_patterns,
                    amplitudes=np.asarray(
                        template.local_operator[0, 1:],
                        dtype=np.complex128,
                    ),
                    operator=directed,
                    hermitian_operator=hermitian,
                    positive_operator=positive,
                )
            )
    return tuple(rows)


def _unique_directed_rows(
    rows: Sequence[DirectedCagingRow],
) -> tuple[DirectedCagingRow, ...]:
    unique: list[DirectedCagingRow] = []
    representatives: list[np.ndarray] = []
    for row in rows:
        representative = _phase_fixed_dense(row.operator)
        if any(np.linalg.norm(representative - existing) <= 1.0e-8 for existing in representatives):
            continue
        unique.append(row)
        representatives.append(representative)
    return tuple(unique)


def _plaquette_flippability_diagonal(
    *,
    model: Any,
    basis_configs: np.ndarray,
    plaquette_id: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the constrained-basis diagonal of the QDM flippability projector."""
    link_ids = tuple(int(value) for value in model.lattice.plaquette_links(plaquette_id))
    variable_indices = tuple(int(model.layout.link_variable_index(link_id)) for link_id in link_ids)
    pattern0, pattern1 = alternating_binary_patterns(len(variable_indices))
    local_values = basis_configs[:, np.asarray(variable_indices, dtype=np.int64)]
    flippable = np.all(local_values == pattern0, axis=1) | np.all(
        local_values == pattern1,
        axis=1,
    )
    return flippable.astype(np.complex128), variable_indices


def _reconstruct_shifted_potential_candidates(
    *,
    records: Sequence[Any],
    build_result: Any,
    search_result: Any,
    model: Any,
) -> tuple[ShiftedPotentialCagingOperator, ...]:
    """Reconstruct companion-style QDM ``Y_R`` candidates from local singlets.

    We deliberately infer the two-plaquette motif from the reduced-IZ local
    structure rather than hard-coding square-lattice coordinates.  A structural
    candidate is returned even if it fails the state-darkness test; the caller
    can therefore distinguish "not defined for this target" from "not searched".
    """
    basis_configs = np.asarray(basis_configs_from_build_result(build_result))
    candidates: list[ShiftedPotentialCagingOperator] = []
    for state_index, record in enumerate(records):
        environment = diagnose_cage_environment_reduction(
            record.cage_state,
            kinetic_matrix=build_result.kinetic,
            basis_configs=basis_configs,
            hilbert_size=search_result.hilbert_size,
            config=EnvironmentReductionConfig(sector_policy="infer_support_component"),
        )
        structure = local_structure_report_from_environment_report(
            environment,
            basis_configs=basis_configs,
            state=record.full_state,
            model=model,
            decomposition="exact_support",
            tolerance=TOLERANCE,
            max_matrix_unit_terms=None,
        )
        for readout_report in structure.readout_reports:
            plaquette_ids = tuple(int(value) for value in readout_report.flippable_plaquette_ids)
            if len(plaquette_ids) != 2 or readout_report.n_singlet_like_pairs != 1:
                continue
            diagonal = -2.0 * np.ones(basis_configs.shape[0], dtype=np.complex128)
            support: set[int] = set()
            for plaquette_id in plaquette_ids:
                contribution, variable_indices = _plaquette_flippability_diagonal(
                    model=model,
                    basis_configs=basis_configs,
                    plaquette_id=plaquette_id,
                )
                diagonal += contribution
                support.update(variable_indices)
            operator = sp.csr_array(sp.diags(diagonal, format="csr"))
            state = np.asarray(record.full_state, dtype=np.complex128)
            candidates.append(
                ShiftedPotentialCagingOperator(
                    state_index=int(state_index),
                    component_index=int(readout_report.readout.component_index),
                    variable_indices=tuple(sorted(support)),
                    plaquette_ids=(int(plaquette_ids[0]), int(plaquette_ids[1])),
                    operator=operator,
                    state_darkness_residual=float(np.linalg.norm(operator @ state)),
                )
            )
    return tuple(candidates)


def _common_dark_span_basis(
    *,
    operators: Sequence[Any],
    target_basis: np.ndarray,
    operator_prefix: str,
) -> tuple[sp.csr_array, ...]:
    """Return a basis for linear combinations dark on the full target."""
    if not operators:
        return ()
    report = diagnose_manifold_dark_operator_basis(
        states=target_basis,
        operators=operators,
        operator_names=tuple(f"{operator_prefix}_{index}" for index in range(len(operators))),
        tolerance=TOLERANCE,
        max_candidates=None,
        candidate_strategy="svd_basis",
    )
    return _deduplicate_operators(
        _combine_detector(candidate.coefficients, operators) for candidate in report.candidates
    )


def _combine_detector(
    coefficients: np.ndarray,
    detector_operators: Sequence[Any],
) -> sp.csr_array:
    if not detector_operators:
        raise ValueError("detector_operators must not be empty.")
    combined = sp.csr_array(detector_operators[0].shape, dtype=np.complex128)
    for coefficient, operator in zip(coefficients, detector_operators, strict=True):
        if abs(coefficient) > 0.0:
            combined = combined + coefficient * sp.csr_array(operator)
    return combined


def _selected_l_family(design: CageLindbladDesignResult) -> tuple[sp.csr_array, ...]:
    report = design.workflow.dark_operator_report
    detector_indices = design.workflow.recycled_selection.selected_detector_indices
    operators = []
    for detector_index in dict.fromkeys(detector_indices):
        candidate = report.candidates[int(detector_index)]
        operators.append(_combine_detector(candidate.coefficients, design.detector_operators))
    return _deduplicate_operators(operators)


def _selected_m_family(design: CageLindbladDesignResult) -> tuple[sp.csr_array, ...]:
    readouts = design.workflow.recycled_recycler_readouts(
        basis_configs=design.problem.basis_configs,
        states=design.problem.target_basis,
    )
    return _deduplicate_operators(
        embed_local_pattern_operator(
            basis_configs=design.problem.basis_configs,
            variable_indices=readout.variable_indices,
            local_patterns=readout.local_patterns,
            local_operator=readout.local_operator,
        )
        for readout in readouts
    )


def _retargeted_a_family(
    *,
    rows: Sequence[DirectedCagingRow],
    build_result: Any,
    target_basis: np.ndarray,
) -> tuple[RetargetedDirectedJump, ...]:
    """Enumerate bounded ``J=|tau><v_R|`` retargetings of directed caging rows.

    ``A_R=|z><v_R|`` supplies the cancellation bra.  Replacing the output
    pattern ``z`` by any locally allowed ``tau`` preserves the same caging
    constraint whenever the compressed operator remains target-dark.  This is
    the direct bridge ``A_R -> J_R^(tau)``; algebraically it is a restricted
    left dressing ``M_{tau,z} A_R`` but no generic detector is introduced.
    """
    basis_configs = basis_configs_from_build_result(build_result)
    candidates: list[RetargetedDirectedJump] = []
    representatives: list[np.ndarray] = []
    for directed_row_index, row in enumerate(rows):
        observed_patterns: list[tuple[int, ...]] = []
        for config in basis_configs:
            pattern = tuple(int(value) for value in config[list(row.variable_indices)])
            if pattern not in observed_patterns:
                observed_patterns.append(pattern)

        for output_pattern in observed_patterns:
            patterns: list[tuple[int, ...]] = []
            for pattern in (output_pattern, *row.source_patterns):
                if pattern not in patterns:
                    patterns.append(pattern)
            index = {pattern: position for position, pattern in enumerate(patterns)}
            local_operator = np.zeros(
                (len(patterns), len(patterns)),
                dtype=np.complex128,
            )
            for source_pattern, amplitude in zip(
                row.source_patterns,
                row.amplitudes,
                strict=True,
            ):
                local_operator[
                    index[output_pattern],
                    index[source_pattern],
                ] += amplitude
            jump = embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=row.variable_indices,
                local_patterns=tuple(patterns),
                local_operator=local_operator,
            )
            if sp.linalg.norm(jump) <= 1.0e-12:
                continue
            if np.linalg.norm(jump @ target_basis) > TOLERANCE:
                continue
            representative = _phase_fixed_dense(jump)
            if any(
                np.linalg.norm(representative - existing) <= 1.0e-8 for existing in representatives
            ):
                continue
            candidates.append(
                RetargetedDirectedJump(
                    directed_row_index=int(directed_row_index),
                    source_state_index=int(row.state_index),
                    zero_index=int(row.zero_index),
                    variable_indices=row.variable_indices,
                    output_pattern=output_pattern,
                    operator=sp.csr_array(jump),
                )
            )
            representatives.append(representative)
    return tuple(candidates)


def _target_inflow_norm(operator: Any, target_basis: np.ndarray) -> float:
    target, _ = np.linalg.qr(np.asarray(target_basis, dtype=np.complex128))
    projector = target @ target.conj().T
    dim = target.shape[0]
    complement = np.eye(dim, dtype=np.complex128) - projector
    action = np.asarray(_as_sparse_operator(operator) @ complement, dtype=np.complex128)
    return float(np.linalg.norm(target.conj().T @ action))


def _sort_retargeted_by_inflow(
    candidates: Sequence[RetargetedDirectedJump],
    target_basis: np.ndarray,
) -> tuple[RetargetedDirectedJump, ...]:
    scored = [
        (_target_inflow_norm(candidate.operator, target_basis), index, candidate)
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored)


def _select_certified_retargeted_single(
    *,
    candidates: Sequence[RetargetedDirectedJump],
    hamiltonian: Any,
    target_basis: np.ndarray,
) -> tuple[RetargetedDirectedJump | None, list[dict[str, object]]]:
    """Return the highest-inflow single direct jump that certifies attraction."""
    rows: list[dict[str, object]] = []
    selected: RetargetedDirectedJump | None = None
    for candidate_index, candidate in enumerate(candidates):
        diagnostics = diagnose_attractive_subspace(
            hamiltonian=hamiltonian,
            jumps=(candidate.operator,),
            target_basis=target_basis,
            tolerance=TOLERANCE,
        )
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "directed_row_index": int(candidate.directed_row_index),
                "source_state_index": int(candidate.source_state_index),
                "zero_index": int(candidate.zero_index),
                "support": repr(candidate.variable_indices),
                "support_size": len(candidate.variable_indices),
                "output_pattern": repr(candidate.output_pattern),
                "target_directed_inflow_norm": _target_inflow_norm(
                    candidate.operator,
                    target_basis,
                ),
                "max_darkness_residual": diagnostics.max_jump_darkness_residual,
                "no_inflow_dimension": diagnostics.no_inflow_dimension,
                "invariant_obstruction_dimension": (diagnostics.invariant_obstruction_dimension),
                "target_attractive_certified": (diagnostics.target_attractive_certified),
            }
        )
        if diagnostics.target_attractive_certified:
            selected = candidate
            break
    return selected, rows


def _retargeted_prefix_scan(
    *,
    operators: Sequence[sp.csr_array],
    hamiltonian: Any,
    target_basis: np.ndarray,
) -> list[dict[str, object]]:
    """Test logarithmic prefixes without assuming monotonicity in jump count."""
    if not operators:
        return []
    budgets = []
    budget = 1
    while budget < len(operators):
        budgets.append(budget)
        budget *= 2
    budgets.append(len(operators))
    rows: list[dict[str, object]] = []
    for budget in dict.fromkeys(budgets):
        diagnostics = diagnose_attractive_subspace(
            hamiltonian=hamiltonian,
            jumps=tuple(operators[:budget]),
            target_basis=target_basis,
            tolerance=TOLERANCE,
        )
        rows.append(
            {
                "n_jumps": int(budget),
                "max_darkness_residual": diagnostics.max_jump_darkness_residual,
                "total_target_directed_inflow_norm": (
                    diagnostics.total_target_directed_inflow_norm
                ),
                "no_inflow_dimension": diagnostics.no_inflow_dimension,
                "invariant_obstruction_dimension": (diagnostics.invariant_obstruction_dimension),
                "target_attractive_certified": (diagnostics.target_attractive_certified),
            }
        )
    return rows


def _family_metrics(
    *,
    hamiltonian: Any,
    target_basis: np.ndarray,
    family_name: str,
    provenance: str,
    operators: Sequence[Any],
    normalization: str,
    construction_seconds: float = 0.0,
) -> tuple[dict[str, object], AttractiveSubspaceDiagnostics]:
    timed = _timed(
        lambda: diagnose_attractive_subspace(
            hamiltonian=hamiltonian,
            jumps=tuple(operators),
            target_basis=target_basis,
            tolerance=TOLERANCE,
        )
    )
    diagnostics = timed.value
    total_nnz = int(sum(_as_sparse_operator(operator).nnz for operator in operators))
    row = {
        "family": family_name,
        "provenance": provenance,
        "normalization": normalization,
        "n_jumps": len(operators),
        "total_jump_nnz": total_nnz,
        "max_jump_nnz": int(
            max((_as_sparse_operator(operator).nnz for operator in operators), default=0)
        ),
        "max_darkness_residual": diagnostics.max_jump_darkness_residual,
        "total_target_directed_inflow_norm": (diagnostics.total_target_directed_inflow_norm),
        "no_inflow_dimension": diagnostics.no_inflow_dimension,
        "invariant_obstruction_dimension": (diagnostics.invariant_obstruction_dimension),
        "target_attractive_certified": diagnostics.target_attractive_certified,
        "common_jump_kernel_dimension": diagnostics.common_jump_kernel_dimension,
        "old_h_invariant_kernel_dimension": (diagnostics.old_h_invariant_kernel_dimension),
        "construction_seconds": float(construction_seconds),
        "diagnostic_seconds": float(timed.seconds),
    }
    return row, diagnostics


def _projection_stats(
    operator: Any,
    basis_operators: Sequence[Any],
) -> tuple[float, float]:
    vector = _phase_fixed_dense(operator).ravel()
    if not basis_operators:
        return 0.0, 1.0
    normalized_basis = [
        _phase_fixed_dense(basis_operator).ravel() for basis_operator in basis_operators
    ]
    max_overlap = float(
        max(abs(np.vdot(basis_vector, vector)) for basis_vector in normalized_basis)
    )
    matrix = np.column_stack(normalized_basis)
    left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return max_overlap, 1.0
    cutoff = 1.0e-10 * max(1.0, float(singular_values[0]))
    rank = int(np.count_nonzero(singular_values > cutoff))
    if rank == 0:
        return max_overlap, 1.0
    span_basis = left_vectors[:, :rank]
    residual = float(np.linalg.norm(vector - span_basis @ (span_basis.conj().T @ vector)))
    return max_overlap, residual


def _operator_provenance_rows(
    *,
    operator_role: str,
    operators: Sequence[Any],
    directed_basis: Sequence[Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operator_index, operator in enumerate(operators):
        overlap, residual = _projection_stats(operator, directed_basis)
        rows.append(
            {
                "operator_role": operator_role,
                "operator_index": int(operator_index),
                "A_span_max_overlap": overlap,
                "A_span_projection_residual": residual,
                "global_rank": int(np.linalg.matrix_rank(sp.csr_array(operator).toarray())),
                "global_kernel_dim": int(
                    sp.csr_array(operator).shape[0]
                    - np.linalg.matrix_rank(sp.csr_array(operator).toarray())
                ),
                "nnz": int(sp.csr_array(operator).nnz),
            }
        )
    return rows


def _directed_action_rows(
    *,
    rows: Sequence[DirectedCagingRow],
    target_basis: np.ndarray,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        column_norms = np.linalg.norm(row.operator @ target_basis, axis=0)
        output.append(
            {
                "directed_row_index": int(row_index),
                "source_state_index": int(row.state_index),
                "zero_index": int(row.zero_index),
                "support": repr(row.variable_indices),
                "support_size": len(row.variable_indices),
                "n_sources": len(row.source_patterns),
                "source_state_darkness_residual": float(column_norms[int(row.state_index)]),
                "max_column_darkness_residual": float(max(column_norms, default=0.0)),
                "total_target_darkness_residual": float(
                    np.linalg.norm(row.operator @ target_basis)
                ),
                "adjoint_target_darkness_residual": float(
                    np.linalg.norm(row.operator.conj().T @ target_basis)
                ),
                "Z_target_darkness_residual": float(
                    np.linalg.norm(row.hermitian_operator @ target_basis)
                ),
                "A_target_directed_inflow_norm": _target_inflow_norm(
                    row.operator,
                    target_basis,
                ),
            }
        )
    return output


def _shifted_potential_action_rows(
    *,
    candidates: Sequence[ShiftedPotentialCagingOperator],
    target_basis: np.ndarray,
) -> list[dict[str, object]]:
    """Report state-level versus full-target darkness for candidate ``Y_R``."""
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        column_norms = np.linalg.norm(candidate.operator @ target_basis, axis=0)
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "source_state_index": int(candidate.state_index),
                "component_index": int(candidate.component_index),
                "support": repr(candidate.variable_indices),
                "support_size": len(candidate.variable_indices),
                "plaquette_ids": repr(candidate.plaquette_ids),
                "source_state_darkness_residual": float(candidate.state_darkness_residual),
                "max_column_darkness_residual": float(max(column_norms, default=0.0)),
                "total_target_darkness_residual": float(
                    np.linalg.norm(candidate.operator @ target_basis)
                ),
                "companion_Y_defined_for_source_state": bool(
                    candidate.state_darkness_residual <= TOLERANCE
                ),
            }
        )
    return rows


def _unrestricted_left_ideal_projection(
    operator: Any,
    generators: Sequence[Any],
) -> tuple[int, float, float]:
    """Project onto the unrestricted left ideal generated by ``generators``.

    For matrices ``L_a``, allowing arbitrary left multipliers produces exactly
    the matrices whose rows lie in the joint row space of the ``L_a``.  This
    space contains every bounded-local left ideal generated from the same
    operators.  Therefore a nonzero residual here is a strong certificate that
    a completion jump is genuinely outside the caging-generated local space;
    no enumeration of a huge local multiplier basis is then necessary.
    """
    dense = np.asarray(_as_sparse_operator(operator).toarray(), dtype=np.complex128)
    operator_norm = float(np.linalg.norm(dense))
    if operator_norm <= 1.0e-14:
        return 0, 0.0, 0.0
    if not generators:
        return 0, operator_norm, 1.0

    joint_range = np.hstack(
        [
            np.asarray(_as_sparse_operator(generator).toarray(), dtype=np.complex128).conj().T
            for generator in generators
        ]
    )
    left_vectors, singular_values, _ = np.linalg.svd(joint_range, full_matrices=False)
    if singular_values.size == 0:
        return 0, operator_norm, 1.0
    cutoff = 1.0e-10 * max(1.0, float(singular_values[0]))
    rank = int(np.count_nonzero(singular_values > cutoff))
    if rank == 0:
        return 0, operator_norm, 1.0
    row_space = left_vectors[:, :rank]
    projected = dense @ row_space @ row_space.conj().T
    residual = float(np.linalg.norm(dense - projected))
    return rank, residual, residual / operator_norm


def _completion_caging_span_rows(
    *,
    completion_operators: Sequence[Any],
    directed_generators: Sequence[Any],
    generic_l_generators: Sequence[Any],
) -> list[dict[str, object]]:
    """Test whether completion jumps can belong to either caging left ideal."""
    rows: list[dict[str, object]] = []
    for operator_index, operator in enumerate(completion_operators):
        a_rank, a_residual, a_relative = _unrestricted_left_ideal_projection(
            operator,
            directed_generators,
        )
        l_rank, l_residual, l_relative = _unrestricted_left_ideal_projection(
            operator,
            generic_l_generators,
        )
        rows.append(
            {
                "completion_index": int(operator_index),
                "A_left_ideal_row_space_dimension": int(a_rank),
                "A_left_ideal_projection_residual": a_residual,
                "A_left_ideal_relative_projection_residual": a_relative,
                "outside_A_generated_left_ideal_certified": bool(a_relative > TOLERANCE),
                "L_left_ideal_row_space_dimension": int(l_rank),
                "L_left_ideal_projection_residual": l_residual,
                "L_left_ideal_relative_projection_residual": l_relative,
                "outside_selected_L_generated_left_ideal_certified": bool(l_relative > TOLERANCE),
                "interpretation": (
                    "nonzero unrestricted-left-ideal residual proves the completion "
                    "is outside every bounded-local left ideal generated by the same caging rows"
                ),
            }
        )
    return rows


def _operator_rank_kernel(operator: Any) -> tuple[int, int]:
    dense = np.asarray(_as_sparse_operator(operator).toarray(), dtype=np.complex128)
    rank = int(np.linalg.matrix_rank(dense))
    return rank, int(dense.shape[1] - rank)


def _caging_operator_map_rows(
    *,
    model: Any,
    directed_rows: Sequence[DirectedCagingRow],
    shifted_potential_candidates: Sequence[ShiftedPotentialCagingOperator],
    target_basis: np.ndarray,
    design: CageLindbladDesignResult,
) -> list[dict[str, object]]:
    """Export reconstructable provenance for the ICQMBS ``A/Z/Y/L`` hierarchy."""
    output: list[dict[str, object]] = []
    for row_index, row in enumerate(directed_rows):
        for role, operator in (("A_R", row.operator), ("Z_R", row.hermitian_operator)):
            rank, kernel_dim = _operator_rank_kernel(operator)
            output.append(
                {
                    "operator_id": f"{role}_{row_index}",
                    "operator_role": role,
                    "provenance": "reduced_IZ_directed_kinetic_caging_relation",
                    "source_state_index": int(row.state_index),
                    "interference_zero_index": int(row.zero_index),
                    "support_variables": row.variable_indices,
                    "support_size": len(row.variable_indices),
                    "support_constraint_graph_diameter": _constraint_graph_diameter(
                        model,
                        row.variable_indices,
                    ),
                    "target_pattern": row.target_pattern,
                    "source_patterns": row.source_patterns,
                    "directed_amplitudes": tuple(complex(value) for value in row.amplitudes),
                    "target_darkness_residual": float(np.linalg.norm(operator @ target_basis)),
                    "global_rank": rank,
                    "global_kernel_dim": kernel_dim,
                }
            )

    for candidate_index, candidate in enumerate(shifted_potential_candidates):
        rank, kernel_dim = _operator_rank_kernel(candidate.operator)
        output.append(
            {
                "operator_id": f"Y_R_candidate_{candidate_index}",
                "operator_role": "Y_R",
                "provenance": "two_flippable_plaquette_shifted_potential_local_singlet",
                "source_state_index": int(candidate.state_index),
                "component_index": int(candidate.component_index),
                "support_variables": candidate.variable_indices,
                "support_size": len(candidate.variable_indices),
                "support_constraint_graph_diameter": _constraint_graph_diameter(
                    model,
                    candidate.variable_indices,
                ),
                "plaquette_ids": candidate.plaquette_ids,
                "source_state_darkness_residual": candidate.state_darkness_residual,
                "target_darkness_residual": float(
                    np.linalg.norm(candidate.operator @ target_basis)
                ),
                "companion_Y_defined_for_source_state": bool(
                    candidate.state_darkness_residual <= TOLERANCE
                ),
                "global_rank": rank,
                "global_kernel_dim": kernel_dim,
            }
        )

    selected_indices = tuple(
        dict.fromkeys(design.workflow.recycled_selection.selected_detector_indices)
    )
    for output_index, detector_index in enumerate(selected_indices):
        candidate = design.workflow.dark_operator_report.candidates[int(detector_index)]
        operator = _combine_detector(candidate.coefficients, design.detector_operators)
        rank, kernel_dim = _operator_rank_kernel(operator)
        nonzero_terms = []
        support: set[int] = set()
        for term_index, coefficient in enumerate(candidate.coefficients):
            if abs(coefficient) <= 1.0e-10:
                continue
            term_payload: dict[str, object] = {
                "detector_operator_index": int(term_index),
                "detector_operator_name": design.detector_operator_names[int(term_index)],
                "coefficient": complex(coefficient),
            }
            if len(design.detector_terms) == len(design.detector_operators):
                descriptor = design.detector_terms[int(term_index)]
                variables = tuple(
                    int(value)
                    for value in (
                        descriptor.support_variables
                        if descriptor.support_variables
                        else descriptor.support_links
                    )
                )
                support.update(variables)
                term_payload["support_variables"] = variables
                term_payload["support_plaquettes"] = descriptor.support_plaquettes
            nonzero_terms.append(term_payload)
        output.append(
            {
                "operator_id": f"L_R_modern_{output_index}",
                "operator_role": "L_R",
                "provenance": "modern_common_dark_kinetic_detector",
                "detector_candidate_index": int(detector_index),
                "support_variables": tuple(sorted(support)),
                "support_size": len(support),
                "support_constraint_graph_diameter": (
                    _constraint_graph_diameter(model, tuple(sorted(support))) if support else 0
                ),
                "nonzero_detector_terms": tuple(nonzero_terms),
                "target_darkness_residual": float(np.linalg.norm(operator @ target_basis)),
                "global_rank": rank,
                "global_kernel_dim": kernel_dim,
            }
        )
    return output


def _basis_rotation_consistency(
    *,
    operators: Sequence[Any],
    target_basis: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(20260815)
    target, _ = np.linalg.qr(np.asarray(target_basis, dtype=np.complex128))
    random_matrix = rng.normal(size=(target.shape[1], target.shape[1])) + 1.0j * rng.normal(
        size=(target.shape[1], target.shape[1])
    )
    rotation, _ = np.linalg.qr(random_matrix)
    rotated = target @ rotation
    original = max(
        (float(np.linalg.norm(sp.csr_array(operator) @ target)) for operator in operators),
        default=0.0,
    )
    rotated_residual = max(
        (float(np.linalg.norm(sp.csr_array(operator) @ rotated)) for operator in operators),
        default=0.0,
    )
    return original, rotated_residual


def _constraint_graph_diameter(model: Any, support: tuple[int, ...]) -> int:
    """Return support diameter in the variable graph induced by local constraints."""
    n_variables = int(model.layout.n_variables)
    adjacency = [set() for _ in range(n_variables)]
    for constraint in model.make_constraints():
        variables = tuple(int(value) for value in constraint.affected_variables())
        for left in variables:
            adjacency[left].update(value for value in variables if value != left)

    max_distance = 0
    support_set = set(support)
    for source in support:
        distance = {int(source): 0}
        frontier = [int(source)]
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[node] + 1
                frontier.append(neighbor)
        if not support_set.issubset(distance):
            raise RuntimeError("selected local support is disconnected in constraint graph.")
        max_distance = max(
            max_distance,
            max(distance[target] for target in support),
        )
    return int(max_distance)


def _direct_jump_locality_certificate(
    *,
    model: Any,
    build_result: Any,
    caging_row: DirectedCagingRow,
    jump: RetargetedDirectedJump,
) -> dict[str, object]:
    """Certify that a compressed QDM direct jump is an environment-local rule.

    Every matrix-unit term is checked against each touched dimer constraint.
    Since exterior variables are unchanged, zero local dimer-count change makes
    the transition constraint preserving independently of the environment.  A
    finite-size exhaustive scan over all compatible source environments then
    verifies that the explicit local rule exactly matches the compressed matrix.
    """
    basis_configs = np.asarray(basis_configs_from_build_result(build_result))
    support = tuple(int(value) for value in jump.variable_indices)
    support_array = np.asarray(support, dtype=np.int64)
    position = {variable: index for index, variable in enumerate(support)}
    output_pattern = tuple(int(value) for value in jump.output_pattern)
    constraints = tuple(model.make_constraints())
    basis_lookup = {
        tuple(int(value) for value in config): int(index)
        for index, config in enumerate(basis_configs)
    }

    term_rows: list[dict[str, object]] = []
    explicit = sp.csr_array(jump.operator.shape, dtype=np.complex128)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[complex] = []
    total_environment_mismatches = 0
    max_constraint_count_delta = 0

    for term_index, (source_pattern, amplitude) in enumerate(
        zip(caging_row.source_patterns, caging_row.amplitudes, strict=True)
    ):
        changed_variables = tuple(
            support[index]
            for index, (source_value, output_value) in enumerate(
                zip(source_pattern, output_pattern, strict=True)
            )
            if source_value != output_value
        )
        touched_constraints = 0
        term_max_delta = 0
        for constraint in constraints:
            variables = tuple(int(value) for value in constraint.affected_variables())
            if not set(changed_variables).intersection(variables):
                continue
            touched_constraints += 1
            delta = sum(
                output_pattern[position[variable]] - source_pattern[position[variable]]
                for variable in variables
                if variable in position
            )
            term_max_delta = max(term_max_delta, abs(int(delta)))

        source_array = np.asarray(source_pattern, dtype=basis_configs.dtype)
        matching = np.flatnonzero(np.all(basis_configs[:, support_array] == source_array, axis=1))
        environment_mismatches = 0
        for source_index in matching:
            target_config = np.asarray(basis_configs[int(source_index)]).copy()
            target_config[support_array] = np.asarray(
                output_pattern,
                dtype=target_config.dtype,
            )
            target_index = basis_lookup.get(tuple(int(value) for value in target_config))
            if target_index is None:
                environment_mismatches += 1
                continue
            row_indices.append(int(target_index))
            column_indices.append(int(source_index))
            values.append(complex(amplitude))

        max_constraint_count_delta = max(
            max_constraint_count_delta,
            term_max_delta,
        )
        total_environment_mismatches += environment_mismatches
        term_rows.append(
            {
                "term_index": int(term_index),
                "source_pattern": source_pattern,
                "output_pattern": output_pattern,
                "amplitude": complex(amplitude),
                "n_changed_variables": len(changed_variables),
                "n_touched_constraints": int(touched_constraints),
                "max_constraint_count_delta": int(term_max_delta),
                "n_compatible_environments": int(matching.size),
                "n_environment_mismatches": int(environment_mismatches),
            }
        )

    if values:
        explicit = sp.csr_array(
            (np.asarray(values, dtype=np.complex128), (row_indices, column_indices)),
            shape=jump.operator.shape,
        )
    mismatch_residual = float(sp.linalg.norm(explicit - jump.operator))
    certified = (
        max_constraint_count_delta == 0
        and total_environment_mismatches == 0
        and mismatch_residual <= TOLERANCE
    )
    return {
        "operator_role": "A_retargeted_single",
        "support": support,
        "support_size": len(support),
        "constraint_graph_diameter": _constraint_graph_diameter(model, support),
        "n_matrix_unit_terms": len(caging_row.source_patterns),
        "max_constraint_count_delta": int(max_constraint_count_delta),
        "n_environment_mismatches": int(total_environment_mismatches),
        "compressed_action_mismatch_residual": mismatch_residual,
        "bounded_support_certified": bool(certified),
        "term_certificates": term_rows,
    }


def _local_matrix_readout_locality_certificate(
    *,
    model: Any,
    build_result: Any,
    readout: Any,
    operator_role: str,
    operator_index: int,
) -> dict[str, object]:
    """Certify one selected local matrix readout against QDM constraints."""
    basis_configs = np.asarray(basis_configs_from_build_result(build_result))
    support = tuple(int(value) for value in readout.variable_indices)
    support_array = np.asarray(support, dtype=np.int64)
    position = {variable: index for index, variable in enumerate(support)}
    patterns = tuple(tuple(int(value) for value in row) for row in readout.local_patterns)
    local_operator = np.asarray(readout.local_operator, dtype=np.complex128)
    constraints = tuple(model.make_constraints())
    basis_lookup = {
        tuple(int(value) for value in config): int(index)
        for index, config in enumerate(basis_configs)
    }

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[complex] = []
    term_rows: list[dict[str, object]] = []
    max_constraint_count_delta = 0
    total_environment_mismatches = 0
    term_index = 0
    for target_index, target_pattern in enumerate(patterns):
        for source_index, source_pattern in enumerate(patterns):
            coefficient = complex(local_operator[target_index, source_index])
            if abs(coefficient) <= 1.0e-12:
                continue
            changed_variables = tuple(
                support[index]
                for index, (source_value, target_value) in enumerate(
                    zip(source_pattern, target_pattern, strict=True)
                )
                if source_value != target_value
            )
            touched_constraints = 0
            term_max_delta = 0
            for constraint in constraints:
                variables = tuple(int(value) for value in constraint.affected_variables())
                if not set(changed_variables).intersection(variables):
                    continue
                touched_constraints += 1
                delta = sum(
                    target_pattern[position[variable]] - source_pattern[position[variable]]
                    for variable in variables
                    if variable in position
                )
                term_max_delta = max(term_max_delta, abs(int(delta)))

            source_array = np.asarray(source_pattern, dtype=basis_configs.dtype)
            matching = np.flatnonzero(
                np.all(basis_configs[:, support_array] == source_array, axis=1)
            )
            environment_mismatches = 0
            for source_global_index in matching:
                target_config = np.asarray(basis_configs[int(source_global_index)]).copy()
                target_config[support_array] = np.asarray(
                    target_pattern,
                    dtype=target_config.dtype,
                )
                target_global_index = basis_lookup.get(tuple(int(value) for value in target_config))
                if target_global_index is None:
                    environment_mismatches += 1
                    continue
                row_indices.append(int(target_global_index))
                column_indices.append(int(source_global_index))
                values.append(coefficient)

            max_constraint_count_delta = max(
                max_constraint_count_delta,
                term_max_delta,
            )
            total_environment_mismatches += environment_mismatches
            term_rows.append(
                {
                    "term_index": int(term_index),
                    "target_pattern": target_pattern,
                    "source_pattern": source_pattern,
                    "coefficient": coefficient,
                    "n_changed_variables": len(changed_variables),
                    "n_touched_constraints": int(touched_constraints),
                    "max_constraint_count_delta": int(term_max_delta),
                    "n_compatible_environments": int(matching.size),
                    "n_environment_mismatches": int(environment_mismatches),
                }
            )
            term_index += 1

    explicit = sp.csr_array(
        (np.asarray(values, dtype=np.complex128), (row_indices, column_indices)),
        shape=build_result.hamiltonian.shape,
    )
    compressed = embed_local_pattern_operator(
        basis_configs=basis_configs,
        variable_indices=support,
        local_patterns=patterns,
        local_operator=local_operator,
    )
    mismatch_residual = float(sp.linalg.norm(explicit - compressed))
    certified = (
        max_constraint_count_delta == 0
        and total_environment_mismatches == 0
        and mismatch_residual <= TOLERANCE
    )
    return {
        "operator_role": operator_role,
        "operator_index": int(operator_index),
        "label": str(readout.label),
        "source": str(readout.source),
        "support": support,
        "support_size": len(support),
        "constraint_graph_diameter": _constraint_graph_diameter(model, support),
        "n_matrix_unit_terms": len(term_rows),
        "max_constraint_count_delta": int(max_constraint_count_delta),
        "n_environment_mismatches": int(total_environment_mismatches),
        "compressed_action_mismatch_residual": mismatch_residual,
        "bounded_support_certified": bool(certified),
        "term_certificates": term_rows,
    }


def _physical_local_matrix_unit_basis(
    *,
    model: Any,
    build_result: Any,
    support: tuple[int, ...],
) -> tuple[
    tuple[str, ...],
    tuple[sp.csr_array, ...],
    tuple[tuple[int, ...], ...],
]:
    """Build linearly independent environment-local dimer-preserving matrix units."""
    basis_configs = np.asarray(basis_configs_from_build_result(build_result))
    support_array = np.asarray(support, dtype=np.int64)
    patterns = tuple(
        sorted({tuple(int(value) for value in config[support_array]) for config in basis_configs})
    )
    position = {variable: index for index, variable in enumerate(support)}
    constraints = tuple(model.make_constraints())
    basis_lookup = {tuple(int(value) for value in config) for config in basis_configs}

    names: list[str] = []
    operators: list[sp.csr_array] = []
    orthonormal_vectors: list[np.ndarray] = []
    local_dim = len(patterns)
    for target_index, target_pattern in enumerate(patterns):
        for source_index, source_pattern in enumerate(patterns):
            changed_variables = {
                support[index]
                for index, (source_value, target_value) in enumerate(
                    zip(source_pattern, target_pattern, strict=True)
                )
                if source_value != target_value
            }
            preserves_constraints = True
            for constraint in constraints:
                variables = tuple(int(value) for value in constraint.affected_variables())
                if not changed_variables.intersection(variables):
                    continue
                delta = sum(
                    target_pattern[position[variable]] - source_pattern[position[variable]]
                    for variable in variables
                    if variable in position
                )
                if delta != 0:
                    preserves_constraints = False
                    break
            if not preserves_constraints:
                continue

            source_array = np.asarray(source_pattern, dtype=basis_configs.dtype)
            matching = np.flatnonzero(
                np.all(basis_configs[:, support_array] == source_array, axis=1)
            )
            if matching.size == 0:
                continue
            valid_in_all_environments = True
            for source_global_index in matching:
                target_config = np.asarray(basis_configs[int(source_global_index)]).copy()
                target_config[support_array] = np.asarray(
                    target_pattern,
                    dtype=target_config.dtype,
                )
                if tuple(int(value) for value in target_config) not in basis_lookup:
                    valid_in_all_environments = False
                    break
            if not valid_in_all_environments:
                continue

            local_operator = np.zeros(
                (local_dim, local_dim),
                dtype=np.complex128,
            )
            local_operator[target_index, source_index] = 1.0
            operator = embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=support,
                local_patterns=patterns,
                local_operator=local_operator,
            )
            dense_vector = (
                np.asarray(operator.toarray())
                .ravel()
                .astype(
                    np.complex128,
                    copy=False,
                )
            )
            norm = float(np.linalg.norm(dense_vector))
            if norm <= 1.0e-12:
                continue
            residual = dense_vector / norm
            for q_vector in orthonormal_vectors:
                residual = residual - q_vector * np.vdot(q_vector, residual)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= 1.0e-10:
                continue
            orthonormal_vectors.append(residual / residual_norm)
            names.append(f"E_{target_index}_{source_index}")
            operators.append(sp.csr_array(operator))

    return tuple(names), tuple(operators), patterns


def _operator_span_dimension(operators: Sequence[Any]) -> int:
    if not operators:
        return 0
    matrix = np.column_stack(
        [np.asarray(_as_sparse_operator(operator).toarray()).ravel() for operator in operators]
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    cutoff = 1.0e-10 * max(1.0, float(singular_values[0]))
    return int(np.count_nonzero(singular_values > cutoff))


def _local_annihilator_space_certificate(
    *,
    model: Any,
    build_result: Any,
    target_basis: np.ndarray,
    caging_row: DirectedCagingRow,
    selected_jump: RetargetedDirectedJump,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare the physical local dark space with the caging-generated left ideal."""
    support = selected_jump.variable_indices
    names, local_operators, local_patterns = _physical_local_matrix_unit_basis(
        model=model,
        build_result=build_result,
        support=support,
    )
    common_report = diagnose_manifold_dark_operator_basis(
        states=target_basis,
        operators=local_operators,
        operator_names=names,
        tolerance=TOLERANCE,
        max_candidates=None,
        candidate_strategy="svd_basis",
    )
    state_nullities = []
    for state_index in range(target_basis.shape[1]):
        report = diagnose_manifold_dark_operator_basis(
            states=target_basis[:, state_index],
            operators=local_operators,
            operator_names=names,
            tolerance=TOLERANCE,
            max_candidates=0,
            candidate_strategy="svd_basis",
        )
        state_nullities.append(int(report.detector_nullity))

    target, _ = np.linalg.qr(np.asarray(target_basis, dtype=np.complex128))
    rng = np.random.default_rng(20260815)
    random_matrix = rng.normal(size=(target.shape[1], target.shape[1])) + 1.0j * rng.normal(
        size=(target.shape[1], target.shape[1])
    )
    rotation, _ = np.linalg.qr(random_matrix)
    rotated_report = diagnose_manifold_dark_operator_basis(
        states=target @ rotation,
        operators=local_operators,
        operator_names=names,
        tolerance=TOLERANCE,
        max_candidates=0,
        candidate_strategy="svd_basis",
    )

    generated_products = _deduplicate_operators(
        multiplier @ caging_row.operator for multiplier in local_operators
    )
    generated_darkness = max(
        (float(np.linalg.norm(operator @ target_basis)) for operator in generated_products),
        default=0.0,
    )
    generated_span_dimension = _operator_span_dimension(generated_products)
    selected_overlap, selected_residual = _projection_stats(
        selected_jump.operator,
        generated_products,
    )
    common_coefficients = (
        np.column_stack([candidate.coefficients for candidate in common_report.candidates])
        if common_report.candidates
        else np.zeros((len(names), 0), dtype=np.complex128)
    )

    scorecard = {
        "support": support,
        "support_size": len(support),
        "n_observed_local_patterns": len(local_patterns),
        "physical_local_operator_basis_dimension": len(local_operators),
        "state_level_annihilator_dimensions": tuple(state_nullities),
        "common_target_annihilator_dimension": int(common_report.detector_nullity),
        "rotated_target_annihilator_dimension": int(rotated_report.detector_nullity),
        "basis_rotation_consistent": (
            common_report.detector_nullity == rotated_report.detector_nullity
        ),
        "caging_generated_left_ideal_dimension": int(generated_span_dimension),
        "caging_generated_max_target_darkness_residual": generated_darkness,
        "selected_direct_jump_A_span_max_overlap": selected_overlap,
        "selected_direct_jump_caging_span_projection_residual": selected_residual,
        "selected_direct_jump_in_caging_generated_span": selected_residual <= TOLERANCE,
    }
    arrays = {
        "common_annihilator_coefficients": common_coefficients,
        "operator_names": np.asarray(names, dtype=str),
        "local_patterns": np.asarray(local_patterns, dtype=np.int8),
    }
    return scorecard, arrays


def _legacy_single_family(
    *,
    model: Any,
    build_result: Any,
    search_result: Any,
    record: Any,
) -> TimedResult:
    basis_configs = basis_configs_from_build_result(build_result)
    environment = diagnose_cage_environment_reduction(
        record.cage_state,
        kinetic_matrix=build_result.kinetic,
        basis_configs=basis_configs,
        hilbert_size=search_result.hilbert_size,
        config=EnvironmentReductionConfig(sector_policy="infer_support_component"),
    )
    state = np.asarray(record.full_state, dtype=np.complex128)
    return _timed(
        lambda: build_type1_cage_lindblad_construction(
            model=model,
            build_result=build_result,
            cage_state=state,
            environment_report=environment,
            z_value=record.signature[1],
            monitor_source="reduced_iz_operators",
            reduced_iz_monitor_content="offdiagonal_only",
            reduced_iz_monitor_decomposition="exact_support",
            jump_operator_design="kinetic_outside_monitor_inside",
            recycling_jump_source="local_rdm_block_reset",
            deduplicate_recycling_regions=True,
            kinetic_jump_grouping="support_greedy",
            max_kinetic_jump_support_size=8,
            check_liouvillian=False,
        )
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_metadata() -> dict[str, str | None]:
    repo_root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": run("status", "--short"),
    }


def run_jump_bridge_case(
    case: JumpBridgeCase,
    *,
    output_dir: Path,
    include_legacy_single: bool,
) -> dict[str, object]:
    """Run one target and write its P0 bridge evidence."""
    model = _model(case.model_name)
    build_timed = _timed(
        lambda: model.build(
            basis_solver="dfs",
            builder="bitmask",
            backend="scipy",
            on_missing="raise",
        )
    )
    build_result = build_timed.value
    search_timed = _search(build_result)
    search_result = search_timed.value
    records = tuple(search_result[case.signature, : case.record_count])
    if len(records) != case.record_count:
        raise RuntimeError(
            f"{case.name} requested {case.record_count} records, got {len(records)}."
        )

    problem = build_cage_lindblad_problem(
        build_result=build_result,
        records=records,
        model=model,
        local_term_kind="plaquette",
    )
    detector_operators = build_cage_lindblad_detector_operators(
        model=model,
        build_result=build_result,
        operator_kind="kinetic",
        builder="sparse",
    )
    design_timed = _timed(
        lambda: problem.design_jumps(
            detector_operators=detector_operators,
            **MODERN_DESIGN_KWARGS,
        )
    )
    design = design_timed.value

    directed_rows = _reconstruct_directed_rows(
        records=records,
        build_result=build_result,
        search_result=search_result,
    )
    unique_directed = _unique_directed_rows(directed_rows)
    a_family = tuple(row.operator for row in unique_directed)
    a_dagger_family = _deduplicate_operators(row.operator.conj().T for row in unique_directed)
    z_family = _deduplicate_operators(row.hermitian_operator for row in unique_directed)
    q_family = _deduplicate_operators(row.positive_operator for row in unique_directed)
    shifted_potential_candidates = _reconstruct_shifted_potential_candidates(
        records=records,
        build_result=build_result,
        search_result=search_result,
        model=model,
    )
    valid_shifted_potential = tuple(
        candidate.operator
        for candidate in shifted_potential_candidates
        if candidate.state_darkness_residual <= TOLERANCE
    )
    y_family = _common_dark_span_basis(
        operators=valid_shifted_potential,
        target_basis=problem.target_basis,
        operator_prefix="Y_R",
    )
    l_family = _selected_l_family(design)
    m_family = _selected_m_family(design)
    retargeted_candidates = _sort_retargeted_by_inflow(
        _retargeted_a_family(
            rows=unique_directed,
            build_result=build_result,
            target_basis=problem.target_basis,
        ),
        problem.target_basis,
    )
    retargeted = tuple(candidate.operator for candidate in retargeted_candidates)
    selected_retargeted, retargeted_single_scan_rows = _select_certified_retargeted_single(
        candidates=retargeted_candidates,
        hamiltonian=build_result.hamiltonian,
        target_basis=problem.target_basis,
    )
    selected_retargeted_family = (
        () if selected_retargeted is None else (selected_retargeted.operator,)
    )
    retargeted_prefix_rows = _retargeted_prefix_scan(
        operators=retargeted,
        hamiltonian=build_result.hamiltonian,
        target_basis=problem.target_basis,
    )

    families = [
        (
            "A_only",
            "directed_reduced_IZ_caging_rows",
            a_family,
            "local_operator_norm",
            0.0,
        ),
        (
            "A_dagger_only",
            "adjoints_of_directed_reduced_IZ_rows",
            a_dagger_family,
            "local_operator_norm",
            0.0,
        ),
        (
            "Z_only",
            "hermitianized_directed_reduced_IZ_rows",
            z_family,
            "local_operator_norm",
            0.0,
        ),
    ]
    if y_family:
        families.append(
            (
                "Y_only",
                "full_target_dark_span_of_companion_shifted_potential_witnesses",
                y_family,
                "native_shifted_flippability_normalization",
                0.0,
            )
        )
    families.extend(
        [
            (
                "Q_only",
                "positive_A_dagger_A_local_witnesses",
                q_family,
                "local_operator_norm",
                0.0,
            ),
            (
                "A_retargeted_single",
                "selected_direct_J_equals_ket_tau_bra_v_from_A",
                selected_retargeted_family,
                "inherited_local_A_operator_norm",
                0.0,
            ),
            (
                "A_retargeted_all",
                "all_direct_J_equals_ket_tau_bra_v_from_A",
                retargeted,
                "inherited_local_A_operator_norm",
                0.0,
            ),
            (
                "L_only",
                "selected_modern_common_dark_kinetic_detectors",
                l_family,
                "construction_native",
                0.0,
            ),
            (
                "M_only",
                "selected_modern_left_multipliers",
                m_family,
                "construction_native",
                0.0,
            ),
            (
                "ML",
                "modern_caging_generated_left_dressed_family",
                design.recycled_jumps,
                "construction_native",
                design_timed.seconds,
            ),
            (
                "final",
                "modern_ML_plus_completion",
                design.jumps,
                "construction_native",
                design_timed.seconds,
            ),
        ]
    )

    family_rows: list[dict[str, object]] = []
    certificates: dict[str, AttractiveSubspaceDiagnostics] = {}
    for family_name, provenance, operators, normalization, construction_seconds in families:
        row, diagnostics = _family_metrics(
            hamiltonian=build_result.hamiltonian,
            target_basis=problem.target_basis,
            family_name=family_name,
            provenance=provenance,
            operators=operators,
            normalization=normalization,
            construction_seconds=construction_seconds,
        )
        row.update(
            {
                "case": case.name,
                "model": type(model).__name__,
                "hilbert_dimension": int(build_result.basis.n_states),
                "target_dimension": int(problem.target_basis.shape[1]),
            }
        )
        family_rows.append(row)
        certificates[family_name] = diagnostics

    legacy_summary: dict[str, object] | None = None
    if include_legacy_single and case.record_count == 1:
        legacy_timed = _legacy_single_family(
            model=model,
            build_result=build_result,
            search_result=search_result,
            record=records[0],
        )
        legacy = legacy_timed.value
        row, diagnostics = _family_metrics(
            hamiltonian=build_result.hamiltonian,
            target_basis=problem.target_basis,
            family_name="legacy_reduced_IZ_block_reset",
            provenance="deprecated_single_cage_constructor",
            operators=legacy.jumps,
            normalization="legacy_construction_native",
            construction_seconds=legacy_timed.seconds,
        )
        row.update(
            {
                "case": case.name,
                "model": type(model).__name__,
                "hilbert_dimension": int(build_result.basis.n_states),
                "target_dimension": 1,
            }
        )
        family_rows.append(row)
        certificates["legacy_reduced_IZ_block_reset"] = diagnostics
        legacy_summary = legacy.to_summary_dict()

    directed_rows = _directed_action_rows(
        rows=directed_rows,
        target_basis=problem.target_basis,
    )
    for row in directed_rows:
        row.update({"case": case.name, "model": type(model).__name__})

    shifted_potential_rows = _shifted_potential_action_rows(
        candidates=shifted_potential_candidates,
        target_basis=problem.target_basis,
    )
    for row in shifted_potential_rows:
        row.update({"case": case.name, "model": type(model).__name__})

    operator_map_rows = _caging_operator_map_rows(
        model=model,
        directed_rows=unique_directed,
        shifted_potential_candidates=shifted_potential_candidates,
        target_basis=problem.target_basis,
        design=design,
    )
    for row in operator_map_rows:
        row.update({"case": case.name, "model": type(model).__name__})

    completion_span_rows = _completion_caging_span_rows(
        completion_operators=design.targeted_jumps,
        directed_generators=a_family,
        generic_l_generators=l_family,
    )
    for row in completion_span_rows:
        row.update({"case": case.name, "model": type(model).__name__})

    provenance_rows = _operator_provenance_rows(
        operator_role="M",
        operators=m_family,
        directed_basis=a_family,
    ) + _operator_provenance_rows(
        operator_role="ML",
        operators=design.recycled_jumps,
        directed_basis=a_family,
    )
    for row in provenance_rows:
        row.update({"case": case.name, "model": type(model).__name__})
    for row in retargeted_prefix_rows:
        row.update({"case": case.name, "model": type(model).__name__})
    for row in retargeted_single_scan_rows:
        row.update({"case": case.name, "model": type(model).__name__})

    rotation_original, rotation_rotated = _basis_rotation_consistency(
        operators=a_family,
        target_basis=problem.target_basis,
    )
    locality_certificate = None
    local_annihilator_scorecard = None
    local_annihilator_arrays: dict[str, np.ndarray] | None = None
    if selected_retargeted is not None:
        selected_caging_row = unique_directed[selected_retargeted.directed_row_index]
        locality_certificate = _direct_jump_locality_certificate(
            model=model,
            build_result=build_result,
            caging_row=selected_caging_row,
            jump=selected_retargeted,
        )
        locality_certificate.update({"case": case.name, "model": type(model).__name__})
        local_annihilator_scorecard, local_annihilator_arrays = (
            _local_annihilator_space_certificate(
                model=model,
                build_result=build_result,
                target_basis=problem.target_basis,
                caging_row=selected_caging_row,
                selected_jump=selected_retargeted,
            )
        )
        local_annihilator_scorecard.update({"case": case.name, "model": type(model).__name__})

    basis_configs = basis_configs_from_build_result(build_result)
    modern_locality_certificates: list[dict[str, object]] = []
    multiplier_readouts = design.workflow.recycled_recycler_readouts(
        basis_configs=basis_configs,
        states=problem.target_basis,
    )
    for operator_index, readout in enumerate(multiplier_readouts):
        certificate = _local_matrix_readout_locality_certificate(
            model=model,
            build_result=build_result,
            readout=readout,
            operator_role="M",
            operator_index=operator_index,
        )
        certificate.update({"case": case.name, "model": type(model).__name__})
        modern_locality_certificates.append(certificate)
    completion_readouts = design.workflow.targeted_operator_readouts(
        basis_configs=basis_configs,
    )
    for operator_index, readout in enumerate(completion_readouts):
        certificate = _local_matrix_readout_locality_certificate(
            model=model,
            build_result=build_result,
            readout=readout,
            operator_role="completion",
            operator_index=operator_index,
        )
        certificate.update({"case": case.name, "model": type(model).__name__})
        modern_locality_certificates.append(certificate)

    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(case_dir / "jump_family_ablation.csv", family_rows)
    _write_csv(case_dir / "directed_caging_action.csv", directed_rows)
    _write_csv(case_dir / "shifted_potential_caging_action.csv", shifted_potential_rows)
    _write_csv(case_dir / "operator_provenance_against_A_span.csv", provenance_rows)
    _write_csv(case_dir / "completion_caging_span_scorecard.csv", completion_span_rows)
    _write_jsonl(case_dir / "caging_to_lindblad_operator_map.jsonl", operator_map_rows)
    _write_csv(case_dir / "retargeted_A_prefix_scan.csv", retargeted_prefix_rows)
    _write_csv(
        case_dir / "retargeted_A_single_selection.csv",
        retargeted_single_scan_rows,
    )
    _write_csv(
        case_dir / "attractivity_scorecard.csv",
        [row for row in family_rows if row["family"] in {"ML", "final"}],
    )
    if locality_certificate is not None:
        _write_json(
            case_dir / "locality_certificate_A_retargeted_single.json",
            locality_certificate,
        )
    _write_jsonl(
        case_dir / "modern_locality_certificates.jsonl",
        modern_locality_certificates,
    )
    if local_annihilator_scorecard is not None:
        _write_json(
            case_dir / "local_annihilator_space_scorecard.json",
            local_annihilator_scorecard,
        )
    if local_annihilator_arrays is not None:
        np.savez_compressed(
            case_dir / "common_local_annihilator_basis.npz",
            **local_annihilator_arrays,
        )
    for family_name, diagnostics in certificates.items():
        _write_json(
            case_dir / f"attractivity_certificate_{family_name}.json",
            diagnostics.to_summary_dict(),
        )
        if diagnostics.invariant_obstruction_dimension > 0:
            np.savez_compressed(
                case_dir / f"invariant_obstruction_{family_name}.npz",
                basis=diagnostics.invariant_obstruction_basis,
            )

    case_summary = {
        "case": case.name,
        "model": type(model).__name__,
        "model_parameters": {
            "lx": model.lx,
            "ly": model.ly,
            "boundary_condition": model.boundary_condition,
            "winding_x": model.winding_x,
            "winding_y": model.winding_y,
            "coup_kin": model.coup_kin,
            "coup_pot": model.coup_pot,
        },
        "hilbert_dimension": int(build_result.basis.n_states),
        "target_dimension": int(problem.target_basis.shape[1]),
        "target_signature": case.signature,
        "target_record_count": case.record_count,
        "n_reconstructed_directed_rows": len(directed_rows),
        "n_unique_directed_rows": len(a_family),
        "n_shifted_potential_structural_candidates": len(shifted_potential_candidates),
        "n_state_dark_shifted_potential_candidates": len(valid_shifted_potential),
        "n_full_target_dark_Y_basis_operators": len(y_family),
        "n_retargeted_A_jumps": len(retargeted),
        "selected_retargeted_A_single": (
            None
            if selected_retargeted is None
            else {
                "directed_row_index": selected_retargeted.directed_row_index,
                "source_state_index": selected_retargeted.source_state_index,
                "zero_index": selected_retargeted.zero_index,
                "support": selected_retargeted.variable_indices,
                "support_size": len(selected_retargeted.variable_indices),
                "A_zero_pattern": unique_directed[
                    selected_retargeted.directed_row_index
                ].target_pattern,
                "A_source_patterns": unique_directed[
                    selected_retargeted.directed_row_index
                ].source_patterns,
                "A_amplitudes": unique_directed[selected_retargeted.directed_row_index].amplitudes,
                "jump_output_pattern": selected_retargeted.output_pattern,
            }
        ),
        "modern_n_ML_jumps": len(design.recycled_jumps),
        "modern_n_completion_jumps": len(design.targeted_jumps),
        "modern_early_stop_reason": design.workflow.early_stop_reason,
        "A_basis_rotation_original_residual": rotation_original,
        "A_basis_rotation_rotated_residual": rotation_rotated,
        "A_basis_rotation_consistent": max(
            rotation_original,
            rotation_rotated,
        )
        <= TOLERANCE,
        "selected_direct_jump_locality_certified": (
            None
            if locality_certificate is None
            else locality_certificate["bounded_support_certified"]
        ),
        "modern_local_matrix_units_all_locality_certified": all(
            bool(item["bounded_support_certified"]) for item in modern_locality_certificates
        ),
        "modern_local_matrix_unit_certificate_count": len(modern_locality_certificates),
        "common_local_annihilator_dimension": (
            None
            if local_annihilator_scorecard is None
            else local_annihilator_scorecard["common_target_annihilator_dimension"]
        ),
        "caging_generated_left_ideal_dimension": (
            None
            if local_annihilator_scorecard is None
            else local_annihilator_scorecard["caging_generated_left_ideal_dimension"]
        ),
        "build_seconds": build_timed.seconds,
        "search_seconds": search_timed.seconds,
        "modern_design_seconds": design_timed.seconds,
        "legacy_summary": legacy_summary,
    }
    _write_json(case_dir / "case_summary.json", case_summary)
    return {
        "case_summary": case_summary,
        "family_rows": family_rows,
        "directed_rows": directed_rows,
        "shifted_potential_rows": shifted_potential_rows,
        "operator_map_rows": operator_map_rows,
        "completion_span_rows": completion_span_rows,
        "provenance_rows": provenance_rows,
        "retargeted_prefix_rows": retargeted_prefix_rows,
        "retargeted_single_scan_rows": retargeted_single_scan_rows,
        "locality_certificate": locality_certificate,
        "local_annihilator_scorecard": local_annihilator_scorecard,
        "modern_locality_certificates": modern_locality_certificates,
    }


def run_jump_bridge_benchmark(
    *,
    output_dir: Path,
    selected_cases: Sequence[str] | None = None,
    include_legacy_single: bool = True,
) -> dict[str, object]:
    """Run the P0 benchmark and write aggregate evidence tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = benchmark_cases()
    if selected_cases:
        selected = set(selected_cases)
        cases = tuple(case for case in cases if case.name in selected)
        unknown = selected - {case.name for case in benchmark_cases()}
        if unknown:
            raise ValueError(f"Unknown case names: {sorted(unknown)}")

    all_family_rows: list[dict[str, object]] = []
    all_directed_rows: list[dict[str, object]] = []
    all_shifted_potential_rows: list[dict[str, object]] = []
    all_operator_map_rows: list[dict[str, object]] = []
    all_completion_span_rows: list[dict[str, object]] = []
    all_provenance_rows: list[dict[str, object]] = []
    all_retargeted_prefix_rows: list[dict[str, object]] = []
    all_retargeted_single_scan_rows: list[dict[str, object]] = []
    locality_certificates: list[dict[str, object]] = []
    modern_locality_certificates: list[dict[str, object]] = []
    local_annihilator_scorecards: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case in cases:
        result = run_jump_bridge_case(
            case,
            output_dir=output_dir,
            include_legacy_single=include_legacy_single,
        )
        all_family_rows.extend(result["family_rows"])
        all_directed_rows.extend(result["directed_rows"])
        all_shifted_potential_rows.extend(result["shifted_potential_rows"])
        all_operator_map_rows.extend(result["operator_map_rows"])
        all_completion_span_rows.extend(result["completion_span_rows"])
        all_provenance_rows.extend(result["provenance_rows"])
        all_retargeted_prefix_rows.extend(result["retargeted_prefix_rows"])
        all_retargeted_single_scan_rows.extend(result["retargeted_single_scan_rows"])
        if result["locality_certificate"] is not None:
            locality_certificates.append(result["locality_certificate"])
        modern_locality_certificates.extend(result["modern_locality_certificates"])
        if result["local_annihilator_scorecard"] is not None:
            local_annihilator_scorecards.append(result["local_annihilator_scorecard"])
        summaries.append(result["case_summary"])

    _write_csv(output_dir / "jump_family_ablation.csv", all_family_rows)
    _write_csv(output_dir / "directed_caging_action.csv", all_directed_rows)
    _write_csv(
        output_dir / "shifted_potential_caging_action.csv",
        all_shifted_potential_rows,
    )
    _write_jsonl(
        output_dir / "caging_to_lindblad_operator_map.jsonl",
        all_operator_map_rows,
    )
    _write_csv(
        output_dir / "completion_caging_span_scorecard.csv",
        all_completion_span_rows,
    )
    _write_csv(
        output_dir / "operator_provenance_against_A_span.csv",
        all_provenance_rows,
    )
    _write_csv(
        output_dir / "retargeted_A_prefix_scan.csv",
        all_retargeted_prefix_rows,
    )
    _write_csv(
        output_dir / "retargeted_A_single_selection.csv",
        all_retargeted_single_scan_rows,
    )
    _write_csv(
        output_dir / "attractivity_scorecard.csv",
        [row for row in all_family_rows if row["family"] in {"ML", "final"}],
    )
    _write_jsonl(
        output_dir / "locality_certificates.jsonl",
        [*locality_certificates, *modern_locality_certificates],
    )
    _write_csv(
        output_dir / "local_dark_jump_space_scorecard.csv",
        local_annihilator_scorecards,
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "benchmark": "P0 ICQMBS directed-caging to Lindblad jump bridge",
            "argv": sys.argv,
            "git": _git_metadata(),
            "tolerance": TOLERANCE,
            "search_seed": SEARCH_SEED,
            "normalization_note": (
                "A/Z/Q and retargeted-A use the local ICQMBS witness normalization; "
                "modern and legacy families retain their construction-native rates."
            ),
            "cases": summaries,
        },
    )
    return {
        "cases": summaries,
        "jump_family_ablation": all_family_rows,
        "directed_caging_action": all_directed_rows,
        "shifted_potential_caging_action": all_shifted_potential_rows,
        "caging_to_lindblad_operator_map": all_operator_map_rows,
        "completion_caging_span_scorecard": all_completion_span_rows,
        "operator_provenance": all_provenance_rows,
        "retargeted_prefix_scan": all_retargeted_prefix_rows,
        "retargeted_single_selection": all_retargeted_single_scan_rows,
        "locality_certificates": locality_certificates,
        "modern_locality_certificates": modern_locality_certificates,
        "local_annihilator_scorecards": local_annihilator_scorecards,
    }
