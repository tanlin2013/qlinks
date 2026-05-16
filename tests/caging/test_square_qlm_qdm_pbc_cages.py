"""Manual integration tests for square-lattice QDM/QLM caged states.

Run manually with:

    pytest tests/caging/manual/test_square_qdm_qlm_cages.py -m manual

These tests are intentionally not part of the normal CI path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging import (
    CageSolverConfig,
    CandidateFilterContext,
    CandidateSubgraph,
    CombinedBoundaryKineticTargetNullityFilter,
    extract_subblocks,
    run_candidate_filters,
    solve_candidate,
    type1_candidates_from_bipartite_self_loops,
    type2_candidates_from_self_loops,
)
from qlinks.models import (
    SquareQDMModel,
    SquareQLMModel,
)

pytestmark = pytest.mark.manual

ModelName = Literal["qdm", "qlm"]


@dataclass(frozen=True)
class SquareCageCase:
    """Expected caging data for one square-lattice model."""

    model_name: ModelName
    lattice_shape: tuple[int, int]
    expected_counts_by_signature: dict[tuple[int, int], int]
    max_support_size: int
    winding_x: int = 0
    winding_y: int = 0


# Fill these with your known small-size counts.
#
# I leave them as explicit test data rather than deriving them inside the test,
# because this is an integration/regression test.
SQUARE_CAGE_CASES = [
    SquareCageCase(
        model_name="qdm",
        lattice_shape=(4, 4),
        winding_x=0,
        winding_y=0,
        expected_counts_by_signature={
            (0, 4): 9,
            (0, 6): 1,
        },
        max_support_size=48,
    ),
    SquareCageCase(
        model_name="qlm",
        lattice_shape=(4, 4),
        winding_x=0,
        winding_y=0,
        expected_counts_by_signature={
            (0, 8): 26,
            (0, 6): 12,
            (2, 8): 6,
            (-2, 8): 6,
        },
        max_support_size=224,
    ),
]


def _build_square_model_result(square_case: SquareCageCase):
    """Build a square QDM/QLM model result."""
    lattice_size_x, lattice_size_y = square_case.lattice_shape

    if square_case.model_name == "qdm":
        model = SquareQDMModel(
            lx=lattice_size_x,
            ly=lattice_size_y,
            boundary_condition="periodic",
            winding_x=square_case.winding_x,
            winding_y=square_case.winding_y,
            winding_convention="electric",
            kinetic=1.0,
            potential=1.0,
        )
        builder_name = "sparse"
    elif square_case.model_name == "qlm":
        model = SquareQLMModel(
            lx=lattice_size_x,
            ly=lattice_size_y,
            boundary_condition="periodic",
            winding_x=square_case.winding_x,
            winding_y=square_case.winding_y,
            charges=0,
            kinetic=1.0,
            potential=1.0,
        )
        builder_name = "bitmask"
    else:
        raise ValueError(f"Unsupported model_name: {square_case.model_name}")

    return model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
    )


def _is_zero_matrix(matrix, *, tolerance: float = 1e-12) -> bool:
    """Return whether a dense or sparse matrix is numerically zero."""
    if matrix is None:
        return True

    if scipy_sparse.issparse(matrix):
        if matrix.nnz == 0:
            return True

        return bool(np.all(np.abs(matrix.data) <= tolerance))

    return bool(np.allclose(matrix, 0.0, atol=tolerance, rtol=0.0))


def _matrix_nnz(matrix) -> int:
    """Return number of nonzero entries."""
    if matrix is None:
        return 0

    if scipy_sparse.issparse(matrix):
        return int(matrix.nnz)

    return int(np.count_nonzero(matrix))


def _debug_build_result(build_result) -> None:
    """Print build-result fields relevant to caging."""
    print()
    print("build_result type =", type(build_result))

    for field_name in ("hamiltonian", "kinetic", "potential", "basis"):
        if not hasattr(build_result, field_name):
            print(f"{field_name}: missing")
            continue

        field_value = getattr(build_result, field_name)

        if hasattr(field_value, "shape"):
            print(f"{field_name}: shape={field_value.shape}, " f"nnz={_matrix_nnz(field_value)}")
        else:
            print(f"{field_name}: type={type(field_value)}")

    for field_name in dir(build_result):
        if field_name.startswith("_"):
            continue

        field_value = getattr(build_result, field_name)

        if hasattr(field_value, "shape"):
            print(
                f"extra field {field_name}: shape={field_value.shape}, "
                f"nnz={_matrix_nnz(field_value)}"
            )


def _debug_square_cage_pipeline(
    *,
    square_case: SquareCageCase,
    hamiltonian_matrix,
    kinetic_matrix,
    potential_matrix,
    self_loop_values: np.ndarray,
    bipartition_labels: np.ndarray,
    type1_candidate_subgraphs: list[CandidateSubgraph],
    type2_candidate_subgraphs: list[CandidateSubgraph],
    tolerance: float,
) -> None:
    """Print useful diagnostics for the square QDM/QLM caging pipeline."""
    print()
    print(f"model_name = {square_case.model_name}")
    print(f"lattice_shape = {square_case.lattice_shape}")
    print(f"H.shape = {hamiltonian_matrix.shape}, H.nnz = {_matrix_nnz(hamiltonian_matrix)}")
    print(f"K.shape = {kinetic_matrix.shape}, K.nnz = {_matrix_nnz(kinetic_matrix)}")
    print(f"V.shape = {potential_matrix.shape}, V.nnz = {_matrix_nnz(potential_matrix)}")

    unique_self_loop_values, self_loop_counts = np.unique(
        np.round(self_loop_values.real, decimals=12),
        return_counts=True,
    )
    print(
        "self_loop histogram =",
        dict(zip(unique_self_loop_values.tolist(), self_loop_counts.tolist())),
    )

    unique_bipartition_labels, bipartition_counts = np.unique(
        bipartition_labels,
        return_counts=True,
    )
    print(
        "bipartition histogram =",
        dict(zip(unique_bipartition_labels.tolist(), bipartition_counts.tolist())),
    )

    print(f"type1 candidate count = {len(type1_candidate_subgraphs)}")
    print(f"type2 candidate count = {len(type2_candidate_subgraphs)}")

    type1_signature_counter = Counter(
        candidate_subgraph.metadata.get("signature")
        for candidate_subgraph in type1_candidate_subgraphs
    )
    type2_signature_counter = Counter(
        candidate_subgraph.metadata.get("signature")
        for candidate_subgraph in type2_candidate_subgraphs
    )

    print("type1 candidate signatures =", dict(type1_signature_counter))
    print("type2 candidate signatures =", dict(type2_signature_counter))

    print(
        "first type1 supports =",
        [
            candidate_subgraph.vertices.tolist()
            for candidate_subgraph in type1_candidate_subgraphs[:10]
        ],
    )

    filter_context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=self_loop_values,
    )
    kappa_filter = CombinedBoundaryKineticTargetNullityFilter(
        target_kappas=(0.0,),
        tolerance=tolerance,
        require_nonzero_kappa=False,
    )

    accepted_type1_candidates = []
    rejected_reasons = Counter()

    for candidate_subgraph in type1_candidate_subgraphs:
        filter_result = run_candidate_filters(
            filter_context,
            candidate_subgraph,
            [kappa_filter],
        )

        if filter_result.accepted:
            accepted_type1_candidates.append(candidate_subgraph)
        else:
            rejected_reasons[filter_result.reason] += 1

    print(f"type1 candidates accepted by kappa=0 filter = {len(accepted_type1_candidates)}")
    print("type1 rejection reasons =", dict(rejected_reasons))

    if accepted_type1_candidates:
        sample_candidate = accepted_type1_candidates[0]
        local_kinetic_matrix, boundary_matrix, _outside_indices = extract_subblocks(
            kinetic_matrix,
            sample_candidate.vertices,
        )
        print("sample accepted type1 support =", sample_candidate.vertices.tolist())
        print("sample K[S,S].nnz =", _matrix_nnz(local_kinetic_matrix))
        print("sample K[out,S].shape =", boundary_matrix.shape)
        print("sample K[out,S].nnz =", _matrix_nnz(boundary_matrix))
        print(
            "sample self_loop values =",
            self_loop_values[sample_candidate.vertices].real.tolist(),
        )


def _count_cages_by_signature_with_source(
    hamiltonian_matrix,
    kinetic_matrix,
    self_loop_values: np.ndarray,
    candidate_subgraphs: list[CandidateSubgraph],
    *,
    tolerance: float,
    allowed_kappas: tuple[int, ...],
) -> Counter[tuple[str, tuple[int, int]]]:
    """Count cages grouped by candidate source and ``(kappa, Z)``."""
    hilbert_size = hamiltonian_matrix.shape[0]
    solver_config = CageSolverConfig(
        tolerance=tolerance,
        validate_full_residual=True,
    )
    filter_context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=self_loop_values,
    )
    candidate_filters = [
        CombinedBoundaryKineticTargetNullityFilter(
            target_kappas=tuple(float(kappa_value) for kappa_value in allowed_kappas),
            tolerance=tolerance,
            require_nonzero_kappa=False,
        ),
    ]

    unique_states_by_source_signature: dict[
        tuple[str, tuple[int, int]],
        list[np.ndarray],
    ] = {}

    for candidate_subgraph in candidate_subgraphs:
        candidate_source = str(candidate_subgraph.metadata.get("candidate_type", "unknown"))

        filter_result = run_candidate_filters(
            filter_context,
            candidate_subgraph,
            candidate_filters,
        )

        if not filter_result.accepted:
            continue

        cage_states = solve_candidate(
            hamiltonian_matrix,
            candidate_subgraph,
            config=solver_config,
        )

        for cage_state in cage_states:
            self_loop_value = self_loop_values[candidate_subgraph.vertices[0]]
            signature = _signature_from_energy_and_self_loop(
                cage_state.energy,
                self_loop_value,
                tolerance=10 * tolerance,
            )

            if signature is None:
                continue

            kinetic_value, _potential_value = signature

            if kinetic_value not in allowed_kappas:
                continue

            full_state = _normalized_full_state(
                cage_state.local_state,
                cage_state.support,
                hilbert_size,
            )

            source_signature = (candidate_source, signature)
            states_for_source_signature = unique_states_by_source_signature.setdefault(
                source_signature,
                [],
            )

            _append_independent_state(
                states_for_source_signature,
                full_state,
                tolerance=100 * tolerance,
            )

    return Counter(
        {
            source_signature: len(unique_states)
            for source_signature, unique_states in unique_states_by_source_signature.items()
        }
    )


def _get_matrix_from_result(build_result, candidate_names: tuple[str, ...]):
    """Read one matrix-like field from a model build result."""
    for candidate_name in candidate_names:
        if hasattr(build_result, candidate_name):
            matrix = getattr(build_result, candidate_name)

            if matrix is not None:
                return matrix

    available_names = sorted(name for name in dir(build_result) if not name.startswith("_"))
    raise AttributeError(
        "Could not find any matrix field from "
        f"{candidate_names}. Available names: {available_names}"
    )


def _extract_hamiltonian_kinetic_potential(build_result):
    """Extract H, K, and V from a model build result."""
    hamiltonian_matrix = build_result.hamiltonian
    kinetic_matrix = build_result.kinetic
    potential_matrix = build_result.potential

    if _is_zero_matrix(kinetic_matrix) and not _is_zero_matrix(hamiltonian_matrix):
        raise AssertionError(
            "build_result.kinetic is zero while build_result.hamiltonian is nonzero. "
            "This means the test is not using the kinetic graph. Check whether the "
            "model was built with kinetic=0, or whether ModelBuildResult assigned "
            "kinetic/potential terms incorrectly."
        )

    if potential_matrix is None:
        raise AssertionError("build_result.potential is None.")

    return hamiltonian_matrix, kinetic_matrix, potential_matrix


def _diagonal_values(matrix) -> np.ndarray:
    """Return the diagonal values of a dense or sparse matrix."""
    if scipy_sparse.issparse(matrix):
        diagonal_values = matrix.diagonal()
    else:
        diagonal_values = np.diag(matrix)

    return np.asarray(diagonal_values, dtype=np.complex128)


def _kinetic_adjacency(kinetic_matrix) -> scipy_sparse.csr_matrix:
    """Return the unweighted adjacency graph of the kinetic matrix."""
    sparse_kinetic_matrix = scipy_sparse.csr_matrix(kinetic_matrix)
    adjacency_matrix = sparse_kinetic_matrix.copy()
    adjacency_matrix.setdiag(0)
    adjacency_matrix.eliminate_zeros()
    adjacency_matrix.data = np.ones_like(adjacency_matrix.data, dtype=np.int8)

    return adjacency_matrix.tocsr()


def _bipartition_labels(kinetic_matrix) -> np.ndarray:
    """Compute bipartition labels of the kinetic graph.

    Raises an assertion error if the graph is not bipartite.
    """
    adjacency_matrix = _kinetic_adjacency(kinetic_matrix)
    hilbert_size = adjacency_matrix.shape[0]
    labels = -np.ones(hilbert_size, dtype=np.int64)

    for start_index in range(hilbert_size):
        if labels[start_index] != -1:
            continue

        labels[start_index] = 0
        search_queue = [start_index]

        while search_queue:
            vertex_index = search_queue.pop(0)
            neighbor_indices = adjacency_matrix.indices[
                adjacency_matrix.indptr[vertex_index] : adjacency_matrix.indptr[vertex_index + 1]
            ]

            for neighbor_index in neighbor_indices:
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = 1 - labels[vertex_index]
                    search_queue.append(int(neighbor_index))
                else:
                    assert labels[neighbor_index] != labels[vertex_index]

    return labels


def _count_cages_by_signature(
    hamiltonian_matrix,
    kinetic_matrix,
    self_loop_values: np.ndarray,
    candidate_subgraphs: list[CandidateSubgraph],
    *,
    tolerance: float,
    allowed_kappas: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    """Count unique cage states grouped by ``(kappa, Z)``."""
    hilbert_size = hamiltonian_matrix.shape[0]
    solver_config = CageSolverConfig(
        tolerance=tolerance,
        validate_full_residual=True,
    )
    filter_context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=self_loop_values,
    )
    candidate_filters = [
        CombinedBoundaryKineticTargetNullityFilter(
            target_kappas=tuple(float(kappa_value) for kappa_value in allowed_kappas),
            tolerance=tolerance,
            require_nonzero_kappa=False,
        ),
    ]

    unique_states_by_signature: dict[tuple[int, int], list[np.ndarray]] = {}

    for candidate_subgraph in candidate_subgraphs:
        filter_result = run_candidate_filters(
            filter_context,
            candidate_subgraph,
            candidate_filters,
        )

        if not filter_result.accepted:
            continue

        cage_states = solve_candidate(
            hamiltonian_matrix,
            candidate_subgraph,
            config=solver_config,
        )

        for cage_state in cage_states:
            self_loop_value = self_loop_values[candidate_subgraph.vertices[0]]
            signature = _signature_from_energy_and_self_loop(
                cage_state.energy,
                self_loop_value,
                tolerance=10 * tolerance,
            )

            if signature is None:
                continue

            kinetic_value, _potential_value = signature

            if kinetic_value not in allowed_kappas:
                continue

            full_state = _normalized_full_state(
                cage_state.local_state,
                cage_state.support,
                hilbert_size,
            )

            states_for_signature = unique_states_by_signature.setdefault(
                signature,
                [],
            )
            _append_independent_state(
                states_for_signature,
                full_state,
                tolerance=100 * tolerance,
            )

    return {
        signature: len(unique_states)
        for signature, unique_states in unique_states_by_signature.items()
    }


def _normalized_full_state(
    local_state: np.ndarray,
    support_indices: np.ndarray,
    hilbert_size: int,
) -> np.ndarray:
    """Lift a local state to the full Hilbert space and normalize it."""
    full_state = np.zeros(hilbert_size, dtype=np.complex128)
    full_state[support_indices] = local_state

    state_norm = np.linalg.norm(full_state)
    assert state_norm > 0.0

    return full_state / state_norm


def _append_independent_state(
    independent_states: list[np.ndarray],
    candidate_state: np.ndarray,
    *,
    tolerance: float,
) -> None:
    """Append a state only if it increases the span dimension."""
    normalized_candidate = candidate_state / np.linalg.norm(candidate_state)

    if len(independent_states) == 0:
        independent_states.append(normalized_candidate)
        return

    existing_matrix = np.column_stack(independent_states)
    old_rank = np.linalg.matrix_rank(existing_matrix, tol=tolerance)

    extended_matrix = np.column_stack(
        [
            existing_matrix,
            normalized_candidate,
        ]
    )
    new_rank = np.linalg.matrix_rank(extended_matrix, tol=tolerance)

    if new_rank > old_rank:
        independent_states.append(normalized_candidate)


def _signature_from_energy_and_self_loop(
    energy_value: complex,
    self_loop_value: complex,
    *,
    tolerance: float,
) -> tuple[int, int] | None:
    """Infer the integer signature ``(kappa, Z)`` from energy and self-loop."""
    potential_value = int(round(self_loop_value.real))
    kinetic_value = energy_value.real - self_loop_value.real
    kinetic_integer = int(round(kinetic_value))

    if not np.isclose(
        self_loop_value.real,
        potential_value,
        atol=tolerance,
        rtol=0.0,
    ):
        return None

    if not np.isclose(
        kinetic_value,
        kinetic_integer,
        atol=tolerance,
        rtol=0.0,
    ):
        return None

    return kinetic_integer, potential_value


def _merge_count_dicts(
    *count_dicts: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    merged_counts: dict[tuple[int, int], int] = {}

    for count_dict in count_dicts:
        for signature, count in count_dict.items():
            merged_counts[signature] = merged_counts.get(signature, 0) + count

    return merged_counts


@pytest.mark.parametrize("square_case", SQUARE_CAGE_CASES)
def test_square_qdm_qlm_cage_counts_by_signature(
    square_case: SquareCageCase,
) -> None:
    """Check known cage counts grouped by ``(kappa, Z)``."""
    tolerance = 1e-10

    build_result = _build_square_model_result(square_case)

    hamiltonian_matrix, kinetic_matrix, potential_matrix = _extract_hamiltonian_kinetic_potential(
        build_result
    )

    self_loop_values = _diagonal_values(potential_matrix)
    bipartition_labels = _bipartition_labels(kinetic_matrix)

    type1_candidate_subgraphs = type1_candidates_from_bipartite_self_loops(
        kinetic_matrix,
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )

    type2_candidate_subgraphs = type2_candidates_from_self_loops(
        kinetic_matrix,
        self_loop_values,
        min_component_size=2,
    )

    if square_case.model_name == "qdm":
        observed_counts_by_signature = _count_cages_by_signature(
            hamiltonian_matrix,
            kinetic_matrix,
            self_loop_values,
            type1_candidate_subgraphs,
            tolerance=tolerance,
            allowed_kappas=(0,),
        )
    else:
        type1_counts_by_signature = _count_cages_by_signature(
            hamiltonian_matrix,
            kinetic_matrix,
            self_loop_values,
            type1_candidate_subgraphs,
            tolerance=tolerance,
            allowed_kappas=(0,),
        )

        type2_counts_by_signature = _count_cages_by_signature(
            hamiltonian_matrix,
            kinetic_matrix,
            self_loop_values,
            type2_candidate_subgraphs,
            tolerance=tolerance,
            allowed_kappas=(2, -2),
        )

        observed_counts_by_signature = _merge_count_dicts(
            type1_counts_by_signature,
            type2_counts_by_signature,
        )

    assert observed_counts_by_signature == square_case.expected_counts_by_signature
