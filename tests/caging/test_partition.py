import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import (
    group_vertices_by_signature,
    type1_candidates_from_bipartite_self_loops,
    type2_candidates_from_self_loops,
)
from qlinks.caging.candidate import (
    BOUNDARY_OVERLAP_MATRIX_METADATA_KEY,
    INTERNAL_KINETIC_MATRIX_METADATA_KEY,
)
from qlinks.caging.partition import SparseKineticNeighborCache


def test_group_vertices_by_signature_uses_bipartition_and_self_loop() -> None:
    self_loop_values = np.array([4.0, 4.0, 6.0, 4.0], dtype=np.complex128)
    bipartition_labels = np.array([0, 0, 0, 1], dtype=np.int64)

    signature_groups = group_vertices_by_signature(
        self_loop_values=self_loop_values,
        bipartition_labels=bipartition_labels,
        include_bipartition=True,
    )

    sorted_group_sizes = sorted(vertex_indices.size for vertex_indices in signature_groups.values())

    assert sorted_group_sizes == [1, 1, 2]


def test_type1_candidates_use_boundary_overlap_components() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([4.0, 4.0, 0.0, 4.0], dtype=np.complex128)
    bipartition_labels = np.array([0, 0, 1, 0], dtype=np.int64)

    candidate_subgraphs = type1_candidates_from_bipartite_self_loops(
        kinetic_matrix,
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )

    assert len(candidate_subgraphs) == 1
    np.testing.assert_array_equal(candidate_subgraphs[0].vertices, np.array([0, 1]))


def test_type2_candidates_use_internal_kinetic_components() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.complex128)

    candidate_subgraphs = type2_candidates_from_self_loops(
        kinetic_matrix,
        self_loop_values,
        min_component_size=2,
    )

    observed_supports = sorted(
        tuple(candidate_subgraph.vertices.tolist()) for candidate_subgraph in candidate_subgraphs
    )

    assert observed_supports == [(0, 1), (2, 3)]


def test_type1_candidates_accept_sparse_kinetic_matrix() -> None:
    kinetic_matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    self_loop_values = np.array([4.0, 4.0, 0.0, 4.0], dtype=np.complex128)
    bipartition_labels = np.array([0, 0, 1, 0], dtype=np.int64)

    candidate_subgraphs = type1_candidates_from_bipartite_self_loops(
        kinetic_matrix,
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )

    assert len(candidate_subgraphs) == 1
    np.testing.assert_array_equal(candidate_subgraphs[0].vertices, np.array([0, 1]))


def test_type2_candidates_accept_sparse_kinetic_matrix() -> None:
    kinetic_matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    self_loop_values = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.complex128)

    candidate_subgraphs = type2_candidates_from_self_loops(
        kinetic_matrix,
        self_loop_values,
        min_component_size=2,
    )

    observed_supports = sorted(
        tuple(candidate_subgraph.vertices.tolist()) for candidate_subgraph in candidate_subgraphs
    )

    assert observed_supports == [(0, 1), (2, 3)]


def test_type1_candidates_cache_fixed_kappa_solver_blocks() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([4.0, 4.0, 0.0, 4.0], dtype=np.complex128)
    bipartition_labels = np.array([0, 0, 1, 0], dtype=np.int64)

    candidate = type1_candidates_from_bipartite_self_loops(
        kinetic_matrix,
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )[0]

    internal_matrix = candidate.metadata[INTERNAL_KINETIC_MATRIX_METADATA_KEY]
    boundary_overlap_matrix = candidate.metadata[BOUNDARY_OVERLAP_MATRIX_METADATA_KEY]

    np.testing.assert_allclose(
        internal_matrix,
        np.zeros((2, 2), dtype=np.complex128),
    )
    np.testing.assert_allclose(
        boundary_overlap_matrix,
        np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.complex128,
        ),
    )


def test_type2_candidates_cache_internal_kinetic_block() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.complex128)

    candidates = type2_candidates_from_self_loops(
        kinetic_matrix,
        self_loop_values,
        min_component_size=2,
    )
    first_candidate = candidates[0]

    internal_matrix = first_candidate.metadata[INTERNAL_KINETIC_MATRIX_METADATA_KEY]

    np.testing.assert_allclose(
        internal_matrix,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        ),
    )


def test_sparse_kinetic_neighbor_cache_matches_boundary_overlap_adjacency() -> None:
    kinetic_matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    vertices = np.array([0, 1, 3, 4], dtype=np.int64)

    cache = SparseKineticNeighborCache.from_matrix(kinetic_matrix)
    adjacency = cache.boundary_overlap_adjacency(vertices).toarray()

    column_block = kinetic_matrix[:, vertices]
    expected = (column_block.conj().T @ column_block).toarray()
    np.fill_diagonal(expected, 0.0)
    expected = (np.abs(expected) > 0).astype(np.int8)

    np.testing.assert_array_equal(adjacency, expected)


def test_type1_sparse_neighbor_cache_matches_dense_candidates() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([4.0, 4.0, 0.0, 4.0, 4.0], dtype=np.complex128)
    bipartition_labels = np.array([0, 0, 1, 1, 0], dtype=np.int64)

    dense_candidates = type1_candidates_from_bipartite_self_loops(
        kinetic_matrix,
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )
    sparse_candidates = type1_candidates_from_bipartite_self_loops(
        scipy_sparse.csr_array(kinetic_matrix),
        self_loop_values,
        bipartition_labels,
        min_component_size=2,
    )

    dense_supports = sorted(tuple(candidate.vertices.tolist()) for candidate in dense_candidates)
    sparse_supports = sorted(tuple(candidate.vertices.tolist()) for candidate in sparse_candidates)

    assert sparse_supports == dense_supports

    dense_overlaps = [
        np.asarray(candidate.metadata[BOUNDARY_OVERLAP_MATRIX_METADATA_KEY])
        for candidate in dense_candidates
    ]
    sparse_overlaps = [
        candidate.metadata[BOUNDARY_OVERLAP_MATRIX_METADATA_KEY].toarray()
        for candidate in sparse_candidates
    ]

    for dense_overlap, sparse_overlap in zip(dense_overlaps, sparse_overlaps, strict=True):
        np.testing.assert_allclose(sparse_overlap, dense_overlap)
