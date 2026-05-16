import numpy as np

from qlinks.caging import (
    group_vertices_by_signature,
    type1_candidates_from_bipartite_self_loops,
    type2_candidates_from_self_loops,
)


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
