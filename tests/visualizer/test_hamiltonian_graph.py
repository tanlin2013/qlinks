import importlib.util

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.visualizer import (
    HamiltonianGraphVisualizer,
    bipartition_labels,
)

igraph_available = importlib.util.find_spec("igraph") is not None


def test_bipartition_labels_for_path_graph() -> None:
    adjacency = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )
    )

    labels = bipartition_labels(adjacency)

    assert labels.tolist() == [0, 1, 0, 1]


def test_bipartition_labels_rejects_triangle() -> None:
    adjacency = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.complex128,
        )
    )

    with pytest.raises(ValueError, match="not bipartite"):
        bipartition_labels(adjacency)


def test_hamiltonian_graph_visualizer_extracts_self_loops_and_degrees() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [2, 1, 0],
                [1, 3, 1],
                [0, 1, 4],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    np.testing.assert_allclose(
        visualizer.graph_data.self_loop_values,
        np.array([2, 3, 4], dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        visualizer.graph_data.degrees,
        np.array([1, 2, 1], dtype=np.int64),
    )


def test_node_values_by_self_loop_degree_and_state_weight() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [2, 1, 0],
                [1, 3, 1],
                [0, 1, 4],
            ],
            dtype=np.complex128,
        )
    )
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    np.testing.assert_allclose(
        visualizer.node_values(color_by="self_loop"),
        np.array([2, 3, 4], dtype=np.float64),
    )
    np.testing.assert_allclose(
        visualizer.node_values(color_by="degree"),
        np.array([1, 2, 1], dtype=np.float64),
    )
    np.testing.assert_allclose(
        visualizer.node_values(
            color_by="state_weight",
            state_vector=np.array([1, 1j, 0], dtype=np.complex128),
        ),
        np.array([1, 1, 0], dtype=np.float64),
    )


def test_to_networkx_has_expected_nodes_and_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.complex128,
        )
    )
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    graph = visualizer.to_networkx()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_to_networkx_uses_real_layout_weight_for_complex_matrix() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0.0, 1.0 + 2.0j],
                [1.0 - 2.0j, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)
    graph = visualizer.to_networkx()

    edge_data = graph.edges[0, 1]

    assert isinstance(edge_data["weight"], float)
    assert edge_data["weight"] == pytest.approx(1.0)
    assert isinstance(edge_data["hamiltonian_weight"], complex)


@pytest.mark.skipif(not igraph_available, reason="python-igraph is not installed.")
def test_to_igraph_has_expected_nodes_and_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.complex128,
        )
    )
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    graph = visualizer.to_igraph()

    assert graph.vcount() == 3
    assert graph.ecount() == 2
