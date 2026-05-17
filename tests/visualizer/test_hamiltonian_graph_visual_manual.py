import os

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.visualizer import HamiltonianGraphVisualizer

pytestmark = pytest.mark.skipif(
    os.environ.get("QLINKS_SHOW_PLOTS") != "1",
    reason="Manual visual tests. Run with QLINKS_SHOW_PLOTS=1.",
)


def test_plot_hamiltonian_graph_bipartition_igraph() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)
    visualizer.plot(
        backend="igraph",
        color_by="bipartition",
        title="Bipartition coloring",
    )


def test_plot_hamiltonian_graph_state_weight_networkx() -> None:
    matrix = scipy_sparse.csr_array(
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.complex128,
        )
    )
    state_vector = np.array([1, 0, -1, 0], dtype=np.complex128) / np.sqrt(2)

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)
    visualizer.plot(
        backend="networkx",
        color_by="state_weight",
        state_vector=state_vector,
        title="State-weight coloring",
    )
