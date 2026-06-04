import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.visualizer import HamiltonianGraphStyle, HamiltonianGraphVisualizer

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


@pytest.mark.parametrize(
    "edge_color_by",
    [
        "constant",
        "weight_abs",
        "weight_real",
        "weight_imag",
        "weight_phase",
        "weight_complex",
    ],
)
def test_manual_hamiltonian_graph_edge_coloring(edge_color_by: str) -> None:
    """Manual visual smoke test for scalar and complex edge coloring."""
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0 + 0.0j, 0.7j, 0.0, 0.0, 0.0],
                [1.0 - 0.0j, 0.0, -0.8 + 0.2j, 0.4 * np.exp(1.0j), 0.0, 0.0],
                [-0.7j, -0.8 - 0.2j, 0.0, 1.2 * np.exp(2.0j), 0.0, 0.0],
                [0.0, 0.4 * np.exp(-1.0j), 1.2 * np.exp(-2.0j), 0.0, -0.6j, 0.9],
                [0.0, 0.0, 0.0, 0.6j, 0.0, -1.1 * np.exp(0.7j)],
                [0.0, 0.0, 0.0, 0.9, -1.1 * np.exp(-0.7j), 0.0],
            ],
            dtype=np.complex128,
        )
    )

    state = np.asarray(
        [0.0, 1.0, -0.5j, 0.3 + 0.2j, 0.0, -0.4],
        dtype=np.complex128,
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
        matrix,
        style=HamiltonianGraphStyle(
            label_vertices=True,
            vertex_size=26.0,
            edge_width=2.8,
            edge_alpha=0.85,
            colorbar=True,
            edge_colorbar=True,
            figure_size=(6.0, 5.5),
        ),
    )

    fig, ax = plt.subplots(figsize=visualizer.style.figure_size)

    visualizer.plot(
        backend="networkx",
        ax=ax,
        show=False,
        layout="spring",
        color_by="state_amplitude_real",
        state_vector=state,
        edge_color_by=edge_color_by,
        title=f"edge_color_by={edge_color_by}",
    )

    plt.show()

    assert fig is not None
