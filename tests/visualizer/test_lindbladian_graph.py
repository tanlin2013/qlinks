import numpy as np
import pytest
from scipy import sparse as scipy_sparse

from qlinks.visualizer.hamiltonian_graph import (
    HamiltonianGraphStyle,
    HamiltonianGraphVisualizer,
)
from qlinks.visualizer.lindbladian_graph import (
    LiouvillianGraphVisualizer,
    flatten_density_matrix,
    operator_space_labels,
    unflatten_operator_index,
)


def test_flatten_density_matrix_column_major() -> None:
    rho = np.asarray(
        [
            [1.0, 2.0 + 1.0j],
            [3.0 - 1.0j, 4.0],
        ],
        dtype=np.complex128,
    )

    flat = flatten_density_matrix(rho, convention="column_major")

    np.testing.assert_allclose(
        flat,
        np.asarray(
            [1.0, 3.0 - 1.0j, 2.0 + 1.0j, 4.0],
            dtype=np.complex128,
        ),
    )


def test_unflatten_operator_index_column_major() -> None:
    assert unflatten_operator_index(
        0,
        hilbert_dim=2,
        convention="column_major",
    ) == (0, 0)
    assert unflatten_operator_index(
        1,
        hilbert_dim=2,
        convention="column_major",
    ) == (1, 0)
    assert unflatten_operator_index(
        2,
        hilbert_dim=2,
        convention="column_major",
    ) == (0, 1)
    assert unflatten_operator_index(
        3,
        hilbert_dim=2,
        convention="column_major",
    ) == (1, 1)


def test_operator_space_labels_column_major() -> None:
    assert operator_space_labels(hilbert_dim=2) == [
        "|0><0|",
        "|1><0|",
        "|0><1|",
        "|1><1|",
    ]


def test_hamiltonian_graph_can_preserve_directed_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0 + 2.0j, 0.0],
                [0.0, 0.0, 3.0],
                [4.0j, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
        matrix,
        directed=True,
    )

    graph = visualizer.to_networkx()

    assert graph.is_directed()
    assert set(graph.edges) == {(0, 1), (1, 2), (2, 0)}

    np.testing.assert_allclose(
        visualizer.edge_weights(),
        np.asarray([1.0 + 2.0j, 3.0, 4.0j], dtype=np.complex128),
    )


def test_liouvillian_graph_uses_directed_graph_and_density_coloring() -> None:
    liouvillian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0j, 0.0],
                [0.0, 0.0, 0.0, -3.0],
                [4.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    rho = np.asarray(
        [
            [1.0, 0.5j],
            [-0.5j, 0.0],
        ],
        dtype=np.complex128,
    )

    visualizer = LiouvillianGraphVisualizer.from_liouvillian(
        liouvillian,
        hilbert_dim=2,
        density_matrix=rho,
    )

    graph = visualizer.to_networkx()

    assert graph.is_directed()
    assert set(graph.edges) == {(0, 1), (1, 2), (2, 3), (3, 0)}

    assert visualizer.graph_data.directed
    assert visualizer.graph_visualizer.vertex_display_labels() == [
        "|0><0|",
        "|1><0|",
        "|0><1|",
        "|1><1|",
    ]

    np.testing.assert_allclose(
        visualizer.graph_data.state_vector,
        flatten_density_matrix(rho, convention="column_major"),
    )

    np.testing.assert_allclose(
        visualizer.graph_visualizer.node_values(
            color_by="state_amplitude_abs",
        ),
        np.asarray([1.0, 0.5, 0.5, 0.0]),
    )


def test_liouvillian_graph_rejects_wrong_dimension() -> None:
    liouvillian = scipy_sparse.identity(5, format="csr", dtype=np.complex128)

    with pytest.raises(ValueError, match="hilbert_dim"):
        LiouvillianGraphVisualizer.from_liouvillian(
            liouvillian,
            hilbert_dim=2,
        )


def test_liouvillian_graph_plot_networkx_smoke() -> None:
    liouvillian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0j, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    rho = np.eye(2, dtype=np.complex128) / 2.0

    visualizer = LiouvillianGraphVisualizer.from_liouvillian(
        liouvillian,
        hilbert_dim=2,
        density_matrix=rho,
        style=HamiltonianGraphStyle(
            label_vertices=True,
            colorbar=False,
            edge_colorbar=False,
        ),
    )

    ax = visualizer.plot(
        backend="networkx",
        layout="circle",
        show=False,
    )

    assert ax is not None


@pytest.mark.manual
def test_manual_liouvillian_graph_visual_effect() -> None:
    import matplotlib.pyplot as plt

    liouvillian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.6j, 0.0],
                [0.0, 0.0, -1.0, 0.3j],
                [0.2, 0.0, 0.0, 1.0],
                [-0.8j, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    rho = np.asarray(
        [
            [0.7, 0.2 + 0.3j],
            [0.2 - 0.3j, 0.3],
        ],
        dtype=np.complex128,
    )

    visualizer = LiouvillianGraphVisualizer.from_liouvillian(
        liouvillian,
        hilbert_dim=2,
        density_matrix=rho,
        style=HamiltonianGraphStyle(
            label_vertices=True,
            vertex_size=30.0,
            edge_width=2.5,
            colorbar=True,
            edge_colorbar=False,
            figure_size=(6.0, 5.0),
        ),
    )

    visualizer.plot(
        backend="networkx",
        layout="circle",
        color_by="state_amplitude_abs",
        edge_color_by="weight_complex",
        title="Liouvillian graph: |rho_ij| nodes, complex L edges",
        show=False,
    )

    plt.show()
