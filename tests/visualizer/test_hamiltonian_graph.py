import importlib.util

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.visualizer import (
    HamiltonianGraphVisualizer,
    bipartition_labels,
)
from qlinks.visualizer.hamiltonian_graph import _normalize_graph_backend

igraph_available = importlib.util.find_spec("igraph") is not None


def _small_hamiltonian():
    return scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )


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


def test_normalize_graph_backend_accepts_legacy_igraph_alias() -> None:
    assert _normalize_graph_backend("igraph") == "igraph-mpl"
    assert _normalize_graph_backend("igraph-mpl") == "igraph-mpl"
    assert _normalize_graph_backend("igraph-cairo") == "igraph-cairo"
    assert _normalize_graph_backend("networkx") == "networkx"


def test_normalize_graph_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="igraph-cairo"):
        _normalize_graph_backend("bad-backend")  # type: ignore[arg-type]


def test_legacy_igraph_backend_alias_plots_with_matplotlib() -> None:
    pytest.importorskip("igraph")

    import matplotlib.pyplot as plt

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    fig, ax = plt.subplots()

    returned_ax = visualizer.plot(
        backend="igraph",
        ax=ax,
        show=False,
        color_by="degree",
    )

    assert returned_ax is ax

    plt.close(fig)


def test_igraph_mpl_backend_plots_with_matplotlib_axis() -> None:
    pytest.importorskip("igraph")

    import matplotlib.pyplot as plt

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    fig, ax = plt.subplots()

    returned_ax = visualizer.plot(
        backend="igraph-mpl",
        ax=ax,
        show=False,
        color_by="degree",
    )

    assert returned_ax is ax

    plt.close(fig)


def test_igraph_cairo_backend_rejects_matplotlib_axes() -> None:
    import matplotlib.pyplot as plt

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="matplotlib Axes"):
        visualizer.plot(
            backend="igraph-cairo",
            ax=ax,
            show=False,
        )

    plt.close(fig)


def test_igraph_cairo_backend_uses_cairo_dependency_check(monkeypatch) -> None:
    pytest.importorskip("igraph")

    import qlinks.visualizer.hamiltonian_graph as module

    def fail_cairo_check() -> None:
        raise ImportError("missing cairo test")

    monkeypatch.setattr(module, "_ensure_cairo_available", fail_cairo_check)

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    with pytest.raises(ImportError, match="missing cairo test"):
        visualizer.plot(
            backend="igraph-cairo",
            show=False,
        )


def test_igraph_cairo_backend_saves_svg_when_available(tmp_path) -> None:
    pytest.importorskip("igraph")

    has_cairo = False
    try:
        __import__("cairo")
        has_cairo = True
    except ImportError:
        try:
            __import__("cairocffi")
            has_cairo = True
        except ImportError:
            pass

    if not has_cairo:
        pytest.skip("No Cairo Python binding installed.")

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph.svg"

    returned = visualizer.plot(
        backend="igraph-cairo",
        target=out,
        show=False,
        color_by="degree",
    )

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_networkx_backend_accepts_layout_kwargs() -> None:
    pytest.importorskip("networkx")

    import matplotlib.pyplot as plt

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    fig, ax = plt.subplots()

    returned_ax = visualizer.plot(
        backend="networkx",
        layout="spring",
        ax=ax,
        show=False,
        seed=123,
        iterations=5,
    )

    assert returned_ax is ax

    plt.close(fig)
