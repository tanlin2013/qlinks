import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.visualizer import (
    HamiltonianGraphStyle,
    HamiltonianGraphVisualizer,
    bipartition_labels,
)
from qlinks.visualizer.hamiltonian_graph import (
    _normalize_graph_backend,
    _orbit_labels_to_hex_colors,
    _orbit_labels_to_rgba_colors,
    _scalar_values_to_colors,
    _scalar_values_to_hex_colors,
)

igraph_available = importlib.util.find_spec("igraph") is not None


def pynauty_is_safe() -> bool:
    code = """
import pynauty
g = pynauty.Graph(3)
g.connect_vertex(0, [1])
g.connect_vertex(1, [0, 2])
g.connect_vertex(2, [1])
pynauty.autgrp(g)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


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


def test_undirected_symmetrization_uses_lower_only_edges_without_warning() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0 + 2.0j, 0.0, 0.0],
                [0.0, -3.0j, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    sparse_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, scipy_sparse.SparseEfficiencyWarning)
    ]

    assert sparse_warnings == []
    assert visualizer.edge_pairs() == [(0, 1), (1, 2)]
    np.testing.assert_allclose(
        visualizer.edge_weights(),
        np.asarray([1.0 - 2.0j, 3.0j], dtype=np.complex128),
    )


def test_directed_hamiltonian_graph_keeps_oriented_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_directed_sparse_matrix(matrix)

    assert visualizer.graph_data.directed is True
    assert visualizer.edge_pairs() == [(0, 1), (1, 0), (1, 2)]
    np.testing.assert_allclose(
        visualizer.edge_weights(),
        np.asarray([2.0, 3.0, 4.0], dtype=np.complex128),
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
    assert edge_data["hamiltonian_weight_real"] == pytest.approx(1.0)
    assert edge_data["hamiltonian_weight_imag"] == pytest.approx(2.0)
    assert edge_data["hamiltonian_weight_abs"] == pytest.approx(np.sqrt(5.0))
    assert edge_data["hamiltonian_weight_phase"] == pytest.approx(np.angle(1.0 + 2.0j))


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


def test_save_plot_with_networkx_backend(tmp_path) -> None:
    pytest.importorskip("networkx")

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph_networkx.png"

    returned = visualizer.save_plot(
        out,
        backend="networkx",
        layout="spring",
        color_by="degree",
        seed=123,
        iterations=5,
    )

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_plot_with_igraph_mpl_backend(tmp_path) -> None:
    pytest.importorskip("igraph")

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph_igraph_mpl.png"

    returned = visualizer.save_plot(
        out,
        backend="igraph-mpl",
        layout="fr",
        color_by="degree",
    )

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_plot_accepts_legacy_igraph_alias(tmp_path) -> None:
    pytest.importorskip("igraph")

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph_igraph_alias.png"

    returned = visualizer.save_plot(
        out,
        backend="igraph",
        layout="fr",
    )

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_plot_with_igraph_cairo_uses_direct_target(
    tmp_path,
    monkeypatch,
) -> None:
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph.svg"
    seen = {}

    def fake_plot_igraph_cairo(self, **kwargs):
        seen.update(kwargs)
        path = Path(kwargs["target"])
        path.write_text("<svg></svg>")
        return path

    monkeypatch.setattr(
        HamiltonianGraphVisualizer,
        "_plot_igraph_cairo",
        fake_plot_igraph_cairo,
    )

    returned = visualizer.save_plot(
        out,
        backend="igraph-cairo",
        layout="fr",
    )

    assert returned == out
    assert seen["target"] == out
    assert seen["save_path"] is None
    assert seen["show"] is False
    assert out.exists()


def test_save_alias_calls_save_plot(tmp_path) -> None:
    pytest.importorskip("networkx")

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    out = tmp_path / "graph.png"

    returned = visualizer.save(
        out,
        backend="networkx",
        layout="spring",
        seed=123,
        iterations=5,
    )

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_state_amplitude_real_colors_are_zero_centered() -> None:
    from matplotlib.colors import to_hex

    values = np.asarray([-0.2, 0.0, 0.1], dtype=np.float64)

    colors = _scalar_values_to_colors(
        values,
        cmap="coolwarm",
        color_by="state_amplitude_real",
    )

    expected_zero = plt.get_cmap("coolwarm")(0.5)

    assert to_hex(colors[1]) == to_hex(expected_zero)


def test_state_amplitude_real_hex_colors_are_zero_centered() -> None:
    from matplotlib.colors import to_hex

    values = np.asarray([-0.2, 0.0, 0.1], dtype=np.float64)

    colors = _scalar_values_to_hex_colors(
        values,
        cmap="coolwarm",
        color_by="state_amplitude_real",
    )

    expected_zero = to_hex(plt.get_cmap("coolwarm")(0.5))

    assert colors[1] == expected_zero


def test_degree_colors_are_not_zero_centered() -> None:
    from matplotlib.colors import to_hex

    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    colors = _scalar_values_to_colors(
        values,
        cmap="coolwarm",
        color_by="degree",
    )

    expected_min = plt.get_cmap("coolwarm")(0.0)

    assert to_hex(colors[0]) == to_hex(expected_min)


def test_dense_orbit_labels_compacts_representatives() -> None:
    raw = np.asarray([0, 0, 2, 2, 0, 5], dtype=np.int64)

    labels = HamiltonianGraphVisualizer._dense_orbit_labels(raw)

    np.testing.assert_array_equal(
        labels,
        np.asarray([0, 0, 1, 1, 0, 2], dtype=np.int64),
    )


def test_orbit_labels_to_hex_colors_groups_equal_labels() -> None:
    colors = _orbit_labels_to_hex_colors(np.asarray([0, 1, 0, 2, 1], dtype=np.int64))

    assert colors[0] == colors[2]
    assert colors[1] == colors[4]
    assert len({colors[0], colors[1], colors[3]}) == 3


def test_igraph_automorphism_orbits_path_graph() -> None:
    pytest.importorskip("igraph")

    hamiltonian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(hamiltonian)

    labels = visualizer.automorphism_orbits(backend="igraph")

    assert labels[0] == labels[2]
    assert labels[1] != labels[0]


@pytest.mark.skipif(
    not pynauty_is_safe(),
    reason="pynauty is not safe on this platform",
)
def test_pynauty_automorphism_orbits_path_graph() -> None:
    pytest.importorskip("pynauty")

    hamiltonian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(hamiltonian)

    labels = visualizer.automorphism_orbits(backend="pynauty")

    assert labels[0] == labels[2]
    assert labels[1] != labels[0]


def test_plot_networkx_colored_by_automorphism_orbit() -> None:
    pytest.importorskip("networkx")
    pytest.importorskip("igraph")

    import matplotlib.pyplot as plt

    hamiltonian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(hamiltonian)

    fig, ax = plt.subplots()

    returned_ax = visualizer.plot(
        backend="networkx",
        color_by="automorphism_orbit",
        automorphism_backend="igraph",
        ax=ax,
        show=False,
    )

    assert returned_ax is ax
    assert len(ax.collections) > 0

    plt.close(fig)


def test_orbit_labels_to_rgba_colors_uses_requested_alpha() -> None:
    colors = _orbit_labels_to_rgba_colors(
        np.asarray([0, 1, 0], dtype=np.int64),
        alpha=0.42,
    )

    assert colors[0] == colors[2]
    assert colors[0][3] == pytest.approx(0.42)
    assert colors[1][3] == pytest.approx(0.42)
    assert colors[0] != colors[1]


def test_orbit_hex_colors_can_be_lightened() -> None:
    labels = np.asarray([0, 1, 2], dtype=np.int64)

    normal = _orbit_labels_to_hex_colors(labels, lightness_boost=0.0)
    light = _orbit_labels_to_hex_colors(labels, lightness_boost=0.2)

    assert len(normal) == len(light)
    assert normal != light


def test_networkx_orbit_colors_are_transparent() -> None:
    pytest.importorskip("networkx")
    pytest.importorskip("igraph")

    import matplotlib.pyplot as plt
    import scipy.sparse as scipy_sparse

    hamiltonian = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    style = HamiltonianGraphStyle(orbit_alpha=0.4)
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
        hamiltonian,
        style=style,
    )

    fig, ax = plt.subplots()

    visualizer.plot(
        backend="networkx",
        color_by="automorphism_orbit",
        automorphism_backend="igraph",
        ax=ax,
        show=False,
    )

    node_collection = ax.collections[0]
    facecolors = node_collection.get_facecolors()

    assert facecolors.shape[0] > 0
    assert np.allclose(facecolors[:, 3], 0.4)

    plt.close(fig)


def test_hamiltonian_graph_cage_subgraph_selects_support_and_zeros() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    vector = np.zeros(5, dtype=np.complex128)
    vector[[1, 2]] = [1.0, -1.0]

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[0, 3],
    )

    np.testing.assert_array_equal(
        subgraph.graph_data.original_indices,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    assert subgraph.graph_data.adjacency.shape == (4, 4)


def test_hamiltonian_graph_cage_subgraph_can_drop_zero_zero_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    vector = np.zeros(4, dtype=np.complex128)
    vector[1] = 1.0

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[0, 2, 3],
        include_zero_edges=False,
    )

    adjacency = subgraph.graph_data.adjacency.toarray()

    # Keep 0-1 and 1-2, drop 2-3 because it is zero-zero.
    assert adjacency[0, 1] != 0
    assert adjacency[1, 2] != 0
    assert adjacency[2, 3] == 0


def test_cage_subgraph_keeps_projected_state_vector_for_coloring() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    vector = np.zeros(4, dtype=np.complex128)
    vector[1] = 1.0
    vector[2] = -0.5

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[0, 3],
    )

    np.testing.assert_array_equal(
        subgraph.graph_data.original_indices,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    np.testing.assert_allclose(
        subgraph.graph_data.state_vector,
        vector,
    )

    values = subgraph.node_values(color_by="state_weight")

    np.testing.assert_allclose(
        values,
        np.asarray([0.0, 1.0, 0.25, 0.0]),
    )

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[0],
    )

    np.testing.assert_array_equal(
        subgraph.graph_data.original_indices,
        np.asarray([0, 1, 2], dtype=np.int64),
    )

    np.testing.assert_allclose(
        subgraph.graph_data.state_vector,
        vector[[0, 1, 2]],
    )


def test_cage_subgraph_vertex_display_labels_use_original_indices() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    vector = np.zeros(5, dtype=np.complex128)
    vector[[2, 3]] = [1.0, -1.0]

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[1, 4],
    )

    np.testing.assert_array_equal(
        subgraph.graph_data.original_indices,
        np.asarray([1, 2, 3, 4], dtype=np.int64),
    )

    assert subgraph.vertex_display_labels() == ["1", "2", "3", "4"]


def test_cage_subgraph_networkx_nodes_have_original_indices() -> None:
    matrix = scipy_sparse.identity(5, format="csr")
    vector = np.zeros(5, dtype=np.complex128)
    vector[3] = 1.0

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[1],
    )

    graph = subgraph.to_networkx()

    assert graph.nodes[0]["original_index"] == 1
    assert graph.nodes[1]["original_index"] == 3
    assert graph.nodes[0]["label"] == "1"
    assert graph.nodes[1]["label"] == "3"


def test_cage_subgraph_plot_labels_show_original_indices() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    vector = np.zeros(3, dtype=np.complex128)
    vector[2] = 1.0

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
        matrix,
        style=HamiltonianGraphStyle(label_vertices=True),
    )

    subgraph = visualizer.subgraph_for_cage_state(
        vector,
        zero_indices=[1],
    )

    fig, ax = plt.subplots()

    subgraph.plot(
        backend="networkx",
        ax=ax,
        show=False,
        layout="spring",
    )

    labels = {text.get_text() for text in ax.texts}

    assert {"1", "2"} <= labels

    plt.close(fig)


def test_hamiltonian_graph_preserves_complex_edge_weights() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0 + 2.0j],
                [1.0 - 2.0j, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    weights = visualizer.edge_weights()

    np.testing.assert_allclose(weights, np.asarray([1.0 + 2.0j]))


def test_hamiltonian_graph_edge_values_real_imag_abs_phase() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0 + 1.0j, 2.0],
                [1.0 - 1.0j, 0.0, -1.0j],
                [2.0, 1.0j, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    np.testing.assert_allclose(
        visualizer.edge_values(color_by="weight_real"),
        np.asarray([1.0, 2.0, 0.0]),
    )
    np.testing.assert_allclose(
        visualizer.edge_values(color_by="weight_imag"),
        np.asarray([1.0, 0.0, -1.0]),
    )
    np.testing.assert_allclose(
        visualizer.edge_values(color_by="weight_abs"),
        np.asarray([np.sqrt(2.0), 2.0, 1.0]),
    )


def test_hamiltonian_graph_networkx_edge_coloring_complex_runs() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 1.0j],
                [1.0, 0.0, -1.0],
                [-1.0j, -1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)

    fig, ax = plt.subplots()

    visualizer.plot(
        backend="networkx",
        ax=ax,
        show=False,
        edge_color_by="weight_complex",
    )

    assert ax.collections

    plt.close(fig)


def test_node_values_state_rules_and_errors() -> None:
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())
    state = np.asarray([1.0, 1.0j, -2.0], dtype=np.complex128)

    np.testing.assert_allclose(visualizer.node_values(color_by="constant"), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        visualizer.node_values(color_by="state_amplitude_real", state_vector=state),
        [1.0, 0.0, -2.0],
    )
    np.testing.assert_allclose(
        visualizer.node_values(color_by="state_amplitude_imag", state_vector=state),
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        visualizer.node_values(color_by="state_amplitude_abs", state_vector=state),
        [1.0, 1.0, 2.0],
    )
    np.testing.assert_allclose(
        visualizer.node_values(color_by="state_phase", state_vector=state),
        np.angle(state),
    )

    with pytest.raises(ValueError, match="state_vector is required"):
        visualizer.node_values(color_by="state_weight")
    with pytest.raises(ValueError, match="Expected 3 node values"):
        visualizer.node_values(color_by="self_loop", self_loop_values=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Unsupported color_by"):
        visualizer.node_values(color_by="not-a-rule")  # type: ignore[arg-type]


def test_edge_values_rules_and_errors() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0 + 1.0j, 0.0],
                [1.0 - 1.0j, 0.0, -2.0j],
                [0.0, 2.0j, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)
    weights = visualizer.edge_weights()

    np.testing.assert_allclose(visualizer.edge_values(color_by="constant"), np.zeros(2))
    np.testing.assert_allclose(visualizer.edge_values(color_by="weight_abs"), np.abs(weights))
    np.testing.assert_allclose(visualizer.edge_values(color_by="weight_real"), np.real(weights))
    np.testing.assert_allclose(visualizer.edge_values(color_by="weight_imag"), np.imag(weights))
    np.testing.assert_allclose(visualizer.edge_values(color_by="weight_phase"), np.angle(weights))
    np.testing.assert_allclose(visualizer.edge_values(color_by="weight_complex"), weights)

    with pytest.raises(ValueError, match="Unsupported edge_color_by"):
        visualizer.edge_values(color_by="bad-rule")  # type: ignore[arg-type]


def test_vertex_display_labels_use_custom_and_original_indices() -> None:
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())
    assert visualizer.vertex_display_labels() == ["0", "1", "2"]

    custom = HamiltonianGraphVisualizer(
        graph_data=visualizer.graph_data.__class__(
            adjacency=visualizer.graph_data.adjacency,
            self_loop_values=visualizer.graph_data.self_loop_values,
            original_indices=np.array([10, 11, 12], dtype=np.int64),
            vertex_labels=("a", "b", "c"),
        ),
        style=visualizer.style,
    )
    assert custom.vertex_display_labels() == ["a", "b", "c"]

    bad = HamiltonianGraphVisualizer(
        graph_data=visualizer.graph_data.__class__(
            adjacency=visualizer.graph_data.adjacency,
            self_loop_values=visualizer.graph_data.self_loop_values,
            original_indices=np.array([0], dtype=np.int64),
        ),
        style=visualizer.style,
    )
    with pytest.raises(ValueError, match="original_indices"):
        bad.vertex_display_labels()


def test_subgraph_for_cage_state_and_zero_edges() -> None:
    matrix = scipy_sparse.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(matrix)
    state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    subgraph = visualizer.subgraph_for_cage_state(
        state,
        zero_indices=[2, 3],
        include_zero_edges=False,
    )

    np.testing.assert_array_equal(subgraph.graph_data.original_indices, np.array([0, 2, 3]))
    assert subgraph.edge_pairs() == []
    np.testing.assert_allclose(subgraph.graph_data.state_vector, [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="No cage support"):
        visualizer.subgraph_for_cage_state(np.zeros(4, dtype=np.complex128))


def test_extract_cage_zero_indices_from_report_like_object() -> None:
    class Report:
        nontrivial_zero_indices = [1]
        projector_like_zero_indices = ((2, 3), np.array([3, 4]))

    extracted = HamiltonianGraphVisualizer._extract_cage_zero_indices(
        zero_indices=[0],
        classification_report=Report(),
    )

    np.testing.assert_array_equal(extracted, np.array([0, 1, 2, 3, 4], dtype=np.int64))


def test_networkx_layout_export_has_node_metadata() -> None:
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    graph = visualizer.to_networkx_with_layout(layout="spring", seed=0)

    for node in graph.nodes:
        assert "x" in graph.nodes[node]
        assert "y" in graph.nodes[node]
        assert "viz" in graph.nodes[node]
        assert "degree" in graph.nodes[node]
        assert "self_loop_real" in graph.nodes[node]


def test_save_graph_rejects_unknown_suffix(tmp_path: Path) -> None:
    visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(_small_hamiltonian())

    with pytest.raises(ValueError, match="Graph export path"):
        visualizer.save_graph(tmp_path / "graph.unknown")
