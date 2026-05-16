"""Fock-space graph visualizer for sparse Hamiltonian matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse
from scipy.sparse.csgraph import connected_components

GraphBackend = Literal["igraph", "networkx"]
NodeColorRule = Literal[
    "constant",
    "bipartition",
    "self_loop",
    "degree",
    "state_weight",
]
LayoutName = Literal[
    "auto",
    "fr",
    "kk",
    "circle",
    "grid_fr",
    "kamada_kawai",
    "spring",
]


@dataclass(frozen=True, slots=True)
class HamiltonianGraphStyle:
    """Style options for drawing Hamiltonian/Fock-space graphs."""

    vertex_size: float = 14.0
    edge_width: float = 0.8
    edge_alpha: float = 0.45
    label_vertices: bool = False
    vertex_label_size: float = 8.0
    colorbar: bool = True
    cmap: str = "viridis"
    default_vertex_color: str = "lightgray"
    edge_color: str = "gray"
    figure_size: tuple[float, float] = (7.0, 7.0)


@dataclass(frozen=True)
class HamiltonianGraphData:
    """Graph data extracted from a sparse Hamiltonian matrix."""

    adjacency: scipy_sparse.csr_array
    self_loop_values: npt.NDArray[np.complex128]

    @property
    def n_vertices(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def degrees(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            self.adjacency.astype(bool).sum(axis=1),
        ).ravel().astype(np.int64)


@dataclass
class HamiltonianGraphVisualizer:
    """Visualizer for Fock-space graphs induced by Hamiltonian matrices."""

    graph_data: HamiltonianGraphData
    style: HamiltonianGraphStyle = HamiltonianGraphStyle()

    @classmethod
    def from_sparse_matrix(
        cls,
        matrix,
        *,
        include_self_loops: bool = False,
        weight_tolerance: float = 0.0,
        style: HamiltonianGraphStyle | None = None,
    ) -> HamiltonianGraphVisualizer:
        """Construct a visualizer from a sparse or dense Hamiltonian matrix.

        Parameters
        ----------
        matrix:
            Hamiltonian or kinetic matrix. Nonzero off-diagonal entries define
            graph edges. Diagonal entries are stored as self-loop values.
        include_self_loops:
            Whether to keep diagonal graph edges. Usually False for drawing.
        weight_tolerance:
            Entries with absolute value <= this threshold are removed.
        style:
            Optional drawing style.
        """
        sparse_matrix = _as_csr_array(matrix)
        self_loop_values = np.asarray(
            sparse_matrix.diagonal(),
            dtype=np.complex128,
        )

        adjacency = sparse_matrix.copy()

        if not include_self_loops:
            adjacency.setdiag(0)

        if weight_tolerance > 0.0:
            adjacency.data[np.abs(adjacency.data) <= weight_tolerance] = 0

        adjacency.eliminate_zeros()

        # For visualization, treat the graph as undirected.
        adjacency = _symmetrized_adjacency(adjacency)

        return cls(
            graph_data=HamiltonianGraphData(
                adjacency=adjacency,
                self_loop_values=self_loop_values,
            ),
            style=HamiltonianGraphStyle() if style is None else style,
        )

    def bipartition_labels(self) -> npt.NDArray[np.int64]:
        """Return bipartition labels for the graph.

        Raises
        ------
        ValueError
            If the graph is not bipartite.
        """
        return bipartition_labels(self.graph_data.adjacency)

    def node_values(
        self,
        *,
        color_by: NodeColorRule,
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return scalar node values used for coloring."""
        if color_by == "constant":
            return np.zeros(self.graph_data.n_vertices, dtype=np.float64)

        if color_by == "bipartition":
            return self.bipartition_labels().astype(np.float64)

        if color_by == "self_loop":
            values = (
                self.graph_data.self_loop_values
                if self_loop_values is None
                else np.asarray(self_loop_values, dtype=np.complex128)
            )
            _validate_node_values(values, self.graph_data.n_vertices)
            return np.real(values).astype(np.float64)

        if color_by == "degree":
            return self.graph_data.degrees.astype(np.float64)

        if color_by == "state_weight":
            if state_vector is None:
                raise ValueError("state_vector is required for color_by='state_weight'.")

            vector = np.asarray(state_vector, dtype=np.complex128)
            _validate_node_values(vector, self.graph_data.n_vertices)

            return np.abs(vector) ** 2

        raise ValueError(f"Unsupported color_by rule: {color_by!r}")

    def plot(
        self,
        *,
        backend: GraphBackend = "igraph",
        color_by: NodeColorRule = "constant",
        layout: LayoutName = "auto",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        title: str | None = None,
        ax=None,
        show: bool = True,
        save_path: str | Path | None = None,
    ):
        """Draw the graph."""
        if backend == "igraph":
            return self._plot_igraph(
                color_by=color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                show=show,
                save_path=save_path,
            )

        if backend == "networkx":
            return self._plot_networkx(
                color_by=color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                ax=ax,
                show=show,
                save_path=save_path,
            )

        raise ValueError("backend must be 'igraph' or 'networkx'.")

    def _plot_igraph(
        self,
        *,
        color_by: NodeColorRule,
        layout: LayoutName,
        self_loop_values: npt.ArrayLike | None,
        state_vector: npt.ArrayLike | None,
        title: str | None,
        show: bool,
        save_path: str | Path | None,
    ):
        """Draw with python-igraph."""
        try:
            import igraph as ig
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "The igraph backend requires python-igraph and matplotlib."
            ) from error

        graph = self.to_igraph()
        values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
        )
        vertex_colors = _scalar_values_to_colors(
            values,
            cmap=self.style.cmap,
        )

        layout_object = _igraph_layout(graph, layout)

        fig, ax = plt.subplots(figsize=self.style.figure_size)
        ig.plot(
            graph,
            target=ax,
            layout=layout_object,
            vertex_size=self.style.vertex_size,
            vertex_color=vertex_colors,
            vertex_label=(
                [str(index) for index in range(graph.vcount())]
                if self.style.label_vertices
                else None
            ),
            vertex_label_size=self.style.vertex_label_size,
            edge_width=self.style.edge_width,
            edge_color=self.style.edge_color,
        )

        if title is not None:
            ax.set_title(title)

        if self.style.colorbar and color_by != "constant":
            _add_colorbar(
                ax=ax,
                values=values,
                cmap=self.style.cmap,
                label=color_by,
            )

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return ax

    def _plot_networkx(
        self,
        *,
        color_by: NodeColorRule,
        layout: LayoutName,
        self_loop_values: npt.ArrayLike | None,
        state_vector: npt.ArrayLike | None,
        title: str | None,
        ax,
        show: bool,
        save_path: str | Path | None,
    ):
        """Draw with networkx."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as error:
            raise ImportError(
                "The networkx backend requires networkx and matplotlib."
            ) from error

        graph = self.to_networkx()
        values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figure_size)

        positions = _networkx_layout(graph, layout)

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=values,
            cmap=self.style.cmap,
            node_size=self.style.vertex_size * 12,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            width=self.style.edge_width,
            edge_color=self.style.edge_color,
            alpha=self.style.edge_alpha,
            ax=ax,
        )

        if self.style.label_vertices:
            nx.draw_networkx_labels(
                graph,
                positions,
                font_size=self.style.vertex_label_size,
                ax=ax,
            )

        if title is not None:
            ax.set_title(title)

        ax.set_axis_off()

        if self.style.colorbar and color_by != "constant":
            _add_colorbar(
                ax=ax,
                values=values,
                cmap=self.style.cmap,
                label=color_by,
            )

        if save_path is not None:
            ax.figure.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return ax

    def to_igraph(self):
        """Convert to an igraph.Graph."""
        try:
            import igraph as ig
        except ImportError as error:
            raise ImportError("python-igraph is required for to_igraph().") from error

        adjacency = self.graph_data.adjacency.tocoo()
        edges = [
            (int(row), int(col))
            for row, col in zip(adjacency.row, adjacency.col, strict=True)
            if int(row) < int(col)
        ]

        graph = ig.Graph(
            n=self.graph_data.n_vertices,
            edges=edges,
            directed=False,
        )
        return graph

    def to_networkx(self):
        """Convert to a networkx.Graph."""
        try:
            import networkx as nx
        except ImportError as error:
            raise ImportError("networkx is required for to_networkx().") from error

        adjacency = self.graph_data.adjacency.tocoo()
        graph = nx.Graph()
        graph.add_nodes_from(range(self.graph_data.n_vertices))

        for row, col, value in zip(
            adjacency.row,
            adjacency.col,
            adjacency.data,
            strict=True,
        ):
            if int(row) < int(col):
                graph.add_edge(
                    int(row),
                    int(col),
                    weight=float(abs(value)),
                    hamiltonian_weight=complex(value),
                )

        return graph


def _as_csr_array(matrix) -> scipy_sparse.csr_array:
    """Convert dense or sparse input to ``csr_array``."""
    if scipy_sparse.issparse(matrix):
        return scipy_sparse.csr_array(matrix)

    return scipy_sparse.csr_array(np.asarray(matrix, dtype=np.complex128))


def _symmetrized_adjacency(
    matrix: scipy_sparse.csr_array,
) -> scipy_sparse.csr_array:
    """Return an unweighted undirected adjacency matrix."""
    adjacency = scipy_sparse.csr_array(matrix.copy())
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency.tocsr()


def _validate_node_values(
    values: npt.ArrayLike,
    n_vertices: int,
) -> None:
    values_array = np.asarray(values)

    if values_array.ndim != 1:
        raise ValueError("Node values must be one-dimensional.")

    if values_array.shape[0] != n_vertices:
        raise ValueError(
            f"Expected {n_vertices} node values, got {values_array.shape[0]}."
        )


def bipartition_labels(
    adjacency_matrix,
) -> npt.NDArray[np.int64]:
    """Compute bipartition labels for an undirected graph.

    Disconnected components are handled independently. Isolated vertices are
    assigned label 0.
    """
    adjacency = _symmetrized_adjacency(_as_csr_array(adjacency_matrix))
    n_vertices = adjacency.shape[0]
    labels = -np.ones(n_vertices, dtype=np.int64)

    for start_index in range(n_vertices):
        if labels[start_index] != -1:
            continue

        labels[start_index] = 0
        queue = [start_index]

        while queue:
            vertex_index = queue.pop(0)
            neighbors = adjacency.indices[
                adjacency.indptr[vertex_index] : adjacency.indptr[vertex_index + 1]
            ]

            for neighbor_index in neighbors:
                neighbor_index = int(neighbor_index)

                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = 1 - labels[vertex_index]
                    queue.append(neighbor_index)
                elif labels[neighbor_index] == labels[vertex_index]:
                    raise ValueError("Graph is not bipartite.")

    return labels


def _scalar_values_to_colors(
    values: npt.ArrayLike,
    *,
    cmap: str,
) -> list:
    """Convert scalar values to matplotlib colors."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return []

    if np.allclose(values, values[0]):
        normalized_values = np.zeros_like(values, dtype=np.float64)
    else:
        normalized_values = Normalize(
            vmin=float(np.min(values)),
            vmax=float(np.max(values)),
        )(values)

    colormap = plt.get_cmap(cmap)

    return [colormap(float(value)) for value in normalized_values]


def _add_colorbar(
    *,
    ax,
    values: npt.ArrayLike,
    cmap: str,
    label: str,
) -> None:
    """Attach a scalar colorbar to an axis."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or np.allclose(values, values[0]):
        return

    norm = Normalize(
        vmin=float(np.min(values)),
        vmax=float(np.max(values)),
    )
    scalar_mappable = ScalarMappable(
        cmap=plt.get_cmap(cmap),
        norm=norm,
    )
    scalar_mappable.set_array([])

    ax.figure.colorbar(
        scalar_mappable,
        ax=ax,
        label=label,
        shrink=0.8,
    )


def _igraph_layout(graph, layout: LayoutName):
    """Return an igraph layout."""
    if layout in ("auto", "fr"):
        return graph.layout_fruchterman_reingold()

    if layout in ("kk", "kamada_kawai"):
        return graph.layout_kamada_kawai()

    if layout == "circle":
        return graph.layout_circle()

    if layout == "grid_fr":
        return graph.layout_grid_fruchterman_reingold()

    raise ValueError(f"Unsupported igraph layout: {layout!r}")


def _networkx_layout(graph, layout: LayoutName):
    """Return networkx positions."""
    import networkx as nx

    if layout in ("auto", "spring", "fr", "grid_fr"):
        return nx.spring_layout(graph, seed=0)

    if layout in ("kk", "kamada_kawai"):
        return nx.kamada_kawai_layout(graph)

    if layout == "circle":
        return nx.circular_layout(graph)

    raise ValueError(f"Unsupported networkx layout: {layout!r}")
