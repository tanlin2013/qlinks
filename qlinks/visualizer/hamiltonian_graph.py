"""Fock-space graph visualizer for sparse Hamiltonian matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

GraphBackend = Literal["igraph", "igraph-mpl", "igraph-cairo", "networkx"]
NormalizedGraphBackend = Literal["igraph-mpl", "igraph-cairo", "networkx"]
NodeColorRule = Literal[
    "constant",
    "bipartition",
    "self_loop",
    "degree",
    "automorphism_orbit",
    "state_weight",
    "state_amplitude_real",
    "state_amplitude_imag",
    "state_amplitude_abs",
    "state_phase",
]
EdgeColorRule = Literal[
    "constant",
    "weight_abs",
    "weight_real",
    "weight_imag",
    "weight_phase",
    "weight_complex",
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
AutomorphismBackend = Literal["auto", "pynauty", "igraph"]


@dataclass(frozen=True, slots=True)
class HamiltonianGraphStyle:
    """Style options for drawing Hamiltonian/Fock-space graphs."""

    # Figure / canvas
    figure_size: tuple[float, float] = (7.0, 7.0)

    # Vertices
    vertex_size: float = 14.0
    default_vertex_color: str = "lightgray"

    # Edges
    edge_width: float = 0.8
    edge_alpha: float = 0.45
    edge_color: str = "gray"

    # Labels
    label_vertices: bool = False
    vertex_label_size: float = 8.0

    # Vertex colormap / colorbar
    cmap: str = "viridis"
    colorbar: bool = True

    # Edge colormap / colorbar
    edge_cmap: str = "coolwarm"
    edge_phase_cmap: str = "twilight"
    edge_colorbar: bool = True

    # Complex edge coloring
    edge_complex_min_alpha: float = 0.20
    edge_complex_max_alpha: float = 0.95

    # Automorphism-orbit coloring
    orbit_alpha: float = 0.65
    orbit_lightness_boost: float = 0.15


@dataclass(frozen=True)
class HamiltonianGraphData:
    """Graph data extracted from a sparse Hamiltonian matrix."""

    adjacency: sp.csr_array
    self_loop_values: npt.NDArray[np.complex128]
    original_indices: npt.NDArray[np.int64] | None = None
    state_vector: npt.NDArray[np.complex128] | None = None
    vertex_labels: Sequence[str] | None = None
    directed: bool = False

    @property
    def n_vertices(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def degrees(self) -> npt.NDArray[np.int64]:
        return (
            np.asarray(
                self.adjacency.astype(bool).sum(axis=1),
            )
            .ravel()
            .astype(np.int64)
        )


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
        directed: bool = False,
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
        directed:
            Whether to treat the matrix as directed (asymmetric) or undirected
            (symmetrized). For an undirected graph, the adjacency is symmetrized
            by A + A^T, which preserves edge weights but may introduce new edges
            if the input matrix is asymmetric.
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

        if not directed:
            adjacency = _symmetrized_weighted_adjacency(adjacency)
        else:
            adjacency = sp.csr_array(adjacency)

        n_vertices = int(adjacency.shape[0])

        return cls(
            graph_data=HamiltonianGraphData(
                adjacency=adjacency,
                self_loop_values=self_loop_values,
                original_indices=np.arange(n_vertices, dtype=np.int64),
                directed=directed,
            ),
            style=HamiltonianGraphStyle() if style is None else style,
        )

    @classmethod
    def from_directed_sparse_matrix(
        cls,
        matrix,
        *,
        include_self_loops: bool = False,
        weight_tolerance: float = 0.0,
        style: HamiltonianGraphStyle | None = None,
    ) -> HamiltonianGraphVisualizer:
        """Construct a directed graph visualizer from a sparse matrix."""
        return cls.from_sparse_matrix(
            matrix,
            include_self_loops=include_self_loops,
            weight_tolerance=weight_tolerance,
            directed=True,
            style=style,
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
        automorphism_backend: AutomorphismBackend = "auto",
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

        if color_by == "automorphism_orbit":
            return self.automorphism_orbits(backend=automorphism_backend).astype(np.float64)

        if color_by in (
            "state_weight",
            "state_amplitude_real",
            "state_amplitude_imag",
            "state_amplitude_abs",
            "state_phase",
        ):
            vector = self._resolve_state_vector(state_vector)

            if color_by == "state_weight":
                return (np.abs(vector) ** 2).astype(np.float64)

            if color_by == "state_amplitude_real":
                return np.real(vector).astype(np.float64)

            if color_by == "state_amplitude_imag":
                return np.imag(vector).astype(np.float64)

            if color_by == "state_amplitude_abs":
                return np.abs(vector).astype(np.float64)

            if color_by == "state_phase":
                return np.angle(vector).astype(np.float64)

        raise ValueError(f"Unsupported color_by rule: {color_by!r}")

    def edge_values(
        self,
        *,
        color_by: EdgeColorRule,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Return edge values in the same order as plotted graph edges."""
        if color_by == "constant":
            return np.zeros(self.n_edges(), dtype=np.float64)

        weights = self.edge_weights()

        if color_by == "weight_abs":
            return np.abs(weights).astype(np.float64)

        if color_by == "weight_real":
            return np.real(weights).astype(np.float64)

        if color_by == "weight_imag":
            return np.imag(weights).astype(np.float64)

        if color_by == "weight_phase":
            return np.angle(weights).astype(np.float64)

        if color_by == "weight_complex":
            return weights.astype(np.complex128)

        raise ValueError(f"Unsupported edge_color_by rule: {color_by!r}")

    def plot(
        self,
        *,
        backend: GraphBackend = "igraph",
        color_by: NodeColorRule = "constant",
        edge_color_by: EdgeColorRule = "constant",
        layout: LayoutName = "auto",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        title: str | None = None,
        ax=None,
        show: bool = True,
        save_path: str | Path | None = None,
        target: str | Path | None = None,
        bbox: tuple[int, int] = (800, 800),
        margin: int = 40,
        **layout_kwargs,
    ):
        """Draw the graph."""
        normalized_backend = _normalize_graph_backend(backend)

        if normalized_backend == "igraph-mpl":
            return self._plot_igraph_matplotlib(
                color_by=color_by,
                edge_color_by=edge_color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                ax=ax,
                show=show,
                save_path=save_path,
                **layout_kwargs,
            )

        if normalized_backend == "igraph-cairo":
            return self._plot_igraph_cairo(
                color_by=color_by,
                edge_color_by=edge_color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                ax=ax,
                show=show,
                save_path=save_path,
                target=target,
                bbox=bbox,
                margin=margin,
                **layout_kwargs,
            )

        if normalized_backend == "networkx":
            return self._plot_networkx(
                color_by=color_by,
                edge_color_by=edge_color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                ax=ax,
                show=show,
                save_path=save_path,
                **layout_kwargs,
            )

        raise ValueError(f"Unsupported backend: {backend!r}")

    def _plot_igraph_matplotlib(
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
        edge_color_by: EdgeColorRule = "constant",
        automorphism_backend: AutomorphismBackend = "auto",
        **layout_kwargs,
    ):
        """Draw with python-igraph layout and Matplotlib rendering."""
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
            automorphism_backend=automorphism_backend,
        )
        if color_by == "automorphism_orbit":
            vertex_colors = _orbit_labels_to_rgba_colors(
                values.astype(np.int64),
                alpha=self.style.orbit_alpha,
            )
        else:
            vertex_colors = _scalar_values_to_colors(
                values,
                cmap=self.style.cmap,
                color_by=color_by,
            )

        edge_colors = self._edge_colors_for_matplotlib(edge_color_by=edge_color_by)

        layout_object = _igraph_layout(graph, layout, **layout_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figure_size)
        else:
            fig = ax.figure

        ig.plot(
            graph,
            target=ax,
            layout=layout_object,
            vertex_size=self.style.vertex_size,
            vertex_color=vertex_colors,
            vertex_label=(self.vertex_display_labels() if self.style.label_vertices else None),
            vertex_label_size=self.style.vertex_label_size,
            edge_width=self.style.edge_width,
            edge_color=edge_colors,
        )

        if title is not None:
            ax.set_title(title)

        if self.style.colorbar and color_by not in ("constant", "automorphism_orbit"):
            _add_colorbar(
                ax=ax,
                values=values,
                cmap=self.style.cmap,
                label=color_by,
                color_by=color_by,
            )

        if self.style.edge_colorbar and edge_color_by not in ("constant", "weight_complex"):
            edge_values = self.edge_values(color_by=edge_color_by)
            edge_cmap = (
                self.style.edge_phase_cmap
                if edge_color_by == "weight_phase"
                else self.style.edge_cmap
            )

            _add_edge_colorbar(
                ax=ax,
                values=edge_values,
                cmap=edge_cmap,
                label=edge_color_by,
                color_by=edge_color_by,
            )

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return ax

    def _plot_igraph_cairo(
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
        target: str | Path | None,
        bbox: tuple[int, int],
        margin: int,
        edge_color_by: EdgeColorRule = "constant",
        automorphism_backend: AutomorphismBackend = "auto",
        **layout_kwargs,
    ):
        """Draw with python-igraph layout and Cairo rendering.

        Cairo rendering is target/file oriented and does not draw into a
        Matplotlib axis.
        """
        if ax is not None:
            raise ValueError(
                "backend='igraph-cairo' does not draw on a matplotlib Axes. "
                "Pass target=... or save_path=... instead."
            )

        try:
            import igraph as ig
        except ImportError as error:
            raise ImportError("The igraph-cairo backend requires python-igraph.") from error

        _ensure_cairo_available()

        if target is not None and save_path is not None:
            raise ValueError("Pass only one of target=... or save_path=..., not both.")

        output_path = target if target is not None else save_path

        graph = self.to_igraph()

        values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            automorphism_backend=automorphism_backend,
        )

        if color_by == "automorphism_orbit":
            vertex_colors = _orbit_labels_to_hex_colors(
                values.astype(np.int64),
                lightness_boost=self.style.orbit_lightness_boost,
            )
        else:
            vertex_colors = _scalar_values_to_hex_colors(
                values,
                cmap=self.style.cmap,
                color_by=color_by,
            )

        layout_object = _igraph_layout(graph, layout, **layout_kwargs)

        vertex_labels = self.vertex_display_labels() if self.style.label_vertices else None

        visual_style = {
            "layout": layout_object,
            "bbox": bbox,
            "margin": margin,
            "vertex_size": self.style.vertex_size,
            "vertex_color": vertex_colors,
            "vertex_label": vertex_labels,
            "vertex_label_size": self.style.vertex_label_size,
            "edge_width": self.style.edge_width,
            "edge_color": self._edge_colors_for_igraph_cairo(
                edge_color_by=edge_color_by,
            ),
        }

        # Optional graph title. Cairo plots do not have a Matplotlib Axes title,
        # so attach a graph label instead.
        if title is not None:
            visual_style["label"] = title

        if output_path is not None:
            ig.plot(
                graph,
                target=str(output_path),
                **visual_style,
            )
            return Path(output_path)

        plot = ig.plot(
            graph,
            **visual_style,
        )

        # For Cairo, show=False simply returns the igraph Plot object.
        # There is no Matplotlib plt.show() equivalent here.
        return plot

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
        edge_color_by: EdgeColorRule = "constant",
        automorphism_backend: AutomorphismBackend = "auto",
        **layout_kwargs,
    ):
        """Draw with networkx."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as error:
            raise ImportError("The networkx backend requires networkx and matplotlib.") from error

        graph = self.to_networkx()
        values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            automorphism_backend=automorphism_backend,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figure_size)

        positions = _networkx_layout(graph, layout, **layout_kwargs)

        if color_by == "automorphism_orbit":
            node_colors = _orbit_labels_to_rgba_colors(
                values.astype(np.int64),
                alpha=self.style.orbit_alpha,
            )
        else:
            node_colors = _scalar_values_to_colors(
                values,
                cmap=self.style.cmap,
                color_by=color_by,
            )

        edge_colors = self._edge_colors_for_matplotlib(edge_color_by=edge_color_by)
        edge_alpha = None if edge_color_by == "weight_complex" else self.style.edge_alpha

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=node_colors,
            node_size=self.style.vertex_size * 12,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            width=self.style.edge_width,
            edge_color=edge_colors,
            alpha=edge_alpha,
            ax=ax,
        )

        if self.style.label_vertices:
            labels = {
                local_index: label for local_index, label in enumerate(self.vertex_display_labels())
            }

            nx.draw_networkx_labels(
                graph,
                positions,
                labels=labels,
                font_size=self.style.vertex_label_size,
                ax=ax,
            )

        if title is not None:
            ax.set_title(title)

        ax.set_axis_off()

        if self.style.colorbar and color_by not in ("constant", "automorphism_orbit"):
            _add_colorbar(
                ax=ax,
                values=values,
                cmap=self.style.cmap,
                label=color_by,
                color_by=color_by,
            )

        if self.style.edge_colorbar and edge_color_by not in ("constant", "weight_complex"):
            edge_values = self.edge_values(color_by=edge_color_by)
            edge_cmap = (
                self.style.edge_phase_cmap
                if edge_color_by == "weight_phase"
                else self.style.edge_cmap
            )

            _add_edge_colorbar(
                ax=ax,
                values=edge_values,
                cmap=edge_cmap,
                label=edge_color_by,
                color_by=edge_color_by,
            )

        if color_by == "automorphism_orbit":
            _add_orbit_legend(ax=ax, labels=values.astype(np.int64), colors=node_colors)

        if save_path is not None:
            ax.figure.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return ax

    def vertex_display_labels(self) -> list[str]:
        """Return vertex labels for plotting.

        For a full graph, these are 0, 1, 2, ...
        For a subgraph, these are the original parent-graph basis indices.
        """
        vertex_labels = self.graph_data.vertex_labels

        if vertex_labels is not None:
            if len(vertex_labels) != self.graph_data.n_vertices:
                raise ValueError("graph_data.vertex_labels must have length n_vertices.")
            return [str(label) for label in vertex_labels]

        original_indices = self.graph_data.original_indices

        if original_indices is None:
            return [str(index) for index in range(self.graph_data.n_vertices)]

        if len(original_indices) != self.graph_data.n_vertices:
            raise ValueError("graph_data.original_indices must have length n_vertices.")

        return [str(int(index)) for index in original_indices]

    def to_igraph(self):
        """Convert to an igraph.Graph."""
        try:
            import igraph as ig
        except ImportError as error:
            raise ImportError("python-igraph is required for to_igraph().") from error

        adjacency = self.graph_data.adjacency.tocoo()

        edges: list[tuple[int, int]] = []
        edge_weights: list[complex] = []

        for row, col, value in zip(
            adjacency.row,
            adjacency.col,
            adjacency.data,
            strict=True,
        ):
            if not self.graph_data.directed and int(row) >= int(col):
                continue

            edges.append((int(row), int(col)))
            edge_weights.append(complex(value))

        graph = ig.Graph(
            n=self.graph_data.n_vertices,
            edges=edges,
            directed=self.graph_data.directed,
        )

        labels = self.vertex_display_labels()

        graph.vs["local_index"] = list(range(self.graph_data.n_vertices))
        graph.vs["label"] = labels
        graph.vs["name"] = labels

        if self.graph_data.original_indices is not None:
            graph.vs["original_index"] = [int(index) for index in self.graph_data.original_indices]

        if self.graph_data.state_vector is not None:
            state_vector = np.asarray(
                self.graph_data.state_vector,
                dtype=np.complex128,
            )
            graph.vs["state_amplitude_real"] = [float(value.real) for value in state_vector]
            graph.vs["state_amplitude_imag"] = [float(value.imag) for value in state_vector]
            graph.vs["state_weight"] = [float(abs(value) ** 2) for value in state_vector]

        graph.es["hamiltonian_weight_real"] = [float(value.real) for value in edge_weights]
        graph.es["hamiltonian_weight_imag"] = [float(value.imag) for value in edge_weights]
        graph.es["hamiltonian_weight_abs"] = [float(abs(value)) for value in edge_weights]
        graph.es["hamiltonian_weight_phase"] = [float(np.angle(value)) for value in edge_weights]

        return graph

    def to_networkx(self):
        """Convert to a networkx.Graph."""
        try:
            import networkx as nx
        except ImportError as error:
            raise ImportError("networkx is required for to_networkx().") from error

        adjacency = self.graph_data.adjacency.tocoo()
        graph = nx.DiGraph() if self.graph_data.directed else nx.Graph()

        original_indices = self.graph_data.original_indices
        state_vector = self.graph_data.state_vector
        labels = self.vertex_display_labels()

        if original_indices is not None and len(original_indices) != self.graph_data.n_vertices:
            raise ValueError("graph_data.original_indices must have length n_vertices.")

        if state_vector is not None and len(state_vector) != self.graph_data.n_vertices:
            raise ValueError("graph_data.state_vector must have length n_vertices.")

        for node in range(self.graph_data.n_vertices):
            graph.add_node(node)
            graph.nodes[node]["local_index"] = int(node)
            graph.nodes[node]["label"] = labels[node]

            if original_indices is not None:
                graph.nodes[node]["original_index"] = int(original_indices[node])

            if state_vector is not None:
                amplitude = complex(state_vector[node])
                graph.nodes[node]["state_amplitude_real"] = float(amplitude.real)
                graph.nodes[node]["state_amplitude_imag"] = float(amplitude.imag)
                graph.nodes[node]["state_weight"] = float(abs(amplitude) ** 2)

        for row, col, value in zip(
            adjacency.row,
            adjacency.col,
            adjacency.data,
            strict=True,
        ):
            if not self.graph_data.directed and int(row) >= int(col):
                continue

            weight = complex(value)

            graph.add_edge(
                int(row),
                int(col),
                weight=1.0,
                hamiltonian_weight_real=float(weight.real),
                hamiltonian_weight_imag=float(weight.imag),
                hamiltonian_weight_abs=float(abs(weight)),
                hamiltonian_weight_phase=float(np.angle(weight)),
            )

        return graph

    def save_graph(
        self,
        path: str | Path,
        *,
        layout_backend: GraphBackend = "networkx",
        layout: LayoutName = "auto",
        color_by: NodeColorRule = "constant",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        **layout_kwargs,
    ) -> None:
        """Save graph with computed layout coordinates.

        The graph is always exported through NetworkX writers.

        Supported formats:
            .graphml
            .gexf

        The layout may be computed with either NetworkX or igraph.
        """
        import networkx as nx

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in {".graphml", ".gexf"}:
            raise ValueError("Graph export path must end with .graphml or .gexf.")

        graph = self.to_networkx_with_layout(
            layout_backend=layout_backend,
            layout=layout,
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            **layout_kwargs,
        )

        if suffix == ".graphml":
            # GraphML cannot serialize nested dict attributes like "viz".
            for node in graph.nodes:
                graph.nodes[node].pop("viz", None)

            nx.write_graphml(graph, path)
            return

        if suffix == ".gexf":
            nx.write_gexf(graph, path)
            return

    def to_networkx_with_layout(
        self,
        *,
        layout_backend: GraphBackend = "networkx",
        layout: LayoutName = "auto",
        color_by: NodeColorRule = "constant",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        automorphism_backend: AutomorphismBackend = "auto",
        **layout_kwargs,
    ):
        """Convert to NetworkX graph and attach computed layout coordinates.

        The graph is always returned as NetworkX, but the layout can be computed
        using either NetworkX or igraph.
        """
        graph = self.to_networkx()

        if layout_backend == "networkx":
            positions = _networkx_layout(
                graph,
                layout,
                **layout_kwargs,
            )
        elif layout_backend == "igraph":
            igraph_graph = self.to_igraph()
            igraph_layout = _igraph_layout(
                igraph_graph,
                layout,
                **layout_kwargs,
            )
            positions = {
                index: np.asarray(position, dtype=float)
                for index, position in enumerate(igraph_layout)
            }
        else:
            raise ValueError("layout_backend must be 'networkx' or 'igraph'.")

        color_values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            automorphism_backend=automorphism_backend,
        )

        degrees = self.graph_data.degrees
        self_loops = self.graph_data.self_loop_values
        orbit_labels = (
            self.automorphism_orbits(backend=automorphism_backend)
            if color_by == "automorphism_orbit"
            else None
        )
        original_indices = self.graph_data.original_indices

        for node in graph.nodes:
            position = np.asarray(positions[node], dtype=float)

            graph.nodes[node]["x"] = float(position[0])
            graph.nodes[node]["y"] = float(position[1])
            graph.nodes[node]["color_value"] = float(color_values[node])
            graph.nodes[node]["degree"] = int(degrees[node])
            graph.nodes[node]["self_loop_real"] = float(np.real(self_loops[node]))
            graph.nodes[node]["self_loop_imag"] = float(np.imag(self_loops[node]))
            if orbit_labels is not None:
                graph.nodes[node]["automorphism_orbit"] = int(orbit_labels[node])
            if original_indices is not None:
                graph.nodes[node]["original_index"] = int(original_indices[node])

            # Gephi GEXF reads this "viz" block as actual node position.
            graph.nodes[node]["viz"] = {
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": 0.0,
                }
            }

        for _source, _target, data in graph.edges(data=True):
            value = data.get("hamiltonian_weight", data.get("weight", 1.0))

            # NetworkX GraphML/GEXF writers cannot safely serialize complex values.
            data.pop("hamiltonian_weight", None)

            data["weight"] = float(abs(value))
            data["hamiltonian_weight_real"] = float(np.real(value))
            data["hamiltonian_weight_imag"] = float(np.imag(value))

        return graph

    def to_igraph_with_layout(
        self,
        *,
        layout: LayoutName = "auto",
        color_by: NodeColorRule = "constant",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        automorphism_backend: AutomorphismBackend = "auto",
        **layout_kwargs,
    ):
        """Convert to an igraph graph and attach computed layout coordinates."""
        graph = self.to_igraph()

        layout_object = _igraph_layout(
            graph,
            layout,
            **layout_kwargs,
        )

        color_values = self.node_values(
            color_by=color_by,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            automorphism_backend=automorphism_backend,
        )

        degrees = self.graph_data.degrees
        self_loops = self.graph_data.self_loop_values

        for vertex_index, position in enumerate(layout_object):
            graph.vs[vertex_index]["x"] = float(position[0])
            graph.vs[vertex_index]["y"] = float(position[1])
            graph.vs[vertex_index]["color_value"] = float(color_values[vertex_index])
            graph.vs[vertex_index]["degree"] = int(degrees[vertex_index])
            graph.vs[vertex_index]["self_loop_real"] = float(np.real(self_loops[vertex_index]))
            graph.vs[vertex_index]["self_loop_imag"] = float(np.imag(self_loops[vertex_index]))
            if self.graph_data.original_indices is not None:
                graph.vs[vertex_index]["original_index"] = int(
                    self.graph_data.original_indices[vertex_index]
                )

        # Attach edge weights from adjacency.
        weight_by_edge = {}
        adjacency = self.graph_data.adjacency.tocoo()
        for row, col, value in zip(adjacency.row, adjacency.col, adjacency.data, strict=True):
            source = int(row)
            target = int(col)
            if source < target:
                weight_by_edge[(source, target)] = complex(value)

        for edge in graph.es:
            source = int(edge.source)
            target = int(edge.target)
            key = (source, target) if source < target else (target, source)
            value = weight_by_edge.get(key, 0.0)
            edge["weight"] = float(abs(value))
            edge["hamiltonian_weight_real"] = float(np.real(value))
            edge["hamiltonian_weight_imag"] = float(np.imag(value))

        return graph

    def save_plot(
        self,
        path: str | Path,
        *,
        backend: GraphBackend = "igraph-cairo",
        color_by: NodeColorRule = "constant",
        layout: LayoutName = "auto",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        title: str | None = None,
        bbox: tuple[int, int] = (800, 800),
        margin: int = 40,
        **layout_kwargs,
    ) -> Path:
        """Save a graph visualization to disk.

        For ``backend='igraph-cairo'``, the plot is rendered directly by
        igraph/Cairo to ``path``.

        For Matplotlib-based backends, the plot is drawn on a Matplotlib figure
        and saved with ``fig.savefig(path)``.
        """
        path = Path(path)
        normalized_backend = _normalize_graph_backend(backend)

        if normalized_backend == "igraph-cairo":
            result = self.plot(
                backend="igraph-cairo",
                target=path,
                color_by=color_by,
                layout=layout,
                self_loop_values=self_loop_values,
                state_vector=state_vector,
                title=title,
                show=False,
                bbox=bbox,
                margin=margin,
                **layout_kwargs,
            )

            return Path(result)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.style.figure_size)

        self.plot(
            backend=normalized_backend,
            ax=ax,
            color_by=color_by,
            layout=layout,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            title=title,
            show=False,
            **layout_kwargs,
        )

        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        return path

    def save(
        self,
        path: str | Path,
        *,
        backend: GraphBackend = "igraph-cairo",
        color_by: NodeColorRule = "constant",
        layout: LayoutName = "auto",
        self_loop_values: npt.ArrayLike | None = None,
        state_vector: npt.ArrayLike | None = None,
        title: str | None = None,
        bbox: tuple[int, int] = (800, 800),
        margin: int = 40,
        **layout_kwargs,
    ) -> Path:
        """Alias for :meth:`save_plot`."""
        return self.save_plot(
            path,
            backend=backend,
            color_by=color_by,
            layout=layout,
            self_loop_values=self_loop_values,
            state_vector=state_vector,
            title=title,
            bbox=bbox,
            margin=margin,
            **layout_kwargs,
        )

    def automorphism_orbits(
        self,
        *,
        backend: AutomorphismBackend = "auto",
    ) -> np.ndarray:
        """Return dense vertex-orbit labels under graph automorphisms.

        The returned array has shape ``(n_vertices,)`` and integer labels
        ``0, 1, ..., n_orbits - 1``.
        """
        if backend == "auto":
            try:
                return self._automorphism_orbits_pynauty()
            except ImportError:
                return self._automorphism_orbits_igraph()

        if backend == "pynauty":
            return self._automorphism_orbits_pynauty()

        if backend == "igraph":
            return self._automorphism_orbits_igraph()

        raise ValueError("backend must be 'auto', 'pynauty', or 'igraph'.")

    @staticmethod
    def _dense_orbit_labels(raw_orbits: npt.ArrayLike) -> np.ndarray:
        """Convert arbitrary orbit representatives to dense labels."""
        raw = np.asarray(raw_orbits, dtype=np.int64)

        representative_to_label: dict[int, int] = {}
        labels = np.empty_like(raw)

        for i, representative in enumerate(raw):
            key = int(representative)

            if key not in representative_to_label:
                representative_to_label[key] = len(representative_to_label)

            labels[i] = representative_to_label[key]

        return labels

    def _automorphism_orbits_pynauty(self) -> np.ndarray:
        """Compute vertex orbits with pynauty."""
        try:
            import pynauty
        except ImportError as error:
            raise ImportError("automorphism_backend='pynauty' requires pynauty.") from error

        adjacency = self.graph_data.adjacency.tocsr()
        n_vertices = int(adjacency.shape[0])

        adjacency_dict: dict[int, list[int]] = {}

        for vertex in range(n_vertices):
            start = int(adjacency.indptr[vertex])
            end = int(adjacency.indptr[vertex + 1])
            neighbors = sorted(
                int(neighbor)
                for neighbor in adjacency.indices[start:end]
                if int(neighbor) != vertex
            )
            adjacency_dict[vertex] = neighbors

        graph = pynauty.Graph(
            number_of_vertices=n_vertices,
            directed=False,
            adjacency_dict=adjacency_dict,
        )

        # pynauty.autgrp returns group data; the fourth item is the orbit array.
        # This mirrors pynauty's documented/common usage: pynauty.autgrp(g)[3].
        raw_orbits = pynauty.autgrp(graph)[3]

        return self._dense_orbit_labels(raw_orbits)

    def _automorphism_orbits_igraph(self) -> np.ndarray:
        """Compute vertex orbits with igraph by enumerating automorphisms."""
        try:
            import igraph as ig  # noqa: F401
        except ImportError as error:
            raise ImportError("automorphism_backend='igraph' requires python-igraph.") from error

        graph = self.to_igraph()
        n_vertices = int(graph.vcount())

        parent = np.arange(n_vertices, dtype=np.int64)

        def find(x: int) -> int:
            while int(parent[x]) != x:
                parent[x] = parent[int(parent[x])]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            if root_b < root_a:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a

        for automorphism in graph.get_automorphisms_vf2():
            for source, target in enumerate(automorphism):
                union(int(source), int(target))

        raw_orbits = np.asarray([find(i) for i in range(n_vertices)], dtype=np.int64)
        return self._dense_orbit_labels(raw_orbits)

    def _edge_colors_for_matplotlib(
        self,
        *,
        edge_color_by: EdgeColorRule,
    ) -> str | list:
        if edge_color_by == "constant":
            return self.style.edge_color

        edge_values = self.edge_values(color_by=edge_color_by)

        if edge_color_by == "weight_complex":
            return _complex_edge_weights_to_rgba(
                edge_values,
                min_alpha=self.style.edge_complex_min_alpha,
                max_alpha=self.style.edge_complex_max_alpha,
            )

        cmap = (
            self.style.edge_phase_cmap if edge_color_by == "weight_phase" else self.style.edge_cmap
        )

        return _edge_scalar_values_to_colors(
            edge_values,
            cmap=cmap,
            color_by=edge_color_by,
        )

    def _edge_colors_for_igraph_cairo(
        self,
        *,
        edge_color_by: EdgeColorRule,
    ) -> str | list[str]:
        if edge_color_by == "constant":
            return self.style.edge_color

        edge_values = self.edge_values(color_by=edge_color_by)

        if edge_color_by == "weight_complex":
            return _complex_edge_weights_to_hex_colors(edge_values)

        cmap = (
            self.style.edge_phase_cmap if edge_color_by == "weight_phase" else self.style.edge_cmap
        )

        return _edge_scalar_values_to_hex_colors(
            edge_values,
            cmap=cmap,
            color_by=edge_color_by,
        )

    def _edge_adjacency_coo(self):
        if self.graph_data.directed:
            return self.graph_data.adjacency.tocoo()

        return sp.triu(
            self.graph_data.adjacency,
            k=1,
            format="coo",
        )

    def edge_weights(self) -> npt.NDArray[np.complex128]:
        """Return edge weights in plotting edge order."""
        adjacency = self._edge_adjacency_coo()
        return np.asarray(adjacency.data, dtype=np.complex128)

    def edge_pairs(self) -> list[tuple[int, int]]:
        """Return edge pairs in plotting edge order."""
        adjacency = self._edge_adjacency_coo()

        return [(int(row), int(col)) for row, col in zip(adjacency.row, adjacency.col, strict=True)]

    def n_edges(self) -> int:
        return len(self.edge_pairs())

    def _resolve_state_vector(
        self,
        state_vector: npt.ArrayLike | None,
    ) -> npt.NDArray[np.complex128]:
        if state_vector is not None:
            vector = np.asarray(state_vector, dtype=np.complex128)
        elif self.graph_data.state_vector is not None:
            vector = np.asarray(self.graph_data.state_vector, dtype=np.complex128)
        else:
            raise ValueError("A state_vector is required for state-based node coloring.")

        _validate_node_values(vector, self.graph_data.n_vertices)
        return vector

    @classmethod
    def cage_subgraph_from_sparse_matrix(
        cls,
        matrix,
        state_vector: npt.ArrayLike,
        *,
        zero_indices: Sequence[int] | None = None,
        classification_report=None,
        support_tolerance: float = 1.0e-10,
        include_zero_edges: bool = True,
        include_self_loops: bool = False,
        weight_tolerance: float = 0.0,
        style: HamiltonianGraphStyle | None = None,
    ) -> HamiltonianGraphVisualizer:
        """Build a cage-support-plus-zero subgraph visualizer from a sparse matrix."""
        full = cls.from_sparse_matrix(
            matrix,
            include_self_loops=include_self_loops,
            weight_tolerance=weight_tolerance,
            style=style,
        )

        return full.subgraph_for_cage_state(
            state_vector,
            zero_indices=zero_indices,
            classification_report=classification_report,
            support_tolerance=support_tolerance,
            include_zero_edges=include_zero_edges,
        )

    def subgraph_for_cage_state(
        self,
        state_vector: npt.ArrayLike,
        *,
        zero_indices: Sequence[int] | None = None,
        classification_report=None,
        support_tolerance: float = 1.0e-10,
        include_zero_edges: bool = True,
    ) -> HamiltonianGraphVisualizer:
        """Return the graph induced by a cage support plus nontrivial zeros.

        Parameters
        ----------
        state_vector:
            Full Hilbert-space vector in the same basis as this graph.
        zero_indices:
            Optional explicit nontrivial-zero node indices.
        classification_report:
            Optional caging classification report. If supplied, this method tries
            to extract zero indices from common report fields.
        support_tolerance:
            Nodes with |psi_i| > support_tolerance are included as cage support.
        include_zero_edges:
            If True, keep all induced edges among support and zero nodes.
            If False, keep only edges incident to at least one support node.
        """
        vector = np.asarray(state_vector, dtype=np.complex128)
        _validate_node_values(vector, self.graph_data.n_vertices)

        support_indices = np.flatnonzero(np.abs(vector) > support_tolerance).astype(
            np.int64,
        )

        extracted_zero_indices = self._extract_cage_zero_indices(
            zero_indices=zero_indices,
            classification_report=classification_report,
        )

        selected_indices = np.unique(
            np.concatenate(
                [
                    support_indices,
                    extracted_zero_indices,
                ]
            )
        ).astype(np.int64)

        if selected_indices.size == 0:
            raise ValueError("No cage support or zero indices were selected.")

        adjacency = self.graph_data.adjacency.tocsr()
        sub_adjacency = adjacency[selected_indices, :][:, selected_indices].copy()

        if not include_zero_edges:
            support_mask = np.isin(selected_indices, support_indices)
            edge_coo = sub_adjacency.tocoo()

            keep = support_mask[edge_coo.row] | support_mask[edge_coo.col]

            sub_adjacency = sp.csr_array(
                (
                    edge_coo.data[keep],
                    (edge_coo.row[keep], edge_coo.col[keep]),
                ),
                shape=sub_adjacency.shape,
            )

        sub_self_loops = self.graph_data.self_loop_values[selected_indices]
        sub_state_vector = vector[selected_indices]

        region_labels = np.zeros(selected_indices.size, dtype=np.int64)
        region_labels[np.isin(selected_indices, support_indices)] = 1
        region_labels[np.isin(selected_indices, extracted_zero_indices)] = 2

        # If a node is both in support and zero_indices, support wins.
        region_labels[np.isin(selected_indices, support_indices)] = 1

        return HamiltonianGraphVisualizer(
            graph_data=HamiltonianGraphData(
                adjacency=sp.csr_array(sub_adjacency),
                self_loop_values=np.asarray(sub_self_loops, dtype=np.complex128),
                original_indices=selected_indices.astype(np.int64),
                state_vector=np.asarray(sub_state_vector, dtype=np.complex128),
            ),
            style=self.style,
        )

    @staticmethod
    def _extract_cage_zero_indices(
        *,
        zero_indices: Sequence[int] | None,
        classification_report,
    ) -> npt.NDArray[np.int64]:
        """Extract nontrivial-zero node indices from explicit input or report."""
        indices: list[int] = []

        if zero_indices is not None:
            indices.extend(int(index) for index in zero_indices)

        if classification_report is not None:
            candidate_field_names = (
                "nontrivial_zero_indices",
                "zero_indices",
                "known_zero_indices",
                "q_empty_zero_indices",
                "closed_by_known_zeros_indices",
                "domain_blocked_zero_indices",
                "projector_like_zero_indices",
                "collective_cancellation_zero_indices",
            )

            for field_name in candidate_field_names:
                if not hasattr(classification_report, field_name):
                    continue

                value = getattr(classification_report, field_name)

                if value is None:
                    continue

                indices.extend(_flatten_int_indices(value))

            # Fallback for report.zero_reports, if present.
            zero_reports = getattr(classification_report, "zero_reports", None)
            if zero_reports is not None:
                for zero_report in zero_reports:
                    zero_index = getattr(zero_report, "zero_index", None)
                    if zero_index is not None:
                        indices.append(int(zero_index))

        if len(indices) == 0:
            return np.asarray([], dtype=np.int64)

        return np.unique(np.asarray(indices, dtype=np.int64))


def _as_csr_array(matrix) -> sp.csr_array:
    """Convert dense or sparse input to ``csr_array``."""
    if sp.issparse(matrix):
        return sp.csr_array(matrix)

    return sp.csr_array(np.asarray(matrix, dtype=np.complex128))


def _symmetrized_adjacency(
    matrix: sp.csr_array,
) -> sp.csr_array:
    """Return an unweighted undirected adjacency matrix."""
    adjacency = sp.csr_array(matrix.copy())
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency.tocsr()


def _symmetrized_weighted_adjacency(
    matrix: sp.csr_array,
) -> sp.csr_array:
    """Return weighted undirected adjacency preserving complex edge weights.

    For Hermitian matrices, the upper/lower entries are conjugates. We keep
    the upper-triangular representative and mirror it, so each undirected
    edge has one stable complex orientation convention.
    """
    matrix = sp.csr_array(matrix.copy())
    matrix.setdiag(0)
    matrix.eliminate_zeros()

    upper = sp.triu(matrix, k=1, format="csr")
    lower = sp.tril(matrix, k=-1, format="csr")

    # Prefer upper-triangular entries. If an edge exists only in lower,
    # use its conjugate as the upper-oriented value.
    upper_from_lower = lower.T.conjugate()
    upper_bool = upper.astype(bool)
    upper_present = upper_from_lower.multiply(upper_bool.astype(upper_from_lower.dtype, copy=False))
    missing_from_upper = upper_from_lower - upper_present
    missing_from_upper.eliminate_zeros()

    upper_combined = upper + missing_from_upper
    adjacency = upper_combined + upper_combined.T.conjugate()

    adjacency.eliminate_zeros()
    return sp.csr_array(adjacency)


def _validate_node_values(
    values: npt.ArrayLike,
    n_vertices: int,
) -> None:
    values_array = np.asarray(values)

    if values_array.ndim != 1:
        raise ValueError("Node values must be one-dimensional.")

    if values_array.shape[0] != n_vertices:
        raise ValueError(f"Expected {n_vertices} node values, got {values_array.shape[0]}.")


def _flatten_int_indices(value) -> list[int]:
    """Flatten nested integer-like containers into a list of ints."""
    if value is None:
        return []

    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, np.ndarray):
        return [int(x) for x in value.ravel().tolist()]

    if isinstance(value, dict):
        out: list[int] = []
        for item in value.values():
            out.extend(_flatten_int_indices(item))
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[int] = []
        for item in value:
            out.extend(_flatten_int_indices(item))
        return out

    return []


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
    color_by: NodeColorRule,
) -> list:
    """Convert scalar values to matplotlib colors."""
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return []

    norm = _node_color_norm(values, color_by=color_by)
    colormap = plt.get_cmap(cmap)

    return [colormap(float(norm(value))) for value in values]


def _scalar_values_to_hex_colors(
    values: npt.ArrayLike,
    *,
    cmap: str,
    color_by: NodeColorRule,
) -> list[str]:
    """Convert scalar values to hex color strings."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return []

    norm = _node_color_norm(values, color_by=color_by)
    colormap = plt.get_cmap(cmap)

    return [to_hex(colormap(float(norm(value)))) for value in values]


def _edge_scalar_values_to_hex_colors(
    values: npt.ArrayLike,
    *,
    cmap: str,
    color_by: EdgeColorRule,
) -> list[str]:
    """Convert scalar edge values to hex color strings."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return []

    norm = _edge_color_norm(values, color_by=color_by)
    colormap = plt.get_cmap(cmap)

    return [to_hex(colormap(float(norm(value)))) for value in values]


def _add_colorbar(
    *,
    ax,
    values: npt.ArrayLike,
    cmap: str,
    label: str,
    color_by: NodeColorRule,
) -> None:
    """Attach a scalar colorbar to an axis."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or np.allclose(values, values[0]):
        return

    norm = _node_color_norm(values, color_by=color_by)

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


def _add_edge_colorbar(
    *,
    ax,
    values: npt.ArrayLike,
    cmap: str,
    label: str,
    color_by: EdgeColorRule,
) -> None:
    """Attach a scalar edge-color colorbar to an axis."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or np.allclose(values, values[0]):
        return

    norm = _edge_color_norm(values, color_by=color_by)

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


def _add_orbit_legend(
    *,
    ax,
    labels: npt.ArrayLike,
    colors: Sequence[str],
    max_entries: int = 12,
) -> None:
    """Add a compact orbit legend for small orbit counts."""
    import matplotlib.patches as mpatches

    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = sorted(int(label) for label in np.unique(labels))

    if len(unique_labels) > max_entries:
        return

    label_to_color = {int(label): color for label, color in zip(labels, colors, strict=True)}

    handles = [
        mpatches.Patch(
            color=label_to_color[label],
            label=f"orbit {label}",
        )
        for label in unique_labels
    ]

    ax.legend(
        handles=handles,
        loc="best",
        fontsize="small",
        frameon=False,
    )


def _uses_zero_centered_colormap(color_by: NodeColorRule) -> bool:
    """Return whether node colors should be centered at zero."""
    return color_by in {
        "state_amplitude_real",
        "state_amplitude_imag",
    }


def _uses_zero_centered_edge_colormap(color_by: EdgeColorRule) -> bool:
    """Return whether edge colors should be centered at zero."""
    return color_by in {
        "weight_real",
        "weight_imag",
    }


def _node_color_norm(
    values: npt.ArrayLike,
    *,
    color_by: NodeColorRule,
):
    """Return a Matplotlib norm for node colors."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    if np.allclose(values, values[0]):
        value = float(values[0])

        if _uses_zero_centered_colormap(color_by):
            scale = max(abs(value), 1.0)
            return TwoSlopeNorm(
                vmin=-scale,
                vcenter=0.0,
                vmax=scale,
            )

        return Normalize(vmin=value - 0.5, vmax=value + 0.5)

    if _uses_zero_centered_colormap(color_by):
        scale = float(np.max(np.abs(values)))

        if scale == 0.0:
            scale = 1.0

        return TwoSlopeNorm(
            vmin=-scale,
            vcenter=0.0,
            vmax=scale,
        )

    return Normalize(
        vmin=float(np.min(values)),
        vmax=float(np.max(values)),
    )


def _edge_color_norm(
    values: npt.ArrayLike,
    *,
    color_by: EdgeColorRule,
):
    """Return a Matplotlib norm for edge colors."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    if color_by == "weight_phase":
        return Normalize(vmin=-np.pi, vmax=np.pi)

    if np.allclose(values, values[0]):
        value = float(values[0])

        if _uses_zero_centered_edge_colormap(color_by):
            scale = max(abs(value), 1.0)
            return TwoSlopeNorm(
                vmin=-scale,
                vcenter=0.0,
                vmax=scale,
            )

        return Normalize(vmin=value - 0.5, vmax=value + 0.5)

    if _uses_zero_centered_edge_colormap(color_by):
        scale = float(np.max(np.abs(values)))
        if scale == 0.0:
            scale = 1.0

        return TwoSlopeNorm(
            vmin=-scale,
            vcenter=0.0,
            vmax=scale,
        )

    return Normalize(
        vmin=float(np.min(values)),
        vmax=float(np.max(values)),
    )


def _complex_edge_weights_to_rgba(
    weights: npt.ArrayLike,
    *,
    min_alpha: float = 0.20,
    max_alpha: float = 0.95,
) -> list[tuple[float, float, float, float]]:
    """Map complex edge weights to RGBA.

    phase -> hue
    relative magnitude -> alpha
    """
    import colorsys

    weights = np.asarray(weights, dtype=np.complex128)

    if weights.size == 0:
        return []

    magnitudes = np.abs(weights)
    max_magnitude = float(np.max(magnitudes))

    if max_magnitude <= 0.0:
        relative = np.zeros_like(magnitudes, dtype=np.float64)
    else:
        relative = magnitudes / max_magnitude

    colors: list[tuple[float, float, float, float]] = []

    for weight, magnitude_ratio in zip(weights, relative, strict=True):
        phase = float(np.angle(weight))
        hue = (phase + np.pi) / (2.0 * np.pi)

        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.90)

        alpha = min_alpha + (max_alpha - min_alpha) * float(magnitude_ratio)
        colors.append((red, green, blue, alpha))

    return colors


def _edge_scalar_values_to_colors(
    values: npt.ArrayLike,
    *,
    cmap: str,
    color_by: EdgeColorRule,
) -> list:
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return []

    norm = _edge_color_norm(values, color_by=color_by)
    colormap = plt.get_cmap(cmap)

    return [colormap(float(norm(value))) for value in values]


def _complex_edge_weights_to_hex_colors(
    weights: npt.ArrayLike,
) -> list[str]:
    from matplotlib.colors import to_hex

    return [to_hex(color, keep_alpha=False) for color in _complex_edge_weights_to_rgba(weights)]


def _orbit_labels_to_hex_colors(
    labels: npt.ArrayLike,
    *,
    lightness_boost: float = 0.0,
) -> list[str]:
    """Return visually separated hex colors for integer orbit labels."""
    import colorsys

    from matplotlib.colors import to_hex

    labels = np.asarray(labels, dtype=np.int64)

    if labels.size == 0:
        return []

    unique_labels = sorted(int(label) for label in np.unique(labels))
    label_to_color: dict[int, str] = {}

    golden_ratio_conjugate = 0.618033988749895

    for rank, label in enumerate(unique_labels):
        hue = (rank * golden_ratio_conjugate) % 1.0

        saturation = 0.70 if rank % 2 == 0 else 0.95
        lightness = 0.52 if (rank // 2) % 2 == 0 else 0.68

        lightness = min(0.95, lightness + lightness_boost)

        red, green, blue = colorsys.hls_to_rgb(
            hue,
            lightness,
            saturation,
        )
        label_to_color[label] = to_hex((red, green, blue))

    return [label_to_color[int(label)] for label in labels]


def _orbit_labels_to_rgba_colors(
    labels: npt.ArrayLike,
    *,
    alpha: float,
) -> list[tuple[float, float, float, float]]:
    """Return visually separated RGBA colors for integer orbit labels."""
    from matplotlib.colors import to_rgba

    hex_colors = _orbit_labels_to_hex_colors(labels)

    return [to_rgba(color, alpha=alpha) for color in hex_colors]


def _igraph_layout(graph, layout: LayoutName, **kwargs):
    """Return an igraph layout."""
    if layout == "auto":
        return graph.layout_auto(**kwargs)

    aliases = {
        "fr": "fruchterman_reingold",
        "kk": "kamada_kawai",
        "grid_fr": "grid_fruchterman_reingold",
    }

    layout_name = aliases.get(layout, layout)

    try:
        return graph.layout(layout_name, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Unsupported igraph layout: {layout!r}. "
            "Pass any layout accepted by igraph.Graph.layout(...)."
        ) from exc


def _networkx_layout(graph, layout: LayoutName, **kwargs):
    """Return networkx positions."""
    import networkx as nx

    if layout in ("auto", "spring", "fr", "grid_fr"):
        kwargs.setdefault("seed", 0)
        return nx.spring_layout(graph, **kwargs)

    if layout in ("kk", "kamada_kawai"):
        return nx.kamada_kawai_layout(graph, **kwargs)

    if layout == "circle":
        return nx.circular_layout(graph, **kwargs)

    raise ValueError(f"Unsupported networkx layout: {layout!r}")


def _ensure_cairo_available() -> None:
    """Raise an informative error if no Cairo Python binding is available."""
    try:
        import cairo  # noqa: F401

        return
    except ImportError:
        pass

    try:
        import cairocffi  # noqa: F401

        return
    except ImportError as error:
        raise ImportError(
            "The igraph-cairo backend requires a Cairo Python binding. "
            "Install either pycairo or cairocffi."
        ) from error


def _normalize_graph_backend(backend: GraphBackend) -> NormalizedGraphBackend:
    """Normalize graph plotting backend names.

    ``"igraph"`` is kept as a backward-compatible alias for ``"igraph-mpl"``.
    """
    if backend == "igraph":
        return "igraph-mpl"

    if backend in ("igraph-mpl", "igraph-cairo", "networkx"):
        return cast(NormalizedGraphBackend, backend)

    raise ValueError(
        "backend must be 'networkx', 'igraph-mpl', or 'igraph-cairo'. "
        "'igraph' is accepted as an alias for 'igraph-mpl'."
    )
