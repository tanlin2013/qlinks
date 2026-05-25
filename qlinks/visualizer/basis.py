from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from qlinks.lattice import (
    BoundaryCondition,
    ChainLattice,
    HoneycombLattice,
    LatticeGraph,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import VariableKind, VariableLayout

BasisConfigLabelStyle = Literal["none", "compact", "array"]
LinkPlotMode = Literal["arrows", "dimers", "values"]
PeriodicImageMode = Literal["none", "positive_patch"]
PlaquetteSymbolMode = Literal["binary", "flux"]
PlaquetteSymbolStyle = Literal["none", "square_qlm", "circulation", "resonance"]
SiteLabelStyle = Literal["cell", "cell_sublattice", "sublattice_cell", "site_id"]
VisualizerBackend = Literal["matplotlib", "networkx"]

# This mapping is copied in spirit from the old square-lattice visualizer.
# Keys are plaquette-link values converted to binary signs in plaquette order.
_SQUARE_QLM_PLAQUETTE_SYMBOLS: dict[str, dict[str, str]] = {
    "1111": {"s": "◩", "color": "silver"},
    "1011": {"s": "↑", "color": "skyblue"},
    "0111": {"s": "→", "color": "salmon"},
    "0011": {"s": "♰", "color": "silver"},
    "1101": {"s": "↓", "color": "salmon"},
    "1001": {"s": "⬔", "color": "silver"},
    "0101": {"s": "↻", "color": "red"},
    "0001": {"s": "←", "color": "salmon"},
    "1110": {"s": "←", "color": "skyblue"},
    "1010": {"s": "↺", "color": "blue"},
    "0110": {"s": "⬕", "color": "silver"},
    "0010": {"s": "↓", "color": "skyblue"},
    "1100": {"s": "♱", "color": "silver"},
    "1000": {"s": "→", "color": "skyblue"},
    "0100": {"s": "↑", "color": "salmon"},
    "0000": {"s": "◪", "color": "silver"},
}


@dataclass(frozen=True, slots=True)
class LinkVisualStyle:
    """
    Basic visual style for link drawing.
    """

    node_size: float = 180.0
    node_color: str = "tab:orange"
    edge_color: str = "black"
    empty_edge_color: str = "lightgray"

    arrow_linewidth: float = 1.1
    arrow_alpha: float = 0.85
    arrow_mutation_scale: float | None = None
    arrow_shrink_points: float | None = None

    occupied_width: float = 2.0
    empty_width: float = 0.8
    occupied_alpha: float = 0.9
    empty_alpha: float = 0.5

    site_label_fontsize: float | None = None
    link_label_fontsize: float | None = None
    plaquette_symbol_fontsize: float = 22.0
    plaquette_symbol_offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class _DrawNode:
    key: tuple[int, tuple[int, ...]]
    site_id: int
    image_shift: tuple[int, ...]
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _DrawLink:
    link_id: int
    source_key: tuple[int, tuple[int, ...]]
    target_key: tuple[int, tuple[int, ...]]
    source_site: int
    target_site: int
    source_position: tuple[float, float]
    target_position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _DrawPlaquette:
    plaquette_id: int
    image_shift: tuple[int, ...]
    visual_cell: tuple[int, ...]
    center: npt.NDArray[np.float64]
    link_ids: tuple[int, ...] = ()
    link_orientations: tuple[int, ...] = ()


@dataclass(frozen=True)
class BasisConfigurationVisualizer:
    """
    Geometry-detached basis-configuration visualizer.

    Parameters
    ----------
    lattice:
        A LatticeGraph, e.g. ChainLattice or SquareLattice.

    layout:
        VariableLayout mapping site/link ids to positions in the raw config
        array. If None, the visualizer assumes link variable index == link id
        for link plotting.

    periodic_image_mode:
        How to draw links that wrap periodic boundaries.

        "none": Don't draw wrapped links.

        "positive": Draw wrapped links with the target site shifted by +1 period
        so they appear next to the source site. This is usually more intuitive
        for small lattices.

        "both": Draw both the original wrapped link and the complementary image
        link. This is more complete but can be visually cluttered.
    """

    lattice: LatticeGraph
    layout: VariableLayout | None = None
    style: LinkVisualStyle = field(default_factory=LinkVisualStyle)
    periodic_image_mode: PeriodicImageMode = "positive_patch"
    collapse_duplicate_visual_links: bool = True
    coordinate_scale: float = 1.0
    coordinate_transform: npt.NDArray[np.float64] | None = None
    site_label_style: SiteLabelStyle = "cell_sublattice"

    def _as_config(self, config: npt.ArrayLike) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)

        if arr.ndim != 1:
            raise ValueError("config must be one-dimensional.")

        if self.layout is not None:
            self.layout.validate_config(arr)
        elif arr.size < self.lattice.num_links:
            raise ValueError(
                "Without a VariableLayout, config must contain at least "
                f"{self.lattice.num_links} link values."
            )

        return arr

    def link_value(self, config: npt.ArrayLike, link_id: int) -> int:
        arr = self._as_config(config)

        if self.layout is None:
            return int(arr[link_id])

        variable_index = self.layout.variable_index(VariableKind.LINK, link_id)
        return int(arr[variable_index])

    def site_value(self, config: npt.ArrayLike, site_id: int) -> int | None:
        if self.layout is None:
            return None

        arr = self._as_config(config)

        try:
            variable_index = self.layout.variable_index(VariableKind.SITE, site_id)
        except KeyError:
            return None

        return int(arr[variable_index])

    def has_link_variables(self) -> bool:
        if self.layout is None:
            return True

        return self.layout.link_variable_indices().size > 0

    def plot(
        self,
        config: npt.ArrayLike,
        *,
        ax=None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        mode: LinkPlotMode = "arrows",
        with_site_labels: bool = True,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_style: PlaquetteSymbolStyle = "square_qlm",
        title: str | None = None,
    ):
        """
        Plot one basis configuration.

        Args:

            mode="arrows":
                QLM-like style. Positive / 1 values point along the stored link
                orientation. Negative / 0 values point opposite.

            mode="dimers":
                QDM-like style. Value 1 links are drawn thick; value 0 links are
                faint.

            mode="values":
                Draw the lattice and place link values at link centers.

            plaquette_symbol_style:

            "circulation": QLM-like signed-flux circulation marker.
             Draws circular arrows only when all nonzero signed link variables circulate
              consistently around a plaquette.

            "resonance": QDM-like binary resonance marker.
            Draws a marker when binary dimer occupations alternate around an even-length plaquette.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        draw_nodes, draw_links = self._draw_primitives()

        draw_plaquettes = None
        if with_plaquette_symbols and plaquette_symbol_style != "none":
            if plaquette_symbol_style == "square_qlm":
                draw_plaquettes = self._draw_square_qlm_plaquette_primitives()
            else:
                draw_plaquettes = self._draw_plaquette_primitives()

        return self._plot_with_primitives(
            config,
            ax=ax,
            draw_nodes=draw_nodes,
            draw_links=draw_links,
            draw_plaquettes=draw_plaquettes,
            show=show,
            backend=backend,
            mode=mode,
            with_site_labels=with_site_labels,
            with_site_values=with_site_values,
            with_link_values=with_link_values,
            with_plaquette_symbols=with_plaquette_symbols,
            plaquette_symbol_style=plaquette_symbol_style,
            title=title,
        )

    def _plot_with_primitives(
        self,
        config: npt.ArrayLike,
        *,
        ax,
        draw_nodes: list[_DrawNode],
        draw_links: list[_DrawLink],
        draw_plaquettes: list[_DrawPlaquette] | None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        mode: LinkPlotMode = "arrows",
        with_site_labels: bool = True,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_style: PlaquetteSymbolStyle = "square_qlm",
        title: str | None = None,
    ):
        import matplotlib.pyplot as plt

        arr = self._as_config(config)

        if mode in ("arrows", "dimers") and not self.has_link_variables():
            raise ValueError(
                f"mode='{mode}' requires link variables in the layout. "
                "For site-only layouts, use mode='values' with with_site_values=True."
            )

        if backend == "matplotlib":
            self._draw_links(
                ax=ax,
                config=arr,
                draw_links=draw_links,
                mode=mode,
            )
            self._draw_nodes(
                ax=ax,
                config=arr,
                draw_nodes=draw_nodes,
                with_site_labels=with_site_labels,
                with_site_values=with_site_values,
            )
            if (with_link_values or mode == "values") and self.has_link_variables():
                self._draw_link_values(
                    ax=ax,
                    config=arr,
                    draw_links=draw_links,
                )
            if with_plaquette_symbols and plaquette_symbol_style != "none":
                self._draw_plaquette_symbols(
                    ax=ax,
                    config=arr,
                    style=plaquette_symbol_style,
                    draw_plaquettes=draw_plaquettes or [],
                )
        else:
            # Keep current path for now, or refactor similarly later.
            self._draw_networkx(
                ax=ax,
                config=arr,
                draw_nodes=draw_nodes,
                draw_links=draw_links,
                draw_plaquettes=draw_plaquettes,
                mode=mode,
                with_site_labels=with_site_labels,
                with_site_values=with_site_values,
                with_link_values=with_link_values,
                with_plaquette_symbols=with_plaquette_symbols,
                plaquette_symbol_style=plaquette_symbol_style,
                title=None,
            )

        self._finish_axes(ax, title=title)

        if show:
            plt.show()

        return ax

    def save(
        self,
        config: npt.ArrayLike,
        path: str | Path,
        *,
        dpi: int = 200,
        show: bool = False,
        **plot_kwargs,
    ) -> None:
        """
        Save a visualization to disk.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self.plot(config, ax=ax, show=show, **plot_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _xy(position: tuple[float, ...]) -> tuple[float, float]:
        """
        Convert a lattice position to 2D plotting coordinates.

        1D:
            (x,) -> (x, 0)

        2D or higher:
            (x, y, ...) -> (x, y)
        """
        if len(position) == 1:
            return float(position[0]), 0.0

        if len(position) >= 2:
            return float(position[0]), float(position[1])

        raise ValueError("Position cannot be empty.")

    def _draw_networkx(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_nodes: list[_DrawNode],
        draw_links: list[_DrawLink],
        draw_plaquettes: list[_DrawPlaquette] | None,
        mode: LinkPlotMode,
        with_site_labels: bool,
        with_site_values: bool,
        with_link_values: bool,
        with_plaquette_symbols: bool,
        plaquette_symbol_style: PlaquetteSymbolStyle,
        title: str | None,
    ) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX backend requires networkx. Install it with `pip install networkx`."
            ) from exc

        if mode == "arrows":
            graph = nx.MultiDiGraph()
        else:
            graph = nx.MultiGraph()

        pos: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}

        for node in draw_nodes:
            graph.add_node(
                node.key,
                site_id=node.site_id,
            )
            pos[node.key] = self._xy(node.position)

        edge_records: list[
            tuple[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]], int, int]
        ] = []

        for link in draw_links:
            value = self.link_value(config, link.link_id)

            source_key = link.source_key
            target_key = link.target_key

            if mode == "arrows" and not self._points_along_link(value):
                source_key, target_key = target_key, source_key

            graph.add_edge(
                source_key,
                target_key,
                link_id=link.link_id,
                value=value,
            )

        node_colors = [self.style.node_color for _ in graph.nodes]

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=self.style.node_size,
            node_color=node_colors,
            linewidths=0.8,
            edgecolors="black",
        )

        if mode == "dimers":
            occupied_edges = []
            empty_edges = []

            for u, v, key, link_id in edge_records:
                value = self.link_value(config, link_id)
                if value != 0:
                    occupied_edges.append((u, v, key))
                else:
                    empty_edges.append((u, v, key))

            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=empty_edges,
                width=self.style.empty_width,
                edge_color=self.style.empty_edge_color,
                alpha=self.style.empty_alpha,
                arrows=False,
                connectionstyle="arc3,rad=0.0",
            )

            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=occupied_edges,
                width=self.style.occupied_width,
                edge_color=self.style.edge_color,
                alpha=self.style.occupied_alpha,
                arrows=False,
                connectionstyle="arc3,rad=0.0",
            )

        elif mode == "arrows":
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                width=self.style.arrow_linewidth,
                edge_color=self.style.edge_color,
                alpha=self.style.arrow_alpha,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=self._resolved_arrow_mutation_scale(),
                connectionstyle="arc3,rad=0.0",
                min_source_margin=self._resolved_arrow_shrink_points(),
                min_target_margin=self._resolved_arrow_shrink_points(),
            )

        elif mode == "values":
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                width=self.style.empty_width,
                edge_color=self.style.empty_edge_color,
                alpha=0.7,
                arrows=False,
                connectionstyle="arc3,rad=0.0",
            )

        else:
            raise ValueError("mode must be one of 'arrows', 'dimers', or 'values'.")

        if with_site_labels or with_site_values:
            labels: dict[tuple[int, tuple[int, ...]], str] = {}

            for node in draw_nodes:
                pieces: list[str] = []

                if with_site_labels:
                    pieces.append(self._format_site_label(node.site_id))

                if with_site_values:
                    value = self.site_value(config, node.site_id)
                    if value is not None:
                        pieces.append(f"{value}")

                if pieces:
                    labels[node.key] = "\n".join(pieces)

            nx.draw_networkx_labels(
                graph,
                pos,
                labels=labels,
                ax=ax,
                font_size=self._resolved_site_label_fontsize(),
                font_color="black",
            )

        if (with_link_values or mode == "values") and self.has_link_variables():
            edge_labels = {}

            for u, v, key, link_id in edge_records:
                value = self.link_value(config, link_id)
                edge_labels[(u, v, key)] = str(value)

            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=self._resolved_link_label_fontsize(),
                rotate=False,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
            )

        # Plaquette symbols are still drawn with the existing matplotlib overlay.
        # This keeps the old square-QLM symbols and generic circulation symbols
        # available for both backends.
        if with_plaquette_symbols and plaquette_symbol_style != "none":
            self._draw_plaquette_symbols(
                ax=ax,
                config=config,
                style=plaquette_symbol_style,
                draw_plaquettes=draw_plaquettes or [],
            )

        self._finish_axes(ax, title=title)

    def _draw_primitives(self) -> tuple[list[_DrawNode], list[_DrawLink]]:
        if (
            self.lattice.boundary_condition != BoundaryCondition.PERIODIC
            or self.periodic_image_mode == "none"
        ):
            return self._draw_primitives_open()

        if self.periodic_image_mode == "positive_patch":
            return self._draw_primitives_positive_patch()

        raise ValueError("periodic_image_mode must be 'none', or 'positive_patch'.")

    def _draw_primitives_open(self) -> tuple[list[_DrawNode], list[_DrawLink]]:
        zero_shift = tuple(0 for _ in range(self.lattice.ndim))
        period_vectors = self._period_vectors_2d()

        nodes: list[_DrawNode] = []
        node_by_key: dict[tuple[int, tuple[int, ...]], _DrawNode] = {}

        for site in self.lattice.sites:
            key = (int(site.id), zero_shift)
            position = self._visual_site_position(
                site_id=int(site.id),
                image_shift=zero_shift,
                period_vectors=period_vectors,
            )
            node = _DrawNode(
                key=key,
                site_id=int(site.id),
                image_shift=zero_shift,
                position=position,
            )
            nodes.append(node)
            node_by_key[key] = node

        links: list[_DrawLink] = []

        for link in self.lattice.links:
            source_key = (int(link.source), zero_shift)
            target_key = (int(link.target), zero_shift)

            source_node = node_by_key[source_key]
            target_node = node_by_key[target_key]

            links.append(
                _DrawLink(
                    link_id=int(link.id),
                    source_key=source_key,
                    target_key=target_key,
                    source_site=int(link.source),
                    target_site=int(link.target),
                    source_position=source_node.position,
                    target_position=target_node.position,
                )
            )

        return nodes, links

    def _draw_primitives_positive_patch(self) -> tuple[list[_DrawNode], list[_DrawLink]]:
        period_vectors = self._period_vectors_2d()
        node_image_shifts = self._positive_patch_node_image_shifts()
        link_source_shifts = self._positive_patch_link_source_shifts()

        nodes: list[_DrawNode] = []
        node_by_key: dict[tuple[int, tuple[int, ...]], _DrawNode] = {}

        def add_node(
            *,
            site_id: int,
            image_shift: tuple[int, ...],
        ) -> _DrawNode:
            key = (int(site_id), tuple(int(x) for x in image_shift))

            existing = node_by_key.get(key)
            if existing is not None:
                return existing

            position = self._visual_site_position(
                site_id=int(site_id),
                image_shift=image_shift,
                period_vectors=period_vectors,
            )

            node = _DrawNode(
                key=key,
                site_id=int(site_id),
                image_shift=image_shift,
                position=position,
            )

            node_by_key[key] = node
            nodes.append(node)

            return node

        # Add all sites in the positive patch:
        #
        #   1D: 0 <= cell <= L
        #   2D: 0 <= cell_x <= Lx, 0 <= cell_y <= Ly
        #
        # This includes the upper-right corner image.
        for image_shift in node_image_shifts:
            for site in self.lattice.sites:
                visual_cell = self._visual_cell(
                    site_id=int(site.id),
                    image_shift=image_shift,
                )

                if not self._is_visual_cell_in_positive_patch(visual_cell):
                    continue

                add_node(
                    site_id=int(site.id),
                    image_shift=image_shift,
                )

        links: list[_DrawLink] = []

        # Lift each physical link into the visual positive patch.
        for source_shift in link_source_shifts:
            for link in self.lattice.links:
                source_visual_cell = self._visual_cell(
                    site_id=int(link.source),
                    image_shift=source_shift,
                )

                if not self._is_visual_cell_in_positive_patch(source_visual_cell):
                    continue

                displacement = self._link_cell_displacement(link)

                target_visual_cell = tuple(
                    int(source_visual_cell[d]) + int(displacement[d])
                    for d in range(self.lattice.ndim)
                )

                if not self._is_visual_cell_in_positive_patch(target_visual_cell):
                    continue

                target_shift = self._image_shift_for_visual_cell(
                    site_id=int(link.target),
                    visual_cell=target_visual_cell,
                )

                if target_shift is None:
                    continue

                source_key = (int(link.source), source_shift)
                target_key = (int(link.target), target_shift)

                if self._should_skip_positive_patch_visual_link(
                    link=link,
                    source_key=source_key,
                    target_key=target_key,
                    source_visual_cell=source_visual_cell,
                    target_visual_cell=target_visual_cell,
                ):
                    continue

                source_node = node_by_key.get(source_key)
                if source_node is None:
                    source_node = add_node(
                        site_id=int(link.source),
                        image_shift=source_shift,
                    )

                target_node = node_by_key.get(target_key)
                if target_node is None:
                    target_node = add_node(
                        site_id=int(link.target),
                        image_shift=target_shift,
                    )

                links.append(
                    _DrawLink(
                        link_id=int(link.id),
                        source_key=source_key,
                        target_key=target_key,
                        source_site=int(link.source),
                        target_site=int(link.target),
                        source_position=source_node.position,
                        target_position=target_node.position,
                    )
                )

        # Keep only base nodes plus image nodes touched by links.
        nodes, links = self._remove_unused_image_nodes(nodes, links)

        if self.collapse_duplicate_visual_links:
            links = self._collapse_duplicate_visual_links(links)

        return nodes, links

    def _positive_patch_image_shifts(self) -> tuple[tuple[int, ...], ...]:
        ndim = self.lattice.ndim

        if ndim == 1:
            return ((0,), (1,))

        if ndim == 2:
            return (
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
            )

        raise NotImplementedError(
            "positive_patch visualization currently supports 1D and 2D lattices."
        )

    def _site_plot_position(self, site_id: int) -> tuple[float, ...]:
        if hasattr(self.lattice, "site_embedded_position"):
            return tuple(self.lattice.site_embedded_position(site_id))

        return tuple(self.lattice.site_positions[site_id])

    def _visual_site_position(
        self,
        *,
        site_id: int,
        image_shift: tuple[int, ...],
        period_vectors: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        xy = np.asarray(
            self._xy(self._site_plot_position(site_id)),
            dtype=float,
        )

        for dim, shift in enumerate(image_shift):
            xy = xy + int(shift) * period_vectors[dim]

        xy = self.coordinate_scale * xy

        if self.coordinate_transform is not None:
            transform = np.asarray(self.coordinate_transform, dtype=float)
            if transform.shape != (2, 2):
                raise ValueError("coordinate_transform must have shape (2, 2).")
            xy = transform @ xy

        return float(xy[0]), float(xy[1])

    def _cell_spans(self) -> npt.NDArray[np.int64]:
        cells = self.lattice.site_cells
        spans = np.max(cells, axis=0) - np.min(cells, axis=0) + 1
        return spans.astype(np.int64)

    def _period_vectors_2d(self) -> npt.NDArray[np.float64]:
        """
        Estimate real-space period vectors for plotting periodic image links.

        The vector for dimension d is:

            average one-cell displacement in real-space embedding
            multiplied by the number of cells in that direction.

        This works for chain, square, triangular, and honeycomb lattices as long as
        the lattice provides consistent site.position metadata.
        """
        ndim = self.lattice.ndim
        spans = self._cell_spans()
        positions = self.lattice.site_positions

        if hasattr(self.lattice, "primitive_vectors"):
            primitive_vectors = self.lattice.primitive_vectors

            vectors = []
            for dim, vec in enumerate(primitive_vectors):
                xy = np.asarray(self._xy(tuple(vec)), dtype=float)
                vectors.append(float(spans[dim]) * xy)

            return np.asarray(vectors, dtype=float)

        vectors = np.zeros((ndim, 2), dtype=float)

        site_by_key: dict[tuple[tuple[int, ...], int], int] = {
            (tuple(site.cell), int(site.sublattice)): int(site.id) for site in self.lattice.sites
        }

        for dim in range(ndim):
            unit = np.zeros(ndim, dtype=np.int64)
            unit[dim] = 1

            displacements: list[npt.NDArray[np.float64]] = []

            for site in self.lattice.sites:
                source_cell = np.asarray(site.cell, dtype=np.int64)
                target_cell = tuple((source_cell + unit).tolist())
                key = (target_cell, int(site.sublattice))

                target_id = site_by_key.get(key)
                if target_id is None:
                    continue

                source_xy = np.asarray(
                    self._xy(tuple(positions[int(site.id)])),
                    dtype=float,
                )
                target_xy = np.asarray(
                    self._xy(tuple(positions[target_id])),
                    dtype=float,
                )

                displacements.append(target_xy - source_xy)

            if displacements:
                step = np.mean(np.asarray(displacements, dtype=float), axis=0)
                vectors[dim] = step * float(spans[dim])
            else:
                if dim == 0:
                    vectors[dim] = np.asarray([float(spans[dim]), 0.0])
                elif dim == 1:
                    vectors[dim] = np.asarray([0.0, float(spans[dim])])
                else:
                    vectors[dim] = np.asarray([0.0, 0.0])

        return vectors

    def _draw_links(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_links: list[_DrawLink],
        mode: LinkPlotMode,
    ) -> None:
        if mode == "arrows":
            self._draw_arrow_links(ax=ax, config=config, draw_links=draw_links)
            return

        if mode == "dimers":
            self._draw_dimer_links(ax=ax, config=config, draw_links=draw_links)
            return

        if mode == "values":
            self._draw_value_backbone(ax=ax, draw_links=draw_links)
            return

        raise ValueError("mode must be one of 'arrows', 'dimers', or 'values'.")

    def _draw_arrow_links(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_links: list[_DrawLink],
    ) -> None:
        from matplotlib.patches import FancyArrowPatch

        for draw_link in draw_links:
            value = self.link_value(config, draw_link.link_id)

            source = self._xy(draw_link.source_position)
            target = self._xy(draw_link.target_position)

            if not self._points_along_link(value):
                source, target = target, source

            arrow = FancyArrowPatch(
                source,
                target,
                arrowstyle="-|>",
                mutation_scale=self._resolved_arrow_mutation_scale(),
                linewidth=self.style.arrow_linewidth,
                color=self.style.edge_color,
                alpha=self.style.arrow_alpha,
                shrinkA=self._resolved_arrow_shrink_points(),
                shrinkB=self._resolved_arrow_shrink_points(),
                zorder=2,
            )

            ax.add_patch(arrow)

    def _draw_dimer_links(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_links: list[_DrawLink],
    ) -> None:
        occupied_segments = []
        empty_segments = []

        for draw_link in draw_links:
            value = self.link_value(config, draw_link.link_id)
            segment = [
                self._xy(draw_link.source_position),
                self._xy(draw_link.target_position),
            ]
            if value != 0:
                occupied_segments.append(segment)
            else:
                empty_segments.append(segment)

        if empty_segments:
            ax.add_collection(
                LineCollection(
                    empty_segments,
                    colors=self.style.empty_edge_color,
                    linewidths=self.style.empty_width,
                    alpha=self.style.empty_alpha,
                    capstyle="round",
                    zorder=1,
                )
            )

        if occupied_segments:
            ax.add_collection(
                LineCollection(
                    occupied_segments,
                    colors=self.style.edge_color,
                    linewidths=self.style.occupied_width,
                    alpha=self.style.occupied_alpha,
                    capstyle="round",
                    zorder=2,
                )
            )

    def _draw_value_backbone(
        self,
        *,
        ax,
        draw_links: list[_DrawLink],
    ) -> None:
        segments = [
            [self._xy(link.source_position), self._xy(link.target_position)] for link in draw_links
        ]

        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=self.style.empty_edge_color,
                    linewidths=self.style.empty_width,
                    alpha=0.7,
                    zorder=1,
                )
            )

    def _draw_nodes(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_nodes: list[_DrawNode],
        with_site_labels: bool,
        with_site_values: bool,
    ) -> None:
        xy = np.asarray([self._xy(node.position) for node in draw_nodes], dtype=float)

        x = xy[:, 0]
        y = xy[:, 1]

        ax.scatter(
            x,
            y,
            s=self.style.node_size,
            color=self.style.node_color,
            zorder=3,
        )

        for node, px, py in zip(draw_nodes, x, y, strict=True):
            pieces: list[str] = []

            if with_site_labels:
                pieces.append(self._format_site_label(node.site_id))

            if with_site_values:
                value = self.site_value(config, node.site_id)
                if value is not None:
                    pieces.append(f"{value}")

            if pieces:
                ax.text(
                    px,
                    py,
                    "\n".join(pieces),
                    ha="center",
                    va="center",
                    fontsize=self._resolved_site_label_fontsize(),
                    color="black",
                    zorder=4,
                )

    def _draw_link_values(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_links: list[_DrawLink],
    ) -> None:
        for draw_link in draw_links:
            value = self.link_value(config, draw_link.link_id)

            sx, sy = self._xy(draw_link.source_position)
            tx, ty = self._xy(draw_link.target_position)

            x = 0.5 * (sx + tx)
            y = 0.5 * (sy + ty)

            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                fontsize=self._resolved_link_label_fontsize(),
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                zorder=5,
            )

    def _draw_square_qlm_plaquette_primitives(self) -> list[_DrawPlaquette]:
        """
        Build visual plaquette centers for square-QLM plaquette symbols.

        For PBC positive_patch, this draws one visual plaquette per cell in the
        enlarged positive patch:

            0 <= x < Lx
            0 <= y < Ly

        with centers at:

            (x + 1/2, y + 1/2)

        The physical plaquette value is taken from the plaquette associated with
        the base cell (x mod Lx, y mod Ly).
        """
        if not isinstance(self.lattice, SquareLattice):
            return []

        if self.lattice.num_plaquettes == 0:
            return []

        if self.lattice.ndim != 2:
            return []

        spans = self._cell_spans()
        lx = int(spans[0])
        ly = int(spans[1])

        period_vectors = self._period_vectors_2d()

        unit_vectors = np.zeros_like(period_vectors)
        unit_vectors[0] = period_vectors[0] / float(lx)
        unit_vectors[1] = period_vectors[1] / float(ly)

        plaquette_by_cell = self._square_plaquette_by_cell_fallback()

        draw_plaquettes: list[_DrawPlaquette] = []

        # Open boundary: use physical plaquette centers only.
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            for plaquette in self.lattice.plaquettes:
                if len(plaquette.links) != 4:
                    continue

                center = self._plaquette_center_2d(plaquette.sites)

                draw_plaquettes.append(
                    _DrawPlaquette(
                        plaquette_id=int(plaquette.id),
                        image_shift=(0, 0),
                        visual_cell=(-1, -1),
                        center=center,
                    )
                )

            return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

        # PBC positive patch: draw one symbol per visual plaquette cell.
        #
        # For a 2x2 torus, this gives cells:
        #   (0,0), (1,0), (0,1), (1,1)
        #
        # For a 4x4 torus, this gives 16 cells.
        for x in range(lx):
            for y in range(ly):
                visual_cell = (x, y)
                base_cell = (x % lx, y % ly)

                plaquette_id = plaquette_by_cell.get(base_cell)

                if plaquette_id is None:
                    # Last-resort fallback:
                    # if plaquettes are ordered by cell, use row-major order.
                    flat_index = x * ly + y
                    if flat_index < self.lattice.num_plaquettes:
                        plaquette_id = int(self.lattice.plaquettes[flat_index].id)

                if plaquette_id is None:
                    continue

                # For square coordinates, site at base_cell should exist.
                lower_left_site_id = self._site_id_from_cell(base_cell)
                if lower_left_site_id is None:
                    continue

                image_shift = self._image_shift_for_visual_cell(
                    site_id=lower_left_site_id,
                    visual_cell=visual_cell,
                )

                if image_shift is None:
                    continue

                lower_left_position = np.asarray(
                    self._visual_site_position(
                        site_id=lower_left_site_id,
                        image_shift=image_shift,
                        period_vectors=period_vectors,
                    ),
                    dtype=float,
                )

                center_arr = lower_left_position + 0.5 * unit_vectors[0] + 0.5 * unit_vectors[1]

                bottom_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="x",
                )
                right_link = self._square_visual_link_id(
                    cell=(visual_cell[0] + 1, visual_cell[1]),
                    kind="y",
                )
                top_link = self._square_visual_link_id(
                    cell=(visual_cell[0], visual_cell[1] + 1),
                    kind="x",
                )
                left_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="y",
                )

                draw_plaquettes.append(
                    _DrawPlaquette(
                        plaquette_id=int(plaquette_id),
                        image_shift=image_shift,
                        visual_cell=visual_cell,
                        center=(float(center_arr[0]), float(center_arr[1])),
                        link_ids=(
                            int(bottom_link),
                            int(right_link),
                            int(top_link),
                            int(left_link),
                        ),
                        link_orientations=(1, 1, -1, -1),
                    )
                )

        return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

    def _site_id_from_cell(
        self,
        cell: tuple[int, ...],
        *,
        sublattice: int = 0,
    ) -> int | None:
        for site in self.lattice.sites:
            if tuple(int(c) for c in site.cell) == tuple(int(c) for c in cell):
                if int(site.sublattice) == int(sublattice):
                    return int(site.id)

        return None

    def _square_plaquette_by_cell_fallback(self) -> dict[tuple[int, int], int]:
        """
        Map square plaquettes to base cells.

        This tries several conventions, because different square-lattice builders
        may store plaquette metadata differently.

        Priority:
            1. plaquette.cell or plaquette.anchor_cell if available
            2. lower-left cell inferred from plaquette sites
            3. row-major plaquette ordering fallback
        """
        if not isinstance(self.lattice, SquareLattice):
            return {}

        spans = self._cell_spans()
        lx = int(spans[0])
        ly = int(spans[1])

        out: dict[tuple[int, int], int] = {}

        # 1. Use explicit plaquette metadata if present.
        for plaquette in self.lattice.plaquettes:
            cell = None

            if hasattr(plaquette, "cell"):
                cell = getattr(plaquette, "cell")

            elif hasattr(plaquette, "anchor_cell"):
                cell = getattr(plaquette, "anchor_cell")

            if cell is None:
                continue

            c = tuple(int(x) for x in cell)
            if len(c) < 2:
                continue

            out[(c[0] % lx, c[1] % ly)] = int(plaquette.id)

        if out:
            return out

        # 2. Try to infer from plaquette sites.
        #
        # For non-wrapping plaquettes this is simply min x, min y.
        # For wrapping plaquettes this may be ambiguous, so this is only a best effort.
        for plaquette in self.lattice.plaquettes:
            if len(plaquette.sites) == 0:
                continue

            cells = np.asarray(
                [self.lattice.sites[int(site_id)].cell for site_id in plaquette.sites],
                dtype=np.int64,
            )

            if cells.shape[1] != 2:
                continue

            xs = cells[:, 0] % lx
            ys = cells[:, 1] % ly

            # If the plaquette spans the PBC seam, the lower-left cell is the
            # largest coordinate before wrapping, not min. Detect this by spread.
            if xs.max() - xs.min() > lx / 2:
                x0 = int(xs.max())
            else:
                x0 = int(xs.min())

            if ys.max() - ys.min() > ly / 2:
                y0 = int(ys.max())
            else:
                y0 = int(ys.min())

            out[(x0 % lx, y0 % ly)] = int(plaquette.id)

        if out:
            return out

        # 3. Last-resort row-major fallback.
        #
        # This assumes plaquette id/order follows:
        #   (0,0), (0,1), ..., (0,ly-1), (1,0), ...
        for x in range(lx):
            for y in range(ly):
                flat_index = x * ly + y
                if flat_index < self.lattice.num_plaquettes:
                    out[(x, y)] = int(self.lattice.plaquettes[flat_index].id)

        return out

    def _canonical_visual_cycle_link_ids(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> tuple[int, ...]:
        """Return link ids in a canonical visual cyclic order.

        The order is determined from the drawn polygon, not from
        ``plaquette.links``. This makes QDM alternating patterns such as
        1010 and 0101 stable across translated/rotated plaquette objects.

        Convention:
        1. sort edge midpoints counterclockwise around the visual center;
        2. rotate so the first edge has the lowest midpoint y, then lowest x.
        """
        center = self._closed_visual_plaquette_center(draw_links)

        records: list[tuple[float, float, float, int]] = []
        for draw_link in draw_links:
            source = np.asarray(draw_link.source_position, dtype=float)
            target = np.asarray(draw_link.target_position, dtype=float)
            midpoint = 0.5 * (source + target)
            angle = math.atan2(
                float(midpoint[1] - center[1]),
                float(midpoint[0] - center[0]),
            )
            records.append(
                (
                    angle,
                    float(midpoint[1]),
                    float(midpoint[0]),
                    int(draw_link.link_id),
                )
            )

        records.sort(key=lambda item: item[0])

        # Use the visually lowest edge as canonical edge 0. The x tie-breaker
        # makes the rule deterministic for symmetric cases.
        start = min(
            range(len(records)),
            key=lambda i: (records[i][1], records[i][2]),
        )

        rotated = records[start:] + records[:start]
        return tuple(int(record[3]) for record in rotated)

    def _canonical_visual_cycle_orientations(
        self,
        *,
        plaquette_id: int,
        canonical_link_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return plaquette orientations reordered to canonical visual link order."""
        plaquette = self.lattice.plaquettes[plaquette_id]

        orientation_by_link_id = {
            int(link_id): int(orientation)
            for link_id, orientation in zip(
                plaquette.links,
                plaquette.orientations,
                strict=True,
            )
        }

        return tuple(int(orientation_by_link_id[int(link_id)]) for link_id in canonical_link_ids)

    def _draw_plaquette_primitives(self) -> list[_DrawPlaquette]:
        """Build visual plaquette centers from actually drawn closed cycles.

        Generic circulation symbols should only be attached to a visual plaquette
        representative that is a single closed polygon. This is especially
        important for triangular and honeycomb PBC positive patches, where a
        physical plaquette may have several drawn images and a greedy nearest-link
        choice can assemble a broken or displaced representative.
        """
        if self.lattice.num_plaquettes == 0:
            return []

        _draw_nodes, draw_links = self._draw_primitives()

        draw_links_by_link_id: dict[int, list[_DrawLink]] = {}
        for draw_link in draw_links:
            draw_links_by_link_id.setdefault(int(draw_link.link_id), []).append(draw_link)

        draw_plaquettes: list[_DrawPlaquette] = []

        for plaquette in self.lattice.plaquettes:
            link_ids = tuple(int(link_id) for link_id in plaquette.links)

            if not self._is_supported_circulation_plaquette(link_ids):
                continue

            candidate_lists = [draw_links_by_link_id.get(link_id, []) for link_id in link_ids]

            if any(len(candidates) == 0 for candidates in candidate_lists):
                continue

            selected = self._select_closed_visual_plaquette(candidate_lists)
            if selected is None:
                continue

            center = self._closed_visual_plaquette_center(selected)

            canonical_link_ids = self._canonical_visual_cycle_link_ids(selected)
            canonical_orientations = self._canonical_visual_cycle_orientations(
                plaquette_id=int(plaquette.id),
                canonical_link_ids=canonical_link_ids,
            )

            draw_plaquettes.append(
                _DrawPlaquette(
                    plaquette_id=int(plaquette.id),
                    image_shift=tuple(0 for _ in range(self.lattice.ndim)),
                    visual_cell=tuple(-1 for _ in range(self.lattice.ndim)),
                    center=(float(center[0]), float(center[1])),
                    link_ids=canonical_link_ids,
                    link_orientations=canonical_orientations,
                )
            )

        return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

    def _is_supported_circulation_plaquette(
        self,
        link_ids: tuple[int, ...],
    ) -> bool:
        """Return whether a plaquette should receive a circulation symbol."""
        n_links = len(link_ids)

        if isinstance(self.lattice, SquareLattice):
            return n_links == 4

        if isinstance(self.lattice, TriangularLattice):
            # For triangular-lattice QDM/QLM resonance, the relevant plaquette is
            # a rhombus, not an elementary triangle.
            return n_links == 4

        if isinstance(self.lattice, HoneycombLattice):
            return n_links == 6

        # Conservative generic fallback.
        return n_links >= 4

    def _select_closed_visual_plaquette(
        self,
        candidate_lists: list[list[_DrawLink]],
    ) -> tuple[_DrawLink, ...] | None:
        """Choose one drawn image per physical link that forms a closed cycle."""
        best: tuple[_DrawLink, ...] | None = None
        best_score = float("inf")

        for candidate_tuple in product(*candidate_lists):
            selected = tuple(candidate_tuple)

            if not self._draw_links_form_closed_cycle(selected):
                continue

            score = self._visual_plaquette_compactness_score(selected)

            if score < best_score:
                best = selected
                best_score = score

        return best

    def _draw_links_form_closed_cycle(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        decimals: int = 10,
    ) -> bool:
        """Return True iff drawn links form one closed polygon.

        This rejects open paths, disconnected pieces, doubled links, and
        incorrectly assembled periodic images.
        """
        if len(draw_links) < 3:
            return False

        def key(position: tuple[float, float]) -> tuple[float, float]:
            return tuple(np.round(np.asarray(position, dtype=float), decimals=decimals))

        adjacency: dict[tuple[float, float], set[tuple[float, float]]] = {}

        for draw_link in draw_links:
            source = key(draw_link.source_position)
            target = key(draw_link.target_position)

            if source == target:
                return False

            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

        # A simple closed n-link polygon has exactly n vertices, and every vertex
        # has degree 2.
        if len(adjacency) != len(draw_links):
            return False

        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False

        # Check connectedness.
        start = next(iter(adjacency))
        visited = {start}
        stack = [start]

        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        return len(visited) == len(adjacency)

    def _visual_plaquette_compactness_score(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> float:
        """Score a closed visual plaquette; smaller means more compact."""
        positions = self._closed_visual_plaquette_vertices(draw_links)
        xy = np.asarray(positions, dtype=float)

        mins = np.min(xy, axis=0)
        maxs = np.max(xy, axis=0)

        # Prefer compact representatives. This avoids choosing a plaquette image
        # stretched across the torus when a local positive-patch representative
        # exists.
        bbox = maxs - mins
        return float(np.dot(bbox, bbox))

    def _closed_visual_plaquette_vertices(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        decimals: int = 10,
    ) -> list[np.ndarray]:
        """Return unique vertices of a closed drawn plaquette."""
        vertices: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()

        for draw_link in draw_links:
            for position in (draw_link.source_position, draw_link.target_position):
                arr = np.asarray(position, dtype=float)
                key = tuple(np.round(arr, decimals=decimals))
                if key in seen:
                    continue
                seen.add(key)
                vertices.append(arr)

        return vertices

    def _closed_visual_plaquette_center(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> np.ndarray:
        """Return the center of a closed drawn plaquette."""
        vertices = self._closed_visual_plaquette_vertices(draw_links)

        if len(vertices) == 0:
            raise ValueError("Cannot compute center of an empty plaquette.")

        return np.mean(np.asarray(vertices, dtype=float), axis=0)

    @staticmethod
    def _draw_link_distance_to_point(
        draw_link: _DrawLink,
        point: npt.ArrayLike,
    ) -> float:
        """Distance from a drawn link midpoint to a point."""
        source = np.asarray(draw_link.source_position, dtype=float)
        target = np.asarray(draw_link.target_position, dtype=float)
        midpoint = 0.5 * (source + target)

        return float(np.linalg.norm(midpoint - np.asarray(point, dtype=float)))

    @staticmethod
    def _unique_positions(
        positions: list[npt.NDArray[np.float64]],
        *,
        decimals: int = 10,
    ) -> list[npt.NDArray[np.float64]]:
        """Remove duplicate plotting positions."""
        out: list[npt.NDArray[np.float64]] = []
        seen: set[tuple[float, float]] = set()

        for position in positions:
            position_array = np.asarray(position, dtype=float)
            key = tuple(np.round(position_array, decimals=decimals).tolist())

            if key in seen:
                continue

            seen.add(key)
            out.append(position_array)

        return out

    def _torus_translation_vectors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return full-system torus translation vectors."""
        primitive_vectors = getattr(self.lattice, "primitive_vectors", None)

        if primitive_vectors is None:
            return None

        primitive_vectors = tuple(np.asarray(vector, dtype=float) for vector in primitive_vectors)

        lattice_x = getattr(self.lattice, "lx", None)
        lattice_y = getattr(self.lattice, "ly", None)

        if lattice_x is None or lattice_y is None:
            shape = getattr(self.lattice, "shape", None)

            if shape is None:
                return None

            lattice_x = shape[0]
            lattice_y = shape[1]

        return (
            int(lattice_x) * primitive_vectors[0],
            int(lattice_y) * primitive_vectors[1],
        )

    def _apply_visual_transform(self, position: npt.ArrayLike) -> np.ndarray:
        """Apply coordinate scale and transform to one position."""
        position_array = np.asarray(position, dtype=float)

        if self.coordinate_transform is not None:
            transform = np.asarray(self.coordinate_transform, dtype=float)
            position_array = transform @ position_array

        return self.coordinate_scale * position_array

    def _nearest_periodic_image(
        self,
        position: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Return the torus image of ``position`` nearest to ``reference``.

        Important:
            For a finite PBC lattice, the periodic translations are the full torus
            periods, not the primitive lattice vectors.
        """
        translations = self._torus_translation_vectors()

        if translations is None:
            return position

        translation_x, translation_y = translations

        best_position = np.asarray(position, dtype=float)
        best_distance = np.linalg.norm(best_position - reference)

        for shift_x in (-1, 0, 1):
            for shift_y in (-1, 0, 1):
                candidate = (
                    np.asarray(position, dtype=float)
                    + shift_x * translation_x
                    + shift_y * translation_y
                )
                distance = np.linalg.norm(candidate - reference)

                if distance < best_distance:
                    best_distance = distance
                    best_position = candidate

        return best_position

    def _collapse_duplicate_draw_plaquettes(
        self,
        draw_plaquettes: list[_DrawPlaquette],
        *,
        atol: float = 1e-9,
    ) -> list[_DrawPlaquette]:
        seen: set[tuple[int, int]] = set()
        out: list[_DrawPlaquette] = []

        def quantize(pos: tuple[float, float]) -> tuple[int, int]:
            return tuple(int(round(float(x) / atol)) for x in pos)

        for draw_plaquette in draw_plaquettes:
            # Collapse by visual center, not by plaquette_id. This avoids duplicate
            # symbols on tiny tori.
            key = quantize(draw_plaquette.center)

            if key in seen:
                continue

            seen.add(key)
            out.append(draw_plaquette)

        return out

    def _draw_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        style: PlaquetteSymbolStyle,
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        if style == "none":
            return

        if style == "square_qlm":
            self._draw_square_qlm_plaquette_symbols(
                ax=ax,
                config=config,
                draw_plaquettes=draw_plaquettes,
            )
            return

        if style == "circulation":
            self._draw_circulation_plaquette_symbols(
                ax=ax,
                config=config,
                draw_plaquettes=draw_plaquettes,
            )
            return

        if style == "resonance":
            self._draw_resonance_plaquette_symbols(
                ax=ax,
                config=config,
                draw_plaquettes=draw_plaquettes,
            )
            return

        raise ValueError(f"Unsupported plaquette symbol style: {style!r}")

    def _draw_square_qlm_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        """Draw the square-QLM-specific 16-symbol plaquette overlay."""
        if not isinstance(self.lattice, SquareLattice):
            return

        for draw_plaquette in draw_plaquettes:
            plaquette = self.lattice.plaquettes[draw_plaquette.plaquette_id]

            if len(plaquette.links) != 4:
                continue

            visual_cell = self._square_visual_cell_from_center(draw_plaquette.center)

            link_values = self._square_visual_qlm_symbol_link_values(
                config,
                tuple(int(value) for value in visual_cell),
            )
            key = self._plaquette_key(link_values)
            symbol_info = _SQUARE_QLM_PLAQUETTE_SYMBOLS.get(key)

            if symbol_info is None:
                continue

            center = draw_plaquette.center

            ax.text(
                center[0],
                center[1],
                symbol_info["s"],
                fontsize=self.style.plaquette_symbol_fontsize,
                color=symbol_info["color"],
                ha="center",
                va="center",
                zorder=6,
            )

    @staticmethod
    def _is_binary_link_pattern(values: Sequence[int]) -> bool:
        return set(int(value) for value in values) <= {0, 1}

    @staticmethod
    def _qdm_resonance_symbol(values: Sequence[int]) -> tuple[str, str] | None:
        """Return a QDM resonance marker for alternating binary dimers.

        The input values must already be in canonical visual cyclic order.

        Pattern 1010... -> blue ◆
        Pattern 0101... -> red ◇
        """
        values_tuple = tuple(int(value) for value in values)

        if len(values_tuple) < 4:
            return None

        if len(values_tuple) % 2 != 0:
            return None

        if not BasisConfigurationVisualizer._is_binary_link_pattern(values_tuple):
            return None

        pattern_a = tuple(1 if i % 2 == 0 else 0 for i in range(len(values_tuple)))
        pattern_b = tuple(0 if i % 2 == 0 else 1 for i in range(len(values_tuple)))

        if values_tuple == pattern_a:
            return "◆", "blue"

        if values_tuple == pattern_b:
            return "◇", "red"

        return None

    @staticmethod
    def _flux_circulation_symbol(
        values: Sequence[int],
        orientations: Sequence[int],
    ) -> tuple[str, str] | None:
        """Return QLM-like flux circulation symbol.

        This is for signed flux values, not binary QDM dimers.
        """
        if len(values) != len(orientations):
            return None

        oriented_values = [
            int(value) * int(orientation)
            for value, orientation in zip(values, orientations, strict=True)
        ]

        # Zero should not count as negative circulation.
        if any(value == 0 for value in oriented_values):
            return None

        if all(value > 0 for value in oriented_values):
            return "↺", "blue"

        if all(value < 0 for value in oriented_values):
            return "↻", "red"

        return None

    def _draw_resonance_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        for draw_plaquette in draw_plaquettes:
            link_ids = tuple(int(link_id) for link_id in draw_plaquette.link_ids)

            if len(link_ids) == 0:
                plaquette = self.lattice.plaquettes[draw_plaquette.plaquette_id]
                link_ids = tuple(int(link_id) for link_id in plaquette.links)

            values = [self.link_value(config, int(link_id)) for link_id in link_ids]

            symbol_info = self._qdm_resonance_symbol(values)
            if symbol_info is None:
                continue

            symbol, color = symbol_info
            center = draw_plaquette.center

            ax.annotate(
                symbol,
                xy=(center[0], center[1]),
                xytext=self.style.plaquette_symbol_offset,
                textcoords="offset points",
                fontsize=self.style.plaquette_symbol_fontsize,
                color=color,
                ha="center",
                va="center",
                zorder=6,
            )

    def _draw_circulation_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        for draw_plaquette in draw_plaquettes:
            link_ids = tuple(int(link_id) for link_id in draw_plaquette.link_ids)
            link_orientations = tuple(
                int(orientation) for orientation in draw_plaquette.link_orientations
            )

            if len(link_ids) == 0:
                plaquette = self.lattice.plaquettes[draw_plaquette.plaquette_id]
                link_ids = tuple(int(link_id) for link_id in plaquette.links)
                link_orientations = tuple(
                    int(orientation) for orientation in plaquette.orientations
                )

            values = [self.link_value(config, int(link_id)) for link_id in link_ids]

            symbol_info = None

            # QLM: signed flux circulation.
            if symbol_info is None:
                symbol_info = self._flux_circulation_symbol(
                    values,
                    link_orientations,
                )

            if symbol_info is None:
                continue

            symbol, color = symbol_info
            center = draw_plaquette.center

            ax.annotate(
                symbol,
                xy=(center[0], center[1]),
                xytext=self.style.plaquette_symbol_offset,
                textcoords="offset points",
                fontsize=self.style.plaquette_symbol_fontsize,
                color=color,
                ha="center",
                va="center",
                zorder=6,
            )

    def _plaquette_center_2d(
        self,
        site_ids: Sequence[int],
    ) -> tuple[float, float]:
        positions = [
            self._xy(tuple(self.lattice.site_positions[int(site_id)])) for site_id in site_ids
        ]
        center = np.mean(np.asarray(positions, dtype=float), axis=0)
        return float(center[0]), float(center[1])

    @staticmethod
    def _points_along_link(value: int) -> bool:
        """
        Link-arrow convention.

        Positive flux or binary 1 points along stored link orientation.
        Negative flux or binary 0 points opposite.
        """

        return value > 0

    def _square_visual_cell_from_center(
        self,
        center: npt.ArrayLike,
    ) -> tuple[int, int]:
        """Infer square-lattice visual cell from a drawn plaquette center.

        In the positive-patch drawing, the visual plaquette at cell (x, y) is
        centered at approximately (x + 1/2, y + 1/2), up to coordinate transforms.
        """
        center_array = np.asarray(center, dtype=float)

        # If coordinate transforms/scales are applied before storing draw centers,
        # this helper assumes draw centers are already in plotting coordinates.
        # For the default square plotting, this is correct.
        cell_x = int(round(float(center_array[0]) - 0.5))
        cell_y = int(round(float(center_array[1]) - 0.5))

        return cell_x, cell_y

    def _square_visual_link_id(
        self,
        *,
        cell: tuple[int, int],
        kind: str,
    ) -> int:
        """Return the square-lattice link id at a visual cell and kind."""
        if not isinstance(self.lattice, SquareLattice):
            raise TypeError("Expected SquareLattice.")

        cell_x = int(cell[0])
        cell_y = int(cell[1])

        lattice_x = cell_x % int(self.lattice.lx)
        lattice_y = cell_y % int(self.lattice.ly)

        for link in self.lattice.links:
            source_site = self.lattice.sites[int(link.source)]

            if tuple(source_site.cell) == (lattice_x, lattice_y) and link.kind == kind:
                return int(link.id)

        raise KeyError(f"No {kind}-link found at cell {(lattice_x, lattice_y)}.")

    def _square_visual_qlm_symbol_link_values(
        self,
        config: npt.ArrayLike,
        visual_cell: tuple[int, int],
    ) -> list[int]:
        """Return square-QLM symbol values from the drawn visual plaquette.

        Key convention:
            bottom, left, right, top

        These values follow the visible positive-patch arrows, not the abstract
        periodic plaquette object's stored boundary. This matters for small PBC
        lattices such as 2x2.
        """
        cell_x = int(visual_cell[0])
        cell_y = int(visual_cell[1])

        bottom_link = self._square_visual_link_id(
            cell=(cell_x, cell_y),
            kind="x",
        )
        left_link = self._square_visual_link_id(
            cell=(cell_x, cell_y),
            kind="y",
        )
        right_link = self._square_visual_link_id(
            cell=(cell_x + 1, cell_y),
            kind="y",
        )
        top_link = self._square_visual_link_id(
            cell=(cell_x, cell_y + 1),
            kind="x",
        )

        return [
            self.link_value(config, bottom_link),
            self.link_value(config, left_link),
            self.link_value(config, right_link),
            self.link_value(config, top_link),
        ]

    @staticmethod
    def _plaquette_key(values: list[int]) -> str:
        bits = [1 if value > 0 else 0 for value in values]
        return "".join(str(bit) for bit in bits)

    def _finish_axes(self, ax, *, title: str | None) -> None:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if title is not None:
            ax.set_title(title)

        self._autoscale_with_padding(ax)

    @staticmethod
    def _autoscale_with_padding(ax, padding: float = 0.5) -> None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        ax.set_ylim(ylim[0] - padding, ylim[1] + padding)

    def _visual_cell(
        self,
        *,
        site_id: int,
        image_shift: tuple[int, ...],
    ) -> tuple[int, ...]:
        spans = self._cell_spans()
        cell = np.asarray(self.lattice.sites[site_id].cell, dtype=np.int64)
        shift = np.asarray(image_shift, dtype=np.int64)
        visual_cell = cell + shift * spans
        return tuple(int(x) for x in visual_cell)

    def _image_shift_for_visual_cell(
        self,
        *,
        site_id: int,
        visual_cell: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """
        Given a physical site and a desired visual cell, return the image shift
        that places the physical site at that visual cell.

        Returns None if the visual cell is not an image of this physical site.
        """
        spans = self._cell_spans()

        base_cell = np.asarray(
            self.lattice.sites[int(site_id)].cell,
            dtype=np.int64,
        )
        visual = np.asarray(visual_cell, dtype=np.int64)

        diff = visual - base_cell

        image_shift = np.zeros(self.lattice.ndim, dtype=np.int64)

        for dim in range(self.lattice.ndim):
            span = int(spans[dim])
            if span <= 0:
                return None

            if diff[dim] % span != 0:
                return None

            image_shift[dim] = diff[dim] // span

        return tuple(int(x) for x in image_shift)

    def _is_visual_site_in_positive_patch(
        self,
        *,
        site_id: int,
        image_shift: tuple[int, ...],
    ) -> bool:
        spans = self._cell_spans()
        visual_cell = np.asarray(
            self._visual_cell(site_id=site_id, image_shift=image_shift),
            dtype=np.int64,
        )

        # Keep 0 <= cell[d] <= span[d].
        # This gives base cell plus one copied positive boundary.
        for dim in range(self.lattice.ndim):
            if visual_cell[dim] < 0:
                return False
            if visual_cell[dim] > spans[dim]:
                return False

        return True

    def _is_visual_cell_in_positive_patch(
        self,
        visual_cell: tuple[int, ...],
    ) -> bool:
        spans = self._cell_spans()

        for dim, value in enumerate(visual_cell):
            if int(value) < 0:
                return False
            if int(value) > int(spans[dim]):
                return False

        return True

    def _positive_patch_node_image_shifts(self) -> tuple[tuple[int, ...], ...]:
        ndim = self.lattice.ndim

        if ndim == 1:
            return ((0,), (1,))

        if ndim == 2:
            return (
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
            )

        raise NotImplementedError(
            "positive_patch node shifts currently support only 1D and 2D lattices."
        )

    def _positive_patch_link_source_shifts(self) -> tuple[tuple[int, ...], ...]:
        ndim = self.lattice.ndim

        if ndim == 1:
            return ((0,),)

        if ndim == 2:
            # Honeycomb needs the corner source shift because the upper apex
            # A-site at image_shift=(1,1) has outgoing A->B links that close
            # the top boundary hexagon.
            if isinstance(self.lattice, HoneycombLattice):
                return (
                    (0, 0),
                    (1, 0),
                    (0, 1),
                    (1, 1),
                )

            # Square and triangular are fine without starting links from the
            # corner image; this avoids overbuilding the outer shell.
            return (
                (0, 0),
                (1, 0),
                (0, 1),
            )

        raise NotImplementedError(
            "positive_patch source shifts currently support only 1D and 2D lattices."
        )

    def _primitive_coordinates_from_position(
        self,
        position: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """
        Express a 2D embedded position in the lattice primitive-vector basis.

        Returns coordinates (u, v) such that:

            position = u * a1 + v * a2

        approximately.
        """
        primitive_vectors = np.asarray(
            [self._xy(tuple(vec)) for vec in self.lattice.primitive_vectors],
            dtype=float,
        )

        if primitive_vectors.shape != (2, 2):
            raise ValueError("Primitive-coordinate clipping only supports 2D embeddings.")

        # Columns are primitive vectors.
        matrix = primitive_vectors.T

        pos = np.asarray(position, dtype=float)

        return np.linalg.solve(matrix, pos)

    def _is_position_in_positive_primitive_patch(
        self,
        position: tuple[float, float],
        *,
        atol: float = 1e-9,
    ) -> bool:
        if self.lattice.ndim != 2:
            return True

        spans = self._cell_spans()
        uv = self._primitive_coordinates_from_position(position)

        for dim in range(2):
            if uv[dim] < -atol:
                return False
            if uv[dim] > float(spans[dim]) + atol:
                return False

        return True

    def _is_honeycomb_origin_a_site(
        self,
        site_id: int,
    ) -> bool:
        if not isinstance(self.lattice, HoneycombLattice):
            return False

        site = self.lattice.sites[int(site_id)]

        return tuple(int(c) for c in site.cell) == (0, 0) and int(site.sublattice) == 0

    def _is_honeycomb_upper_apex_node(
        self,
        node: _DrawNode,
    ) -> bool:
        return (
            isinstance(self.lattice, HoneycombLattice)
            and self._is_honeycomb_origin_a_site(node.site_id)
            and node.image_shift == (1, 1)
        )

    def _is_honeycomb_lower_apex_node(
        self,
        node: _DrawNode,
    ) -> bool:
        return (
            isinstance(self.lattice, HoneycombLattice)
            and self._is_honeycomb_origin_a_site(node.site_id)
            and node.image_shift == (0, 0)
        )

    def _should_skip_positive_patch_visual_link(
        self,
        *,
        link,
        source_key: tuple[int, tuple[int, ...]],
        target_key: tuple[int, tuple[int, ...]],
        source_visual_cell: tuple[int, ...],
        target_visual_cell: tuple[int, ...],
    ) -> bool:
        """
        Filter visual links that are artifacts of the finite positive patch.

        For honeycomb, the upper apex A-site image at visual cell (Lx, Ly)
        is kept to close the top boundary hexagon. However, its z-link
        A(Lx,Ly) -> B(Lx,Ly) points outside the desired patch and creates
        an extra top node. We skip only that link.
        """
        if not isinstance(self.lattice, HoneycombLattice):
            return False

        kind = str(getattr(link, "kind", ""))

        # Honeycomb convention:
        #   z: A(x,y) -> B(x,y)
        if kind != "z":
            return False

        spans = self._cell_spans()

        source_cell = np.asarray(source_visual_cell, dtype=np.int64)
        target_cell = np.asarray(target_visual_cell, dtype=np.int64)

        # Skip z-link from upper apex:
        #   A(Lx,Ly) -> B(Lx,Ly)
        if np.array_equal(source_cell, spans) and np.array_equal(target_cell, spans):
            return True

        return False

    def _remove_unused_image_nodes(
        self,
        nodes: list[_DrawNode],
        links: list[_DrawLink],
    ) -> tuple[list[_DrawNode], list[_DrawLink]]:
        used_keys: set[tuple[int, tuple[int, ...]]] = set()

        for link in links:
            used_keys.add(link.source_key)
            used_keys.add(link.target_key)

        base_shift = tuple(0 for _ in range(self.lattice.ndim))
        spans = self._cell_spans()

        filtered_nodes: list[_DrawNode] = []

        for node in nodes:
            # Remove the lower honeycomb apex even though it is a base node.
            if self._is_honeycomb_lower_apex_node(node):
                continue

            # Keep the upper honeycomb apex. It visually closes the top boundary.
            if self._is_honeycomb_upper_apex_node(node):
                filtered_nodes.append(node)
                continue

            # Keep base physical nodes.
            if node.image_shift == base_shift:
                filtered_nodes.append(node)
                continue

            # Keep image nodes touched by displayed links.
            if node.key in used_keys:
                filtered_nodes.append(node)
                continue

            # Square lattice keeps extra unused boundary image nodes to complete
            # the rectangular positive patch.
            if isinstance(self.lattice, SquareLattice):
                visual_cell = np.asarray(
                    self._visual_cell(
                        site_id=node.site_id,
                        image_shift=node.image_shift,
                    ),
                    dtype=np.int64,
                )

                if np.any(visual_cell == spans):
                    filtered_nodes.append(node)
                    continue

        kept_keys = {node.key for node in filtered_nodes}

        filtered_links = [
            link for link in links if link.source_key in kept_keys and link.target_key in kept_keys
        ]

        return filtered_nodes, filtered_links

    def _collapse_duplicate_visual_links(
        self,
        draw_links: list[_DrawLink],
        *,
        atol: float = 1e-9,
    ) -> list[_DrawLink]:
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        out: list[_DrawLink] = []

        def quantize(pos: tuple[float, float]) -> tuple[int, int]:
            return tuple(int(round(float(x) / atol)) for x in pos)

        for link in draw_links:
            p0 = quantize(link.source_position)
            p1 = quantize(link.target_position)

            # Undirected key avoids visually doubled arrows on tiny tori.
            key = tuple(sorted((p0, p1)))

            if key in seen:
                continue

            seen.add(key)
            out.append(link)

        return out

    def _link_cell_displacement(self, link) -> tuple[int, ...]:
        """
        Return the intended displacement of a link on the infinite covering lattice.

        This is different from the stored finite-torus target_cell - source_cell.
        For example, on a square torus:

            stored:  (Lx - 1, y) -> (0, y)
            visual:  displacement should be (+1, 0)
        """
        kind = str(getattr(link, "kind", ""))

        if isinstance(self.lattice, ChainLattice):
            return (1,)

        if isinstance(self.lattice, SquareLattice):
            if kind in ("x", "a"):
                return (1, 0)
            if kind in ("y", "b"):
                return (0, 1)

        if isinstance(self.lattice, TriangularLattice):
            if kind == "a":
                return (1, 0)
            if kind == "b":
                return (0, 1)
            if kind == "c":
                return (-1, 1)

        if isinstance(self.lattice, HoneycombLattice):
            if kind == "z":
                return (0, 0)
            if kind == "x":
                return (-1, 0)
            if kind == "y":
                return (0, -1)

        return self._infer_link_cell_displacement(link)

    def _node_radius_points(self) -> float:
        """
        Approximate scatter-marker radius in points.

        Matplotlib scatter size is area in points^2.
        """
        return float(np.sqrt(float(self.style.node_size) / np.pi))

    def _resolved_arrow_shrink_points(self) -> float:
        """
        Infer arrow shrink so links visually connect sites.

        For lattice plots, links should look connected, so the default shrink is
        intentionally much smaller than the full node radius.
        """
        if self.style.arrow_shrink_points is not None:
            return float(self.style.arrow_shrink_points)

        radius = self._node_radius_points()

        # Small fraction of radius: avoids visible gaps but prevents arrowheads
        # from being too deeply hidden by nodes.
        return max(0.0, 0.8 * radius)

    def _resolved_arrow_mutation_scale(self) -> float:
        if self.style.arrow_mutation_scale is not None:
            return float(self.style.arrow_mutation_scale)

        radius = self._node_radius_points()

        # Keep arrowhead size visually compatible with node size.
        return max(4.0, min(14.0, 2.0 * radius))

    def _resolved_site_label_fontsize(self) -> float:
        if self.style.site_label_fontsize is not None:
            return float(self.style.site_label_fontsize)

        radius = self._node_radius_points()

        # A label like "(3, 2)" is wider than a single character, so use a
        # conservative fraction of the marker radius.
        return max(4.0, min(10.0, 0.85 * radius))

    def _resolved_link_label_fontsize(self) -> float:
        if self.style.link_label_fontsize is not None:
            return float(self.style.link_label_fontsize)

        return max(4.0, 0.85 * self._resolved_site_label_fontsize())

    def _format_site_label(self, site_id: int) -> str:
        site = self.lattice.sites[int(site_id)]
        cell = tuple(int(c) for c in site.cell)
        sublattice = int(site.sublattice)

        if self.site_label_style == "cell":
            return str(cell)

        if self.site_label_style == "cell_sublattice":
            if len(self.lattice.basis_offsets) == 1:
                return str(cell)
            return f"{cell}, {self._format_sublattice(sublattice)}"

        if self.site_label_style == "sublattice_cell":
            if len(self.lattice.basis_offsets) == 1:
                return str(cell)
            return f"{self._format_sublattice(sublattice)}{cell}"

        if self.site_label_style == "site_id":
            return str(int(site_id))

        raise ValueError(
            "site_label_style must be 'cell', 'cell_sublattice', "
            "'sublattice_cell', or 'site_id'."
        )

    @staticmethod
    def _format_sublattice(sublattice: int) -> str:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if 0 <= sublattice < len(labels):
            return labels[sublattice]

        return str(sublattice)


def plot_basis_config(
    lattice: LatticeGraph,
    config: npt.ArrayLike,
    *,
    layout: VariableLayout | None = None,
    ax=None,
    show: bool = True,
    backend: VisualizerBackend = "matplotlib",
    mode: LinkPlotMode = "arrows",
    with_site_labels: bool = True,
    with_site_values: bool = False,
    with_link_values: bool = False,
    with_plaquette_symbols: bool = True,
    plaquette_symbol_style: PlaquetteSymbolStyle = "square_qlm",
    title: str | None = None,
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "cell_sublattice",
    style: LinkVisualStyle | None = None,
):
    """
    Functional convenience wrapper around BasisConfigurationVisualizer.
    """

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=style if style is not None else LinkVisualStyle(),
        periodic_image_mode=periodic_image_mode,
        collapse_duplicate_visual_links=collapse_duplicate_visual_links,
        coordinate_scale=coordinate_scale,
        coordinate_transform=coordinate_transform,
        site_label_style=site_label_style,
    )

    return visualizer.plot(
        config,
        ax=ax,
        show=show,
        backend=backend,
        mode=mode,
        with_site_labels=with_site_labels,
        with_site_values=with_site_values,
        with_link_values=with_link_values,
        with_plaquette_symbols=with_plaquette_symbols,
        plaquette_symbol_style=plaquette_symbol_style,
        title=title,
    )


def format_basis_config(
    config: npt.ArrayLike,
    *,
    style: BasisConfigLabelStyle = "compact",
    max_length: int = 48,
) -> str:
    """
    Format one basis configuration for subplot labels.

    style="compact":
        binary configs are printed like 010101.
        other configs are printed like 1,-1,1,-1.

    style="array":
        use numpy array formatting.

    style="none":
        return an empty string.
    """
    arr = np.asarray(config, dtype=np.int64)

    if style == "none":
        return ""

    if style == "array":
        text = np.array2string(arr, separator=", ")

    elif style == "compact":
        values = set(arr.tolist())

        if values <= {0, 1}:
            text = "".join(str(int(x)) for x in arr)
        else:
            text = ",".join(str(int(x)) for x in arr)

    else:
        raise ValueError("style must be 'none', 'compact', or 'array'.")

    if len(text) > max_length:
        return text[: max_length - 3] + "..."

    return text


def automatic_grid_shape(
    n_items: int, *, ncols: int | None = None, nrows: int | None = None
) -> tuple[int, int]:
    """
    Decide a reasonable grid shape.

    If both nrows and ncols are given, they must fit n_items.
    If only one is given, the other is inferred.
    If neither is given, use a near-square grid.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")

    if n_items == 0:
        return 0, 0

    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive.")

    if ncols is not None and ncols <= 0:
        raise ValueError("ncols must be positive.")

    if nrows is not None and ncols is not None:
        if nrows * ncols < n_items:
            raise ValueError("nrows * ncols is smaller than the number of states.")
        return nrows, ncols

    if ncols is not None:
        return math.ceil(n_items / ncols), ncols

    if nrows is not None:
        return nrows, math.ceil(n_items / nrows)

    ncols_auto = math.ceil(math.sqrt(n_items))
    nrows_auto = math.ceil(n_items / ncols_auto)
    return nrows_auto, ncols_auto


def _select_cage_record(
    result_or_record,
    *,
    signature: tuple[int, int] | None = None,
    record_index: int = 0,
):
    """Return a CageRecord from either a CageRecord or CageSearchResult.

    This intentionally uses duck typing to avoid making the visualizer module
    depend directly on qlinks.caging.
    """
    if hasattr(result_or_record, "support") and hasattr(
        result_or_record,
        "local_state",
    ):
        return result_or_record

    if signature is None:
        return result_or_record[record_index]

    return result_or_record[signature, record_index]


def _amplitude_label(
    *,
    basis_index: int,
    amplitude: complex,
    digits: int = 3,
) -> str:
    real = float(np.real(amplitude))
    imag = float(np.imag(amplitude))

    if abs(imag) < 10 ** (-digits):
        amp_text = f"{real:.{digits}g}"
    elif abs(real) < 10 ** (-digits):
        amp_text = f"{imag:.{digits}g}j"
    else:
        amp_text = f"{real:.{digits}g}{imag:+.{digits}g}j"

    return f"basis {basis_index}\namp={amp_text}"


def _zero_mechanism_label_map(report) -> dict[int, str]:
    """Map zero index to its zero-level mechanism label."""
    labels: dict[int, str] = {}

    for zero_report in report.zero_reports:
        labels[int(zero_report.zero_index)] = str(zero_report.probe_mechanism_label)

    return labels


def _zero_indices_for_mechanism(
    report,
    mechanism: str,
) -> npt.NDArray[np.int64]:
    """Return zero indices selected by mechanism name."""
    if mechanism == "all":
        return np.array(
            [int(zero.zero_index) for zero in report.zero_reports],
            dtype=np.int64,
        )

    field_name_by_mechanism = {
        "q_empty": "q_empty_zero_indices",
        "closed_by_known_zeros": "closed_by_known_zero_indices",
        "projector_like": "projector_like_zero_indices",
        "unexplained_leakage": "unexplained_leakage_zero_indices",
        "regional": "regional_mechanism_zero_indices",
        "extended": "extended_mechanism_zero_indices",
        "failure": "failure_mechanism_zero_indices",
    }

    try:
        field_name = field_name_by_mechanism[mechanism]
    except KeyError as exc:
        allowed = ", ".join(["all", *field_name_by_mechanism])
        raise ValueError(
            f"Unknown zero mechanism {mechanism!r}. " f"Expected one of: {allowed}."
        ) from exc

    return np.asarray(getattr(report, field_name), dtype=np.int64)


@dataclass(frozen=True)
class BasisGridVisualizer:
    """
    Plot many basis configurations as an n by m grid.

    Parameters
    ----------
    lattice:
        Geometry/topology object.

    layout:
        Variable layout used to interpret each config array.

    single_visualizer:
        Optional custom single-state visualizer. If omitted, this class creates
        BasisConfigurationVisualizer(lattice, layout).
    """

    lattice: LatticeGraph
    layout: VariableLayout | None = None
    style: LinkVisualStyle = field(default_factory=LinkVisualStyle)
    periodic_image_mode: PeriodicImageMode = "positive_patch"
    collapse_duplicate_visual_links: bool = True
    coordinate_scale: float = 1.0
    coordinate_transform: npt.ArrayLike | None = None
    site_label_style: SiteLabelStyle = "cell_sublattice"

    def plot(
        self,
        states: npt.ArrayLike,
        *,
        nrows: int | None = None,
        ncols: int | None = None,
        start_index: int = 0,
        labels: Sequence[str] | None = None,
        show_config_label: bool = False,
        config_label_style: BasisConfigLabelStyle = "compact",
        config_label_max_length: int = 48,
        mode: str = "arrows",
        plaquette_symbols: PlaquetteSymbolStyle = "none",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
    ):
        """
        Plot a batch of basis states.

        Parameters
        ----------
        states:
            Either a single config with shape (n_variables,) or a batch with
            shape (n_states, n_variables). Slices like basis.states[:12] work.

        nrows, ncols:
            Optional grid shape. If not provided, a near-square shape is chosen.

        start_index:
            Index offset used in automatic labels. For example, if plotting
            basis.states[20:30], pass start_index=20.

        labels:
            Optional explicit labels for each subplot.

        show_config_label:
            Whether to include the raw config/binary string below the state
            index label.

        mode:
            Passed to BasisConfigurationVisualizer.plot.
            Common values: "arrows", "dimers", "values".

        plaquette_symbols:
            "none":
                draw no plaquette symbols.

            "square_qlm":
                use the old 16-symbol square-QLM dictionary. This is only
                meaningful for SquareLattice four-link plaquettes.

            "circulation":
                generic QLM-like circulation marker. Draws circular arrows when
                all link variables circulate consistently around a plaquette.
        """
        import matplotlib.pyplot as plt

        arr = np.asarray(states, dtype=np.int64)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("states must have shape (n_variables,) or (n_states, n_variables).")

        n_states = arr.shape[0]
        rows, cols = automatic_grid_shape(n_states, nrows=nrows, ncols=ncols)

        if labels is not None and len(labels) != n_states:
            raise ValueError("labels must have the same length as states.")

        if figsize is None:
            figsize = (3.0 * cols, 3.0 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        single_visualizer = BasisConfigurationVisualizer(
            lattice=self.lattice,
            layout=self.layout,
            style=self.style,
            periodic_image_mode=self.periodic_image_mode,
            collapse_duplicate_visual_links=self.collapse_duplicate_visual_links,
            coordinate_scale=self.coordinate_scale,
            coordinate_transform=self.coordinate_transform,
            site_label_style=self.site_label_style,
        )

        draw_nodes, draw_links = single_visualizer._draw_primitives()

        if plaquette_symbols == "none":
            draw_plaquettes = None
        elif plaquette_symbols == "square_qlm":
            draw_plaquettes = single_visualizer._draw_square_qlm_plaquette_primitives()
        else:
            draw_plaquettes = single_visualizer._draw_plaquette_primitives()

        if single_plot_kwargs is None:
            single_plot_kwargs = {}

        for k in range(rows * cols):
            ax = axes.flat[k]

            if k >= n_states:
                ax.axis("off")
                continue

            config = arr[k]

            if labels is None:
                title = f"state {start_index + k}"
            else:
                title = labels[k]

            if show_config_label:
                config_text = format_basis_config(
                    config,
                    style=config_label_style,
                    max_length=config_label_max_length,
                )
                if config_text:
                    title = f"{title}\n{config_text}"

            plot_kwargs = dict(single_plot_kwargs)
            plot_kwargs.pop("with_plaquette_symbols", None)
            plot_kwargs.pop("plaquette_symbol_style", None)
            plot_kwargs.pop("title", None)
            plot_kwargs.pop("show", None)
            plot_kwargs.pop("backend", None)
            plot_kwargs.pop("ax", None)
            plot_kwargs.pop("mode", None)

            # Constructor-only options; do not pass to BasisConfigurationVisualizer.plot().
            plot_kwargs.pop("style", None)
            plot_kwargs.pop("periodic_image_mode", None)
            plot_kwargs.pop("collapse_duplicate_visual_links", None)
            plot_kwargs.pop("coordinate_scale", None)
            plot_kwargs.pop("coordinate_transform", None)
            plot_kwargs.pop("site_label_style", None)

            single_visualizer._plot_with_primitives(
                config,
                ax=ax,
                draw_nodes=draw_nodes,
                draw_links=draw_links,
                draw_plaquettes=draw_plaquettes,
                show=False,
                backend=backend,
                mode=mode,
                with_plaquette_symbols=(plaquette_symbols != "none"),
                plaquette_symbol_style=plaquette_symbols,
                title=title,
                **plot_kwargs,
            )

        if suptitle is not None:
            fig.suptitle(suptitle, y=suptitle_y)

        if tight_layout_rect is None:
            if suptitle is None:
                tight_layout_rect = (0.0, 0.0, 1.0, 1.0)
            else:
                tight_layout_rect = (0.0, 0.0, 1.0, 0.96)

        fig.tight_layout(rect=tight_layout_rect)

        if show:
            plt.show()

        return fig, axes

    def plot_cage_support(
        self,
        result_or_record,
        *,
        basis_configs: npt.ArrayLike,
        signature: tuple[int, int] | None = None,
        record_index: int = 0,
        max_states: int | None = None,
        show_amplitudes: bool = True,
        amplitude_digits: int = 3,
        labels: Sequence[str] | None = None,
        suptitle: str | None = None,
        **plot_kwargs,
    ):
        """Plot the support basis states of one cage record.

        Parameters
        ----------
        result_or_record:
            Either a CageSearchResult or a CageRecord.
        basis_configs:
            Basis configuration array with shape (hilbert_size, n_variables).
        signature:
            Optional cage signature (kappa, Z). If provided, select
            result_or_record[signature, record_index].
        record_index:
            Record index among all records, or among records with the given
            signature.
        max_states:
            Optional cap on the number of support states to plot.
        show_amplitudes:
            Whether subplot labels include local-state amplitudes.
        """
        basis_configs = np.asarray(basis_configs)
        record = _select_cage_record(
            result_or_record,
            signature=signature,
            record_index=record_index,
        )

        support = np.asarray(record.support, dtype=np.int64)
        local_state = np.asarray(record.local_state, dtype=np.complex128)

        if max_states is not None:
            support = support[:max_states]
            local_state = local_state[:max_states]

        states = basis_configs[support]

        if labels is None:
            if show_amplitudes:
                labels = [
                    _amplitude_label(
                        basis_index=int(index),
                        amplitude=complex(amplitude),
                        digits=amplitude_digits,
                    )
                    for index, amplitude in zip(support, local_state, strict=True)
                ]
            else:
                labels = [f"basis {int(index)}" for index in support]

        if suptitle is None:
            suptitle = (
                f"Cage support, signature={record.signature}, "
                f"support size={record.support.size}"
            )

        return self.plot(
            states,
            labels=labels,
            suptitle=suptitle,
            **plot_kwargs,
        )

    def plot_interference_zeros(
        self,
        classification_report,
        *,
        basis_configs: npt.ArrayLike,
        mechanism: str = "all",
        max_states: int | None = None,
        labels: Sequence[str] | None = None,
        suptitle: str | None = None,
        **plot_kwargs,
    ):
        """Plot basis states corresponding to nontrivial interference zeros.

        Parameters
        ----------
        classification_report:
            CageClassificationReport returned by classify_cage_state or
            classify_full_state.
        basis_configs:
            Basis configuration array with shape (hilbert_size, n_variables).
        mechanism:
            One of:
                "all",
                "q_empty",
                "closed_by_known_zeros",
                "projector_like",
                "unexplained_leakage",
                "regional",
                "extended",
                "failure".
        max_states:
            Optional cap on the number of zero states to plot.
        """
        basis_configs = np.asarray(basis_configs)
        zero_indices = _zero_indices_for_mechanism(
            classification_report,
            mechanism,
        )

        if max_states is not None:
            zero_indices = zero_indices[:max_states]

        states = basis_configs[zero_indices]
        mechanism_labels = _zero_mechanism_label_map(classification_report)

        if labels is None:
            labels = [
                f"zero {int(index)}\n{mechanism_labels.get(int(index), mechanism)}"
                for index in zero_indices
            ]

        if suptitle is None:
            if mechanism == "all":
                suptitle = "Nontrivial interference zeros " f"({zero_indices.size} states)"
            else:
                suptitle = (
                    f"Nontrivial interference zeros: {mechanism} " f"({zero_indices.size} states)"
                )

        return self.plot(
            states,
            labels=labels,
            suptitle=suptitle,
            **plot_kwargs,
        )


def plot_basis_grid(
    lattice: LatticeGraph,
    states: npt.ArrayLike,
    *,
    layout: VariableLayout | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    start_index: int = 0,
    labels: Sequence[str] | None = None,
    show_config_label: bool = False,
    config_label_style: BasisConfigLabelStyle = "compact",
    config_label_max_length: int = 48,
    backend: VisualizerBackend = "matplotlib",
    mode: LinkPlotMode = "arrows",
    plaquette_symbols: PlaquetteSymbolStyle = "none",
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "cell_sublattice",
    style: LinkVisualStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
):
    """
    Functional wrapper around BasisGridVisualizer.
    """

    visualizer = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        style=style if style is not None else LinkVisualStyle(),
        periodic_image_mode=periodic_image_mode,
        collapse_duplicate_visual_links=collapse_duplicate_visual_links,
        coordinate_scale=coordinate_scale,
        coordinate_transform=coordinate_transform,
        site_label_style=site_label_style,
    )

    return visualizer.plot(
        states,
        nrows=nrows,
        ncols=ncols,
        start_index=start_index,
        labels=labels,
        show_config_label=show_config_label,
        config_label_style=config_label_style,
        config_label_max_length=config_label_max_length,
        mode=mode,
        plaquette_symbols=plaquette_symbols,
        figsize=figsize,
        show=show,
        backend=backend,
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
    )
