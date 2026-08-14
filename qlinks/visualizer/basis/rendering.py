from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from qlinks.lattice import BoundaryCondition
from qlinks.visualizer.basis.render_cache import _DrawLink, _DrawNode, _DrawPlaquette
from qlinks.visualizer.basis.styles import LinkPlotMode, PlaquetteSymbolStyle


class _BasisConfigurationRenderingMixin:
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

                if not self._is_visual_cell_in_positive_patch_closure_shell(source_visual_cell):
                    continue

                displacement = self._link_cell_displacement(link)

                target_visual_cell = tuple(
                    int(source_visual_cell[d]) + int(displacement[d])
                    for d in range(self.lattice.ndim)
                )

                if not self._is_visual_cell_in_positive_patch_closure_shell(target_visual_cell):
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

    def _draw_link_ids(
        self,
        *,
        ax,
        draw_links: list[_DrawLink],
    ) -> None:
        """Overlay physical link ids at drawn-link midpoints."""
        for draw_link in draw_links:
            sx, sy = self._xy(draw_link.source_position)
            tx, ty = self._xy(draw_link.target_position)

            x = 0.5 * (sx + tx)
            y = 0.5 * (sy + ty)

            ax.text(
                x,
                y,
                str(int(draw_link.link_id)),
                ha="center",
                va="center",
                fontsize=self._resolved_link_label_fontsize(),
                color="purple",
                zorder=20,
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.7,
                },
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
            **self._node_scatter_kwargs(zorder=3),
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
