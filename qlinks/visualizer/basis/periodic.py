from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    KagomeLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.visualizer.basis.render_cache import _DrawLink, _DrawNode


class _BasisConfigurationPeriodicMixin:
    def _finish_axes(
        self,
        ax,
        *,
        title: str | None,
        with_coordinate_labels: bool = False,
        draw_nodes: Sequence[_DrawNode] | None = None,
    ) -> None:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if title is not None:
            title_fontsize = self._theme_defaults.title_fontsize
            if title_fontsize is None:
                ax.set_title(title)
            else:
                ax.set_title(title, fontsize=title_fontsize)

        self._autoscale_with_padding(
            ax,
            padding=self._theme_defaults.axes_padding,
        )

        if with_coordinate_labels and draw_nodes:
            self._draw_coordinate_labels(ax, draw_nodes=draw_nodes)

    @staticmethod
    def _autoscale_with_padding(ax, padding: float = 0.5) -> None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        ax.set_ylim(ylim[0] - padding, ylim[1] + padding)

    def _draw_coordinate_labels(
        self,
        ax,
        *,
        draw_nodes: Sequence[_DrawNode],
    ) -> None:
        annotation_data = self._coordinate_annotation_data(draw_nodes)
        if annotation_data is None:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span_x = max(float(xlim[1] - xlim[0]), 1e-9)
        span_y = max(float(ylim[1] - ylim[0]), 1e-9)

        label_color = self._theme_defaults.coordinate_label_color
        label_fontsize = self._resolved_coordinate_label_fontsize()
        axis_label_fontsize = self._resolved_coordinate_axis_label_fontsize()

        bottom_offset = 0.08 * span_y
        left_offset = 0.08 * span_x
        axis_extra_x = 0.05 * span_x
        axis_extra_y = 0.05 * span_y

        label_y = annotation_data["min_y"] - bottom_offset
        label_x = annotation_data["min_x"] - left_offset

        x_label_positions = [annotation_data["min_x"]]
        y_label_positions = [annotation_data["min_y"]]

        for x_pos, label in annotation_data["x_labels"]:
            ax.text(
                x_pos,
                label_y,
                str(label),
                ha="center",
                va="top",
                fontsize=label_fontsize,
                color=label_color,
                clip_on=False,
            )
            x_label_positions.append(float(x_pos))

        if self.lattice.ndim >= 2:
            for y_pos, label in annotation_data["y_labels"]:
                ax.text(
                    label_x,
                    y_pos,
                    str(label),
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                    color=label_color,
                    clip_on=False,
                )
                y_label_positions.append(float(y_pos))

        axis_text_positions_x = list(x_label_positions)
        axis_text_positions_y = list(y_label_positions)

        if annotation_data["x_labels"]:
            ax.text(
                max(x_label_positions) + axis_extra_x,
                label_y,
                r"$x$",
                ha="left",
                va="top",
                fontsize=axis_label_fontsize,
                color=label_color,
                clip_on=False,
            )
            axis_text_positions_x.append(max(x_label_positions) + axis_extra_x)

        if self.lattice.ndim >= 2 and annotation_data["y_labels"]:
            ax.text(
                label_x,
                max(y_label_positions) + axis_extra_y,
                r"$y$",
                ha="right",
                va="bottom",
                fontsize=axis_label_fontsize,
                color=label_color,
                clip_on=False,
            )
            axis_text_positions_y.append(max(y_label_positions) + axis_extra_y)

        new_xlim = (
            min(float(xlim[0]), label_x - 0.5 * left_offset),
            max(float(xlim[1]), max(axis_text_positions_x) + 0.6 * axis_extra_x),
        )
        new_ylim = (
            min(float(ylim[0]), label_y - 0.6 * bottom_offset),
            max(float(ylim[1]), max(axis_text_positions_y) + 0.6 * axis_extra_y),
        )
        ax.set_xlim(*new_xlim)
        ax.set_ylim(*new_ylim)

    def _coordinate_annotation_data(
        self,
        draw_nodes: Sequence[_DrawNode],
    ) -> dict[str, Any] | None:
        base_nodes = [
            node for node in draw_nodes if all(int(shift) == 0 for shift in node.image_shift)
        ]
        if not base_nodes:
            base_nodes = list(draw_nodes)

        if not base_nodes:
            return None

        x_groups: dict[int, list[tuple[float, float]]] = {}
        y_groups: dict[int, list[tuple[float, float]]] = {}
        all_x: list[float] = []
        all_y: list[float] = []

        for node in base_nodes:
            site = self.lattice.sites[int(node.site_id)]
            cell = tuple(int(v) for v in site.cell)
            x_coord = cell[0] if cell else 0
            y_coord = cell[1] if len(cell) >= 2 else 0
            x_groups.setdefault(x_coord, []).append(node.position)
            y_groups.setdefault(y_coord, []).append(node.position)
            all_x.append(float(node.position[0]))
            all_y.append(float(node.position[1]))

        x_labels: list[tuple[float, int]] = []
        for x_coord in sorted(x_groups):
            points = np.asarray(x_groups[x_coord], dtype=float)
            min_y = float(np.min(points[:, 1]))
            x_pos = float(np.mean(points[np.isclose(points[:, 1], min_y), 0]))
            x_labels.append((x_pos, int(x_coord)))

        y_labels: list[tuple[float, int]] = []
        if self.lattice.ndim >= 2:
            for y_coord in sorted(y_groups):
                points = np.asarray(y_groups[y_coord], dtype=float)
                min_x = float(np.min(points[:, 0]))
                y_pos = float(np.mean(points[np.isclose(points[:, 0], min_x), 1]))
                y_labels.append((y_pos, int(y_coord)))

        return {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "min_x": min(all_x),
            "min_y": min(all_y),
        }

    def _resolved_coordinate_label_fontsize(self) -> float:
        fontsize = self._theme_defaults.coordinate_label_fontsize
        if fontsize is not None:
            return float(fontsize)
        return max(self._resolved_site_label_fontsize() - 0.5, 6.0)

    def _resolved_coordinate_axis_label_fontsize(self) -> float:
        fontsize = self._theme_defaults.coordinate_axis_label_fontsize
        if fontsize is not None:
            return float(fontsize)
        return self._resolved_coordinate_label_fontsize() + 1.5

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

    def _is_visual_cell_in_positive_patch_closure_shell(
        self,
        visual_cell: tuple[int, ...],
    ) -> bool:
        """Return whether a visual cell may be used to close boundary plaquettes.

        For triangular lattices, boundary rhombi may need a one-cell halo on the
        positive side. We only allow the top/right halo, not the left/bottom halo,
        because positive-patch drawing should show each periodic object once using
        positive-side images.
        """
        if self.periodic_image_mode != "positive_patch":
            return self._is_visual_cell_in_positive_patch(visual_cell)

        if not isinstance(self.lattice, TriangularLattice):
            return self._is_visual_cell_in_positive_patch(visual_cell)

        spans = self._cell_spans()

        return all(
            0 <= int(cell) <= int(span) + 1 for cell, span in zip(visual_cell, spans, strict=True)
        )

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
            # Honeycomb and triangular lattices can require corner-source links
            # to close boundary plaquettes in the positive patch.
            if isinstance(self.lattice, (HoneycombLattice, KagomeLattice, TriangularLattice)):
                return (
                    (0, 0),
                    (1, 0),
                    (0, 1),
                    (1, 1),
                )

            # Square is fine without starting links from the
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

        if isinstance(self.lattice, KagomeLattice):
            return self.lattice.link_cell_displacement(kind)

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
            "site_label_style must be 'cell', 'cell_sublattice', 'sublattice_cell', or 'site_id'."
        )

    @staticmethod
    def _format_sublattice(sublattice: int) -> str:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if 0 <= sublattice < len(labels):
            return labels[sublattice]

        return str(sublattice)
