from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.lattice import BoundaryCondition, LatticeGraph, SquareLattice
from qlinks.variables import VariableKind, VariableLayout


LinkPlotMode = Literal["arrows", "dimers", "values"]
PlaquetteSymbolMode = Literal["binary", "flux"]


@dataclass(frozen=True, slots=True)
class LinkVisualStyle:
    """
    Basic visual style for link drawing.
    """

    occupied_width: float = 3.0
    empty_width: float = 0.8
    arrow_width: float = 1.4
    occupied_alpha: float = 0.9
    empty_alpha: float = 0.25
    arrow_alpha: float = 0.65
    node_size: float = 700.0
    node_color: str = "tab:orange"
    edge_color: str = "black"
    empty_edge_color: str = "lightgray"
    arrow_mutation_scale: float = 18.0


@dataclass(frozen=True, slots=True)
class _DrawNode:
    key: tuple[int, tuple[int, ...]]
    site_id: int
    position: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _DrawLink:
    link_id: int
    source_site: int
    target_site: int
    source_position: tuple[float, ...]
    target_position: tuple[float, ...]


@dataclass(frozen=True, slots=True)
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
    """

    lattice: LatticeGraph
    layout: VariableLayout | None = None
    style: LinkVisualStyle = field(default_factory=LinkVisualStyle)

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
        mode: LinkPlotMode = "arrows",
        with_site_labels: bool = True,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_mode: PlaquetteSymbolMode = "binary",
        title: str | None = None,
    ):
        """
        Plot one basis configuration.

        mode="arrows":
            QLM-like style. Positive / 1 values point along the stored link
            orientation. Negative / 0 values point opposite.

        mode="dimers":
            QDM-like style. Value 1 links are drawn thick; value 0 links are
            faint.

        mode="values":
            Draw the lattice and place link values at link centers.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        arr = self._as_config(config)

        if mode in ("arrows", "dimers") and not self.has_link_variables():
            raise ValueError(
                f"mode='{mode}' requires link variables in the layout. "
                "For site-only layouts, use mode='values' with with_site_values=True."
            )

        draw_nodes, draw_links = self._draw_primitives()

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

        if with_plaquette_symbols:
            self._draw_plaquette_symbols(
                ax=ax,
                config=arr,
                mode=plaquette_symbol_mode,
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

    def _draw_primitives(self) -> tuple[list[_DrawNode], list[_DrawLink]]:
        nodes: dict[tuple[int, tuple[int, ...]], _DrawNode] = {}
        links: list[_DrawLink] = []

        base_positions = self.lattice.site_positions

        def add_node(site_id: int, shift: tuple[int, ...] | None = None) -> tuple[float, ...]:
            if shift is None:
                shift = tuple(0 for _ in range(self.lattice.ndim))

            pos = tuple(float(x) + float(dx) for x, dx in zip(base_positions[site_id], shift))
            key = (site_id, shift)

            if key not in nodes:
                nodes[key] = _DrawNode(
                    key=key,
                    site_id=site_id,
                    position=pos,
                )

            return pos

        for site_id in range(self.lattice.num_sites):
            add_node(site_id)

        for link in self.lattice.links:
            source_shift = tuple(0 for _ in range(self.lattice.ndim))
            target_shift = tuple(0 for _ in range(self.lattice.ndim))

            if isinstance(self.lattice, SquareLattice):
                target_shift = self._square_periodic_target_shift(link_id=link.id)

            source_pos = add_node(link.source, source_shift)
            target_pos = add_node(link.target, target_shift)

            links.append(
                _DrawLink(
                    link_id=link.id,
                    source_site=link.source,
                    target_site=link.target,
                    source_position=source_pos,
                    target_position=target_pos,
                )
            )

        return list(nodes.values()), links

    def _square_periodic_target_shift(self, link_id: int) -> tuple[int, int]:
        """
        Flatten square-lattice periodic boundaries in the same spirit as the old
        visualizer: wrapping +x links are drawn to x=Lx, and wrapping +y links
        are drawn to y=Ly.
        """

        if not isinstance(self.lattice, SquareLattice):
            return (0, 0)

        link = self.lattice.links[link_id]

        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return (0, 0)

        if not link.wrap:
            return (0, 0)

        if link.kind == "x":
            return (self.lattice.lx, 0)

        if link.kind == "y":
            return (0, self.lattice.ly)

        return (0, 0)

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
                mutation_scale=self.style.arrow_mutation_scale,
                linewidth=self.style.arrow_width,
                color=self.style.edge_color,
                alpha=self.style.arrow_alpha,
                shrinkA=12,
                shrinkB=12,
            )

            ax.add_patch(arrow)

    def _draw_dimer_links(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_links: list[_DrawLink],
    ) -> None:
        for draw_link in draw_links:
            value = self.link_value(config, draw_link.link_id)

            occupied = value != 0

            sx, sy = self._xy(draw_link.source_position)
            tx, ty = self._xy(draw_link.target_position)

            x = [sx, tx]
            y = [sy, ty]

            ax.plot(
                x,
                y,
                color=self.style.edge_color if occupied else self.style.empty_edge_color,
                linewidth=self.style.occupied_width if occupied else self.style.empty_width,
                alpha=self.style.occupied_alpha if occupied else self.style.empty_alpha,
                solid_capstyle="round",
            )

    def _draw_value_backbone(
        self,
        *,
        ax,
        draw_links: list[_DrawLink],
    ) -> None:
        for draw_link in draw_links:
            sx, sy = self._xy(draw_link.source_position)
            tx, ty = self._xy(draw_link.target_position)

            x = [sx, tx]
            y = [sy, ty]

            ax.plot(
                x,
                y,
                color=self.style.empty_edge_color,
                linewidth=self.style.empty_width,
                alpha=0.7,
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
                site = self.lattice.sites[node.site_id]
                pieces.append(str(site.cell))

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
                    fontsize=9,
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
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                zorder=5,
            )

    def _draw_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        mode: PlaquetteSymbolMode,
    ) -> None:
        if self.lattice.num_plaquettes == 0:
            return

        for plaquette in self.lattice.plaquettes:
            if len(plaquette.links) != 4:
                continue

            link_values = [
                self.link_value(config, int(link_id))
                for link_id in plaquette.links
            ]

            key = self._plaquette_key(link_values, mode=mode)
            symbol_info = _SQUARE_PLAQUETTE_SYMBOLS.get(key)

            if symbol_info is None:
                continue

            positions = self.lattice.site_positions[list(plaquette.sites)]
            center = np.mean(positions[:, :2], axis=0)

            ax.text(
                center[0],
                center[1],
                symbol_info["s"],
                fontsize=22,
                color=symbol_info["color"],
                ha="center",
                va="center",
                zorder=6,
            )

    @staticmethod
    def _points_along_link(value: int) -> bool:
        """
        Link-arrow convention.

        Positive flux or binary 1 points along stored link orientation.
        Negative flux or binary 0 points opposite.
        """

        return value > 0

    @staticmethod
    def _plaquette_key(values: list[int], *, mode: PlaquetteSymbolMode) -> str:
        if mode == "binary":
            bits = [1 if value > 0 else 0 for value in values]
        elif mode == "flux":
            bits = [1 if value > 0 else 0 for value in values]
        else:
            raise ValueError("plaquette symbol mode must be 'binary' or 'flux'.")

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


# This mapping is copied in spirit from the old square-lattice visualizer.
# Keys are plaquette-link values converted to binary signs in plaquette order.
_SQUARE_PLAQUETTE_SYMBOLS: dict[str, dict[str, str]] = {
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


def plot_basis_config(
    lattice: LatticeGraph,
    config: npt.ArrayLike,
    *,
    layout: VariableLayout | None = None,
    ax=None,
    show: bool = True,
    mode: LinkPlotMode = "arrows",
    with_site_labels: bool = True,
    with_site_values: bool = False,
    with_link_values: bool = False,
    with_plaquette_symbols: bool = True,
    title: str | None = None,
):
    """
    Functional convenience wrapper.
    """

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    return visualizer.plot(
        config,
        ax=ax,
        show=show,
        mode=mode,
        with_site_labels=with_site_labels,
        with_site_values=with_site_values,
        with_link_values=with_link_values,
        with_plaquette_symbols=with_plaquette_symbols,
        title=title,
    )
