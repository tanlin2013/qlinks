from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.lattice import BoundaryCondition, LatticeGraph, SquareLattice
from qlinks.variables import VariableKind, VariableLayout

LinkPlotMode = Literal["arrows", "dimers", "values"]
PlaquetteSymbolMode = Literal["binary", "flux"]

PlaquetteSymbolStyle = Literal["none", "square_qlm", "circulation"]
BasisConfigLabelStyle = Literal["none", "compact", "array"]

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
        plaquette_symbol_style: PlaquetteSymbolStyle = "square_qlm",
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

        if with_plaquette_symbols and plaquette_symbol_style != "none":
            self._draw_plaquette_symbols(
                ax=ax,
                config=arr,
                style=plaquette_symbol_style,
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
        style: PlaquetteSymbolStyle,
    ) -> None:
        if style == "none":
            return

        if self.lattice.num_plaquettes == 0:
            return

        if style == "square_qlm":
            self._draw_square_qlm_plaquette_symbols(
                ax=ax,
                config=config,
            )
            return

        if style == "circulation":
            self._draw_circulation_plaquette_symbols(
                ax=ax,
                config=config,
            )
            return

        raise ValueError("plaquette symbol style must be 'none', 'square_qlm', or 'circulation'.")

    def _draw_square_qlm_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
    ) -> None:
        """
        Draw the old 16-symbol square-QLM plaquette overlay.

        This is intentionally square-lattice specific. It assumes four-link
        square plaquettes and a QLM-style variable pattern.
        """
        if not isinstance(self.lattice, SquareLattice):
            return

        for plaquette in self.lattice.plaquettes:
            if len(plaquette.links) != 4:
                continue

            link_values = [self.link_value(config, int(link_id)) for link_id in plaquette.links]

            key = self._plaquette_key(link_values)
            symbol_info = _SQUARE_QLM_PLAQUETTE_SYMBOLS.get(key)

            if symbol_info is None:
                continue

            center = self._plaquette_center_2d(plaquette.sites)

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

    def _draw_circulation_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
    ) -> None:
        """
        Generic QLM-like circulation marker.

        Draws a circular arrow if all link variables align with the plaquette
        orientation, or all oppose it.

        This works for generic even/odd plaquettes as long as the lattice stores
        plaquette.orientations.
        """
        for plaquette in self.lattice.plaquettes:
            if len(plaquette.links) == 0:
                continue

            signs: list[int] = []

            for link_id, orientation in zip(plaquette.links, plaquette.orientations, strict=True):
                value = self.link_value(config, int(link_id))

                if value == 0:
                    signs.append(-1)
                elif value > 0:
                    signs.append(1)
                else:
                    signs.append(-1)

                signs[-1] *= int(orientation)

            if all(s > 0 for s in signs):
                symbol = "↻"
                color = "red"
            elif all(s < 0 for s in signs):
                symbol = "↺"
                color = "blue"
            else:
                continue

            center = self._plaquette_center_2d(plaquette.sites)

            ax.text(
                center[0],
                center[1],
                symbol,
                fontsize=22,
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
        suptitle: str | None = None,
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
        )

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
            plot_kwargs.pop("ax", None)
            plot_kwargs.pop("mode", None)

            single_visualizer.plot(
                config,
                ax=ax,
                show=False,
                mode=mode,
                with_plaquette_symbols=(plaquette_symbols != "none"),
                plaquette_symbol_style=plaquette_symbols,
                title=title,
                **plot_kwargs,
            )

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


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
    mode: str = "arrows",
    plaquette_symbols: PlaquetteSymbolStyle = "none",
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
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
    )
