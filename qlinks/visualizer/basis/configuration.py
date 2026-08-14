from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from qlinks.lattice import LatticeGraph, SquareLattice
from qlinks.variables import VariableKind, VariableLayout
from qlinks.visualizer.basis.periodic import _BasisConfigurationPeriodicMixin
from qlinks.visualizer.basis.plaquette_geometry import _BasisConfigurationPlaquetteGeometryMixin
from qlinks.visualizer.basis.plaquette_symbols import (
    _SQUARE_QLM_PLAQUETTE_SYMBOLS,
    _BasisConfigurationPlaquetteSymbolMixin,
)
from qlinks.visualizer.basis.render_cache import (
    _BasisGridRenderCache,
    _DrawLink,
    _DrawNode,
    _DrawPlaquette,
)
from qlinks.visualizer.basis.rendering import _BasisConfigurationRenderingMixin
from qlinks.visualizer.basis.styles import (
    BasisVisualizerTheme,
    LinkPlotMode,
    LinkVisualStyle,
    LocalBasisShadowStyle,
    PeriodicImageMode,
    PlaquetteSymbolStyle,
    SiteLabelStyle,
    VisualizerBackend,
    _basis_visualizer_theme_defaults,
    _BasisVisualizerThemeDefaults,
)


@dataclass(frozen=True)
class BasisConfigurationVisualizer(
    _BasisConfigurationRenderingMixin,
    _BasisConfigurationPlaquetteGeometryMixin,
    _BasisConfigurationPlaquetteSymbolMixin,
    _BasisConfigurationPeriodicMixin,
):
    """Draw one basis configuration on a lattice geometry.

    The visualizer is model-agnostic: it reads variable values from a
    :class:`VariableLayout` and renders them as QLM arrows, QDM dimers, or
    generic values.

    Attributes:
        lattice: Lattice graph, such as :class:`ChainLattice` or
            :class:`SquareLattice`.
        layout: Optional variable layout.  If omitted, link plotting assumes
            ``link_variable_index == link_id``.
        theme: Named presentation theme. ``"research"`` preserves the
            historical qlinks styling; ``"paper"`` uses compact publication
            defaults.
        style: Optional explicit visual style. When provided, it overrides the
            link/site style supplied by ``theme`` while retaining the theme's
            presentation defaults.
        periodic_image_mode: How to draw links that wrap periodic boundaries.
            ``"none"`` omits wrapped links; ``"positive_patch"`` draws the
            positive image patch; ``"both"`` draws both images.
        collapse_duplicate_visual_links: Whether to collapse duplicate periodic
            visual links.
        coordinate_scale: Uniform coordinate scaling.
        coordinate_transform: Optional 2x2 coordinate transform.
        site_label_style: How to label lattice sites.
    """

    lattice: LatticeGraph
    layout: VariableLayout | None = None
    style: LinkVisualStyle | None = None
    theme: BasisVisualizerTheme = "research"
    periodic_image_mode: PeriodicImageMode = "positive_patch"
    collapse_duplicate_visual_links: bool = True
    coordinate_scale: float = 1.0
    coordinate_transform: npt.NDArray[np.float64] | None = None
    site_label_style: SiteLabelStyle = "cell_sublattice"

    def __post_init__(self) -> None:
        defaults = _basis_visualizer_theme_defaults(self.theme)
        if self.style is None:
            object.__setattr__(self, "style", defaults.style)

    @property
    def _theme_defaults(self) -> _BasisVisualizerThemeDefaults:
        return _basis_visualizer_theme_defaults(self.theme)

    def _infer_link_plot_mode(
        self,
        config: npt.ArrayLike | None = None,
    ) -> Literal["arrows", "dimers", "values"]:
        """Infer a plotting mode from the layout, falling back to config values.

        Convention:
            {-1, +1} or {-1, 0, +1} -> QLM-like arrows
            {0, 1}                   -> QDM-like dimers
            site-only layout          -> values
        """
        if not self.has_link_variables():
            return "values"

        if self.layout is not None:
            link_variable_indices = self.layout.link_variable_indices()

            if link_variable_indices.size == 0:
                return "values"

            link_spaces = [
                tuple(int(value) for value in self.layout.local_space(int(index)).values)
                for index in link_variable_indices
            ]

            unique_spaces = set(link_spaces)

            if unique_spaces == {(-1, 1)}:
                return "arrows"

            if unique_spaces == {(-1, 0, 1)}:
                return "arrows"

            if unique_spaces == {(0, 1)}:
                return "dimers"

            # Conservative fallback for mixed/unknown link spaces.
            return "values"

        if config is not None:
            arr = np.asarray(config, dtype=np.int64)

            if arr.size >= self.lattice.num_links:
                link_values = set(int(value) for value in arr[: self.lattice.num_links])

                if link_values <= {-1, 1}:
                    return "arrows"

                if link_values <= {-1, 0, 1} and any(value < 0 for value in link_values):
                    return "arrows"

                if link_values <= {0, 1}:
                    return "dimers"

        return "arrows"

    def _resolve_link_plot_mode(
        self,
        *,
        config: npt.ArrayLike,
        mode: LinkPlotMode,
    ) -> Literal["arrows", "dimers", "values"]:
        if mode != "auto":
            return mode

        return self._infer_link_plot_mode(config)

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

    def _link_variable_index(self, link_id: int) -> int:
        if self.layout is None:
            return int(link_id)

        try:
            return int(self.layout.variable_index(VariableKind.LINK, int(link_id)))
        except KeyError:
            return -1

    def _site_variable_index_or_missing(self, site_id: int) -> int:
        if self.layout is None:
            return -1

        try:
            return int(self.layout.variable_index(VariableKind.SITE, int(site_id)))
        except KeyError:
            return -1

    def _validate_config_batch_for_cached_grid(
        self,
        configs: npt.NDArray[np.int64],
    ) -> None:
        if configs.ndim != 2:
            raise ValueError("states must have shape (n_variables,) or (n_states, n_variables).")

        if self.layout is not None:
            self.layout.validate_batch(configs)
        elif configs.shape[1] < self.lattice.num_links:
            raise ValueError(
                "Without a VariableLayout, configs must contain at least "
                f"{self.lattice.num_links} link values."
            )

    def build_grid_render_cache(
        self,
        *,
        reference_config: npt.ArrayLike,
        mode: LinkPlotMode = "auto",
        plaquette_symbols: PlaquetteSymbolStyle = "auto",
    ) -> _BasisGridRenderCache:
        """Build a reusable cache for fast repeated grid plotting.

        The cache resolves the plotting mode once, precomputes visual geometry,
        and converts physical site/link ids to raw configuration indices.  The
        resulting object is specific to this visualizer's lattice/layout/style
        options and to the resolved ``mode``/``plaquette_symbols`` pair.
        """
        reference = self._as_config(reference_config)
        resolved_mode = self._resolve_link_plot_mode(
            config=reference,
            mode=mode,
        )
        resolved_plaquette_symbols = self._resolve_plaquette_symbol_style(
            mode=resolved_mode,
            plaquette_symbol_style=plaquette_symbols,
        )

        draw_nodes_list, draw_links_list = self._draw_primitives()
        draw_nodes = tuple(draw_nodes_list)
        draw_links = tuple(draw_links_list)

        if resolved_plaquette_symbols == "none":
            draw_plaquettes = ()
        else:
            draw_plaquettes = tuple(self._draw_plaquette_primitives())

        link_variable_indices = np.asarray(
            [self._link_variable_index(draw_link.link_id) for draw_link in draw_links],
            dtype=np.int64,
        )
        site_variable_indices = np.asarray(
            [self._site_variable_index_or_missing(node.site_id) for node in draw_nodes],
            dtype=np.int64,
        )

        node_xy = np.asarray([self._xy(node.position) for node in draw_nodes], dtype=float)
        link_source_xy = np.asarray(
            [self._xy(draw_link.source_position) for draw_link in draw_links],
            dtype=float,
        )
        link_target_xy = np.asarray(
            [self._xy(draw_link.target_position) for draw_link in draw_links],
            dtype=float,
        )

        if draw_links:
            link_segments = np.stack((link_source_xy, link_target_xy), axis=1)
            link_midpoints = 0.5 * (link_source_xy + link_target_xy)
        else:
            link_segments = np.empty((0, 2, 2), dtype=float)
            link_midpoints = np.empty((0, 2), dtype=float)

        site_labels = tuple(self._format_site_label(node.site_id) for node in draw_nodes)

        plaquette_link_variable_indices: list[tuple[int, ...]] = []
        plaquette_orientations: list[tuple[int, ...]] = []
        plaquette_midpoints: list[tuple[tuple[float, float], ...]] = []
        square_qlm_link_variable_indices: list[tuple[int, ...] | None] = []

        for draw_plaquette in draw_plaquettes:
            link_ids = tuple(int(link_id) for link_id in draw_plaquette.link_ids)

            if len(link_ids) == 0:
                plaquette = self.lattice.plaquettes[int(draw_plaquette.plaquette_id)]
                link_ids = tuple(int(link_id) for link_id in plaquette.links)

            plaquette_link_variable_indices.append(
                tuple(self._link_variable_index(link_id) for link_id in link_ids)
            )
            plaquette_orientations.append(
                tuple(int(orientation) for orientation in draw_plaquette.link_orientations)
            )
            plaquette_midpoints.append(
                tuple((float(point[0]), float(point[1])) for point in draw_plaquette.link_midpoints)
            )

            if isinstance(self.lattice, SquareLattice) and len(draw_plaquette.link_ids) == 4:
                if len(draw_plaquette.visual_cell) >= 2 and all(
                    int(value) >= 0 for value in draw_plaquette.visual_cell[:2]
                ):
                    visual_cell = (
                        int(draw_plaquette.visual_cell[0]),
                        int(draw_plaquette.visual_cell[1]),
                    )
                else:
                    visual_cell = self._square_visual_cell_from_center(
                        draw_plaquette.center,
                    )

                bottom_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="x",
                )
                left_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="y",
                )
                right_link = self._square_visual_link_id(
                    cell=(visual_cell[0] + 1, visual_cell[1]),
                    kind="y",
                )
                top_link = self._square_visual_link_id(
                    cell=(visual_cell[0], visual_cell[1] + 1),
                    kind="x",
                )
                square_qlm_link_variable_indices.append(
                    tuple(
                        self._link_variable_index(link_id)
                        for link_id in (bottom_link, left_link, right_link, top_link)
                    )
                )
            else:
                square_qlm_link_variable_indices.append(None)

        if draw_plaquettes:
            plaquette_centers = np.asarray(
                [draw_plaquette.center[:2] for draw_plaquette in draw_plaquettes],
                dtype=float,
            )
        else:
            plaquette_centers = np.empty((0, 2), dtype=float)

        return _BasisGridRenderCache(
            mode=resolved_mode,
            plaquette_symbol_style=resolved_plaquette_symbols,
            draw_nodes=draw_nodes,
            draw_links=draw_links,
            draw_plaquettes=draw_plaquettes,
            link_variable_indices=link_variable_indices,
            site_variable_indices=site_variable_indices,
            node_xy=node_xy,
            link_source_xy=link_source_xy,
            link_target_xy=link_target_xy,
            link_segments=link_segments,
            link_midpoints=link_midpoints,
            site_labels=site_labels,
            plaquette_link_variable_indices=tuple(plaquette_link_variable_indices),
            plaquette_orientations=tuple(plaquette_orientations),
            plaquette_centers=plaquette_centers,
            plaquette_midpoints=tuple(plaquette_midpoints),
            square_qlm_link_variable_indices=tuple(square_qlm_link_variable_indices),
        )

    def plot(
        self,
        config: npt.ArrayLike,
        *,
        ax=None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        mode: LinkPlotMode = "auto",
        with_site_labels: bool | None = None,
        with_coordinate_labels: bool | None = None,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_link_ids: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_style: PlaquetteSymbolStyle = "auto",
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
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
        if ax is None:
            _, ax = plt.subplots()

        if with_site_labels is None:
            with_site_labels = self._theme_defaults.with_site_labels
        if with_coordinate_labels is None:
            with_coordinate_labels = self._theme_defaults.with_coordinate_labels

        resolved_mode = self._resolve_link_plot_mode(
            config=config,
            mode=mode,
        )
        resolved_plaquette_symbol_style = self._resolve_plaquette_symbol_style(
            mode=resolved_mode,
            plaquette_symbol_style=plaquette_symbol_style,
        )

        draw_nodes, draw_links = self._draw_primitives()

        draw_plaquettes = None
        if with_plaquette_symbols and resolved_plaquette_symbol_style != "none":
            draw_plaquettes = self._draw_plaquette_primitives()

        return self._plot_with_primitives(
            config,
            ax=ax,
            draw_nodes=draw_nodes,
            draw_links=draw_links,
            draw_plaquettes=draw_plaquettes,
            show=show,
            backend=backend,
            mode=resolved_mode,
            with_site_labels=with_site_labels,
            with_coordinate_labels=with_coordinate_labels,
            with_site_values=with_site_values,
            with_link_values=with_link_values,
            with_link_ids=with_link_ids,
            with_plaquette_symbols=with_plaquette_symbols,
            plaquette_symbol_style=resolved_plaquette_symbol_style,
            plaquette_symbol_values=plaquette_symbol_values,
            title=title,
        )

    def _resolve_plaquette_symbol_style(
        self,
        *,
        mode: LinkPlotMode,
        plaquette_symbol_style: PlaquetteSymbolStyle,
    ) -> PlaquetteSymbolStyle:
        """Resolve automatic plaquette-symbol style.

        Concrete meaning:
            arrows  -> QLM-like circulation, except square uses square_qlm
            dimers  -> QDM-like resonance
            values  -> none
        """
        if plaquette_symbol_style != "auto":
            return plaquette_symbol_style

        if mode == "arrows":
            return "circulation"

        if mode == "dimers":
            return "resonance"

        if mode == "values":
            return "none"

        raise ValueError("mode must be one of 'arrows', 'dimers', or 'values'.")

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
        mode: LinkPlotMode = "auto",
        with_site_labels: bool = True,
        with_coordinate_labels: bool = False,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_link_ids: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_style: PlaquetteSymbolStyle = "auto",
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
        title: str | None = None,
    ):
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
            if with_link_ids:
                self._draw_link_ids(
                    ax=ax,
                    draw_links=draw_links,
                )
            if with_plaquette_symbols and plaquette_symbol_style != "none":
                self._draw_plaquette_symbols(
                    ax=ax,
                    config=arr,
                    style=plaquette_symbol_style,
                    draw_plaquettes=draw_plaquettes or [],
                    plaquette_symbol_values=plaquette_symbol_values,
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

        self._finish_axes(
            ax,
            title=title,
            with_coordinate_labels=with_coordinate_labels,
            draw_nodes=draw_nodes,
        )

        if show:
            plt.show()

        return ax

    def _plot_with_grid_render_cache(
        self,
        config: npt.NDArray[np.int64],
        *,
        ax,
        render_cache: _BasisGridRenderCache,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        with_site_labels: bool = True,
        with_coordinate_labels: bool = False,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_link_ids: bool = False,
        with_plaquette_symbols: bool = True,
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
        title: str | None = None,
    ):
        """Plot one already-validated config using cached grid geometry."""
        if render_cache.mode in ("arrows", "dimers") and not self.has_link_variables():
            raise ValueError(
                f"mode='{render_cache.mode}' requires link variables in the layout. "
                "For site-only layouts, use mode='values' with with_site_values=True."
            )

        if backend != "matplotlib":
            self._draw_networkx(
                ax=ax,
                config=config,
                draw_nodes=list(render_cache.draw_nodes),
                draw_links=list(render_cache.draw_links),
                draw_plaquettes=list(render_cache.draw_plaquettes),
                mode=render_cache.mode,
                with_site_labels=with_site_labels,
                with_site_values=with_site_values,
                with_link_values=with_link_values,
                with_plaquette_symbols=with_plaquette_symbols,
                plaquette_symbol_style=render_cache.plaquette_symbol_style,
                title=None,
            )
        else:
            self._draw_links_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
            )
            self._draw_nodes_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
                with_site_labels=with_site_labels,
                with_site_values=with_site_values,
            )
            if (with_link_values or render_cache.mode == "values") and self.has_link_variables():
                self._draw_link_values_from_grid_render_cache(
                    ax=ax,
                    config=config,
                    render_cache=render_cache,
                )
            if with_link_ids:
                self._draw_link_ids_from_grid_render_cache(
                    ax=ax,
                    render_cache=render_cache,
                )
            if with_plaquette_symbols and render_cache.plaquette_symbol_style != "none":
                self._draw_plaquette_symbols_from_grid_render_cache(
                    ax=ax,
                    config=config,
                    render_cache=render_cache,
                    plaquette_symbol_values=plaquette_symbol_values,
                )

        self._finish_axes(
            ax,
            title=title,
            with_coordinate_labels=with_coordinate_labels,
            draw_nodes=render_cache.draw_nodes,
        )

        if show:
            plt.show()

        return ax

    def _plot_local_basis_with_grid_render_cache(
        self,
        config: npt.NDArray[np.int64],
        *,
        ax,
        render_cache: _BasisGridRenderCache,
        active_link_mask: npt.NDArray[np.bool_],
        active_node_mask: npt.NDArray[np.bool_],
        shadow_style: LocalBasisShadowStyle,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        with_site_labels: bool = True,
        with_coordinate_labels: bool = False,
        with_site_values: bool = False,
        with_link_values: bool = False,
        with_link_ids: bool = False,
        with_plaquette_symbols: bool = False,
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
        title: str | None = None,
    ):
        """Plot one embedded local-basis pattern using cached geometry.

        The ``active_*_mask`` arrays select the site/link artists associated
        with the local variable support.  The rest of the lattice is still drawn
        using ``shadow_style``.
        """
        if backend != "matplotlib":
            raise ValueError("Local-basis shadow plotting currently supports backend='matplotlib'.")

        if active_link_mask.shape != (len(render_cache.draw_links),):
            raise ValueError("active_link_mask has an incompatible shape.")

        if active_node_mask.shape != (len(render_cache.draw_nodes),):
            raise ValueError("active_node_mask has an incompatible shape.")

        if render_cache.mode in ("arrows", "dimers") and not self.has_link_variables():
            raise ValueError(
                f"mode='{render_cache.mode}' requires link variables in the layout. "
                "For site-only layouts, use mode='values' with with_site_values=True."
            )

        self._draw_local_basis_links_from_grid_render_cache(
            ax=ax,
            config=config,
            render_cache=render_cache,
            active_link_mask=active_link_mask,
            shadow_style=shadow_style,
        )
        self._draw_local_basis_nodes_from_grid_render_cache(
            ax=ax,
            config=config,
            render_cache=render_cache,
            active_node_mask=active_node_mask,
            shadow_style=shadow_style,
            with_site_labels=with_site_labels,
            with_site_values=with_site_values,
        )
        if (with_link_values or render_cache.mode == "values") and self.has_link_variables():
            self._draw_local_basis_link_values_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
                active_link_mask=active_link_mask,
                shadow_style=shadow_style,
            )
        if with_link_ids:
            self._draw_local_basis_link_ids_from_grid_render_cache(
                ax=ax,
                render_cache=render_cache,
                active_link_mask=active_link_mask,
                shadow_style=shadow_style,
            )
        if with_plaquette_symbols and render_cache.plaquette_symbol_style != "none":
            self._draw_plaquette_symbols_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
                plaquette_symbol_values=plaquette_symbol_values,
            )

        self._finish_axes(
            ax,
            title=title,
            with_coordinate_labels=with_coordinate_labels,
            draw_nodes=render_cache.draw_nodes,
        )

        if show:
            plt.show()

        return ax

    def _draw_local_basis_links_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        active_link_mask: npt.NDArray[np.bool_],
        shadow_style: LocalBasisShadowStyle,
    ) -> None:
        if render_cache.mode == "arrows":
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[~active_link_mask],
                color=shadow_style.shadow_link_color,
                linewidth=self.style.arrow_linewidth * shadow_style.shadow_link_width_scale,
                alpha=shadow_style.shadow_link_alpha,
                zorder=1,
            )

            values = config[render_cache.link_variable_indices]

            for index, value in enumerate(values):
                if not bool(active_link_mask[index]):
                    continue

                source = tuple(float(x) for x in render_cache.link_source_xy[index])
                target = tuple(float(x) for x in render_cache.link_target_xy[index])

                if not self._points_along_link(int(value)):
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
            return

        if render_cache.mode == "dimers":
            values = config[render_cache.link_variable_indices]
            occupied_mask = values != 0
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[~active_link_mask],
                color=shadow_style.shadow_link_color,
                linewidth=self.style.empty_width * shadow_style.shadow_link_width_scale,
                alpha=shadow_style.shadow_link_alpha,
                zorder=1,
            )
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[active_link_mask & ~occupied_mask],
                color=self.style.empty_edge_color,
                linewidth=self.style.empty_width,
                alpha=self.style.empty_alpha,
                zorder=2,
            )
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[active_link_mask & occupied_mask],
                color=self.style.edge_color,
                linewidth=self.style.occupied_width,
                alpha=self.style.occupied_alpha,
                zorder=3,
            )
            return

        if render_cache.mode == "values":
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[~active_link_mask],
                color=shadow_style.shadow_link_color,
                linewidth=self.style.empty_width * shadow_style.shadow_link_width_scale,
                alpha=shadow_style.shadow_link_alpha,
                zorder=1,
            )
            self._add_local_basis_link_collection(
                ax=ax,
                segments=render_cache.link_segments[active_link_mask],
                color=self.style.empty_edge_color,
                linewidth=self.style.empty_width,
                alpha=0.7,
                zorder=2,
            )
            return

        raise ValueError("mode must be one of 'arrows', 'dimers', or 'values'.")

    @staticmethod
    def _add_local_basis_link_collection(
        *,
        ax,
        segments: npt.NDArray[np.float64],
        color: str,
        linewidth: float,
        alpha: float,
        zorder: int,
    ) -> None:
        if segments.size == 0:
            return

        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                alpha=alpha,
                capstyle="round",
                zorder=zorder,
            )
        )

    def _draw_local_basis_nodes_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        active_node_mask: npt.NDArray[np.bool_],
        shadow_style: LocalBasisShadowStyle,
        with_site_labels: bool,
        with_site_values: bool,
    ) -> None:
        if render_cache.node_xy.size == 0:
            return

        inactive_xy = render_cache.node_xy[~active_node_mask]
        active_xy = render_cache.node_xy[active_node_mask]

        if inactive_xy.size:
            ax.scatter(
                inactive_xy[:, 0],
                inactive_xy[:, 1],
                s=self.style.node_size,
                color=shadow_style.shadow_node_color,
                alpha=shadow_style.shadow_node_alpha,
                zorder=2,
            )

        if active_xy.size:
            ax.scatter(
                active_xy[:, 0],
                active_xy[:, 1],
                **self._node_scatter_kwargs(zorder=4),
            )

        if not (with_site_labels or with_site_values):
            return

        for node_index, (px, py) in enumerate(render_cache.node_xy):
            active = bool(active_node_mask[node_index])
            if not active and not shadow_style.label_shadowed_variables:
                continue

            pieces: list[str] = []

            if with_site_labels:
                pieces.append(render_cache.site_labels[node_index])

            if with_site_values:
                variable_index = int(render_cache.site_variable_indices[node_index])
                if variable_index >= 0:
                    pieces.append(str(int(config[variable_index])))

            if pieces:
                ax.text(
                    float(px),
                    float(py),
                    "\n".join(pieces),
                    ha="center",
                    va="center",
                    fontsize=self._resolved_site_label_fontsize(),
                    color="black" if active else shadow_style.shadow_link_color,
                    alpha=1.0 if active else shadow_style.shadow_node_alpha,
                    zorder=5 if active else 3,
                )

    def _draw_local_basis_link_values_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        active_link_mask: npt.NDArray[np.bool_],
        shadow_style: LocalBasisShadowStyle,
    ) -> None:
        values = config[render_cache.link_variable_indices]

        for index, (midpoint, value) in enumerate(
            zip(render_cache.link_midpoints, values, strict=True)
        ):
            active = bool(active_link_mask[index])
            if not active and not shadow_style.label_shadowed_variables:
                continue

            ax.text(
                float(midpoint[0]),
                float(midpoint[1]),
                str(int(value)),
                ha="center",
                va="center",
                fontsize=self._resolved_link_label_fontsize(),
                color="black" if active else shadow_style.shadow_link_color,
                alpha=1.0 if active else shadow_style.shadow_link_alpha,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8 if active else 0.35,
                },
                zorder=6 if active else 3,
            )

    def _draw_local_basis_link_ids_from_grid_render_cache(
        self,
        *,
        ax,
        render_cache: _BasisGridRenderCache,
        active_link_mask: npt.NDArray[np.bool_],
        shadow_style: LocalBasisShadowStyle,
    ) -> None:
        for index, (midpoint, draw_link) in enumerate(
            zip(render_cache.link_midpoints, render_cache.draw_links, strict=True)
        ):
            active = bool(active_link_mask[index])
            if not active and not shadow_style.label_shadowed_variables:
                continue

            ax.text(
                float(midpoint[0]),
                float(midpoint[1]),
                str(int(draw_link.link_id)),
                ha="center",
                va="center",
                fontsize=self._resolved_link_label_fontsize(),
                color="purple" if active else shadow_style.shadow_link_color,
                alpha=1.0 if active else shadow_style.shadow_link_alpha,
                zorder=20 if active else 3,
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.7 if active else 0.35,
                },
            )

    def _draw_links_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
    ) -> None:
        if render_cache.mode == "arrows":
            values = config[render_cache.link_variable_indices]

            for index, value in enumerate(values):
                source = tuple(float(x) for x in render_cache.link_source_xy[index])
                target = tuple(float(x) for x in render_cache.link_target_xy[index])

                if not self._points_along_link(int(value)):
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
            return

        if render_cache.mode == "dimers":
            values = config[render_cache.link_variable_indices]
            occupied_mask = values != 0
            empty_segments = render_cache.link_segments[~occupied_mask]
            occupied_segments = render_cache.link_segments[occupied_mask]

            if empty_segments.size:
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

            if occupied_segments.size:
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
            return

        if render_cache.mode == "values":
            if render_cache.link_segments.size:
                ax.add_collection(
                    LineCollection(
                        render_cache.link_segments,
                        colors=self.style.empty_edge_color,
                        linewidths=self.style.empty_width,
                        alpha=0.7,
                        zorder=1,
                    )
                )
            return

        raise ValueError("mode must be one of 'arrows', 'dimers', or 'values'.")

    def _node_scatter_kwargs(self, *, zorder: int) -> dict[str, Any]:
        """Return scatter kwargs while preserving legacy filled-node behavior."""
        kwargs: dict[str, Any] = {
            "s": self.style.node_size,
            "zorder": zorder,
        }

        if self.style.node_face_color is None and self.style.node_edge_color is None:
            kwargs["color"] = self.style.node_color
            return kwargs

        kwargs["facecolors"] = (
            self.style.node_color
            if self.style.node_face_color is None
            else self.style.node_face_color
        )
        kwargs["edgecolors"] = (
            self.style.node_color
            if self.style.node_edge_color is None
            else self.style.node_edge_color
        )

        if self.style.node_linewidth is not None:
            kwargs["linewidths"] = self.style.node_linewidth

        return kwargs

    def _draw_nodes_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        with_site_labels: bool,
        with_site_values: bool,
    ) -> None:
        if render_cache.node_xy.size == 0:
            return

        x = render_cache.node_xy[:, 0]
        y = render_cache.node_xy[:, 1]

        ax.scatter(
            x,
            y,
            **self._node_scatter_kwargs(zorder=3),
        )

        if not (with_site_labels or with_site_values):
            return

        for node_index, (px, py) in enumerate(zip(x, y, strict=True)):
            pieces: list[str] = []

            if with_site_labels:
                pieces.append(render_cache.site_labels[node_index])

            if with_site_values:
                variable_index = int(render_cache.site_variable_indices[node_index])
                if variable_index >= 0:
                    pieces.append(str(int(config[variable_index])))

            if pieces:
                ax.text(
                    float(px),
                    float(py),
                    "\n".join(pieces),
                    ha="center",
                    va="center",
                    fontsize=self._resolved_site_label_fontsize(),
                    color="black",
                    zorder=4,
                )

    def _draw_link_values_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
    ) -> None:
        values = config[render_cache.link_variable_indices]

        for midpoint, value in zip(render_cache.link_midpoints, values, strict=True):
            ax.text(
                float(midpoint[0]),
                float(midpoint[1]),
                str(int(value)),
                ha="center",
                va="center",
                fontsize=self._resolved_link_label_fontsize(),
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
                zorder=5,
            )

    def _draw_link_ids_from_grid_render_cache(
        self,
        *,
        ax,
        render_cache: _BasisGridRenderCache,
    ) -> None:
        for midpoint, draw_link in zip(
            render_cache.link_midpoints,
            render_cache.draw_links,
            strict=True,
        ):
            ax.text(
                float(midpoint[0]),
                float(midpoint[1]),
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

    def _draw_plaquette_symbols_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        if render_cache.plaquette_symbol_style == "circulation":
            self._draw_circulation_plaquette_symbols_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
            )
            return

        if render_cache.plaquette_symbol_style == "resonance":
            self._draw_resonance_plaquette_symbols_from_grid_render_cache(
                ax=ax,
                config=config,
                render_cache=render_cache,
                plaquette_symbol_values=plaquette_symbol_values,
            )
            return

        if render_cache.plaquette_symbol_style != "none":
            raise ValueError(
                "plaquette_symbol_style must be 'auto', 'none', 'circulation', or 'resonance'."
            )

    def _draw_resonance_plaquette_symbols_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        for index, draw_plaquette in enumerate(render_cache.draw_plaquettes):
            plaquette_id = int(draw_plaquette.plaquette_id)
            center = render_cache.plaquette_centers[index]

            if plaquette_symbol_values is not None:
                symbol_info = plaquette_symbol_values.get(plaquette_id)

                if symbol_info is None:
                    continue

                symbol, color = symbol_info
                ax.annotate(
                    symbol,
                    xy=(float(center[0]), float(center[1])),
                    xytext=self.style.plaquette_symbol_offset,
                    textcoords="offset points",
                    fontsize=self.style.plaquette_symbol_fontsize,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=6,
                )
                continue

            values = tuple(
                int(config[variable_index])
                for variable_index in render_cache.plaquette_link_variable_indices[index]
            )

            symbol_info = self._theme_qdm_resonance_symbol(values)

            if symbol_info is not None:
                symbol, color = symbol_info
                ax.annotate(
                    symbol,
                    xy=(float(center[0]), float(center[1])),
                    xytext=self.style.plaquette_symbol_offset,
                    textcoords="offset points",
                    fontsize=self.style.plaquette_symbol_fontsize,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=6,
                )
                continue

            vulnerable_info = self._qdm_one_vulnerable_link(values)

            if vulnerable_info is None:
                self._draw_theme_qdm_nonflippable_symbol(
                    ax=ax,
                    center=center,
                )
                continue

            vulnerable_index, color = vulnerable_info
            color = self._theme_qdm_vulnerable_color(color)
            plaquette_midpoints = render_cache.plaquette_midpoints[index]

            if vulnerable_index >= len(plaquette_midpoints):
                continue

            self._draw_vulnerable_link_arrow(
                ax=ax,
                center=center,
                link_midpoint=plaquette_midpoints[vulnerable_index],
                color=color,
            )

    def _draw_circulation_plaquette_symbols_from_grid_render_cache(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        render_cache: _BasisGridRenderCache,
    ) -> None:
        text_items: list[tuple[Sequence[float], str, str]] = []

        for index, _draw_plaquette in enumerate(render_cache.draw_plaquettes):
            square_indices = render_cache.square_qlm_link_variable_indices[index]

            if square_indices is not None:
                values = tuple(int(config[variable_index]) for variable_index in square_indices)
                key = self._plaquette_key(values)
                payload = _SQUARE_QLM_PLAQUETTE_SYMBOLS.get(key)

                if payload is None:
                    continue

                text_items.append(
                    (
                        render_cache.plaquette_centers[index],
                        str(payload["s"]),
                        str(payload["color"]),
                    )
                )
                continue

            values = tuple(
                int(config[variable_index])
                for variable_index in render_cache.plaquette_link_variable_indices[index]
            )
            orientations = render_cache.plaquette_orientations[index]

            symbol_info = self._flux_circulation_symbol(values, orientations)

            if symbol_info is not None:
                symbol, color = symbol_info
                text_items.append((render_cache.plaquette_centers[index], symbol, color))
                continue

            vulnerable_info = self._flux_one_vulnerable_link(values, orientations)

            if vulnerable_info is None:
                continue

            vulnerable_index, color = vulnerable_info
            plaquette_midpoints = render_cache.plaquette_midpoints[index]

            if vulnerable_index >= len(plaquette_midpoints):
                continue

            self._draw_vulnerable_link_arrow(
                ax=ax,
                center=render_cache.plaquette_centers[index],
                link_midpoint=plaquette_midpoints[vulnerable_index],
                color=color,
            )

        for center, symbol, color in text_items:
            ax.annotate(
                symbol,
                xy=(float(center[0]), float(center[1])),
                xytext=self.style.plaquette_symbol_offset,
                textcoords="offset points",
                fontsize=self.style.plaquette_symbol_fontsize,
                color=color,
                ha="center",
                va="center",
                zorder=6,
            )

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
        fig, ax = plt.subplots()
        self.plot(config, ax=ax, show=show, **plot_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
