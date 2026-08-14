from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout
from qlinks.visualizer.basis.configuration import BasisConfigurationVisualizer
from qlinks.visualizer.basis.formatting import (
    _amplitude_label,
    _select_cage_record,
    _zero_indices_for_mechanism,
    _zero_mechanism_label_map,
    automatic_grid_shape,
    format_basis_config,
)
from qlinks.visualizer.basis.render_cache import _BasisGridRenderCache
from qlinks.visualizer.basis.styles import (
    BasisConfigLabelStyle,
    BasisVisualizerTheme,
    LinkPlotMode,
    LinkVisualStyle,
    PeriodicImageMode,
    PlaquetteSymbolStyle,
    SiteLabelStyle,
    VisualizerBackend,
    _basis_visualizer_theme_defaults,
    _BasisVisualizerThemeDefaults,
)


@dataclass(frozen=True)
class BasisGridVisualizer:
    """Plot many basis configurations as a grid of lattice panels.

    The grid visualizer reuses the same drawing primitives as
    :class:`BasisConfigurationVisualizer` and can build an internal render cache
    for repeated plotting on the same geometry.

    Attributes:
        lattice: Geometry/topology object.
        layout: Variable layout used to interpret each configuration array.
        theme: Named presentation theme. ``"research"`` preserves the
            historical qlinks styling; ``"paper"`` uses compact publication
            defaults.
        style: Optional explicit visual style. When provided, it overrides the
            link/site style supplied by ``theme``.
        periodic_image_mode: How to draw periodic links.
        collapse_duplicate_visual_links: Whether duplicate periodic visual
            links are collapsed.
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
    coordinate_transform: npt.ArrayLike | None = None
    site_label_style: SiteLabelStyle = "cell_sublattice"

    def __post_init__(self) -> None:
        defaults = _basis_visualizer_theme_defaults(self.theme)
        if self.style is None:
            object.__setattr__(self, "style", defaults.style)

    @property
    def _theme_defaults(self) -> _BasisVisualizerThemeDefaults:
        return _basis_visualizer_theme_defaults(self.theme)

    def _single_visualizer(self) -> BasisConfigurationVisualizer:
        return BasisConfigurationVisualizer(
            lattice=self.lattice,
            layout=self.layout,
            theme=self.theme,
            style=self.style,
            periodic_image_mode=self.periodic_image_mode,
            collapse_duplicate_visual_links=self.collapse_duplicate_visual_links,
            coordinate_scale=self.coordinate_scale,
            coordinate_transform=self.coordinate_transform,
            site_label_style=self.site_label_style,
        )

    def build_render_cache(
        self,
        *,
        reference_config: npt.ArrayLike,
        mode: LinkPlotMode = "auto",
        plaquette_symbols: PlaquetteSymbolStyle = "auto",
    ) -> _BasisGridRenderCache:
        """Build a reusable render cache for this grid visualizer.

        Pass the returned cache to :meth:`plot` when plotting several batches
        with the same lattice/layout/style and plotting mode.
        """
        return self._single_visualizer().build_grid_render_cache(
            reference_config=reference_config,
            mode=mode,
            plaquette_symbols=plaquette_symbols,
        )

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
        mode: str = "auto",
        plaquette_symbols: PlaquetteSymbolStyle = "auto",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
        render_cache: _BasisGridRenderCache | None = None,
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

            "circulation":
                generic QLM-like circulation marker. Draws circular arrows when
                all link variables circulate consistently around a plaquette.
        """
        arr = np.asarray(states, dtype=np.int64)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("states must have shape (n_variables,) or (n_states, n_variables).")

        n_states = arr.shape[0]

        if n_states == 0:
            raise ValueError("states must contain at least one configuration.")

        single_visualizer = self._single_visualizer()
        single_visualizer._validate_config_batch_for_cached_grid(arr)

        if render_cache is None:
            render_cache = single_visualizer.build_grid_render_cache(
                reference_config=arr[0],
                mode=mode,
                plaquette_symbols=plaquette_symbols,
            )

        rows, cols = automatic_grid_shape(n_states, nrows=nrows, ncols=ncols)

        if labels is not None and len(labels) != n_states:
            raise ValueError("labels must have the same length as states.")

        if figsize is None:
            panel_size = self._theme_defaults.panel_size
            figsize = (panel_size * cols, panel_size * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        resolved_plaquette_symbols = render_cache.plaquette_symbol_style

        if single_plot_kwargs is None:
            single_plot_kwargs = {}

        plot_kwargs = dict(single_plot_kwargs)
        with_site_labels = bool(
            plot_kwargs.pop(
                "with_site_labels",
                self._theme_defaults.with_site_labels,
            )
        )
        with_coordinate_labels = bool(
            plot_kwargs.pop(
                "with_coordinate_labels",
                self._theme_defaults.with_coordinate_labels,
            )
        )
        with_site_values = bool(plot_kwargs.pop("with_site_values", False))
        with_link_values = bool(plot_kwargs.pop("with_link_values", False))
        with_link_ids = bool(plot_kwargs.pop("with_link_ids", False))
        with_plaquette_symbols = bool(plot_kwargs.pop("with_plaquette_symbols", True))
        plaquette_symbol_values = plot_kwargs.pop(
            "plaquette_symbol_values",
            None,
        )
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

            single_visualizer._plot_with_grid_render_cache(
                config,
                ax=ax,
                render_cache=render_cache,
                show=False,
                backend=backend,
                with_site_labels=with_site_labels,
                with_coordinate_labels=with_coordinate_labels,
                with_site_values=with_site_values,
                with_link_values=with_link_values,
                with_link_ids=with_link_ids,
                with_plaquette_symbols=with_plaquette_symbols
                and resolved_plaquette_symbols != "none",
                plaquette_symbol_values=plaquette_symbol_values,
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
                f"Cage support, signature={record.signature}, support size={record.support.size}"
            )

        return self.plot(
            states,
            labels=labels,
            suptitle=suptitle,
            **plot_kwargs,
        )

    def plot_interference_zeros(
        self,
        environment_report,
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
        environment_report:
            EnvironmentReductionReport returned by diagnose_cage_environment_reduction or
            diagnose_environment_reduction.
        basis_configs:
            Basis configuration array with shape (hilbert_size, n_variables).
        mechanism:
            One of:
                "all",
                "q_empty",
                "closed_by_same_pattern_zeros",
                "domain_blocked",
                "projector_like",
                "collective_cancellation",
                "unexplained_leakage",
                or one of the four coarse environment-removal mechanisms.
        max_states:
            Optional cap on the number of zero states to plot.
        """
        basis_configs = np.asarray(basis_configs)
        zero_indices = _zero_indices_for_mechanism(
            environment_report,
            mechanism,
        )

        if max_states is not None:
            zero_indices = zero_indices[:max_states]

        states = basis_configs[zero_indices]
        mechanism_labels = _zero_mechanism_label_map(environment_report)

        if labels is None:
            labels = [
                f"zero {int(index)}\n{mechanism_labels.get(int(index), mechanism)}"
                for index in zero_indices
            ]

        if suptitle is None:
            if mechanism == "all":
                suptitle = f"Nontrivial interference zeros ({zero_indices.size} states)"
            else:
                suptitle = (
                    f"Nontrivial interference zeros: {mechanism} ({zero_indices.size} states)"
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
    mode: LinkPlotMode = "auto",
    plaquette_symbols: PlaquetteSymbolStyle = "auto",
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "cell_sublattice",
    theme: BasisVisualizerTheme = "research",
    style: LinkVisualStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
    render_cache: _BasisGridRenderCache | None = None,
):
    """
    Functional wrapper around BasisGridVisualizer.
    """

    visualizer = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        theme=theme,
        style=style,
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
        render_cache=render_cache,
    )
