from __future__ import annotations

import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout
from qlinks.visualizer.basis.configuration import BasisConfigurationVisualizer
from qlinks.visualizer.basis.styles import (
    BasisVisualizerTheme,
    LinkPlotMode,
    LinkVisualStyle,
    PeriodicImageMode,
    PlaquetteSymbolStyle,
    SiteLabelStyle,
    VisualizerBackend,
)


def plot_basis_config(
    lattice: LatticeGraph,
    config: npt.ArrayLike,
    *,
    layout: VariableLayout | None = None,
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
    title: str | None = None,
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "cell_sublattice",
    theme: BasisVisualizerTheme = "research",
    style: LinkVisualStyle | None = None,
):
    """
    Functional convenience wrapper around BasisConfigurationVisualizer.
    """

    visualizer = BasisConfigurationVisualizer(
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
        config,
        ax=ax,
        show=show,
        backend=backend,
        mode=mode,
        with_site_labels=with_site_labels,
        with_coordinate_labels=with_coordinate_labels,
        with_site_values=with_site_values,
        with_link_values=with_link_values,
        with_link_ids=with_link_ids,
        with_plaquette_symbols=with_plaquette_symbols,
        plaquette_symbol_style=plaquette_symbol_style,
        title=title,
    )
