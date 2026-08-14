"""Basis-configuration and basis-grid visualization APIs.

Implementation is split by rendering responsibility. Public callers may import the
curated API from this package; implementation modules import siblings directly.
"""

from qlinks.visualizer.basis.api import plot_basis_config
from qlinks.visualizer.basis.configuration import BasisConfigurationVisualizer
from qlinks.visualizer.basis.formatting import automatic_grid_shape, format_basis_config
from qlinks.visualizer.basis.grid import BasisGridVisualizer, plot_basis_grid
from qlinks.visualizer.basis.local_grid import (
    LocalBasisGridVisualizer,
    plot_local_basis_grid,
    plot_local_structure_readout,
    plot_local_structure_report,
)
from qlinks.visualizer.basis.styles import (
    BasisConfigLabelStyle,
    BasisVisualizerTheme,
    LinkPlotMode,
    LinkVisualStyle,
    LocalBasisShadowStyle,
    PeriodicImageMode,
    PlaquetteSymbolMode,
    PlaquetteSymbolStyle,
    SiteLabelStyle,
    VisualizerBackend,
    basis_visual_style,
)

__all__ = [
    "BasisConfigLabelStyle",
    "BasisConfigurationVisualizer",
    "BasisGridVisualizer",
    "BasisVisualizerTheme",
    "LinkPlotMode",
    "LinkVisualStyle",
    "LocalBasisGridVisualizer",
    "LocalBasisShadowStyle",
    "PeriodicImageMode",
    "PlaquetteSymbolMode",
    "PlaquetteSymbolStyle",
    "SiteLabelStyle",
    "VisualizerBackend",
    "automatic_grid_shape",
    "basis_visual_style",
    "format_basis_config",
    "plot_basis_config",
    "plot_basis_grid",
    "plot_local_basis_grid",
    "plot_local_structure_readout",
    "plot_local_structure_report",
]
