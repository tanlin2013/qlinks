from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    BasisGridVisualizer,
    LinkVisualStyle,
    automatic_grid_shape,
    format_basis_config,
    plot_basis_config,
    plot_basis_grid,
)
from qlinks.visualizer.basis import _SQUARE_QLM_PLAQUETTE_SYMBOLS

matplotlib.use("Agg")



def test_style_infers_site_label_fontsize_from_node_size() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    small = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=25.0),
    )

    large = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=400.0),
    )

    assert small._resolved_site_label_fontsize() < large._resolved_site_label_fontsize()



def test_style_explicit_site_label_fontsize_overrides_inference() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(
            node_size=25.0,
            site_label_fontsize=7.5,
        ),
    )

    assert visualizer._resolved_site_label_fontsize() == 7.5



def test_style_infers_arrow_parameters_from_node_size() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    small = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=25.0),
    )

    large = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=400.0),
    )

    assert small._resolved_arrow_mutation_scale() < large._resolved_arrow_mutation_scale()
    assert small._resolved_arrow_shrink_points() < large._resolved_arrow_shrink_points()



def test_style_explicit_arrow_parameters_override_inference() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(
            node_size=25.0,
            arrow_mutation_scale=9.0,
            arrow_shrink_points=0.0,
        ),
    )

    assert visualizer._resolved_arrow_mutation_scale() == 9.0
    assert visualizer._resolved_arrow_shrink_points() == 0.0



def test_honeycomb_site_label_includes_sublattice() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        site_label_style="sublattice_cell",
    )

    labels = {
        visualizer._format_site_label(site.id)
        for site in lattice.sites
        if tuple(site.cell) == (0, 0)
    }

    assert "A(0, 0)" in labels
    assert "B(0, 0)" in labels
