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



def test_square_qlm_symbols_draw_one_symbol_per_open_square_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())
    config = np.array([1, -1, 1, -1], dtype=np.int64)

    fig, ax = plt.subplots()
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="square_qlm",
    )

    assert len(ax.texts) == lattice.num_plaquettes

    plt.close(fig)



def test_circulation_symbols_run_on_square() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    states = np.array(
        [
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        mode="arrows",
        plaquette_symbols="circulation",
        show=False,
    )

    assert axes.shape == (1, 1)

    plt.close(fig)



def test_square_2_by_2_positive_patch_draws_four_plaquette_visual_cells() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = visualizer._draw_square_qlm_plaquette_primitives()

    visual_cells = {p.visual_cell for p in draw_plaquettes}

    assert {(0, 0), (1, 0), (0, 1), (1, 1)} <= visual_cells



def test_square_qlm_visual_symbol_key_accepts_circulating_open_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.zeros(lattice.num_links, dtype=np.int64)
    plaquette = lattice.plaquettes[0]

    for link_id, orientation in zip(
        plaquette.links,
        plaquette.orientations,
        strict=True,
    ):
        config[int(link_id)] = int(orientation)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS



def test_square_qlm_visual_symbol_key_accepts_anticirculating_open_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.zeros(lattice.num_links, dtype=np.int64)
    plaquette = lattice.plaquettes[0]

    for link_id, orientation in zip(
        plaquette.links,
        plaquette.orientations,
        strict=True,
    ):
        config[int(link_id)] = -int(orientation)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS



def test_square_qlm_visual_symbol_values_2x2_pbc_use_visual_cell() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.array([1, -1, -1, 1, 1, 1, -1, -1], dtype=np.int64)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS



def test_square_qlm_visual_symbol_key_order_bottom_left_right_top() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.array([1, -1, -1, 1, 1, 1, -1, -1], dtype=np.int64)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    # bottom, left, right, top for visual cell (0, 0)
    assert values == [1, -1, 1, -1]



def test_honeycomb_drawn_plaquettes_carry_link_metadata() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        assert len(draw_plaquette.link_ids) == 6
        assert len(draw_plaquette.link_orientations) == 6
        assert all(isinstance(link_id, int) for link_id in draw_plaquette.link_ids)
        assert set(draw_plaquette.link_orientations) <= {-1, 1}
