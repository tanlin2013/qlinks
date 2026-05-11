import os

import numpy as np
import pytest

from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    LinkVisualStyle,
    plot_basis_grid,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("QLINKS_SHOW_PLOTS") != "1",
    reason="Manual visual tests. Run with QLINKS_SHOW_PLOTS=1.",
)


def test_show_chain_pbc_positive_patch() -> None:
    lattice = ChainLattice(6, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    config = np.ones(layout.n_variables, dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_plaquette_symbols=False,
        title="Chain L=6 PBC, positive_patch",
    )


def test_show_square_pbc_positive_patch_arrows_with_symbols() -> None:
    lattice = SquareLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    rng = np.random.default_rng(0)
    config = rng.choice(np.array([-1, 1], dtype=np.int64), size=layout.n_variables)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_plaquette_symbols=True,
        plaquette_symbol_style="square_qlm",
        title="Square 4x4 PBC, QLM arrows + square symbols",
    )


def test_show_square_2_by_2_pbc_positive_patch_arrows() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    config = np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_plaquette_symbols=True,
        plaquette_symbol_style="square_qlm",
        title="Square 2x2 PBC, positive_patch",
    )


def test_show_square_pbc_dimers() -> None:
    lattice = SquareLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    rng = np.random.default_rng(1)
    config = rng.choice(np.array([0, 1], dtype=np.int64), size=layout.n_variables)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        show=True,
        mode="dimers",
        with_plaquette_symbols=False,
        title="Square 4x4 PBC, dimers",
    )


def test_show_triangular_pbc_positive_patch() -> None:
    lattice = TriangularLattice(
        4,
        4,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    rng = np.random.default_rng(2)
    config = rng.choice(np.array([-1, 1], dtype=np.int64), size=layout.n_variables)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_plaquette_symbols=False,
        title="Triangular 4x4 PBC, positive_patch",
    )


def test_show_honeycomb_pbc_positive_patch() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    rng = np.random.default_rng(3)
    config = rng.choice(np.array([-1, 1], dtype=np.int64), size=layout.n_variables)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        style=LinkVisualStyle(
            node_size=50.0,
            arrow_linewidth=1.2,
            arrow_mutation_scale=10.0,
            arrow_shrink_points=2.0,
            site_label_fontsize=5.0,
        ),
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=False,
        title="Honeycomb 4x4 PBC, positive_patch",
    )


def test_show_square_basis_grid_positive_patch() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    states = np.array(
        [
            [1, -1, 1, -1, -1, 1, -1, 1],
            [-1, 1, -1, 1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
        ],
        dtype=np.int64,
    )

    plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=2,
        show=True,
        mode="arrows",
        plaquette_symbols="square_qlm",
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        show_config_label=True,
        suptitle="Square 2x2 PBC basis grid",
    )
