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
    BasisGridVisualizer,
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
        site_label_style="sublattice_cell",
        coordinate_transform=np.array(
            [
                [1.0, 0.0],
                [0.0, 0.72],
            ],
            dtype=float,
        ),
        style=LinkVisualStyle(
            node_size=100.0,
        ),
    )

    visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_site_labels=True,
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


def _force_drawn_plaquettes_to_circulate(
    *,
    visualizer: BasisConfigurationVisualizer,
    layout: VariableLayout,
    config: np.ndarray,
    n_plaquettes: int,
    sign: int = 1,
) -> list[int]:
    """Force several actually-drawn plaquettes to have circulation symbols."""
    draw_plaquettes = visualizer._draw_plaquette_primitives()

    forced_ids: list[int] = []

    for draw_plaquette in draw_plaquettes[:n_plaquettes]:
        plaquette_id = int(draw_plaquette.plaquette_id)
        plaquette = visualizer.lattice.plaquettes[plaquette_id]

        for link_id, orientation in zip(
            plaquette.links,
            plaquette.orientations,
            strict=True,
        ):
            variable_index = layout.link_variable_index(int(link_id))
            config[variable_index] = sign * int(orientation)

        forced_ids.append(plaquette_id)

    return forced_ids


def _debug_drawn_plaquette_centers(
    visualizer: BasisConfigurationVisualizer,
) -> None:
    print()
    print("drawn plaquette centers:")

    for draw_plaquette in visualizer._draw_plaquette_primitives():
        print(
            "plaquette_id =",
            int(draw_plaquette.plaquette_id),
            "image_shift =",
            draw_plaquette.image_shift,
            "visual_cell =",
            draw_plaquette.visual_cell,
            "center =",
            np.asarray(draw_plaquette.center, dtype=float),
        )


def test_show_honeycomb_grid_multiple_circulation_symbols() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    coordinate_transform = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.72],
        ],
        dtype=float,
    )

    style = LinkVisualStyle(
        node_size=100.0,
        plaquette_symbol_fontsize=22.0,
        # Keep this zero while debugging center correctness.
        plaquette_symbol_offset=(0.0, 0.0),
    )

    single_visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        site_label_style="sublattice_cell",
        coordinate_transform=coordinate_transform,
        style=style,
    )

    config = -np.ones(layout.n_variables, dtype=np.int64)

    forced_ids = _force_drawn_plaquettes_to_circulate(
        visualizer=single_visualizer,
        layout=layout,
        config=config,
        n_plaquettes=4,
        sign=1,
    )

    print("forced plaquette ids =", forced_ids)
    _debug_drawn_plaquette_centers(single_visualizer)

    # Single plot: this should be correct first.
    single_visualizer.plot(
        config,
        show=True,
        mode="arrows",
        with_site_labels=True,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
        title="Single visualizer: multiple forced circulation symbols",
    )

    # Grid plot: same state repeated. Symbols should match the single plot.
    grid_visualizer = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        site_label_style="sublattice_cell",
        coordinate_transform=coordinate_transform,
        style=style,
    )

    states = np.stack([config, config], axis=0)

    grid_visualizer.plot(
        states,
        ncols=2,
        mode="arrows",
        plaquette_symbols="circulation",
        show_config_label=False,
        suptitle="Grid visualizer: repeated forced circulation state",
        single_plot_kwargs={
            "with_site_labels": True,
            "with_site_values": False,
            "with_link_values": False,
        },
    )
