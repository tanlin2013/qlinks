import os

import matplotlib.pyplot as plt
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


def _force_circulating_plaquette(
    lattice,
    config: np.ndarray,
    plaquette_id: int,
    *,
    sign: int,
) -> None:
    """Force one plaquette to have uniform oriented circulation.

    sign=+1 should draw one circulation direction.
    sign=-1 should draw the opposite direction.
    """
    plaquette = lattice.plaquettes[plaquette_id]

    for link_id, orientation in zip(
        plaquette.links,
        plaquette.orientations,
        strict=True,
    ):
        config[int(link_id)] = int(sign) * int(orientation)


def _plaquette_ids_with_n_links(lattice, n_links: int) -> list[int]:
    return [
        int(plaquette.id) for plaquette in lattice.plaquettes if len(plaquette.links) == n_links
    ]


def _non_overlapping_plaquette_ids(
    lattice,
    *,
    n_links: int,
    count: int,
) -> list[int]:
    selected: list[int] = []
    used_links: set[int] = set()

    for plaquette in lattice.plaquettes:
        link_ids = {int(link_id) for link_id in plaquette.links}

        if len(link_ids) != n_links:
            continue

        if link_ids & used_links:
            continue

        selected.append(int(plaquette.id))
        used_links.update(link_ids)

        if len(selected) == count:
            break

    if len(selected) < count:
        raise AssertionError(
            f"Could only find {len(selected)} non-overlapping "
            f"{n_links}-link plaquettes; requested {count}."
        )

    return selected


def _force_alternating_dimer_plaquette(
    lattice,
    config: np.ndarray,
    plaquette_id: int,
    *,
    phase: int,
) -> None:
    """Force a QDM flippable plaquette.

    phase=0 gives 1010... around the plaquette.
    phase=1 gives 0101... around the plaquette.

    This assumes plaquette.links are stored in cyclic boundary order.
    """
    plaquette = lattice.plaquettes[plaquette_id]

    if len(plaquette.links) % 2 != 0:
        raise ValueError("Alternating dimer plaquette must have even length.")

    for i, link_id in enumerate(plaquette.links):
        config[int(link_id)] = 1 if (i + phase) % 2 == 0 else 0


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


def test_show_triangular_rhombus_circulation_symbols() -> None:
    lattice = TriangularLattice(
        4,
        4,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    rhombus_ids = _plaquette_ids_with_n_links(lattice, 4)
    assert len(rhombus_ids) >= 6

    config = np.ones(layout.n_variables, dtype=np.int64)

    # Pick separated-ish rhombi so the visual result is easy to inspect.
    selected = [
        rhombus_ids[0],
        rhombus_ids[len(rhombus_ids) // 4],
        rhombus_ids[len(rhombus_ids) // 2],
        rhombus_ids[-1],
    ]

    signs = [1, -1, 1, -1]

    for plaquette_id, sign in zip(selected, signs, strict=True):
        _force_circulating_plaquette(
            lattice,
            config,
            plaquette_id,
            sign=sign,
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
        title="Triangular lattice: circulation symbols only on rhombi",
    )

    plt.show()
    plt.close(fig)


def test_show_triangular_rhombus_circulation_symbol_grid() -> None:
    lattice = TriangularLattice(
        4,
        4,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    rhombus_ids = _plaquette_ids_with_n_links(lattice, 4)
    assert len(rhombus_ids) >= 6

    states = []

    choices = [
        [(rhombus_ids[0], 1), (rhombus_ids[1], -1)],
        [(rhombus_ids[len(rhombus_ids) // 3], 1)],
        [(rhombus_ids[len(rhombus_ids) // 2], -1)],
        [(rhombus_ids[-2], 1), (rhombus_ids[-1], -1)],
    ]

    for forced_plaquettes in choices:
        config = np.ones(layout.n_variables, dtype=np.int64)
        for plaquette_id, sign in forced_plaquettes:
            _force_circulating_plaquette(
                lattice,
                config,
                plaquette_id,
                sign=sign,
            )
        states.append(config)

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=np.asarray(states, dtype=np.int64),
        ncols=2,
        mode="arrows",
        show=False,
        plaquette_symbols="circulation",
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        show_config_label=False,
        single_plot_kwargs={
            "with_site_labels": False,
        },
    )

    fig.suptitle("Triangular lattice: rhombus circulation symbols in grid")
    plt.show()
    plt.close(fig)


def test_show_honeycomb_hexagon_circulation_symbols() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    hexagon_ids = _plaquette_ids_with_n_links(lattice, 6)
    assert len(hexagon_ids) >= 4

    config = np.ones(layout.n_variables, dtype=np.int64)

    selected = [
        hexagon_ids[0],
        hexagon_ids[len(hexagon_ids) // 4],
        hexagon_ids[len(hexagon_ids) // 2],
        hexagon_ids[-1],
    ]

    signs = [1, -1, 1, -1]

    for plaquette_id, sign in zip(selected, signs, strict=True):
        _force_circulating_plaquette(
            lattice,
            config,
            plaquette_id,
            sign=sign,
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
        title="Honeycomb lattice: hexagon circulation symbols",
    )

    plt.show()
    plt.close(fig)


def test_show_honeycomb_hexagon_circulation_symbol_grid() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    hexagon_ids = _plaquette_ids_with_n_links(lattice, 6)
    assert len(hexagon_ids) >= 6

    states = []

    choices = [
        [(hexagon_ids[0], 1), (hexagon_ids[1], -1)],
        [(hexagon_ids[len(hexagon_ids) // 3], 1)],
        [(hexagon_ids[len(hexagon_ids) // 2], -1)],
        [(hexagon_ids[-2], 1), (hexagon_ids[-1], -1)],
    ]

    for forced_plaquettes in choices:
        config = np.ones(layout.n_variables, dtype=np.int64)
        for plaquette_id, sign in forced_plaquettes:
            _force_circulating_plaquette(
                lattice,
                config,
                plaquette_id,
                sign=sign,
            )
        states.append(config)

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=np.asarray(states, dtype=np.int64),
        ncols=2,
        mode="arrows",
        show=False,
        plaquette_symbols="circulation",
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
        show_config_label=False,
        single_plot_kwargs={
            "with_site_labels": False,
        },
    )

    fig.suptitle("Honeycomb lattice: hexagon circulation symbols in grid")
    plt.show()
    plt.close(fig)


def test_show_triangular_qdm_rhombus_dimer_symbols() -> None:
    lattice = TriangularLattice(
        4,
        4,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    selected = _non_overlapping_plaquette_ids(
        lattice,
        n_links=4,
        count=4,
    )

    # Start from all-empty dimers. Only explicitly forced rhombi should be
    # flippable.
    config = np.zeros(layout.n_variables, dtype=np.int64)

    phases = [0, 1, 0, 1]
    for plaquette_id, phase in zip(selected, phases, strict=True):
        _force_alternating_dimer_plaquette(
            lattice,
            config,
            plaquette_id,
            phase=phase,
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
        title="Triangular QDM: alternating dimers on rhombi",
    )

    plt.show()
    plt.close(fig)


def test_show_honeycomb_qdm_hexagon_dimer_symbols() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    selected = _non_overlapping_plaquette_ids(
        lattice,
        n_links=6,
        count=4,
    )

    # Start from all-empty dimers. Only explicitly forced hexagons should be
    # flippable.
    config = np.zeros(layout.n_variables, dtype=np.int64)

    phases = [0, 1, 0, 1]
    for plaquette_id, phase in zip(selected, phases, strict=True):
        _force_alternating_dimer_plaquette(
            lattice,
            config,
            plaquette_id,
            phase=phase,
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
        title="Honeycomb QDM: alternating dimers on hexagons",
    )

    plt.show()
    plt.close(fig)
