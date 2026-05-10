from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    BasisGridVisualizer,
    automatic_grid_shape,
    format_basis_config,
    plot_basis_config,
    plot_basis_grid,
)

matplotlib.use("Agg")


def test_square_qdm_dimer_plot_runs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    config = np.array([1, 0, 1, 0], dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)
    returned_ax = visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_plaquette_symbols=True,
    )

    assert returned_ax is ax

    # Nodes are drawn as one PathCollection.
    assert len(ax.collections) >= 1

    # Dimer links are drawn as Line2D objects.
    assert len(ax.lines) == lattice.num_links

    # Plaquette symbol should add text beyond node labels.
    assert len(ax.texts) >= lattice.num_sites + 1

    plt.close(fig)


def test_square_qlm_arrow_plot_runs() -> None:
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
        with_plaquette_symbols=True,
    )

    # Arrow links are matplotlib patches.
    assert len(ax.patches) == lattice.num_links

    # Nodes.
    assert len(ax.collections) >= 1

    # Plaquette symbol + site labels.
    assert len(ax.texts) >= lattice.num_sites + 1

    plt.close(fig)


def test_functional_plot_wrapper_runs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    config = np.array([1, 0, 1, 0], dtype=np.int64)

    fig, ax = plt.subplots()

    returned_ax = plot_basis_config(
        lattice,
        config,
        layout=layout,
        ax=ax,
        show=False,
        mode="dimers",
    )

    assert returned_ax is ax

    plt.close(fig)


def test_chain_site_values_plot_runs() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    config = np.array([1, 0, 1, 0], dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)
    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="values",
        with_site_values=True,
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) >= 1
    assert len(ax.texts) >= lattice.num_sites

    plt.close(fig)


def test_periodic_square_plot_flattens_wrapping_links() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    config = np.ones(lattice.num_links, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)
    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_plaquette_symbols=False,
    )

    # Periodic square 2x2 has original nodes plus duplicate boundary nodes
    # introduced by wrapping links.
    assert len(ax.collections) >= 1
    assert len(ax.lines) == lattice.num_links

    plt.close(fig)


def test_save_plot(tmp_path: Path) -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    config = np.array([1, 0, 1, 0], dtype=np.int64)

    out = tmp_path / "basis.png"

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)
    visualizer.save(
        config,
        out,
        show=False,
        mode="dimers",
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_link_value_without_layout_assumes_link_index_order() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    config = np.array([1, 0, 1, 0], dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    assert visualizer.link_value(config, 0) == 1
    assert visualizer.link_value(config, 1) == 0


def test_automatic_grid_shape_near_square() -> None:
    assert automatic_grid_shape(1) == (1, 1)
    assert automatic_grid_shape(4) == (2, 2)
    assert automatic_grid_shape(5) == (2, 3)
    assert automatic_grid_shape(10) == (3, 4)


def test_automatic_grid_shape_with_ncols() -> None:
    assert automatic_grid_shape(10, ncols=3) == (4, 3)


def test_automatic_grid_shape_with_nrows() -> None:
    assert automatic_grid_shape(10, nrows=2) == (2, 5)


def test_automatic_grid_shape_rejects_too_small_grid() -> None:
    with pytest.raises(ValueError, match="smaller"):
        automatic_grid_shape(10, nrows=2, ncols=4)


def test_format_basis_config_binary_compact() -> None:
    config = np.array([0, 1, 0, 1], dtype=np.int64)
    assert format_basis_config(config, style="compact") == "0101"


def test_format_basis_config_flux_compact() -> None:
    config = np.array([1, -1, 1, -1], dtype=np.int64)
    assert format_basis_config(config, style="compact") == "1,-1,1,-1"


def test_basis_grid_visualizer_chain_site_values() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    states = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    visualizer = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
    )

    fig, axes = visualizer.plot(
        states,
        ncols=2,
        mode="values",
        show=False,
        show_config_label=True,
        single_plot_kwargs={
            "with_site_values": True,
            "with_plaquette_symbols": False,
        },
    )

    assert axes.shape == (2, 2)

    # Three populated panels, one empty panel.
    populated_titles = [ax.get_title() for ax in axes.flat if ax.get_title()]
    assert len(populated_titles) == 3
    assert "state 0" in populated_titles[0]
    assert "0000" in populated_titles[0]

    plt.close(fig)


def test_plot_basis_grid_functional_wrapper() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    states = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=2,
        mode="values",
        show=False,
        show_config_label=True,
        single_plot_kwargs={
            "with_site_values": True,
            "with_plaquette_symbols": False,
        },
    )

    assert axes.shape == (1, 2)

    plt.close(fig)


def test_basis_grid_accepts_single_config() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    config = np.array([0, 1, 0, 1], dtype=np.int64)

    visualizer = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
    )

    fig, axes = visualizer.plot(
        config,
        mode="values",
        show=False,
        show_config_label=True,
        single_plot_kwargs={
            "with_site_values": True,
            "with_plaquette_symbols": False,
        },
    )

    assert axes.shape == (1, 1)
    assert "0101" in axes.flat[0].get_title()

    plt.close(fig)


def test_square_qlm_symbols_are_square_specific() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    states = np.array(
        [
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
        ],
        dtype=np.int64,
    )

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=2,
        mode="arrows",
        plaquette_symbols="square_qlm",
        show=False,
    )

    assert axes.shape == (1, 2)

    # At least node labels plus plaquette symbol texts should exist.
    assert sum(len(ax.texts) for ax in axes.flat) > 0

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
