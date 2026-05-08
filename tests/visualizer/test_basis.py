from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import BasisConfigurationVisualizer, plot_basis_config


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
