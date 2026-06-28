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
from qlinks.models import KagomeQDMModel, KagomeQLMModel
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    BasisGridVisualizer,
    format_basis_config,
    plot_basis_config,
)

matplotlib.use("Agg")


def line_collection_segment_count(ax) -> int:
    return sum(
        len(collection.get_segments())
        for collection in ax.collections
        if isinstance(collection, LineCollection)
    )


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
        with_plaquette_symbols=False,
    )

    assert returned_ax is ax

    # Nodes are drawn as one PathCollection.
    assert len(ax.collections) >= 1

    # Dimer links are drawn as Line2D objects.
    assert line_collection_segment_count(ax) == lattice.num_links

    # Plaquette symbol should add text beyond node labels.
    assert len(ax.texts) >= lattice.num_sites

    plt.close(fig)


def test_link_value_plot_draws_batched_backbone() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    config = np.array([1, 0, 1, 0], dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)
    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="values",
        with_link_values=True,
        with_plaquette_symbols=False,
    )

    assert line_collection_segment_count(ax) == lattice.num_links
    assert len(ax.texts) >= lattice.num_links

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


def test_format_basis_config_binary_compact() -> None:
    config = np.array([0, 1, 0, 1], dtype=np.int64)
    assert format_basis_config(config, style="compact") == "0101"


def test_format_basis_config_flux_compact() -> None:
    config = np.array([1, -1, 1, -1], dtype=np.int64)
    assert format_basis_config(config, style="compact") == "1,-1,1,-1"


@pytest.mark.parametrize(
    "lattice",
    [
        TriangularLattice(
            3,
            3,
            boundary_condition="open",
            include_triangles=True,
            include_rhombi=True,
        ),
        HoneycombLattice(3, 3, boundary_condition="open"),
    ],
)
def test_non_square_open_lattice_visualizer_runs(lattice) -> None:
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) > 0

    plt.close(fig)


def test_auto_mode_uses_dimers_for_binary_link_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    config = layout.default_config()

    assert (
        visualizer._resolve_link_plot_mode(
            config=config,
            mode="auto",
        )
        == "dimers"
    )


def test_auto_mode_uses_arrows_for_spin_half_flux_link_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    config = layout.default_config()

    assert (
        visualizer._resolve_link_plot_mode(
            config=config,
            mode="auto",
        )
        == "arrows"
    )


def test_auto_symbols_follow_inferred_mode_for_binary_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    assert (
        visualizer._resolve_plaquette_symbol_style(
            mode=visualizer._resolve_link_plot_mode(
                config=layout.default_config(),
                mode="auto",
            ),
            plaquette_symbol_style="auto",
        )
        == "resonance"
    )


def test_auto_symbols_follow_inferred_mode_for_flux_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    assert (
        visualizer._resolve_plaquette_symbol_style(
            mode=visualizer._resolve_link_plot_mode(
                config=layout.default_config(),
                mode="auto",
            ),
            plaquette_symbol_style="auto",
        )
        == "circulation"
    )


def test_kagome_qdm_basis_visualizer_smoke() -> None:
    model = KagomeQDMModel(lx=2, ly=3, boundary_condition="periodic")
    build = model.build(builder="sparse", basis_solver="dfs")

    visualizer = BasisConfigurationVisualizer(
        lattice=model.lattice,
        layout=build.layout,
        periodic_image_mode="positive_patch",
    )
    ax = visualizer.plot(
        build.basis.states[0],
        mode="dimers",
        with_plaquette_symbols=True,
        show=False,
    )

    assert ax is not None


def test_kagome_qlm_basis_grid_visualizer_smoke() -> None:
    model = KagomeQLMModel(lx=2, ly=2, boundary_condition="periodic")
    build = model.build(builder="sparse", basis_solver="dfs")

    visualizer = BasisGridVisualizer(
        lattice=model.lattice,
        layout=build.layout,
        periodic_image_mode="positive_patch",
    )
    fig, axes = visualizer.plot(
        build.basis.states[:2],
        mode="arrows",
        plaquette_symbols="auto",
        show=False,
    )

    assert fig is not None
    assert axes.size >= 2
