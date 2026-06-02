import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qlinks.lattice import (
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
)

matplotlib.use("Agg")


def test_networkx_backend_runs_for_square_arrows() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        backend="networkx",
        mode="arrows",
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) > 0

    plt.close(fig)


def test_networkx_backend_runs_for_dimers() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        backend="networkx",
        mode="dimers",
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) > 0

    plt.close(fig)


def test_networkx_backend_runs_for_triangular_positive_patch() -> None:
    lattice = TriangularLattice(
        3,
        3,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        backend="networkx",
        mode="arrows",
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) > 0

    plt.close(fig)


def test_networkx_backend_runs_for_honeycomb_positive_patch() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        backend="networkx",
        mode="arrows",
        with_plaquette_symbols=False,
    )

    assert len(ax.collections) > 0

    plt.close(fig)


def test_square_generic_visual_plaquette_links_match_lattice_links() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    by_id = {
        int(draw_plaquette.plaquette_id): draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
    }

    for plaquette in lattice.plaquettes:
        plaquette_id = int(plaquette.id)

        assert by_id[plaquette_id].link_ids == tuple(int(link_id) for link_id in plaquette.links)

        assert by_id[plaquette_id].link_orientations == tuple(
            int(orientation) for orientation in plaquette.orientations
        )
