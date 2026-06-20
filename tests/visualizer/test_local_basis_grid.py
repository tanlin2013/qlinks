from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import qlinks.visualizer as visualizer_api
from qlinks.lattice import SquareLattice
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    LocalBasisGridVisualizer,
    LocalBasisShadowStyle,
    plot_local_basis_grid,
)

matplotlib.use("Agg")


@dataclass(frozen=True)
class DummyLocalRDMReadout:
    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    component_index: int | None = None


def test_plot_local_basis_grid_shadows_links_outside_support() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    reference = np.zeros(layout.n_variables, dtype=np.int64)
    shadow_style = LocalBasisShadowStyle(shadow_link_alpha=0.17)

    fig, axes = plot_local_basis_grid(
        lattice=lattice,
        layout=layout,
        local_patterns=np.array([[1], [0]], dtype=np.int64),
        variable_indices=(0,),
        reference_config=reference,
        ncols=2,
        mode="dimers",
        shadow_style=shadow_style,
        show=False,
        single_plot_kwargs={
            "with_site_labels": False,
            "with_link_values": True,
        },
    )

    assert axes.shape == (1, 2)
    assert "local 0" in axes.flat[0].get_title()
    assert "1" in axes.flat[0].get_title()

    line_collections = [
        collection
        for collection in axes.flat[0].collections
        if isinstance(collection, LineCollection)
    ]
    assert line_collections
    assert line_collections[0].get_alpha() == pytest.approx(0.17)
    assert len(line_collections[0].get_segments()) == lattice.num_links - 1

    plt.close(fig)


def test_local_basis_grid_can_use_layout_default_reference_config() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = LocalBasisGridVisualizer(lattice=lattice, layout=layout)

    fig, axes = visualizer.plot(
        np.array([-1, 1], dtype=np.int64),
        variable_indices=(0,),
        mode="arrows",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert len(axes.flat[0].patches) == lattice.num_links

    plt.close(fig)


def test_local_basis_grid_plot_readout_uses_readout_metadata() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    reference = np.zeros(layout.n_variables, dtype=np.int64)
    readout = DummyLocalRDMReadout(
        variable_indices=(0, 1),
        local_patterns=((1, 0), (0, 1)),
        component_index=3,
    )

    visualizer = LocalBasisGridVisualizer(lattice=lattice, layout=layout)
    fig, axes = visualizer.plot_readout(
        readout,
        reference_config=reference,
        ncols=2,
        mode="dimers",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert fig._suptitle is not None
    assert "component 3" in fig._suptitle.get_text()

    plt.close(fig)


def test_local_basis_grid_rejects_duplicate_variable_indices() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = LocalBasisGridVisualizer(lattice=lattice, layout=layout)

    with pytest.raises(ValueError, match="duplicates"):
        visualizer.plot(
            np.array([[1, 0]], dtype=np.int64),
            variable_indices=(0, 0),
            mode="dimers",
            show=False,
        )


def test_local_basis_grid_api_is_exported_in_all() -> None:
    assert visualizer_api.LocalBasisGridVisualizer is LocalBasisGridVisualizer
    assert visualizer_api.LocalBasisShadowStyle is LocalBasisShadowStyle
    assert visualizer_api.plot_local_basis_grid is plot_local_basis_grid
    assert "LocalBasisGridVisualizer" in visualizer_api.__all__
    assert "LocalBasisShadowStyle" in visualizer_api.__all__
    assert "plot_local_basis_grid" in visualizer_api.__all__
