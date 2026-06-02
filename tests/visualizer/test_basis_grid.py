import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qlinks.lattice import (
    ChainLattice,
    SquareLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    BasisGridVisualizer,
    automatic_grid_shape,
    plot_basis_grid,
)
from qlinks.visualizer.basis import _zero_indices_for_mechanism

matplotlib.use("Agg")


class DummyZeroReport:
    def __init__(self, zero_index, mechanism_label):
        self.zero_index = zero_index
        self.mechanism_label = mechanism_label


class DummyClassificationReport:
    zero_reports = (
        DummyZeroReport(2, "q_empty"),
        DummyZeroReport(5, "projector_like"),
    )

    q_empty_zero_indices = np.array([2], dtype=np.int64)
    closed_by_known_zero_indices = np.array([], dtype=np.int64)
    projector_like_zero_indices = np.array([5], dtype=np.int64)
    unexplained_leakage_zero_indices = np.array([], dtype=np.int64)

    regional_mechanism_zero_indices = np.array([2], dtype=np.int64)
    extended_mechanism_zero_indices = np.array([5], dtype=np.int64)
    failure_mechanism_zero_indices = np.array([], dtype=np.int64)


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
        plaquette_symbols="none",
        single_plot_kwargs={
            "with_site_values": True,
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
        plaquette_symbols="none",
        single_plot_kwargs={
            "with_site_values": True,
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
        plaquette_symbols="none",
        single_plot_kwargs={
            "with_site_values": True,
        },
    )

    assert axes.shape == (1, 1)
    assert "0101" in axes.flat[0].get_title()

    plt.close(fig)


def test_basis_grid_networkx_backend_runs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    states = np.ones((2, layout.n_variables), dtype=np.int64)

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=2,
        backend="networkx",
        mode="dimers",
        show=False,
        plaquette_symbols="none",
    )

    assert axes.shape == (1, 2)

    plt.close(fig)


def test_basis_grid_reuses_draw_primitives(monkeypatch) -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    states = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    call_count = 0
    original = BasisConfigurationVisualizer._draw_primitives

    def counted_draw_primitives(self):
        nonlocal call_count
        call_count += 1
        return original(self)

    monkeypatch.setattr(
        BasisConfigurationVisualizer,
        "_draw_primitives",
        counted_draw_primitives,
    )

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=2,
        mode="dimers",
        show=False,
        plaquette_symbols="none",
    )

    assert axes.shape == (2, 2)
    assert call_count == 1

    plt.close(fig)


def test_basis_grid_reuses_plaquette_primitives(monkeypatch) -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    states = np.ones((3, layout.n_variables), dtype=np.int64)

    call_count = 0
    original = BasisConfigurationVisualizer._draw_square_generic_plaquette_primitives

    def counted_draw_square_plaquettes(self):
        nonlocal call_count
        call_count += 1
        return original(self)

    monkeypatch.setattr(
        BasisConfigurationVisualizer,
        "_draw_square_generic_plaquette_primitives",
        counted_draw_square_plaquettes,
    )

    fig, _ = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        ncols=3,
        mode="arrows",
        show=False,
        periodic_image_mode="positive_patch",
    )

    assert call_count == 1

    plt.close(fig)


def test_zero_indices_for_mechanism_all():
    report = DummyClassificationReport()

    indices = _zero_indices_for_mechanism(report, "all")

    assert indices.tolist() == [2, 5]


def test_zero_indices_for_mechanism_projector_like():
    report = DummyClassificationReport()

    indices = _zero_indices_for_mechanism(report, "projector_like")

    assert indices.tolist() == [5]


def test_zero_indices_for_mechanism_extended_group():
    report = DummyClassificationReport()

    indices = _zero_indices_for_mechanism(report, "extended")

    assert indices.tolist() == [5]


def test_basis_grid_auto_mode_resolves_before_plaquette_symbols() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    states = [
        np.zeros(layout.n_variables, dtype=np.int64),
        np.ones(layout.n_variables, dtype=np.int64),
    ]

    grid = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
    )

    fig, axes = grid.plot(
        states,
        mode="auto",
        plaquette_symbols="auto",
        show=False,
    )

    assert fig is not None
    plt.close(fig)


def test_basis_grid_auto_mode_resolves_flux_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    states = [
        np.ones(layout.n_variables, dtype=np.int64),
    ]

    grid = BasisGridVisualizer(
        lattice=lattice,
        layout=layout,
    )

    fig, axes = grid.plot(
        states,
        mode="auto",
        plaquette_symbols="auto",
        show=False,
    )

    assert fig is not None
    plt.close(fig)
