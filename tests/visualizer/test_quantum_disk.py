from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import qlinks.visualizer as visualizer_api
from qlinks.lattice import SquareLattice
from qlinks.models import SquareQuantumDiskModel
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    QuantumDiskBasisGridVisualizer,
    QuantumDiskConfigurationVisualizer,
    QuantumDiskVisualStyle,
    plot_quantum_disk_basis_grid,
)

matplotlib.use("Agg")


@dataclass(frozen=True)
class DummyCageRecord:
    support: np.ndarray
    local_state: np.ndarray
    signature: tuple[int, int]


@dataclass(frozen=True)
class DummyZeroReport:
    zero_index: int
    probe_mechanism_label: str

    @property
    def removal_mechanism(self) -> str:
        if self.probe_mechanism_label == "q_empty":
            return "no_environment_weight"
        if self.probe_mechanism_label in {"domain_blocked", "projector_like"}:
            return "projective_annihilation"
        if self.probe_mechanism_label in {
            "closed_by_same_pattern_zeros",
            "collective_cancellation",
        }:
            return "same_local_cancellation_pattern"
        return "unsafe"


@dataclass(frozen=True)
class DummyEnvironmentReport:
    zero_reports: tuple[DummyZeroReport, ...]


def test_quantum_disk_visualizer_api_is_exported() -> None:
    assert visualizer_api.QuantumDiskConfigurationVisualizer is QuantumDiskConfigurationVisualizer
    assert visualizer_api.QuantumDiskBasisGridVisualizer is QuantumDiskBasisGridVisualizer
    assert visualizer_api.QuantumDiskVisualStyle is QuantumDiskVisualStyle
    assert visualizer_api.plot_quantum_disk_basis_grid is plot_quantum_disk_basis_grid
    assert "QuantumDiskConfigurationVisualizer" in visualizer_api.__all__
    assert "QuantumDiskBasisGridVisualizer" in visualizer_api.__all__
    assert "plot_quantum_disk_basis_grid" in visualizer_api.__all__


def test_quantum_disk_configuration_visualizer_draws_occupied_disks() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())
    visualizer = QuantumDiskConfigurationVisualizer(lattice=lattice, layout=layout)

    fig, ax = plt.subplots()
    visualizer.plot(
        np.array([1, 0, 0, 1], dtype=np.int64),
        ax=ax,
        show=False,
        with_site_values=True,
        with_hop_bonds=False,
    )

    disks = [patch for patch in ax.patches if isinstance(patch, Circle)]
    assert len(disks) == 2
    assert ax.get_aspect() == 1.0

    plt.close(fig)


def test_quantum_disk_configuration_visualizer_can_draw_hop_bonds() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    visualizer = QuantumDiskConfigurationVisualizer.from_model(model)

    fig, ax = plt.subplots()
    visualizer.plot(
        np.array([0, 1, 1, 0], dtype=np.int64),
        ax=ax,
        show=False,
        with_hop_bonds=True,
        with_site_labels=False,
    )

    line_collections = [
        collection for collection in ax.collections if isinstance(collection, LineCollection)
    ]
    assert len(line_collections) == 2  # blockade edges plus diagonal hop bond collection
    assert len(line_collections[0].get_segments()) == model.lattice.num_links
    assert len(line_collections[1].get_segments()) == 1

    plt.close(fig)


def test_quantum_disk_grid_visualizer_plots_basis_states() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    basis = model.build_basis(sort=True)
    visualizer = QuantumDiskBasisGridVisualizer.from_model(model)

    fig, axes = visualizer.plot(
        basis.states[:3],
        ncols=2,
        show=False,
        show_config_label=True,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (2, 2)
    populated_titles = [ax.get_title() for ax in axes.flat if ax.get_title()]
    assert len(populated_titles) == 3
    assert "state 0" in populated_titles[0]
    assert any("0000" in title for title in populated_titles)

    plt.close(fig)


def test_quantum_disk_grid_functional_wrapper_accepts_model() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    basis = model.build_basis(sort=True)

    fig, axes = plot_quantum_disk_basis_grid(
        basis.states[:2],
        model=model,
        ncols=2,
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    plt.close(fig)


def test_quantum_disk_grid_plot_cage_support() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    basis = model.build_basis(sort=True)
    visualizer = QuantumDiskBasisGridVisualizer.from_model(model)
    record = DummyCageRecord(
        support=np.array([1, 2], dtype=np.int64),
        local_state=np.array([1.0, -1.0], dtype=np.complex128),
        signature=(0, 2),
    )

    fig, axes = visualizer.plot_cage_support(
        record,
        basis_configs=basis.states,
        ncols=2,
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert "basis 1" in axes.flat[0].get_title()
    assert "amp=1" in axes.flat[0].get_title()
    assert fig._suptitle is not None
    assert "signature=(0, 2)" in fig._suptitle.get_text()

    plt.close(fig)


def test_quantum_disk_grid_plot_interference_zeros() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    basis = model.build_basis(sort=True)
    visualizer = QuantumDiskBasisGridVisualizer.from_model(model)
    report = DummyEnvironmentReport(
        zero_reports=(
            DummyZeroReport(1, "q_empty"),
            DummyZeroReport(3, "projector_like"),
        ),
    )

    fig, axes = visualizer.plot_interference_zeros(
        report,
        basis_configs=basis.states,
        mechanism="all",
        ncols=2,
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert "zero 1" in axes.flat[0].get_title()
    assert "q_empty" in axes.flat[0].get_title()
    assert "zero 3" in axes.flat[1].get_title()
    assert "projector_like" in axes.flat[1].get_title()

    plt.close(fig)


def test_quantum_disk_visualizer_rejects_non_matplotlib_backend() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = QuantumDiskConfigurationVisualizer(lattice=lattice)

    with pytest.raises(ValueError, match="backend='matplotlib'"):
        visualizer.plot(np.array([0, 0, 0, 0], dtype=np.int64), backend="networkx")
