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
    plot_local_structure_readout,
    plot_local_structure_report,
)

matplotlib.use("Agg")


@dataclass(frozen=True)
class DummyLocalRDMReadout:
    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    component_index: int | None = None
    density_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class DummyMatrixUnitTerm:
    target_pattern: tuple[int, ...]
    source_pattern: tuple[int, ...]


@dataclass(frozen=True)
class DummyCoherentPair:
    pattern_a: tuple[int, ...]
    pattern_b: tuple[int, ...]
    weight: float
    sign_label: str = "-"
    is_singlet_like: bool = True
    relative_phase: complex = -1.0 + 0.0j


@dataclass(frozen=True)
class DummyClassicalSector:
    pattern: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class DummyLocalStructureReadoutReport:
    readout: DummyLocalRDMReadout
    coherent_pairs: tuple[DummyCoherentPair, ...]
    classical_sectors: tuple[DummyClassicalSector, ...]


@dataclass(frozen=True)
class DummyCageLocalStructureReport:
    readout_reports: tuple[DummyLocalStructureReadoutReport, ...]


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
    assert len(axes.flat[0].patches) == 1
    line_collections = [
        collection
        for collection in axes.flat[0].collections
        if isinstance(collection, LineCollection)
    ]
    assert line_collections
    assert len(line_collections[0].get_segments()) == lattice.num_links - 1

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


def test_local_basis_grid_can_plot_without_reference_or_layout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    visualizer = LocalBasisGridVisualizer(lattice=lattice)
    fig, axes = visualizer.plot(
        np.array([[1], [0]], dtype=np.int64),
        variable_indices=(0,),
        ncols=2,
        mode="dimers",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    line_collections = [
        collection
        for collection in axes.flat[0].collections
        if isinstance(collection, LineCollection)
    ]
    assert line_collections
    assert len(line_collections[0].get_segments()) == lattice.num_links - 1

    plt.close(fig)


def test_plot_readout_filters_to_patterns_with_nonzero_density_matrix_entries() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    readout = DummyLocalRDMReadout(
        variable_indices=(0, 1),
        local_patterns=((0, 0), (1, 0), (0, 1)),
        density_matrix=np.array(
            [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),
    )

    visualizer = LocalBasisGridVisualizer(lattice=lattice)
    fig, axes = visualizer.plot_readout(
        readout,
        ncols=2,
        mode="values",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert "local 0" in axes.flat[0].get_title()
    assert "local 2" in axes.flat[1].get_title()

    plt.close(fig)


def test_plot_local_basis_grid_filters_to_nonzero_local_operator_patterns() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    local_operator = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)

    fig, axes = plot_local_basis_grid(
        lattice=lattice,
        local_patterns=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64),
        variable_indices=(0, 1),
        local_operator=local_operator,
        show_only_nonzero_matrix_elements=True,
        mode="values",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 1)
    assert "local 1" in axes.flat[0].get_title()
    assert "10" in axes.flat[0].get_title()

    plt.close(fig)


def test_local_basis_grid_can_plot_structure_readout() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    reference = np.zeros(layout.n_variables, dtype=np.int64)

    readout = DummyLocalRDMReadout(
        variable_indices=(0, 1),
        local_patterns=((1, 0), (0, 1), (1, 1)),
        component_index=2,
    )
    structure = DummyLocalStructureReadoutReport(
        readout=readout,
        coherent_pairs=(DummyCoherentPair(pattern_a=(1, 0), pattern_b=(0, 1), weight=0.25),),
        classical_sectors=(DummyClassicalSector(pattern=(1, 1), weight=0.5),),
    )

    visualizer = LocalBasisGridVisualizer(lattice=lattice, layout=layout)
    fig, axes = visualizer.plot_structure_readout(
        structure,
        reference_config=reference,
        include_frozen=True,
        max_structures=1,
        max_basis_states=2,
        max_frozen=1,
        ncols=2,
        mode="dimers",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (2, 2)
    assert fig._suptitle is not None
    assert "component 2" in fig._suptitle.get_text()
    titles = [ax.get_title() for ax in axes.flat[:3]]
    assert any("singlet 0" in title for title in titles)
    assert any("frozen 0" in title for title in titles)

    plt.close(fig)


def test_plot_local_structure_report_wrapper_is_exported() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    reference = np.zeros(layout.n_variables, dtype=np.int64)

    readout = DummyLocalRDMReadout(
        variable_indices=(0, 1),
        local_patterns=((1, 0), (0, 1)),
        component_index=1,
    )
    structure = DummyCageLocalStructureReport(
        readout_reports=(
            DummyLocalStructureReadoutReport(
                readout=readout,
                coherent_pairs=(
                    DummyCoherentPair(pattern_a=(1, 0), pattern_b=(0, 1), weight=0.25),
                ),
                classical_sectors=(),
            ),
        ),
    )

    fig, axes = plot_local_structure_report(
        lattice=lattice,
        layout=layout,
        structure_report=structure,
        reference_config=reference,
        max_readouts=1,
        max_structures_per_readout=1,
        ncols=2,
        mode="values",
        show=False,
        single_plot_kwargs={"with_site_labels": False},
    )

    assert axes.shape == (1, 2)
    assert fig._suptitle is not None
    assert "Local entangled structures" in fig._suptitle.get_text()
    assert visualizer_api.plot_local_structure_readout is plot_local_structure_readout
    assert visualizer_api.plot_local_structure_report is plot_local_structure_report
    assert "plot_local_structure_readout" in visualizer_api.__all__
    assert "plot_local_structure_report" in visualizer_api.__all__

    plt.close(fig)
