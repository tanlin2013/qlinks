from pathlib import Path

import matplotlib
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
    BasisGridVisualizer,
    automatic_grid_shape,
    format_basis_config,
    plot_basis_config,
    plot_basis_grid,
)

matplotlib.use("Agg")


def visual_cell_of_node(visualizer, node):
    return visualizer._visual_cell(
        site_id=node.site_id,
        image_shift=node.image_shift,
    )


def link_visual_cells(visualizer, nodes, link):
    node_by_key = {node.key: node for node in nodes}
    source_node = node_by_key[link.source_key]
    target_node = node_by_key[link.target_key]

    return (
        visual_cell_of_node(visualizer, source_node),
        visual_cell_of_node(visualizer, target_node),
    )


def visual_edges(visualizer, nodes, links):
    return {link_visual_cells(visualizer, nodes, link) for link in links}


def undirected_visual_edges(visualizer, nodes, links):
    return {frozenset(link_visual_cells(visualizer, nodes, link)) for link in links}


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
    assert len(ax.lines) == lattice.num_links

    # Plaquette symbol should add text beyond node labels.
    assert len(ax.texts) >= lattice.num_sites

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


def test_periodic_square_positive_patch_draws_boundary_images() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    nodes, links = visualizer._draw_primitives()

    assert any(any(node.image_shift) for node in nodes)

    assert any(any(link.source_key[1]) or any(link.target_key[1]) for link in links)


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


def test_square_pbc_without_images_draws_each_link_once() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="none",
    )

    _, draw_links = visualizer._draw_primitives()

    assert len(draw_links) == lattice.num_links


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


def test_square_2_by_2_positive_patch_connects_boundary_images() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    nodes, links = visualizer._draw_primitives()

    # There should be image nodes on the right/top boundary.
    image_nodes = [node for node in nodes if any(node.image_shift)]
    assert len(image_nodes) > 0

    # At least one link must touch an image node.
    assert any(any(link.source_key[1]) or any(link.target_key[1]) for link in links)

    # The upper-right corner image of site (0,0) should exist or be used if
    # positive patch needs it.
    spans = visualizer._cell_spans()

    assert any(
        np.array_equal(
            np.asarray(
                visualizer._visual_cell(
                    site_id=node.site_id,
                    image_shift=node.image_shift,
                ),
                dtype=np.int64,
            ),
            spans,
        )
        for node in nodes
    )


def test_square_2_by_2_positive_patch_places_wrapping_links_on_right_and_top() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()

    visual_edges = {link_visual_cells(visualizer, nodes, link) for link in links}

    assert ((1, 0), (2, 0)) in visual_edges or ((2, 0), (1, 0)) in visual_edges
    assert ((0, 1), (0, 2)) in visual_edges or ((0, 2), (0, 1)) in visual_edges


def test_square_2_by_2_positive_patch_draws_four_plaquette_visual_cells() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = visualizer._draw_square_qlm_plaquette_primitives()

    visual_cells = {p.visual_cell for p in draw_plaquettes}

    assert {(0, 0), (1, 0), (0, 1), (1, 1)} <= visual_cells


def test_chain_pbc_positive_patch_places_wrapping_link_on_right() -> None:
    lattice = ChainLattice(4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()
    edges = undirected_visual_edges(visualizer, nodes, links)

    assert any(any(node.image_shift) for node in nodes)
    assert frozenset({(3,), (4,)}) in edges


def test_triangular_pbc_positive_patch_has_positive_images() -> None:
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

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()

    assert any(any(node.image_shift) for node in nodes)

    for node in nodes:
        assert all(shift >= 0 for shift in node.image_shift)

    assert any(any(link.source_key[1]) or any(link.target_key[1]) for link in links)


def test_honeycomb_pbc_positive_patch_has_positive_images() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()

    assert any(any(node.image_shift) for node in nodes)

    for node in nodes:
        assert all(shift >= 0 for shift in node.image_shift)

    assert any(any(link.source_key[1]) or any(link.target_key[1]) for link in links)


def test_honeycomb_pbc_positive_patch_has_boundary_visual_cells() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, _ = visualizer._draw_primitives()
    spans = visualizer._cell_spans()

    visual_cells = {
        visualizer._visual_cell(
            site_id=node.site_id,
            image_shift=node.image_shift,
        )
        for node in nodes
    }

    assert any(cell[0] == spans[0] for cell in visual_cells)
    assert any(cell[1] == spans[1] for cell in visual_cells)


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


@pytest.mark.parametrize(
    "lattice",
    [
        TriangularLattice(
            3, 3, boundary_condition="open", include_triangles=True, include_rhombi=True
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


def test_visualizer_period_vectors_use_lattice_primitives() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    periods = visualizer._period_vectors_2d()
    spans = visualizer._cell_spans()

    expected = np.asarray(
        [
            spans[0] * np.asarray(lattice.primitive_vectors[0], dtype=float),
            spans[1] * np.asarray(lattice.primitive_vectors[1], dtype=float),
        ],
        dtype=float,
    )

    np.testing.assert_allclose(periods, expected)


def test_triangular_pbc_positive_patch_places_a_and_b_boundary_links() -> None:
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

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()
    edges = undirected_visual_edges(visualizer, nodes, links)

    assert frozenset({(3, 0), (4, 0)}) in edges
    assert frozenset({(0, 3), (0, 4)}) in edges


def test_honeycomb_positive_patch_upper_apex_has_two_links_not_three() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=False,
    )

    nodes, links = visualizer._draw_primitives()
    node_keys = {node.key for node in nodes}

    origin_a_site_id = None
    for site in lattice.sites:
        if tuple(int(c) for c in site.cell) == (0, 0) and int(site.sublattice) == 0:
            origin_a_site_id = int(site.id)
            break

    assert origin_a_site_id is not None

    upper_apex_key = (origin_a_site_id, (1, 1))
    assert upper_apex_key in node_keys

    incident_links = [
        link
        for link in links
        if link.source_key == upper_apex_key or link.target_key == upper_apex_key
    ]

    # The upper apex should close the top hexagon with two links.
    # The outward z-link should be skipped.
    assert len(incident_links) == 2
