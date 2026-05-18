import matplotlib
import numpy as np

from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
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
