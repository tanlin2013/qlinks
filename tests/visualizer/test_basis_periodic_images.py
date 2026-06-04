from itertools import product

import matplotlib
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


def test_triangular_positive_patch_uses_positive_side_source_shifts_only() -> None:
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
    )

    shifts = set(visualizer._positive_patch_link_source_shifts())

    assert (1, 1) in shifts
    assert (-1, 0) not in shifts
    assert (0, -1) not in shifts
    assert (-1, -1) not in shifts


def test_triangular_pbc_positive_patch_draws_all_rhombus_plaquettes() -> None:
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
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    drawn_ids = {int(draw_plaquette.plaquette_id) for draw_plaquette in draw_plaquettes}
    expected_ids = {
        int(plaquette.id) for plaquette in lattice.plaquettes if len(plaquette.links) == 4
    }

    assert drawn_ids == expected_ids, {
        "missing": sorted(expected_ids - drawn_ids),
        "extra": sorted(drawn_ids - expected_ids),
    }


def test_triangular_pbc_positive_patch_has_no_negative_visual_cells() -> None:
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
        collapse_duplicate_visual_links=True,
    )

    nodes, links = visualizer._draw_primitives()

    for node in nodes:
        visual_cell = visualizer._visual_cell(
            site_id=node.site_id,
            image_shift=node.image_shift,
        )
        assert all(int(cell) >= 0 for cell in visual_cell)

    for link in links:
        source_cell, target_cell = link_visual_cells(
            visualizer,
            nodes,
            link,
        )
        assert all(int(cell) >= 0 for cell in source_cell)
        assert all(int(cell) >= 0 for cell in target_cell)


def test_triangular_closed_plaquette_representative_minimizes_score() -> None:
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
        collapse_duplicate_visual_links=True,
    )

    _nodes, draw_links = visualizer._draw_primitives()

    draw_links_by_link_id: dict[int, list] = {}
    for draw_link in draw_links:
        draw_links_by_link_id.setdefault(int(draw_link.link_id), []).append(draw_link)

    found_multi_representative = False

    for plaquette in lattice.plaquettes:
        if len(plaquette.links) != 4:
            continue

        physical_link_ids = tuple(int(link_id) for link_id in plaquette.links)

        candidate_lists = [draw_links_by_link_id.get(link_id, []) for link_id in physical_link_ids]

        if any(len(candidates) == 0 for candidates in candidate_lists):
            continue

        closed_representatives = [
            tuple(candidate_tuple)
            for candidate_tuple in product(*candidate_lists)
            if visualizer._draw_links_form_closed_cycle(tuple(candidate_tuple))
        ]

        if len(closed_representatives) <= 1:
            continue

        selected = visualizer._select_closed_visual_plaquette(
            candidate_lists,
            physical_link_ids=physical_link_ids,
            preferred_center=None,
        )

        if selected is None:
            continue

        selected_score = visualizer._visual_plaquette_representative_score_for_physical_links(
            selected,
            physical_link_ids=physical_link_ids,
            preferred_center=None,
        )

        all_scores = [
            visualizer._visual_plaquette_representative_score_for_physical_links(
                representative,
                physical_link_ids=physical_link_ids,
                preferred_center=None,
            )
            for representative in closed_representatives
        ]

        assert selected_score == min(all_scores)

        # Additional regression: among equal order/center-distance classes,
        # the selected representative should be bottom-left.
        best_prefix = selected_score[:3]
        tied = [
            representative
            for representative, score in zip(closed_representatives, all_scores, strict=True)
            if score[:3] == best_prefix
        ]

        tied_centers = [
            visualizer._closed_visual_plaquette_center(representative) for representative in tied
        ]

        selected_center = visualizer._closed_visual_plaquette_center(selected)

        min_y = min(float(center[1]) for center in tied_centers)
        assert float(selected_center[1]) == pytest.approx(min_y)

        tol = 1e-9
        min_x_at_min_y = min(
            float(center[0]) for center in tied_centers if abs(float(center[1]) - min_y) <= tol
        )
        assert float(selected_center[0]) == pytest.approx(min_x_at_min_y)

        found_multi_representative = True
        break

    assert found_multi_representative


def test_honeycomb_small_torus_visual_plaquettes_match_lattice_links() -> None:
    lattice = HoneycombLattice(
        2,
        2,
        boundary_condition="periodic",
    )
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

    by_id = {
        int(draw_plaquette.plaquette_id): draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
    }

    for plaquette in lattice.plaquettes:
        plaquette_id = int(plaquette.id)

        assert plaquette_id in by_id
        assert set(by_id[plaquette_id].link_ids) == {int(link_id) for link_id in plaquette.links}
