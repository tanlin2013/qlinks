import numpy as np
import pytest

from qlinks.lattice import BoundaryCondition, LatticeGraph, Link, Plaquette, Site


def make_triangle_graph() -> LatticeGraph:
    sites = (
        Site(id=0, cell=(0,), position=(0.0,)),
        Site(id=1, cell=(1,), position=(1.0,)),
        Site(id=2, cell=(2,), position=(2.0,)),
    )

    links = (
        Link(id=0, source=0, target=1, kind="a"),
        Link(id=1, source=1, target=2, kind="a"),
        Link(id=2, source=2, target=0, kind="a"),
    )

    plaquettes = (
        Plaquette(
            id=0,
            links=(0, 1, 2),
            orientations=(1, 1, 1),
            sites=(0, 1, 2),
            kind="triangle",
        ),
    )

    return LatticeGraph(
        sites=sites,
        links=links,
        plaquettes=plaquettes,
        boundary_condition=BoundaryCondition.PERIODIC,
        translations={
            (0, (1,)): 1,
            (1, (1,)): 2,
            (2, (1,)): 0,
        },
    )


def test_graph_basic_counts() -> None:
    graph = make_triangle_graph()

    assert graph.ndim == 1
    assert graph.num_sites == 3
    assert graph.num_links == 3
    assert graph.num_plaquettes == 1

    np.testing.assert_array_equal(graph.site_ids, np.array([0, 1, 2]))
    np.testing.assert_array_equal(graph.link_ids, np.array([0, 1, 2]))
    np.testing.assert_array_equal(graph.plaquette_ids, np.array([0]))


def test_link_endpoints() -> None:
    graph = make_triangle_graph()

    np.testing.assert_array_equal(
        graph.link_endpoints,
        np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ]
        ),
    )


def test_incidence_matrix() -> None:
    graph = make_triangle_graph()

    incidence = graph.incidence_matrix().toarray()

    expected = np.array(
        [
            [-1, 0, +1],
            [+1, -1, 0],
            [0, +1, -1],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(incidence, expected)


def test_unoriented_adjacency_matrix() -> None:
    graph = make_triangle_graph()

    adjacency = graph.unoriented_adjacency_matrix().toarray()

    expected = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(adjacency, expected)


def test_incident_incoming_outgoing_links() -> None:
    graph = make_triangle_graph()

    np.testing.assert_array_equal(graph.incident_links(0), np.array([0, 2]))
    np.testing.assert_array_equal(graph.outgoing_links(0), np.array([0]))
    np.testing.assert_array_equal(graph.incoming_links(0), np.array([2]))


def test_neighbors() -> None:
    graph = make_triangle_graph()

    np.testing.assert_array_equal(graph.neighbors(0), np.array([1, 2]))
    np.testing.assert_array_equal(graph.neighbors(1), np.array([0, 2]))
    np.testing.assert_array_equal(graph.neighbors(2), np.array([0, 1]))


def test_plaquette_accessors() -> None:
    graph = make_triangle_graph()

    np.testing.assert_array_equal(graph.plaquette_links(0), np.array([0, 1, 2]))
    np.testing.assert_array_equal(graph.plaquette_orientations(0), np.array([1, 1, 1]))
    np.testing.assert_array_equal(graph.plaquette_sites(0), np.array([0, 1, 2]))


def test_translate_site() -> None:
    graph = make_triangle_graph()

    assert graph.translate_site(0, (1,)) == 1
    assert graph.translate_site(1, (1,)) == 2
    assert graph.translate_site(2, (1,)) == 0
    assert graph.translate_site(0, (-1,)) is None


def test_oriented_link_between() -> None:
    graph = make_triangle_graph()

    assert graph.oriented_link_between(0, 1) == (0, +1)
    assert graph.oriented_link_between(1, 0) == (0, -1)

    assert graph.oriented_link_between(2, 0) == (2, +1)
    assert graph.oriented_link_between(0, 2) == (2, -1)


def test_as_metadata() -> None:
    graph = make_triangle_graph()

    metadata = graph.as_metadata()

    assert metadata["ndim"] == 1
    assert metadata["num_sites"] == 3
    assert metadata["num_links"] == 3
    assert metadata["num_plaquettes"] == 1
    assert metadata["boundary_condition"] == "periodic"


def test_reject_unordered_site_ids() -> None:
    with pytest.raises(ValueError, match="Site ids"):
        LatticeGraph(
            sites=(
                Site(id=1, cell=(0,)),
                Site(id=0, cell=(1,)),
            ),
            links=(),
        )


def test_reject_unordered_link_ids() -> None:
    with pytest.raises(ValueError, match="Link ids"):
        LatticeGraph(
            sites=(
                Site(id=0, cell=(0,)),
                Site(id=1, cell=(1,)),
            ),
            links=(Link(id=1, source=0, target=1),),
        )


def test_reject_bad_link_endpoint() -> None:
    with pytest.raises(IndexError, match="site_id"):
        LatticeGraph(
            sites=(Site(id=0, cell=(0,)),),
            links=(Link(id=0, source=0, target=1),),
        )


def test_reject_bad_plaquette_link_id() -> None:
    with pytest.raises(IndexError, match="link_id"):
        LatticeGraph(
            sites=(
                Site(id=0, cell=(0,)),
                Site(id=1, cell=(1,)),
                Site(id=2, cell=(2,)),
            ),
            links=(
                Link(id=0, source=0, target=1),
                Link(id=1, source=1, target=2),
            ),
            plaquettes=(Plaquette(id=0, links=(0, 1, 2), orientations=(1, 1, 1), sites=(0, 1, 2)),),
        )


def test_invalid_query_ids_raise() -> None:
    graph = make_triangle_graph()

    with pytest.raises(IndexError, match="site_id"):
        graph.neighbors(3)

    with pytest.raises(IndexError, match="plaquette_id"):
        graph.plaquette_links(1)

    with pytest.raises(KeyError, match="No link"):
        graph.oriented_link_between(0, 0)
