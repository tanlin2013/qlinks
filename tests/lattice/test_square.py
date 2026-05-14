import numpy as np
import pytest

from qlinks.lattice import BoundaryCondition, SquareLattice


def test_open_square_counts_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    assert lattice.ndim == 2
    assert lattice.num_sites == 4
    assert lattice.num_links == 4
    assert lattice.num_plaquettes == 1
    assert lattice.boundary_condition == BoundaryCondition.OPEN


def test_open_square_site_id() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="open")

    assert lattice.site_id(0, 0) == 0
    assert lattice.site_id(0, 1) == 1
    assert lattice.site_id(1, 0) == 2
    assert lattice.site_id(2, 1) == 5

    with pytest.raises(IndexError):
        lattice.site_id(3, 0)


def test_periodic_square_site_id_wraps() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="periodic")

    assert lattice.site_id(3, 0) == lattice.site_id(0, 0)
    assert lattice.site_id(-1, 0) == lattice.site_id(2, 0)
    assert lattice.site_id(0, 2) == lattice.site_id(0, 0)


def test_open_square_links_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    # site convention:
    #   (0,0) -> 0
    #   (0,1) -> 1
    #   (1,0) -> 2
    #   (1,1) -> 3
    #
    # link construction order:
    #   for each site: +x link, then +y link if present
    expected = np.array(
        [
            [0, 2],  # (0,0) -> (1,0), x
            [0, 1],  # (0,0) -> (0,1), y
            [1, 3],  # (0,1) -> (1,1), x
            [2, 3],  # (1,0) -> (1,1), y
        ]
    )

    np.testing.assert_array_equal(lattice.link_endpoints, expected)

    assert [link.kind for link in lattice.links] == ["x", "y", "x", "y"]
    assert not any(link.wrap for link in lattice.links)


def test_open_square_incidence_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    expected = np.array(
        [
            [-1, -1, 0, 0],
            [0, +1, -1, 0],
            [+1, 0, 0, -1],
            [0, 0, +1, +1],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(lattice.incidence_matrix().toarray(), expected)


def test_open_square_neighbors_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    np.testing.assert_array_equal(lattice.neighbors(0), np.array([1, 2]))
    np.testing.assert_array_equal(lattice.neighbors(1), np.array([0, 3]))
    np.testing.assert_array_equal(lattice.neighbors(2), np.array([0, 3]))
    np.testing.assert_array_equal(lattice.neighbors(3), np.array([1, 2]))


def test_open_square_plaquette_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    assert lattice.num_plaquettes == 1

    # Counter-clockwise around the single square:
    #   0 -> 2 uses x-link id 0, orientation +1
    #   2 -> 3 uses y-link id 3, orientation +1
    #   3 -> 1 uses x-link id 2, orientation -1
    #   1 -> 0 uses y-link id 1, orientation -1
    np.testing.assert_array_equal(lattice.plaquette_sites(0), np.array([0, 2, 3, 1]))
    np.testing.assert_array_equal(lattice.plaquette_links(0), np.array([0, 3, 2, 1]))
    np.testing.assert_array_equal(lattice.plaquette_orientations(0), np.array([1, 1, -1, -1]))


def test_open_square_translation() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="open")

    center = lattice.site_id(1, 1)

    assert lattice.translate_site(center, (1, 0)) == lattice.site_id(2, 1)
    assert lattice.translate_site(center, (-1, 0)) == lattice.site_id(0, 1)
    assert lattice.translate_site(center, (0, 1)) == lattice.site_id(1, 2)
    assert lattice.translate_site(center, (0, -1)) == lattice.site_id(1, 0)

    left_boundary = lattice.site_id(0, 1)
    assert lattice.translate_site(left_boundary, (-1, 0)) is None


def test_periodic_square_counts_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")

    assert lattice.num_sites == 4

    # 2 links per site: +x and +y
    assert lattice.num_links == 8

    # One plaquette per unit cell under this simple periodic convention.
    assert lattice.num_plaquettes == 4


def test_periodic_square_has_wrapping_links() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")

    assert any(link.wrap for link in lattice.links)


def test_periodic_square_translation_wraps() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="periodic")

    right_boundary = lattice.site_id(2, 1)
    left_boundary = lattice.site_id(0, 1)

    assert lattice.translate_site(right_boundary, (1, 0)) == left_boundary
    assert lattice.translate_site(left_boundary, (-1, 0)) == right_boundary

    top_boundary = lattice.site_id(1, 2)
    bottom_boundary = lattice.site_id(1, 0)

    assert lattice.translate_site(top_boundary, (0, 1)) == bottom_boundary
    assert lattice.translate_site(bottom_boundary, (0, -1)) == top_boundary


def test_periodic_square_site_degrees_3_by_3() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="periodic")

    for site_id in lattice.site_ids:
        assert lattice.incident_links(int(site_id)).size == 4
        assert lattice.neighbors(int(site_id)).size == 4


def test_open_square_counts_3_by_2() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="open")

    assert lattice.num_sites == 6

    # horizontal links: (lx - 1) * ly = 2 * 2 = 4
    # vertical links: lx * (ly - 1) = 3 * 1 = 3
    assert lattice.num_links == 7

    # square plaquettes: (lx - 1) * (ly - 1) = 2 * 1 = 2
    assert lattice.num_plaquettes == 2


def test_oriented_link_between_square() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    assert lattice.oriented_link_between(0, 2) == (0, +1)
    assert lattice.oriented_link_between(2, 0) == (0, -1)

    assert lattice.oriented_link_between(0, 1) == (1, +1)
    assert lattice.oriented_link_between(1, 0) == (1, -1)


def test_square_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="lx must be positive"):
        SquareLattice(0, 2)

    with pytest.raises(ValueError, match="ly must be positive"):
        SquareLattice(2, 0)


def test_square_metadata() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    metadata = lattice.as_metadata()

    assert metadata["ndim"] == 2
    assert metadata["num_sites"] == 4
    assert metadata["num_links"] == 4
    assert metadata["num_plaquettes"] == 1
    assert metadata["boundary_condition"] == "open"


def test_square_lattice_2x2_pbc_plaquette_boundary_matches_legacy_fields() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")

    for plaquette_id in lattice.plaquette_ids:
        boundary = lattice.plaquette_boundary(int(plaquette_id))

        observed_links = tuple(oriented_link.link_id for oriented_link in boundary)
        observed_orientations = tuple(
            oriented_link.orientation for oriented_link in boundary
        )

        expected_links = tuple(lattice.plaquette_links(int(plaquette_id)).tolist())
        expected_orientations = tuple(
            lattice.plaquette_orientations(int(plaquette_id)).tolist()
        )

        assert observed_links == expected_links
        assert observed_orientations == expected_orientations


def test_square_lattice_2x2_pbc_boundary_of_boundary_is_zero() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")

    boundary_of_boundary = (
        lattice.incidence_matrix() @ lattice.plaquette_incidence_matrix()
    )

    np.testing.assert_array_equal(
        boundary_of_boundary.toarray(),
        np.zeros(
            (lattice.num_sites, lattice.num_plaquettes),
            dtype=np.int8,
        ),
    )
