from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import Site
from qlinks.lattice.square_lattice import Plaquette, SquareLattice


class TestSquareLattice:
    @pytest.fixture(scope="function")
    def lattice(self, request):
        return SquareLattice(*request.param)

    @pytest.fixture(scope="function")
    def preset_lattice(self, request) -> SquareLattice:
        fixture = f"preset_{'x'.join(map(str, request.param))}_lattice"
        return request.getfixturevalue(fixture)

    def test_constructor(self):
        assert SquareLattice(2, 2).shape == (2, 2)
        with pytest.raises(InvalidArgumentError):
            _ = SquareLattice(1, 2)
        with pytest.raises(InvalidArgumentError):
            _ = SquareLattice(2, 2, np.array([1, 2, -1]))
        with pytest.raises(InvalidArgumentError):
            _ = SquareLattice(2, 2, np.array([0, 1, 1, 0]))
        with does_not_raise():
            _ = SquareLattice(2, 2, np.array([0, 1, 1, 0, 0, 1, 1, 0]))

    @pytest.mark.parametrize("length_x, length_y", [(2, 2), (3, 3), (6, 8)])
    def test_links(self, length_x: int, length_y: int):
        lattice = SquareLattice(length_x, length_y)
        assert lattice.links.ndim == 1
        assert lattice.links.size == 2 * length_x * length_y == lattice.n_links
        assert np.all(lattice.links == -1)  # default value

    @pytest.mark.parametrize("lattice", [(2, 2), (3, 3), (4, 6)], indirect=True)
    def test_index(self, lattice: SquareLattice):
        with pytest.raises(ValueError):
            _ = lattice.index  # links are not set yet
        (bin_num,) = np.random.randint(
            2**lattice.n_links, size=1, dtype=int
        )  # warn: randint is bound by int64 limit, use with caution
        lattice.links = np.array(list(bin(bin_num).lstrip("0b").zfill(lattice.n_links)), dtype=int)
        assert lattice.index == bin_num

    @pytest.mark.parametrize("lattice", [(4, 4)], indirect=True)
    def test_get_item(self, lattice: SquareLattice):
        assert lattice[0, 0] == Site(0, 0)
        assert lattice[0, 4] == Site(0, 0)  # assume periodic b.c.
        assert lattice[0, 5] == Site(0, 1)
        assert lattice[Site(4, 0)] == Site(0, 0)
        assert lattice[Site(5, 0)] == Site(1, 0)
        assert lattice[-1, 0] == Site(3, 0)

    @pytest.mark.parametrize("lattice", [(4, 4)], indirect=True)
    def test_site_index(self, lattice: SquareLattice):
        assert lattice.site_index(Site(0, 0)) // 2 == 0  # each site associated with two links
        assert lattice.site_index(Site(1, 0)) // 2 == 1
        assert lattice.site_index(Site(3, 0)) // 2 == 3
        assert lattice.site_index(Site(4, 0)) // 2 == 0
        assert lattice.site_index(Site(0, 1)) // 2 == 4
        assert lattice.site_index(Site(0, 3)) // 2 == 12
        assert lattice.site_index(Site(0, 4)) // 2 == 0
        assert lattice.site_index(Site(4, 4)) // 2 == 0

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    def test_iter(self, lattice: SquareLattice):
        it = iter(lattice)
        assert next(it) == Site(0, 0)
        assert next(it) == Site(1, 0)
        assert next(it) == Site(0, 1)
        assert next(it) == Site(1, 1)
        with pytest.raises(StopIteration):
            _ = next(it)

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    def test_iter_plaquettes(self, lattice: SquareLattice):
        it = lattice.iter_plaquettes()
        assert next(it) == Plaquette(lattice, Site(0, 0))
        assert next(it) == Plaquette(lattice, Site(1, 0))
        assert next(it) == Plaquette(lattice, Site(0, 1))
        assert next(it) == Plaquette(lattice, Site(1, 1))
        with pytest.raises(StopIteration):
            _ = next(it)

    def test_set_vertex_links(self):
        """
              ▲      │
              │      ▼
        ◄─────o◄────►o◄─────
              │      ▲
              ▼      │
        """
        ...

    @pytest.mark.parametrize("preset_lattice", [(2, 2)], indirect=True)
    def test_charge(self, preset_lattice: SquareLattice):
        assert preset_lattice.charge(Site(0, 0)) == 0
        assert preset_lattice.charge(Site(1, 0)) == 2
        assert preset_lattice.charge(Site(0, 1)) == -2
        assert preset_lattice.charge(Site(1, 1)) == 0
        assert preset_lattice.charge(Site(2, 0)) == 0
        assert preset_lattice.charge(Site(0, -1)) == -2
        empty_lattice = SquareLattice(2, 2)
        assert np.isnan(empty_lattice.charge(Site(0, 0)))
        empty_lattice.links = np.array([0, 1, 1, -1, -1, 0, -1, -1])
        assert empty_lattice.charge(Site(0, 0)) == 0
        assert np.isnan(empty_lattice.charge(Site(1, 0)))
        assert np.isnan(empty_lattice.charge(Site(0, 1)))
        assert np.isnan(empty_lattice.charge(Site(1, 1)))

    @pytest.mark.parametrize("preset_lattice", [(2, 2)], indirect=True)
    def test_axial_flux(self, preset_lattice: SquareLattice):
        assert preset_lattice.axial_flux(0, axis=0) == 0.5 * -2
        assert preset_lattice.axial_flux(1, axis=0) == 0.5 * 2
        assert preset_lattice.axial_flux(0, axis=1) == 0.5 * 2
        assert preset_lattice.axial_flux(1, axis=1) == 0.5 * -2
        assert preset_lattice.axial_flux(-1, axis=0) == 0.5 * 2
        assert preset_lattice.axial_flux(2, axis=1) == 0.5 * 2
        empty_lattice = SquareLattice(2, 2)
        assert np.isnan(empty_lattice.axial_flux(0, axis=0))

    @pytest.mark.parametrize("lattice", [(2, 2), (4, 2), (4, 4)], indirect=True)
    @pytest.mark.parametrize("axis", [0, 1])
    def test_bipartite_index(self, lattice: SquareLattice, axis: int):
        for idx in range(lattice.shape[axis]):
            first_partition_idx, second_partition_idx = lattice.bipartite_index(idx, axis)
            first_partition_idx[1::2] -= 1
            second_partition_idx[1::2] -= 1
            first_partition_idx //= 2
            second_partition_idx //= 2
            if axis == 0:
                assert np.all(first_partition_idx % lattice.length_x <= idx)
                assert np.all(second_partition_idx % lattice.length_x > idx)
            elif axis == 1:
                assert np.all(first_partition_idx // lattice.length_x <= idx)
                assert np.all(second_partition_idx // lattice.length_x > idx)

    @pytest.mark.parametrize("preset_lattice", [(2, 2)], indirect=True)
    def test_adjacency_matrix(self, preset_lattice: SquareLattice):
        adj_mat = preset_lattice.adjacency_matrix()
        np.testing.assert_array_equal(
            adj_mat,
            np.array(
                [
                    [0, 0, 2, 0],
                    [2, 0, 0, 2],
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                ]
            ),
        )  # parallel edges coz of periodic b.c.
        assert np.all(np.sum(adj_mat, axis=0) + np.sum(adj_mat, axis=1) == 4), f"got {adj_mat}"
        assert adj_mat.dtype == np.int64
        assert np.all(adj_mat >= 0)

    @pytest.mark.parametrize("preset_lattice", [(2, 2)], indirect=True)
    def test_as_graph(self, preset_lattice: SquareLattice):
        graph = preset_lattice.as_graph()
        assert graph.number_of_nodes() == preset_lattice.size
        assert graph.number_of_edges() == preset_lattice.n_links
        for _, degree in graph.degree:
            assert degree == 4


class TestPlaquette:
    @pytest.fixture(scope="function")
    def lattice(self, request):
        return SquareLattice(*request.param)

    @pytest.mark.parametrize(
        "lattice, site, expect",
        [
            ((2, 2), Site(0, 0), int("11011000", 2)),
            ((2, 2), Site(1, 0), int("01110010", 2)),
            ((2, 2), Site(0, 1), int("10001101", 2)),
            ((2, 2), Site(1, 1), int("00100111", 2)),
            ((4, 8), Site(0, 0), int("110100001".ljust(2 * 4 * 8, "0"), 2)),
            (
                (4, 8),
                Site(3, 7),
                int("".join("1" if i in [6, 57, 62, 63] else "0" for i in range(2 * 4 * 8)), 2),
            ),
            (
                (6, 8),
                Site(0, 7),
                int("".join("1" if i in [0, 84, 85, 87] else "0" for i in range(2 * 6 * 8)), 2),
            ),
            (
                (6, 8),
                Site(5, 7),
                int("".join("1" if i in [10, 85, 94, 95] else "0" for i in range(2 * 6 * 8)), 2),
            ),
        ],
        indirect=["lattice"],
    )
    def test_mask(self, lattice: SquareLattice, site: Site, expect: int):
        plaquette = Plaquette(lattice, site)
        assert plaquette._mask == expect

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    @pytest.mark.parametrize(
        "site, expect",
        [
            (Site(0, 0), np.array([0, 1, 3, 4])),
            (Site(1, 0), np.array([1, 2, 3, 6])),
            (Site(0, 1), np.array([0, 4, 5, 7])),
            (Site(1, 1), np.array([2, 5, 6, 7])),
        ],
    )
    def test_link_index(self, lattice: SquareLattice, site: Site, expect):
        plaquette = Plaquette(lattice, site)
        np.testing.assert_array_equal(np.sort(plaquette.link_index()), expect)

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    @pytest.mark.parametrize(
        "site, expect",
        [
            (Site(0, 0), np.array([False, True, True, True, True, False])),
            (Site(1, 0), np.array([True, False, True, True, False, True])),
            (Site(0, 1), np.array([True, False, True, True, False, True])),
            (Site(1, 1), np.array([False, True, True, True, True, False])),
        ],
    )
    def test_flippable(
        self, qlm_2x2_basis: ComputationBasis, lattice: SquareLattice, site: Site, expect
    ):
        plaquette = Plaquette(lattice, site)
        np.testing.assert_array_equal(plaquette.flippable(qlm_2x2_basis), expect)

    @pytest.mark.parametrize(
        "lattice, site, expect",
        [
            ((2, 2), Site(0, 0), np.array([27, 150, 177, 78, 105, 228])),
            ((2, 2), Site(1, 0), np.array([105, 78, 27, 228, 177, 150])),
            ((2, 2), Site(0, 1), np.array([150, 78, 228, 27, 177, 105])),
            ((2, 2), Site(1, 1), np.array([27, 105, 78, 177, 150, 228])),
        ],
        indirect=["lattice"],
    )
    def test_matrix_multiplication(
        self, qlm_2x2_basis: ComputationBasis, lattice: SquareLattice, site: Site, expect
    ):
        plaquette = Plaquette(lattice, site)
        np.testing.assert_array_equal(plaquette @ qlm_2x2_basis, expect)

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    @pytest.mark.parametrize(
        "site, expect_basis_idx",
        [
            (Site(0, 0), ([1, 2, 3, 4], [3, 4, 1, 2])),
            (Site(1, 0), ([0, 2, 3, 5], [2, 0, 5, 3])),
            (Site(0, 1), ([0, 2, 3, 5], [3, 5, 0, 2])),
            (Site(1, 1), ([1, 2, 3, 4], [2, 1, 4, 3])),
        ],
    )
    def test_matrix_element(
        self,
        qlm_2x2_basis: ComputationBasis,
        lattice: SquareLattice,
        site: Site,
        expect_basis_idx,
    ):
        plaquette = Plaquette(lattice, site)
        mat = plaquette[qlm_2x2_basis].toarray()
        assert np.all(mat[expect_basis_idx] == 1)
        mask = np.zeros_like(mat, dtype=bool)
        mask[expect_basis_idx] = True
        assert np.all(mat[~mask] == 0)

    @pytest.mark.parametrize("lattice", [(2, 2)], indirect=True)
    @pytest.mark.parametrize("site", [Site(0, 0), Site(1, 0), Site(0, 1), Site(1, 1)])
    def test_power(self, qlm_2x2_basis, lattice, site):
        plaquette = Plaquette(lattice, site)
        for power in range(5):
            plaquette_power = plaquette**power
            mat = plaquette_power[qlm_2x2_basis].toarray()
            if power % 2 == 0:
                assert plaquette_power._mask == 0
                assert np.count_nonzero(mat - np.diag(np.diagonal(mat))) == 0  # diagonal matrix
            else:
                assert plaquette_power._mask == plaquette._mask
                assert np.count_nonzero(np.diagonal(mat)) == 0  # off-diagonal matrix
