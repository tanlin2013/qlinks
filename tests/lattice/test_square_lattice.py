import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import ishermitian

from qlinks.exceptions import InvalidArgumentError, InvalidOperationError, LinkOverridingError
from qlinks.lattice.square_lattice import Cross, Plaquette, SquareLattice
from qlinks.lattice.component import Site, UnitVectors
from qlinks.spin_object import SpinOperator, SpinOperators, SpinConfigs, Link


class TestSquareLattice:
    def test_shape(self):
        assert SquareLattice(2, 2).shape == (2, 2)
        with pytest.raises(InvalidArgumentError):
            _ = SquareLattice(1, 2).shape

    @pytest.fixture(scope="class")
    def lattice(self):
        return SquareLattice(4, 4)

    def test_get_item(self, lattice):
        assert lattice[0, 0] == Site(0, 0)
        assert lattice[0, 4] == Site(0, 0)  # assume periodic b.c.
        assert lattice[0, 5] == Site(0, 1)
        assert lattice[Site(4, 0)] == Site(0, 0)
        assert lattice[Site(5, 0)] == Site(1, 0)
        assert lattice[-1, 0] == Site(3, 0)

    def test_iter(self):
        lattice = SquareLattice(2, 2)
        it = iter(lattice)
        assert next(it) == Site(0, 0)
        assert next(it) == Site(1, 0)
        assert next(it) == Site(0, 1)
        assert next(it) == Site(1, 1)

    def test_iter_links(self):
        lattice = SquareLattice(2, 2)
        it = lattice.iter_links()
        assert next(it) == Link(Site(0, 0), UnitVectors.rightward)
        assert next(it) == Link(Site(0, 0), UnitVectors.upward)
        assert next(it) == Link(Site(1, 0), UnitVectors.rightward)

    def test_iter_plaquettes(self, lattice):
        lattice = SquareLattice(2, 2)
        it = lattice.iter_plaquettes()
        assert next(it).site == Site(0, 0)
        assert next(it).site == Site(1, 0)
        assert next(it).site == Site(0, 1)

    @pytest.mark.parametrize("shape, expected", [((2, 2), 2**8), ((4, 2), 2**16)])
    def test_hilbert_dims(self, shape, expected):
        assert SquareLattice(*shape).hilbert_dims[0] == expected


class TestPlaquette:
    @pytest.fixture(scope="class")
    def lattice(self):
        return SquareLattice(2, 2)

    @pytest.fixture(scope="class")
    def plaquette(self, lattice):
        return Plaquette(lattice, Site(0, 0))

    def test_iter(self, lattice):
        plaquette = Plaquette(lattice, Site(0, 0))
        for link in plaquette:
            print(link)

    def test_array(self, lattice, plaquette):
        arr = np.array(plaquette)
        assert np.allclose(arr, np.triu(arr), atol=1e-12)
        assert arr.shape == lattice.hilbert_dims
        unique_value_counts = dict(zip(*np.unique(arr, return_counts=True)))
        assert tuple(unique_value_counts) == (0, 1)  # dict keys to tuple
        assert unique_value_counts[1] == 2 ** (lattice.num_links - 4)
        plt.matshow(arr)
        plt.colorbar()
        plt.show()

    def test_conj(self, plaquette):
        arr = plaquette.conj()
        assert np.allclose(arr, np.tril(arr), atol=1e-12)
        flipper = plaquette + plaquette.conj()
        assert ishermitian(flipper)
        assert np.all(
            np.linalg.eigvals(np.linalg.matrix_power(flipper, 2)) >= 0
        )  # positive semi-definite

    def test_addition(self, lattice, plaquette):
        assert isinstance(plaquette + plaquette.conj(), SpinOperator)
        with pytest.raises(InvalidOperationError):
            _ = Plaquette(lattice, Site(0, 0)) + Plaquette(lattice, Site(0, 1))
