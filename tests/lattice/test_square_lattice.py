import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import ishermitian

from qlinks.lattice import Plaquette, SquareLattice
from qlinks.lattice.component import Site
from qlinks.spin_object import SpinOperator


class TestSquareLattice:
    def test_get_item(self):
        lattice = SquareLattice(4, 4)  # assume periodic b.c.
        assert lattice[0, 0] == Site(0, 0)
        assert lattice[0, 4] == Site(0, 0)
        assert lattice[0, 5] == Site(0, 1)
        assert lattice[Site(4, 0)] == Site(0, 0)
        assert lattice[Site(5, 0)] == Site(1, 0)
        assert lattice[-1, 0] == Site(3, 0)

    def test_iter(self):
        lattice = SquareLattice(4, 4)
        for site in lattice:
            print(site)
        for link in lattice.iter_links():
            print(link)
        for plaquette in lattice.iter_plaquettes():
            print(plaquette)

    def test_shape(self):
        assert SquareLattice(2, 2).shape == (2, 2)
        with pytest.raises(ValueError):
            _ = SquareLattice(1, 2).shape

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

    def test_array(self, plaquette):
        arr = np.array(plaquette)
        assert np.allclose(arr, np.triu(arr), atol=1e-12)
        plt.matshow(arr)
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
        with pytest.raises(ValueError):
            _ = Plaquette(lattice, Site(0, 0)) + Plaquette(lattice, Site(0, 1))
