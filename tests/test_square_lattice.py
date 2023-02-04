import matplotlib.pyplot as plt
import numpy as np
import pytest

from qlinks.coordinate import Site
from qlinks.square_lattice import Plaquette, SquareLattice


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
        for site in SquareLattice(4, 4):
            print(site)
        for link in SquareLattice(4, 4).iter_links():
            print(link)

    def test_shape(self):
        assert SquareLattice(2, 2).shape == (2, 2)
        with pytest.raises(ValueError):
            _ = SquareLattice(1, 2).shape

    def test_hilbert_dims(self):
        lattice = SquareLattice(2, 2)
        assert lattice.hilbert_dims[0] == 2**8


class TestPlaquette:
    @pytest.fixture(scope="class")
    def lattice(self):
        return SquareLattice(2, 2)

    def test_iter(self, lattice):
        plaquette = Plaquette(lattice, Site(0, 0))
        for link in plaquette:
            print(link)

    def test_array(self, lattice):
        plaquette = Plaquette(lattice, Site(0, 0))
        arr = np.array(plaquette)
        plt.matshow(arr)
        plt.show()
