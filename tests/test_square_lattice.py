import pytest  # noqa: F401

from qlinks.square_lattice import Site, UnitVector


def test_unit_vector():
    site = Site(0, 0)
    site2 = site + UnitVector().leftward
    print(site2)
