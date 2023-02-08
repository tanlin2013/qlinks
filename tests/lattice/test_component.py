import numpy as np
import pytest

from qlinks.lattice.component import Site, UnitVector, UnitVectors


class TestSite:
    def test_comparison(self):
        assert Site(1, 2) > Site(1, 1)  # fix x, compare y
        assert Site(2, 2) > Site(1, 2)  # fix y, compare x
        assert Site(1, 2) < Site(2, 3)  # should compare y first
        assert Site(2, 3) < Site(1, 4)  # should compare y first
        assert Site(2, 2) == Site(2, 2)
        assert Site(1, 1) != Site(2, 2)
        assert Site(0, 0) > Site(-1, 0)

    def test_addition(self):
        assert Site(1, 1) + UnitVector(0, 1) == Site(1, 2)  # fix y, add x
        assert Site(1, 1) + UnitVector(1, 0) == Site(2, 1)  # fix x, add y
        assert Site(1, 1) + Site(2, 3) == Site(3, 4)

    def test_subtraction(self):
        assert Site(1, 1) - Site(1, 0) == UnitVector(0, 1)  # fix x, sub y
        assert Site(1, 1) - Site(0, 1) == UnitVector(1, 0)  # fix y, sub x
        assert Site(3, 3) - Site(1, 1) == UnitVector(2, 2)

    def test_get_item(self):
        assert Site(1, 2)[0] == 1
        assert Site(1, 2)[1] == 2
        with pytest.raises(KeyError):
            _ = Site(1, 2)[3]


class TestUnitVector:
    def test_scalar_multiplication(self):
        assert -1 * UnitVector(1, 0) == UnitVector(-1, 0)
        assert -1 * UnitVector(0, 1) == UnitVector(0, -1)
        assert 3 * UnitVector(0, 1) == UnitVector(0, 3)

    def test_comparison(self):
        assert UnitVector(1, 0) < UnitVector(0, 1)
        assert UnitVector(1, 0) > UnitVector(-1, 0)
        assert UnitVector(1, 0) > UnitVector(0, -1)
        assert UnitVector(-1, 0) > UnitVector(0, -1)

    def test_length(self):
        assert abs(UnitVector(1, 0)) == 1
        assert abs(UnitVector(0, 1)) == 1
        assert abs(UnitVector(-1, 0)) == 1
        assert abs(UnitVector(1, 1)) == np.sqrt(2)

    def test_sign(self):
        assert UnitVector(1, 0).sign == 1
        assert UnitVector(0, 1).sign == 1
        assert UnitVector(-1, 0).sign == -1
        assert UnitVector(0, -1).sign == -1
        with pytest.raises(ValueError):
            _ = UnitVector(1, 1).sign


class TestUnitVectorCollection:
    def test_instance(self):
        assert UnitVectors.rightward == -1 * UnitVectors.leftward
        assert UnitVectors.upward == -1 * UnitVectors.downward
        with pytest.raises(TypeError):
            _ = UnitVectors.upward + UnitVectors.rightward

    def test_iter(self):
        it = iter(UnitVectors)
        assert next(it) == UnitVectors.rightward
        assert next(it) == UnitVectors.upward

    def test_iter_all_directions(self):
        it = UnitVectors.iter_all_directions()
        assert next(it) == UnitVectors.downward
        assert next(it) == UnitVectors.leftward
        assert next(it) == UnitVectors.rightward
        assert next(it) == UnitVectors.upward
        with pytest.raises(StopIteration):
            _ = next(it)
