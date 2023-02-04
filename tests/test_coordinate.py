import pytest

from qlinks.coordinate import Site, UnitVector


class TestSite:

    def test_comparison(self):
        assert Site(1, 2) > Site(1, 1)  # fix x, compare y
        assert Site(2, 2) > Site(1, 2)  # fix y, compare x
        assert Site(1, 2) < Site(2, 3)  # should compare y first
        assert Site(2, 3) < Site(1, 4)  # should compare y first
        assert Site(2, 2) == Site(2, 2)
        assert Site(1, 1) != Site(2, 2)

    def test_addition(self):
        assert Site(1, 1) + UnitVector(0, 1) == Site(1, 2)  # fix y, add x
        assert Site(1, 1) + UnitVector(1, 0) == Site(2, 1)  # fix x, add y
        assert Site(1, 1) + Site(2, 3) == Site(3, 4)

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
        assert UnitVector(-1, 0) < UnitVector(1, 0)

    def test_sign(self):
        assert UnitVector(1, 0).sign == 1
        assert UnitVector(0, 1).sign == 1
        assert UnitVector(-1, 0).sign == -1
        assert UnitVector(0, -1).sign == -1
