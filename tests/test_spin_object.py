import numpy as np
import pytest

from qlinks.coordinate import UnitVectorCollection, Site
from qlinks.spin_object import Spin, SpinOperator, SpinOperatorCollection, Link


class TestSpin:

    def test_get_item(self):
        assert Spin([0, 1])[0] == 0
        assert Spin([0, 1])[1] == 1

    def test_overwrite_elems(self):
        spin = Spin([0, 1])
        with pytest.raises(ValueError):
            spin[1] = 2

    def test_tensor_product(self):
        assert Spin([1, 0]) ^ Spin([0, 1]) == Spin([0, 1, 0, 0])


class TestSpinOperator:

    def test_constructor(self):
        spin = SpinOperator([[0, 1], [0, 0]], dtype=float)
        assert spin.dtype == np.float64

    @pytest.fixture(scope="class")
    def spin_opts(self):
        return SpinOperatorCollection()

    def test_overwrite_elems(self, spin_opts):
        opt = spin_opts.Sp
        with pytest.raises(ValueError):
            opt[0, 1] = 0

    def test_conj(self, spin_opts):
        assert spin_opts.Sp.conj == spin_opts.Sm
        assert spin_opts.Sm.conj == spin_opts.Sp
        assert spin_opts.I2.conj == spin_opts.I2

    def test_tensor_product(self, spin_opts):
        assert spin_opts.I2 ^ spin_opts.I2 == SpinOperator(np.identity(4))
        assert spin_opts.Sp ^ spin_opts.O2 == SpinOperator(np.zeros((4, 4)))
        assert spin_opts.Sm ^ spin_opts.O2 == SpinOperator(np.zeros((4, 4)))


class TestLink:

    @pytest.fixture(scope="class")
    def spin_opts(self):
        return SpinOperatorCollection()

    @pytest.fixture(scope="class")
    def unit_vectors(self):
        return UnitVectorCollection()

    def test_comparison(self, spin_opts, unit_vectors):
        assert Link(Site(1, 1), unit_vectors.rightward) < Link(Site(2, 1), unit_vectors.rightward)
        assert Link(Site(1, 1), unit_vectors.rightward) < Link(Site(1, 1), unit_vectors.upward)
        assert Link(Site(1, 1), unit_vectors.rightward) == Link(Site(2, 1), unit_vectors.leftward)
        assert Link(Site(1, 1), unit_vectors.rightward) != Link(Site(1, 1), unit_vectors.rightward,
                                                                spin_opts.Sp)

    def test_sorting(self, unit_vectors):
        assert sorted([
            Link(Site(2, 1), unit_vectors.leftward),
            Link(Site(2, 1), unit_vectors.rightward),
            Link(Site(1, 1), unit_vectors.upward),
            Link(Site(1, 1), unit_vectors.rightward)
        ]) == [
                   Link(Site(1, 1), unit_vectors.rightward),
                   Link(Site(1, 1), unit_vectors.rightward),
                   Link(Site(1, 1), unit_vectors.upward),
                   Link(Site(2, 1), unit_vectors.rightward)
               ]

    def test_set_method(self, spin_opts, unit_vectors):
        assert {
                   Link(Site(1, 1), unit_vectors.rightward),
                   Link(Site(1, 1), unit_vectors.rightward),
                   Link(Site(1, 1), unit_vectors.rightward, spin_opts.Sp),
                   Link(Site(1, 1), unit_vectors.rightward, spin_opts.I2),
                   Link(Site(1, 1), unit_vectors.upward)
               } == {
                   Link(Site(1, 1), unit_vectors.rightward, spin_opts.Sp),
                   Link(Site(1, 1), unit_vectors.rightward),
                   Link(Site(1, 1), unit_vectors.upward)
               }

    def test_conj(self, spin_opts, unit_vectors):
        assert Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sp).conj() == Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sm)
        assert Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sm).conj() == Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sp)
        assert Link(
            Site(1, 1), unit_vectors.upward, spin_opts.Sp).conj() != Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sm)
        assert Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sp).conj() != Link(
            Site(2, 2), unit_vectors.rightward, spin_opts.Sm)

    def test_reset(self, spin_opts, unit_vectors):
        assert Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sp).reset() == Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.I2)
        assert Link(
            Site(1, 1), unit_vectors.upward, spin_opts.Sp).reset() != Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.I2)
        assert Link(
            Site(1, 1), unit_vectors.rightward, spin_opts.Sp).reset() != Link(
            Site(2, 2), unit_vectors.rightward, spin_opts.Sm)
