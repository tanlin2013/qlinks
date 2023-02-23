import numpy as np
import pytest

from qlinks.exceptions import InvalidArgumentError, InvalidOperationError
from qlinks.lattice.component import Site, UnitVector, UnitVectors
from qlinks.lattice.spin_object import (
    Link,
    Spin,
    SpinConfigs,
    SpinOperator,
    SpinOperators,
)


class TestSpin:
    def test_constructor(self):
        assert SpinConfigs.up.dtype == np.float64
        assert SpinConfigs.down.dtype == np.float64

    def test_get_item(self):
        assert Spin([0, 1])[0] == 0
        assert Spin([0, 1])[1] == 1

    def test_overwrite_elems(self):
        spin = Spin([0, 1], read_only=True)
        with pytest.raises(ValueError):
            spin[1] = 2

    def test_tensor_product(self):
        assert Spin([1, 0]) ^ Spin([0, 1]) == Spin([0, 1, 0, 0])

    @pytest.mark.parametrize(
        "spin, expected",
        [(SpinConfigs.up, 0.5), (SpinConfigs.down, -0.5), (Spin([[0], [0]]), np.nan)],
    )
    def test_magnetization(self, spin, expected):
        if np.isnan(expected):
            assert np.isnan(spin.magnetization)
        else:
            assert spin.magnetization == expected

    def test_equality(self):
        assert Spin([1, 0]) == Spin([1, 0])
        assert Spin([1, 0]) != Spin([0, 1])

    def test_inner_product(self):
        assert (SpinConfigs.up.T @ SpinConfigs.up).item() == 1
        assert (SpinConfigs.down.T @ SpinConfigs.down).item() == 1
        assert (SpinConfigs.up.T @ SpinConfigs.down).item() == 0


class TestSpinOperator:
    def test_constructor(self):
        spin = SpinOperator([[0, 1], [0, 0]], dtype=float)
        assert spin.dtype == np.float64

    def test_overwrite_elems(self):
        opt = SpinOperators.Sp
        with pytest.raises(ValueError):
            opt[0, 1] = 0

    def test_equality(self):
        assert SpinOperator(np.eye(2)) == SpinOperators.I2
        assert np.all(SpinOperators.I2 == np.eye(2))  # elementwise
        assert np.any(SpinOperators.Sp != np.zeros((2, 2)))  # elementwise

    def test_conj(self):
        assert SpinOperators.Sp.conj == SpinOperators.Sm
        assert SpinOperators.Sm.conj == SpinOperators.Sp
        assert SpinOperators.I2.conj == SpinOperators.I2

    def test_tensor_product(self):
        assert SpinOperators.I2 ^ SpinOperators.I2 == SpinOperator(np.identity(4))
        assert SpinOperators.Sp ^ SpinOperators.O2 == SpinOperator(np.zeros((4, 4)))
        assert SpinOperators.Sm ^ SpinOperators.O2 == SpinOperator(np.zeros((4, 4)))

    def test_power(self):
        assert SpinOperators.Sz**2 == SpinOperators.I2 * (0.5**2)
        assert (SpinOperators.Sp + SpinOperators.Sm) ** 2 == SpinOperators.I2
        assert -1 * (SpinOperators.Sp - SpinOperators.Sm) ** 2 == SpinOperators.I2


class TestLink:
    def test_constructor(self):
        with pytest.raises(InvalidArgumentError):
            _ = Link(Site(1, 1), UnitVector(1, 1))

    def test_mutation(self):
        link = Link(Site(1, 1), UnitVectors.rightward)
        assert link.operator == SpinOperators.I2
        link.operator = SpinOperators.Sp
        assert link.operator == SpinOperators.Sp
        assert link.state is None
        link.state = SpinConfigs.up
        assert link.state == SpinConfigs.up

    def test_comparison(self):
        assert Link(Site(1, 1), UnitVectors.rightward) < Link(Site(2, 1), UnitVectors.rightward)
        assert Link(Site(1, 1), UnitVectors.rightward) < Link(Site(1, 1), UnitVectors.upward)
        assert Link(Site(1, 1), UnitVectors.rightward) == Link(Site(2, 1), UnitVectors.leftward)
        assert Link(Site(1, 1), UnitVectors.rightward) != Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.Sp
        )
        with pytest.raises(InvalidOperationError):
            assert Link(Site(1, 1), UnitVectors.rightward) > Link(
                Site(1, 1), UnitVectors.rightward, SpinOperators.Sp
            )

    def test_sorting(self):
        assert sorted(
            [
                Link(Site(2, 1), UnitVectors.leftward),
                Link(Site(2, 1), UnitVectors.rightward),
                Link(Site(1, 1), UnitVectors.upward),
                Link(Site(1, 1), UnitVectors.rightward),
            ]
        ) == [
            Link(Site(1, 1), UnitVectors.rightward),
            Link(Site(1, 1), UnitVectors.rightward),
            Link(Site(1, 1), UnitVectors.upward),
            Link(Site(2, 1), UnitVectors.rightward),
        ]

    def test_set_method(self):
        assert {
            Link(Site(1, 1), UnitVectors.rightward),
            Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp, SpinConfigs.up),
            Link(Site(1, 1), UnitVectors.rightward),
            Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp),
            Link(Site(1, 1), UnitVectors.rightward, SpinOperators.I2),
            Link(Site(1, 1), UnitVectors.upward),
        } == {
            Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp, SpinConfigs.up),
            Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp),
            Link(Site(1, 1), UnitVectors.rightward),
            Link(Site(1, 1), UnitVectors.upward),
        }

    def test_tensor_product(self):
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp) ^ Link(
            Site(1, 1), UnitVectors.upward, SpinOperators.Sp
        ) == SpinOperator([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sm) ^ Link(
            Site(0, 0), UnitVectors.upward, SpinOperators.Sp
        ) == SpinOperator(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )  # auto sorted

    def test_conj(self):
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp).conj() == Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.Sm
        )
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sm).conj() == Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.Sp
        )
        assert Link(Site(1, 1), UnitVectors.upward, SpinOperators.Sp).conj() != Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.Sm
        )
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp).conj() != Link(
            Site(2, 2), UnitVectors.rightward, SpinOperators.Sm
        )

    def test_reset(self):
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp).reset() == Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.I2
        )
        assert Link(Site(1, 1), UnitVectors.upward, SpinOperators.Sp).reset() != Link(
            Site(1, 1), UnitVectors.rightward, SpinOperators.I2
        )
        assert Link(Site(1, 1), UnitVectors.rightward, SpinOperators.Sp).reset() != Link(
            Site(2, 2), UnitVectors.rightward, SpinOperators.Sm
        )
