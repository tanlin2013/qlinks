import numpy as np
import pytest

from qlinks.lattice.spin_operators import SpinOperators


def commutator(a, b):
    return a @ b - b @ a


@pytest.mark.parametrize("s", [0.5, 1, 1.5, 2])
def test_spin_operators(s):
    sop = SpinOperators(s)
    s_plus, s_minus, s_z = sop.s_plus, sop.s_minus, sop.s_z
    np.testing.assert_allclose(commutator(s_plus, s_minus), 2 * s_z)
    np.testing.assert_allclose(commutator(s_z, s_plus), s_plus)
    np.testing.assert_allclose(commutator(s_z, s_minus), -s_minus)

    s_x, s_y = sop.s_x, sop.s_y
    np.testing.assert_allclose(commutator(s_x, s_y), 1j * s_z, atol=1e-12)
    np.testing.assert_allclose(commutator(s_y, s_z), 1j * s_x, atol=1e-12)
    np.testing.assert_allclose(commutator(s_z, s_x), 1j * s_y, atol=1e-12)
