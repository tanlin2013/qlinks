import numpy as np
import pytest

from qlinks.lattice.spin_operators import spin_operators


def commutator(a, b):
    return a @ b - b @ a


@pytest.mark.parametrize("s", [0.5, 1, 1.5, 2])
def test_spin_operators(s):
    s_plus, s_minus, s_z = spin_operators(s)
    np.testing.assert_allclose(commutator(s_plus, s_minus), 2 * s_z)
    np.testing.assert_allclose(commutator(s_z, s_plus), s_plus)
    np.testing.assert_allclose(commutator(s_z, s_minus), -s_minus)
