import numpy as np
import pytest

from qlinks.symmetry.computation_basis import ComputationBasis


class TestComputationBasis:
    @pytest.mark.parametrize(
        "basis, expect",
        [
            (
                np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]),
                np.array([1, 6, 9, 3]),
            ),
            (np.array([[1] * 63]), np.array([np.iinfo(np.int64).max])),
            (np.array([[1] * 64]), np.array([int("1" * 64, 2)])),
        ],
    )
    def test_index(self, basis, expect):
        basis = ComputationBasis(basis)
        np.testing.assert_array_equal(basis.index, expect)

    @pytest.mark.parametrize(
        "basis, expect",
        [
            (
                np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]),
                np.array([1, 3, 6, 9]),
            ),
            (
                np.array([[0] + [1] * 63, [1] * 64]),
                np.array([np.iinfo(np.int64).max, int("1" * 64, 2)]),
            ),
        ],
    )
    def test_sort(self, basis, expect):
        basis = ComputationBasis(basis)
        basis.sort()
        np.testing.assert_array_equal(basis.index, expect)
