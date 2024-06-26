import numpy as np
import pytest

from qlinks.computation_basis import ComputationBasis


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
            (
                np.array([[0] * (100 - 70) + [1] * 70, [0] * (100 - 90) + [1] * 90]),
                np.array([int("1" * 70, 2), int("1" * 90, 2)]),
            ),
        ],
    )
    def test_index(self, basis, expect):
        basis = ComputationBasis(basis)
        np.testing.assert_array_equal(basis.index, expect)
        assert isinstance(basis.index.dtype, (np.int64, object))

    @pytest.mark.parametrize(
        "basis, expect",
        [
            (
                np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]),
                np.array([1, 3, 6, 9]),
            ),
            (
                np.array([[1] * 64, [0] + [1] * 63]),
                np.array([np.iinfo(np.int64).max, int("1" * 64, 2)], dtype=object),
            ),
        ],
    )
    def test_sort(self, basis, expect):
        basis = ComputationBasis(basis)
        assert not np.array_equal(basis.index, expect)  # not sorted
        basis.sort()
        np.testing.assert_array_equal(basis.index, expect)

    def test_from_index(self, qlm_2x2_basis):
        np.testing.assert_equal(
            ComputationBasis.from_index(np.array([27, 78, 105, 150, 177, 228]), 8).links,
            qlm_2x2_basis.links,
        )
