import pytest  # noqa: F401

from qlinks.lattice.component import UnitVector
from qlinks.symmetry.translation import Translation
from tests.lattice.test_square_lattice import (  # noqa: F401
    anti_clockwise_state,
    clockwise_state,
    zero_clock_state,
)


class TestTranslation:
    def test_matrix_multiplication(
        self, clockwise_state, anti_clockwise_state, zero_clock_state  # noqa: F811
    ):
        for state in [clockwise_state, anti_clockwise_state, zero_clock_state]:
            # translate by lattice size
            assert Translation(UnitVector(0, state.shape[1])) @ state == state
            assert Translation(UnitVector(state.shape[0], 0)) @ state == state
            assert Translation(UnitVector(*state.shape)) @ state == state
            # translate back and forth in x
            assert Translation(UnitVector(1, 0)) @ state != state
            assert Translation(UnitVector(-1, 0)) @ (Translation(UnitVector(1, 0)) @ state) == state
            # translate back and forth in y
            assert Translation(UnitVector(0, 1)) @ state != state
            assert Translation(UnitVector(0, -1)) @ (Translation(UnitVector(0, 1)) @ state) == state
