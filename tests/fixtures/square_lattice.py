import numpy as np
import pytest

from qlinks.lattice.square_lattice import SquareLattice


@pytest.fixture(scope="function")
def preset_2x2_lattice() -> SquareLattice:
    """
    o◄──────o──────►o
    │       │       │
    │       │       │
    ▼       ▼       ▼
    o◄──────o──────►o
    ▲       ▲       ▲
    │       │       │
    │       │       │
    o◄──────o──────►o
    """
    return SquareLattice(2, 2, np.array([0, 1, 1, 1, 0, 0, 1, 0]))
