import numpy as np
import pytest

from qlinks.computation_basis import ComputationBasis


@pytest.fixture(scope="function")
def four_link_basis():
    return ComputationBasis(np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]))


@pytest.fixture(scope="function")
def lattice_2x2_basis() -> ComputationBasis:
    """
    o◄──────o◄──────o    o◄──────o◄──────o   o◄──────o──────►o
    │       ▲       │    ▲       │       ▲   │       ▲       │
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    │       ▼       │   ▼       │       ▼
    o──────►o──────►o    o──────►o──────►o   o──────►o◄──────o
    │       ▲       │    ▲       │       ▲   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    │       ▼       │   │       ▼       │
    o◄──────o◄──────o    o◄──────o◄──────o   o◄──────o──────►o

            27                   78                 105

    o──────►o◄──────o    o──────►o──────►o   o──────►o──────►o
    ▲       │       ▲    │       ▲       │   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    │       ▼       │    ▼       │       ▼   │       ▼       │
    o◄──────o──────►o    o◄──────o◄──────o   o◄──────o◄──────o
    │       ▲       │    │       ▲       │   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    ▼       │       ▼   │       ▼       │
    o──────►o◄──────o    o──────►o──────►o   o──────►o──────►o

           150                  177                 228
    """
    basis = ComputationBasis(
        np.array(
            [
                [0, 0, 0, 1, 1, 0, 1, 1],  # 27
                [0, 1, 0, 0, 1, 1, 1, 0],  # 78
                [0, 1, 1, 0, 1, 0, 0, 1],  # 105
                [1, 0, 0, 1, 0, 1, 1, 0],  # 150
                [1, 0, 1, 1, 0, 0, 0, 1],  # 177
                [1, 1, 1, 0, 0, 1, 0, 0],  # 228
            ]
        )
    )
    basis.sort()
    return basis


@pytest.fixture(scope="function")
def lattice_6x4_basis() -> ComputationBasis:
    """ """
    return ComputationBasis(...)
