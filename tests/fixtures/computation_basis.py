import pytest

import numpy as np

from qlinks.computation_basis import ComputationBasis


@pytest.fixture(scope="function")
def four_link_basis():
    return ComputationBasis(np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]))


@pytest.fixture(scope="function")
def lattice_2x2_basis():
    """

    """
    return ComputationBasis(
        np.array(
            [
                [0, 0, 0, 1, 1, 0, 1, 1],
                [0, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 0],
                [1, 0, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 1, 0, 0],
            ]
        )
    )


@pytest.fixture(scope="function")
def lattice_6x4_basis():
    """

    """
    return ComputationBasis(...)


