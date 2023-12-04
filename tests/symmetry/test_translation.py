import numpy as np
import pytest

from qlinks.computation_basis import ComputationBasis
from qlinks.lattice.square_lattice import SquareLattice
from qlinks.symmetry.translation import Translation


class TestTranslation:
    @pytest.fixture(scope="function")
    def lattice(self):
        return SquareLattice(2, 2)

    @pytest.fixture(
        scope="function",
        params=[
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
        ],
    )
    def basis(self, request):
        basis = ComputationBasis(request.param)
        basis.sort()
        return basis

    def test(self, lattice, basis):
        translation = Translation(lattice, basis)
        df = translation._df  # noqa: F841
