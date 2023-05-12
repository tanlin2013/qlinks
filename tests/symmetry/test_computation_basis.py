import pytest

from qlinks.symmetry.computation_basis import ComputationBasis


class TestComputationBasis:
    @pytest.fixture(scope="function")
    def basis(self):
        return ComputationBasis()

    def test_bases(self):
        pass
