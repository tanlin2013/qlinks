import pytest
from matplotlib import pyplot as plt
from scipy.linalg import ishermitian

from qlinks.model import QuantumLinkModel


class TestQuantumLinkModel:
    @pytest.mark.parametrize("length, width", [(2, 2)])
    @pytest.mark.parametrize("coup_j, coup_rk", [(10, 1), (1, -10)])
    def test_hamiltonian(self, length, width, coup_j, coup_rk):
        model = QuantumLinkModel(length, width, coup_j, coup_rk)
        ham = model.hamiltonian
        assert ishermitian(ham)
        # assert np.all(eigh(ham, eigvals_only=True) >= -1e-12)  # positive semi-definite
        plt.matshow(ham)
        plt.colorbar()
        plt.show()
