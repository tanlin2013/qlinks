import pytest
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import ishermitian, eigh

from qlinks.symmetry.computation_basis import ComputationBasis
from qlinks.symmetry.gauss_law import GaussLaw
from qlinks.model import QuantumLinkModel


class TestQuantumLinkModel:
    @pytest.mark.parametrize("length, width", [(2, 2)])
    @pytest.mark.parametrize("coup_j, coup_rk", [(10, 1), (1, -10)])
    def test_hamiltonian(self, length, width, coup_j, coup_rk):
        model = QuantumLinkModel(length, width, coup_j, coup_rk)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals = eigh(ham, eigvals_only=True)
        print(evals)
        plt.matshow(ham)
        plt.colorbar()
        plt.show()

    @pytest.mark.parametrize(
        "charge_distri, flux_sector",
        [
            (GaussLaw.staggered_charge_distri(2, 2), (0, 0)),
            (GaussLaw.staggered_charge_distri(4, 4), (0, 0)),  # 26 secs
            # (GaussLaw.staggered_charge_distri(4, 4), None),  # 1 mins 20 secs
            (np.zeros((2, 2)), None),
            # (np.zeros((4, 4)), (0, 0)),  # 9 mins 26 secs
        ]
    )
    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0.9)])
    def test_symmetry_sector_hamiltonian(self, charge_distri, flux_sector, coup_j, coup_rk):
        length, width = charge_distri.shape
        basis = ComputationBasis(length, width, charge_distri, flux_sector).get()
        model = QuantumLinkModel(length, width, coup_j, coup_rk, basis)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals = eigh(ham, eigvals_only=True)  # ground state energy = 0, when coup_rk = 1
        print(evals)
        plt.matshow(ham)
        plt.colorbar()
        plt.show()
