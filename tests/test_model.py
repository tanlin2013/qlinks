import numpy as np
import pytest
from scipy.linalg import eigh, ishermitian

from qlinks.computation_basis import ComputationBasis
from qlinks.model import QuantumLinkModel
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw


class TestQuantumLinkModel:
    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 1), (1, 0.1)])
    @pytest.mark.parametrize("length_x, length_y", [(2, 2)])
    @pytest.mark.parametrize(
        "basis",
        [
            ComputationBasis(
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
        ],
    )
    def test_hamiltonian(self, coup_j, coup_rk, length_x, length_y, basis):
        assert basis.n_links == 2 * length_x * length_y
        model = QuantumLinkModel(coup_j, coup_rk, (length_x, length_y), basis)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals, evecs = eigh(ham)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1

    @pytest.fixture(scope="class")
    def basis_from_solver(self):
        gauss_law = GaussLaw.from_zero_charge_distri(6, 4)
        gauss_law.flux_sector = (0, 0)
        dfs = DeepFirstSearch(gauss_law, max_steps=int(1e8))
        return gauss_law.to_basis(dfs.solve(n_solution=32810))  # 10 secs

    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 1)])
    def test_with_solver(self, coup_j, coup_rk, basis_from_solver):
        model = QuantumLinkModel(coup_j, coup_rk, (6, 4), basis_from_solver)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals, evecs = eigh(ham)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1
