import numpy as np
import pytest
from scipy.linalg import eigh, ishermitian

from qlinks.model import QuantumLinkModel
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw


def is_spectral_reflection_symmetric(evals):
    mid_point = len(evals) // 2 if len(evals) % 2 == 0 else len(evals) // 2 + 1
    return np.allclose(evals[:mid_point], -np.flip(evals[mid_point:]), atol=1e-12)


class TestQuantumLinkModel:
    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0), (1, 1), (1, 0.1)])
    @pytest.mark.parametrize("length_x, length_y", [(2, 2)])
    def test_hamiltonian(self, coup_j, coup_rk, length_x, length_y, lattice_2x2_basis):
        assert lattice_2x2_basis.n_links == 2 * length_x * length_y
        model = QuantumLinkModel(coup_j, coup_rk, (length_x, length_y), lattice_2x2_basis)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals, evecs = eigh(ham)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1
        if coup_rk == 0:
            assert is_spectral_reflection_symmetric(evals)
        else:
            assert not is_spectral_reflection_symmetric(evals)

    @pytest.fixture(scope="class")
    def basis_from_solver(self):
        gauss_law = GaussLaw.from_zero_charge_distri(6, 4)
        gauss_law.flux_sector = (0, 0)
        dfs = DeepFirstSearch(gauss_law, max_steps=int(1e8))
        return gauss_law.to_basis(dfs.solve(n_solution=32810))  # 10 secs

    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0), (1, 1)])
    def test_with_solver(self, coup_j, coup_rk, basis_from_solver):
        model = QuantumLinkModel(coup_j, coup_rk, (6, 4), basis_from_solver)
        ham = model.hamiltonian
        assert ishermitian(ham)
        evals, evecs = eigh(ham)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1
        if coup_rk == 0:
            assert is_spectral_reflection_symmetric(evals)
        else:
            assert not is_spectral_reflection_symmetric(evals)
