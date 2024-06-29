import networkx as nx
import numpy as np
import pytest
from scipy.linalg import eigh, ishermitian

from qlinks.model.quantum_link_model import QuantumLinkModel
from qlinks.symmetry.gauss_law import GaussLaw


def is_spectral_reflection_symmetric(evals):
    positive_part = evals[evals > 1e-12]
    negative_part = evals[evals < -1e-12]
    if len(positive_part) != len(negative_part):
        return False
    else:
        return np.allclose(
            np.sort(evals[evals > 1e-12]), np.sort(-(evals[evals < -1e-12])), atol=1e-12
        )


def isin_with_tolerance(a, b, tol=1e-12):
    result = np.zeros_like(a, dtype=bool)
    for i, val_a in enumerate(a):
        result[i] = any(abs(val_a - val_b) < tol for val_b in b)
    return np.array(result)


class TestQuantumLinkModel:
    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0), (1, 1), (1, 0.1)])
    @pytest.mark.parametrize("length_x, length_y", [(2, 2)])
    def test_hamiltonian(self, coup_j, coup_rk, length_x, length_y, qlm_2x2_basis):
        assert qlm_2x2_basis.n_links == 2 * length_x * length_y
        model = QuantumLinkModel(coup_j, coup_rk, (length_x, length_y), qlm_2x2_basis)
        ham = model.hamiltonian.todense()
        assert ishermitian(ham)
        evals = eigh(ham, eigvals_only=True)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1
        if coup_rk == 0:
            assert is_spectral_reflection_symmetric(evals)
        else:
            assert not is_spectral_reflection_symmetric(evals)

    @pytest.fixture(scope="class")
    def basis_from_solver(self):
        gauss_law = GaussLaw.from_zero_charge_distri(4, 4, (0, 0))
        return gauss_law.solve()

    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0), (1, 1)])
    def test_with_solver(self, coup_j, coup_rk, basis_from_solver):
        model = QuantumLinkModel(coup_j, coup_rk, (4, 4), basis_from_solver)
        ham = model.hamiltonian.todense()
        assert ishermitian(ham)
        evals = eigh(ham, eigvals_only=True)
        if np.isclose(coup_rk / coup_j, 1):
            assert np.isclose(evals[0], 0)  # E0 = 0, when coup_rk / coup_j = 1
        if coup_rk == 0:
            assert is_spectral_reflection_symmetric(evals)
        else:
            assert not is_spectral_reflection_symmetric(evals)

    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 0), (1, 1)])
    @pytest.mark.parametrize("length_x, length_y", [(2, 2), (4, 2)])
    @pytest.mark.parametrize("momenta", [(0, 0)])  # TODO: non-zero momenta is still problematic
    def test_momentum_hamiltonian(self, coup_j, coup_rk, length_x, length_y, momenta):
        gauss_law = GaussLaw.from_zero_charge_distri(length_x, length_y, (0, 0))
        basis = gauss_law.solve()
        k_model = QuantumLinkModel(coup_j, coup_rk, (length_x, length_y), basis, momenta)
        k_ham = k_model.hamiltonian.todense()
        assert ishermitian(k_ham)
        g = nx.from_numpy_array(k_ham)
        k_evals = eigh(k_ham, eigvals_only=True)
        if coup_rk == 0:
            assert nx.is_bipartite(g)
            assert is_spectral_reflection_symmetric(k_evals)
        else:
            assert not nx.is_bipartite(g)
            assert not is_spectral_reflection_symmetric(k_evals)

        model = QuantumLinkModel(coup_j, coup_rk, (length_x, length_y), basis)
        evals = eigh(model.hamiltonian.todense(), eigvals_only=True)
        assert isin_with_tolerance(k_evals, evals).all()

    def test__bipartite_sorting_index(self):
        ...

    @pytest.mark.parametrize("coup_j, coup_rk", [(1, 1)])
    def test_entropy(self, coup_j, coup_rk, qdm_4x2_basis):
        model = QuantumLinkModel(coup_j, coup_rk, (4, 2), qdm_4x2_basis)
        evecs = np.array([0, 1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        evecs /= np.linalg.norm(evecs)
        _ = model.entropy(evecs, 1, 0)
        ...
