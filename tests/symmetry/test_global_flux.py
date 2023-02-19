from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from qlinks.exceptions import InvalidArgumentError
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.global_flux import FluxSectorSnapshot, GlobalFlux


class TestGlobalFlux:
    def test_get_item(self):
        assert GlobalFlux(1, 0)[0] == 1
        assert GlobalFlux(1, 0)[1] == 0
        with pytest.raises(KeyError):
            _ = GlobalFlux(1, 0)[3]

    def test_quantum_numbers(self):
        assert GlobalFlux(1, 2).quantum_numbers == (1, 2)
        assert GlobalFlux(-2, 2).quantum_numbers == (-2, 2)


class TestFluxSectorSnapshot:
    @pytest.mark.parametrize(
        "charge_distri, n_expected_solution, expectation",
        [
            (np.zeros((2, 2)), 10, does_not_raise()),
            ([[1, -1], [-1, 1]], 6, does_not_raise()),
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], None, pytest.raises(InvalidArgumentError)),
            # (np.zeros((4, 4)), 990, does_not_raise()),  # 2 mins 42 secs
            # (GaussLaw.staggered_charge_distri(4, 4), 132, does_not_raise()),  # 25 secs
        ],
    )
    def test_multi_solutions(self, charge_distri, n_expected_solution, expectation):
        width, length = np.asarray(charge_distri).shape
        with expectation:
            snapshot = FluxSectorSnapshot(length, width, charge_distri, (0, 0))
            dfs = DeepFirstSearch(snapshot, max_steps=10000)
            filled_snapshots = dfs.search(n_solution=1000)  # far more than all possibilities
            assert all(isinstance(s, FluxSectorSnapshot) for s in filled_snapshots)
            assert len(filled_snapshots) == n_expected_solution
