import pytest
import numpy as np

from qlinks.solver.constraint_programming import CpModel


@pytest.mark.parametrize(
    "shape, n_solutions",
    [((4, 2), 38), ((4, 4), 990), ((8, 2), 2214)]
)
def test_cp_model(shape, n_solutions):
    charge_distri = np.zeros(shape, dtype=int)
    cp_model = CpModel(shape, charge_distri, (0, 0))
    cp_model.solve()
    assert cp_model.n_solutions == n_solutions
