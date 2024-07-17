import pytest
from hypothesis import given, strategies as st

from qlinks.model.pxp_model_1d import CpModel, fibonacci, PauliX, PXPModel1D


@given(st.integers(min_value=2, max_value=20))
def test_fibonacci(n):
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)


class TestCpModel:
    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_solve(self, n, periodic):
        model = CpModel(n, periodic)
        model.solve()
        if periodic:
            assert model._callback.n_solutions == fibonacci(n - 1) + fibonacci(n + 1)
        else:
            assert model._callback.n_solutions == fibonacci(n + 2)


class TestPauliX:
    def test_mask(self):
        opt = PauliX(4, 2)
        ...


class TestPXPModel1D:
    def test_hamiltonian(self):
        model = PXPModel1D(4, periodic=True)
        ...
