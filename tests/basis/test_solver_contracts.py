import pytest

from qlinks.basis import BruteForceBasisSolver, CPSATBasisSolver, DFSBasisSolver
from qlinks.constraints import TotalValueSector


def _cpsat_solver_factory():
    pytest.importorskip("ortools.sat.python.cp_model")
    return CPSATBasisSolver(sort=True)


@pytest.mark.parametrize(
    "solver_factory",
    [
        lambda: BruteForceBasisSolver(sort=True),
        _cpsat_solver_factory,
        lambda: DFSBasisSolver(sort=True),
    ],
    ids=["brute_force", "cpsat", "dfs"],
)
def test_solver_binary_no_constraints(solver_factory, binary_site_layout_3) -> None:
    basis = solver_factory().solve(binary_site_layout_3)
    assert basis.n_states == 8


@pytest.mark.parametrize(
    "solver_factory",
    [
        lambda: BruteForceBasisSolver(sort=True),
        _cpsat_solver_factory,
        lambda: DFSBasisSolver(sort=True),
    ],
    ids=["brute_force", "cpsat", "dfs"],
)
def test_solver_total_value_sector(solver_factory, binary_site_layout_5) -> None:
    sectors = [TotalValueSector(layout=binary_site_layout_5, target=2)]
    basis = solver_factory().solve(binary_site_layout_5, sectors=sectors)
    assert basis.n_states == 10
