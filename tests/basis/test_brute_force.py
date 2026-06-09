import numpy as np

from qlinks.basis import BruteForceBasisSolver
from qlinks.constraints import (
    FixedValueConstraint,
    NearestNeighborBlockadeConstraint,
    TotalValueSector,
)
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_brute_force_fixed_value_constraint() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    constraints = [
        FixedValueConstraint.single(layout, variable_index=0, value=1),
    ]

    basis = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 4
    assert all(state[0] == 1 for state in basis.states)


def test_brute_force_total_value_sector() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    sectors = [
        TotalValueSector(layout=layout, target=2),
    ]

    basis = BruteForceBasisSolver(sort=True).solve(layout, sectors=sectors)

    assert basis.n_states == 6
    assert all(np.sum(state) == 2 for state in basis.states)


def test_brute_force_pxp_chain_length_4() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)

    basis = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)

    # Number of binary strings of length 4 without adjacent 1s is F_6 = 8.
    assert basis.n_states == 8

    for state in basis.states:
        assert not any(state[i] == 1 and state[i + 1] == 1 for i in range(3))


def test_brute_force_basis_solver_respects_max_states() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    basis = BruteForceBasisSolver(sort=False).solve(
        layout,
        max_states=1,
    )

    assert basis.n_states == 1
