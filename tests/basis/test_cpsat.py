import numpy as np
import pytest

from qlinks.basis import BruteForceBasisSolver, CPSATBasisSolver
from qlinks.constraints import (
    DimerCoveringConstraint,
    FixedValueConstraint,
    GaussLawConstraint,
    NearestNeighborBlockadeConstraint,
    ParitySector,
    TotalValueSector,
)
from qlinks.lattice import ChainLattice
from qlinks.variables import LocalSpace, VariableLayout


def assert_same_basis(basis_a, basis_b) -> None:
    set_a = {tuple(state.tolist()) for state in basis_a.states}
    set_b = {tuple(state.tolist()) for state in basis_b.states}
    assert set_a == set_b


def test_cpsat_binary_no_constraints() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    basis = CPSATBasisSolver(sort=True).solve(layout)

    assert basis.n_states == 8


def test_cpsat_matches_brute_force_fixed_value() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    constraints = [
        FixedValueConstraint.single(layout, variable_index=0, value=1),
        FixedValueConstraint.single(layout, variable_index=3, value=0),
    ]

    brute = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)
    cpsat = CPSATBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert_same_basis(brute, cpsat)


def test_cpsat_matches_brute_force_total_value_sector() -> None:
    layout = VariableLayout.from_sites(5, LocalSpace.binary())

    sectors = [
        TotalValueSector(layout=layout, target=2),
    ]

    brute = BruteForceBasisSolver(sort=True).solve(layout, sectors=sectors)
    cpsat = CPSATBasisSolver(sort=True).solve(layout, sectors=sectors)

    assert_same_basis(brute, cpsat)
    assert cpsat.n_states == 10


def test_cpsat_parity_sector() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    sectors = [
        ParitySector(layout=layout, target=0),
    ]

    basis = CPSATBasisSolver(sort=True).solve(layout, sectors=sectors)

    assert basis.n_states == 8
    assert all(np.sum(state) % 2 == 0 for state in basis.states)


def test_cpsat_pxp_chain_length_5() -> None:
    lattice = ChainLattice(5, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)

    brute = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)
    cpsat = CPSATBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert_same_basis(brute, cpsat)
    assert cpsat.n_states == 13


def test_cpsat_dimer_chain_length_4() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    constraints = DimerCoveringConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        required_counts=1,
    )

    basis = CPSATBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 1
    np.testing.assert_array_equal(basis.states[0], np.array([1, 0, 1]))


def test_cpsat_gauss_law_chain_length_3() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    constraints = GaussLawConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        charges=np.array([-1, 0, 1]),
    )

    basis = CPSATBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 1
    np.testing.assert_array_equal(basis.states[0], np.array([1, 1]))


def test_cpsat_max_solutions() -> None:
    layout = VariableLayout.from_sites(5, LocalSpace.binary())

    basis = CPSATBasisSolver(max_solutions=3).solve(layout)

    assert basis.n_states == 3
