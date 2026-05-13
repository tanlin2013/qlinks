from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from qlinks.basis import BruteForceBasisSolver, DFSBasisSolver
from qlinks.constraints import (
    BaseConstraint,
    ConstraintResult,
    DimerCoveringConstraint,
    FixedValueConstraint,
    GaussLawConstraint,
    NearestNeighborBlockadeConstraint,
    TotalValueSector,
)
from qlinks.lattice import ChainLattice
from qlinks.models import PXPModel, SquareQDMModel
from qlinks.variables import LocalSpace, VariableKind, VariableLayout, VariableSpec


def assert_same_basis(basis_a, basis_b) -> None:
    set_a = {tuple(state.tolist()) for state in basis_a.states}
    set_b = {tuple(state.tolist()) for state in basis_b.states}
    assert set_a == set_b


def test_dfs_binary_no_constraints() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    basis = DFSBasisSolver(sort=True).solve(layout)

    assert basis.n_states == 8


def test_dfs_matches_brute_force_fixed_value() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    constraints = [
        FixedValueConstraint.single(layout, variable_index=0, value=1),
        FixedValueConstraint.single(layout, variable_index=3, value=0),
    ]

    brute = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)
    dfs = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert_same_basis(brute, dfs)


def test_dfs_matches_brute_force_total_value_sector() -> None:
    layout = VariableLayout.from_sites(5, LocalSpace.binary())

    sectors = [
        TotalValueSector(layout=layout, target=2),
    ]

    brute = BruteForceBasisSolver(sort=True).solve(layout, sectors=sectors)
    dfs = DFSBasisSolver(sort=True).solve(layout, sectors=sectors)

    assert_same_basis(brute, dfs)
    assert dfs.n_states == 10


def test_dfs_pxp_chain_length_5() -> None:
    lattice = ChainLattice(5, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    # Binary strings of length 5 with no adjacent 1s: F_7 = 13.
    assert basis.n_states == 13

    for state in basis.states:
        assert not any(state[i] == 1 and state[i + 1] == 1 for i in range(4))


def test_dfs_dimer_chain_length_4() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    constraints = DimerCoveringConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        required_counts=1,
    )

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    # Open chain with 4 sites has one perfect matching: links [1, 0, 1].
    assert basis.n_states == 1
    np.testing.assert_array_equal(basis.states[0], np.array([1, 0, 1]))


def test_dfs_gauss_law_chain_length_3() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    constraints = GaussLawConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        charges=np.array([-1, 0, 1]),
        charge_normalization="integer_flux",
    )

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 1
    np.testing.assert_array_equal(basis.states[0], np.array([1, 1]))


def test_dfs_custom_variable_order() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    basis = DFSBasisSolver(
        variable_order=np.array([2, 1, 0]),
        sort=True,
    ).solve(layout)

    assert basis.n_states == 8


def test_dfs_does_not_validate_uninitialized_memory() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    constraints = [
        FixedValueConstraint.single(layout, variable_index=0, value=1),
    ]

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 8
    assert all(state[0] == 1 for state in basis.states)
    assert np.all((basis.states == 0) | (basis.states == 1))


def test_dfs_partial_pxp_matches_expected_count() -> None:
    model = PXPModel.chain(
        length=8,
        boundary_condition="open",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    # Number of binary strings length 8 with no adjacent 1s: F_10 = 55.
    assert basis.n_states == 55


def test_dfs_partial_square_qdm_4x4_total_count() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        required_count=1,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 272


def test_dfs_partial_square_qdm_4x4_electric_sector_count() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        required_count=1,
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 132


def test_dfs_partial_bruteforce_agree_small_qdm() -> None:
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        required_count=1,
    )

    dfs_basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    brute_basis = model.build_basis(
        solver="brute_force",
        sort=True,
    )

    assert dfs_basis.n_states == brute_basis.n_states
    np.testing.assert_array_equal(dfs_basis.states, brute_basis.states)


@dataclass
class CountingConstraint:
    layout: VariableLayout
    affected: tuple[int, ...]
    target_sum: int
    name: str = "counting_constraint"

    def __post_init__(self) -> None:
        self.partial_calls = 0
        self.full_calls = 0

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.affected, dtype=np.int64)

    def _unique_affected(self) -> npt.NDArray[np.int64]:
        return np.unique(np.asarray(self.affected, dtype=np.int64))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        self.full_calls += 1
        arr = np.asarray(config, dtype=np.int64)
        affected = self._unique_affected()
        value = int(np.sum(arr[affected]))

        return ConstraintResult(
            satisfied=value == self.target_sum,
            name=self.name,
            residual=value,
        )

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.check(config).satisfied

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        self.partial_calls += 1

        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)
        affected = self._unique_affected()

        if np.all(assigned[affected]):
            return int(np.sum(arr[affected])) == self.target_sum

        return True


def _binary_site_layout(n: int) -> VariableLayout:
    local_space = LocalSpace.binary()

    return VariableLayout(
        specs=tuple(
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=i,
                local_space=local_space,
            )
            for i in range(n)
        )
    )


def test_dfs_only_checks_conditions_affected_by_assigned_variable() -> None:
    layout = _binary_site_layout(3)

    c0 = CountingConstraint(
        layout=layout,
        affected=(0,),
        target_sum=1,
        name="c0",
    )
    c2 = CountingConstraint(
        layout=layout,
        affected=(2,),
        target_sum=1,
        name="c2",
    )

    solver = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    )

    basis = solver.solve(
        layout,
        constraints=(c0, c2),
    )

    # Correct basis: variable 0 and 2 fixed to 1, variable 1 free.
    assert basis.n_states == 2

    # c0 is checked only after assigning variable 0.
    # It should not be checked again after assigning variable 1 or 2.
    assert c0.partial_calls == 2

    # c2 is reached only on branches surviving c0 and after variable 1 choices.
    assert c2.partial_calls == 4


@dataclass
class BadAffectedConstraint:
    layout: VariableLayout
    affected: object
    name: str = "bad_affected"

    def affected_variables(self):
        return self.affected

    def check(self, config):
        return ConstraintResult(True, name=self.name)

    def is_satisfied(self, config):
        return True

    def partial_check(self, config, assigned_mask):
        return True


def test_dfs_rejects_out_of_range_affected_variables() -> None:
    layout = _binary_site_layout(2)

    constraint = BadAffectedConstraint(
        layout=layout,
        affected=np.array([2], dtype=np.int64),
    )

    solver = DFSBasisSolver()

    with pytest.raises(ValueError, match="outside"):
        solver.solve(layout, constraints=(constraint,))


def test_dfs_deduplicates_affected_variables() -> None:
    layout = _binary_site_layout(2)

    constraint = CountingConstraint(
        layout=layout,
        affected=(0, 0),
        target_sum=1,
    )

    solver = DFSBasisSolver(
        variable_order=np.array([0, 1], dtype=np.int64),
    )

    basis = solver.solve(layout, constraints=(constraint,))

    assert basis.n_states == 2

    # If affected_variables were not deduplicated, this would be called twice
    # per assignment of variable 0.
    assert constraint.partial_calls == 2


class AlwaysTrueGlobalConstraint(BaseConstraint):
    layout: VariableLayout
    name: str = "always_true"

    def __init__(self, layout: VariableLayout):
        self.layout = layout
        self.name = "always_true"

    def check(self, config):
        self._as_config(config)
        return ConstraintResult(True, name=self.name)


def test_dfs_base_constraint_default_affects_all_variables() -> None:
    layout = _binary_site_layout(3)

    constraint = AlwaysTrueGlobalConstraint(layout)

    solver = DFSBasisSolver(sort=True)
    basis = solver.solve(layout, constraints=(constraint,))

    assert basis.n_states == 8

    affected = constraint.affected_variables()
    np.testing.assert_array_equal(
        affected,
        np.arange(layout.n_variables, dtype=np.int64),
    )
