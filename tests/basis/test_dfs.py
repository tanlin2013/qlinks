from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from qlinks.basis import BruteForceBasisSolver, DFSBasisSolver, DFSStatistics
from qlinks.constraints import (
    BaseConstraint,
    BoundedLocalCountConstraint,
    ConstraintPropagation,
    ConstraintResult,
    DimerCoveringConstraint,
    FixedValueConstraint,
    GaussLawConstraint,
    NearestNeighborBlockadeConstraint,
)
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.models import PXPModel, SquareQDMModel
from qlinks.variables import LocalSpace, VariableKind, VariableLayout, VariableSpec


def assert_same_basis(basis_a, basis_b) -> None:
    set_a = {tuple(state.tolist()) for state in basis_a.states}
    set_b = {tuple(state.tolist()) for state in basis_b.states}
    assert set_a == set_b


def test_dfs_matches_brute_force_fixed_value() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    constraints = [
        FixedValueConstraint.single(layout, variable_index=0, value=1),
        FixedValueConstraint.single(layout, variable_index=3, value=0),
    ]

    brute = BruteForceBasisSolver(sort=True).solve(layout, constraints=constraints)
    dfs = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert_same_basis(brute, dfs)


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


# ------------------------------------------------------------------
# Test that DFSBasisSolver only calls partial_check for constraints that are affected by the
# currently assigned variable. This is important for performance,
# as partial_check can be expensive and should only be called when necessary.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Test that invalid affected variables are rejected. The affected_variables method should return
# valid variable indices within the layout, and the solver should validate this before starting DFS.
# This is important to prevent out-of-bounds errors and ensure that constraints are properly
# integrated into the solving process.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Duplicate affected variables are harmless.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Test that DFSBasisSolver treats constraints with no affected variables as global constants,
# checking them once before starting DFS and not calling partial_check for them at all.
# ------------------------------------------------------------------


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


@dataclass
class LeafExplodingConstraint:
    layout: VariableLayout
    affected: tuple[int, ...]
    target_sum: int
    name: str = "leaf_exploding"

    def __post_init__(self) -> None:
        self.partial_calls = 0
        self.full_calls = 0

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.affected, dtype=np.int64)

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        self.full_calls += 1
        raise AssertionError("DFS should not re-run full leaf checks.")

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
        affected = np.asarray(self.affected, dtype=np.int64)

        if np.all(assigned[affected]):
            return int(np.sum(arr[affected])) == self.target_sum

        return True


def test_dfs_does_not_repeat_full_checks_at_leaves() -> None:
    layout = _binary_site_layout(2)
    constraint = LeafExplodingConstraint(
        layout=layout,
        affected=(0, 1),
        target_sum=1,
    )

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=(constraint,))

    assert basis.n_states == 2
    np.testing.assert_array_equal(
        basis.states,
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    assert constraint.full_calls == 0


class PropagationCountingCountConstraint(BoundedLocalCountConstraint):
    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "propagation_calls", 0)

    def propagate(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> ConstraintPropagation:
        object.__setattr__(self, "propagation_calls", self.propagation_calls + 1)
        return super().propagate(config, assigned_mask)


def test_dfs_propagates_bounded_local_count_constraint() -> None:
    layout = _binary_site_layout(3)
    constraint = PropagationCountingCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=1,
        name="exactly_one",
    )

    basis = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    ).solve(layout, constraints=(constraint,))

    np.testing.assert_array_equal(
        basis.states,
        np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    assert constraint.propagation_calls > 0


def test_dfs_propagation_respects_conflicting_forced_assignments() -> None:
    layout = _binary_site_layout(2)
    constraints = (
        BoundedLocalCountConstraint.exact(
            layout=layout,
            variable_indices=np.array([0, 1], dtype=np.int64),
            count=2,
            name="both_occupied",
        ),
        FixedValueConstraint.single(layout, variable_index=1, value=0),
    )

    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    assert basis.n_states == 0


def test_dfs_propagated_assignments_are_undone_between_branches() -> None:
    layout = _binary_site_layout(3)
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=2,
        name="exactly_two",
    )

    basis = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    ).solve(layout, constraints=(constraint,))

    np.testing.assert_array_equal(
        basis.states,
        np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.int64,
        ),
    )


# ------------------------------------------------------------------
# Test variable ordering strategies in DFSBasisSolver. The variable order can affect the number of
# partial checks and thus the performance of the solver, but should not affect the final basis
# (up to sorting).
# ------------------------------------------------------------------


@dataclass
class SupportOnlyConstraint:
    layout: VariableLayout
    affected: tuple[int, ...]
    name: str = "support_only"

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.affected, dtype=np.int64)

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        return ConstraintResult(True, name=self.name)

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return True

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        return True


def test_dfs_layout_variable_order_strategy() -> None:
    layout = _binary_site_layout(4)

    solver = DFSBasisSolver(
        variable_order_strategy="layout",
    )

    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=(),
    )

    order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="layout",
    )

    np.testing.assert_array_equal(
        order,
        np.array([0, 1, 2, 3], dtype=np.int64),
    )


def test_dfs_degree_variable_order_strategy() -> None:
    layout = _binary_site_layout(4)

    # variable 2 appears in three constraints
    # variable 0 appears in two constraints
    # variable 1 appears in one constraint
    # variable 3 appears in zero constraints
    conditions = (
        SupportOnlyConstraint(layout, affected=(2,)),
        SupportOnlyConstraint(layout, affected=(2, 0)),
        SupportOnlyConstraint(layout, affected=(2, 0, 1)),
    )

    solver = DFSBasisSolver(
        variable_order_strategy="degree",
    )

    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=conditions,
    )

    order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="degree",
    )

    np.testing.assert_array_equal(
        order,
        np.array([2, 0, 1, 3], dtype=np.int64),
    )


def test_dfs_weighted_degree_variable_order_strategy() -> None:
    layout = _binary_site_layout(4)

    conditions = (
        SupportOnlyConstraint(layout, affected=(0, 1, 2, 3)),
        SupportOnlyConstraint(layout, affected=(2,)),
    )

    solver = DFSBasisSolver(variable_order_strategy="weighted_degree")
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=conditions,
    )

    order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="weighted_degree",
    )

    np.testing.assert_array_equal(
        order,
        np.array([2, 0, 1, 3], dtype=np.int64),
    )


def test_dfs_constraint_closure_variable_order_strategy() -> None:
    layout = _binary_site_layout(5)

    conditions = (
        SupportOnlyConstraint(layout, affected=(0, 1, 2)),
        SupportOnlyConstraint(layout, affected=(2, 3)),
        SupportOnlyConstraint(layout, affected=(3, 4)),
    )

    solver = DFSBasisSolver(variable_order_strategy="constraint_closure")
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=conditions,
    )

    order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="constraint_closure",
    )

    # The greedy closure heuristic first picks the high weighted-degree bridge
    # variable 3, then completes the size-2 supports before finishing the
    # remaining size-3 support.
    np.testing.assert_array_equal(
        order,
        np.array([3, 2, 4, 0, 1], dtype=np.int64),
    )


def test_dfs_auto_variable_order_uses_constraint_closure() -> None:
    layout = _binary_site_layout(5)
    conditions = (
        SupportOnlyConstraint(layout, affected=(0, 1, 2)),
        SupportOnlyConstraint(layout, affected=(2, 3)),
        SupportOnlyConstraint(layout, affected=(3, 4)),
    )

    solver = DFSBasisSolver(variable_order_strategy="auto")
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=conditions,
    )

    auto_order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="auto",
    )
    closure_order = solver._choose_variable_order(
        layout=layout,
        condition_infos=condition_infos,
        strategy="constraint_closure",
    )

    np.testing.assert_array_equal(auto_order, closure_order)


def test_dfs_explicit_variable_order_overrides_strategy() -> None:
    layout = _binary_site_layout(3)

    solver = DFSBasisSolver(
        sort=False,
        variable_order=np.array([2, 1, 0], dtype=np.int64),
        variable_order_strategy="degree",
    )

    basis = solver.solve(layout)

    # With sort=False and order [2,1,0], the first generated state is still
    # all zeros, but the discovery order differs from layout order.
    assert basis.n_states == 8

    expected_order = np.array([2, 1, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        solver._validate_variable_order(
            variable_order=solver.variable_order,
            n_variables=layout.n_variables,
        ),
        expected_order,
    )


def test_dfs_ordered_local_values_follow_variable_order() -> None:
    layout = VariableLayout(
        specs=(
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=0,
                local_space=LocalSpace.from_values([0, 1]),
            ),
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=1,
                local_space=LocalSpace.from_values([-1, 0, 1]),
            ),
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=2,
                local_space=LocalSpace.from_values([2, 4]),
            ),
        )
    )

    ordered_values = DFSBasisSolver._ordered_local_values(
        layout=layout,
        variable_order=np.array([2, 0, 1], dtype=np.int64),
    )

    assert len(ordered_values) == 3
    np.testing.assert_array_equal(
        ordered_values[0],
        np.array([2, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        ordered_values[1],
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        ordered_values[2],
        np.array([-1, 0, 1], dtype=np.int64),
    )


def test_dfs_invalid_variable_order_strategy_raises() -> None:
    layout = _binary_site_layout(2)
    solver = DFSBasisSolver()
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=(),
    )

    with pytest.raises(ValueError, match="variable_order_strategy"):
        solver._choose_variable_order(
            layout=layout,
            condition_infos=condition_infos,
            strategy="bad",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_order",
    [
        np.array([0, 1, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 3], dtype=np.int64),
    ],
)
def test_dfs_invalid_explicit_variable_order_raises(
    bad_order: np.ndarray,
) -> None:
    layout = _binary_site_layout(3)

    solver = DFSBasisSolver(variable_order=bad_order)

    with pytest.raises(ValueError, match="variable_order"):
        solver.solve(layout)


def test_dfs_variable_order_does_not_affect_sorted_basis() -> None:
    layout = _binary_site_layout(3)

    solver_a = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    )
    solver_b = DFSBasisSolver(
        sort=True,
        variable_order=np.array([2, 1, 0], dtype=np.int64),
    )

    basis_a = solver_a.solve(layout)
    basis_b = solver_b.solve(layout)

    np.testing.assert_array_equal(basis_a.states, basis_b.states)


def test_dfs_variable_order_can_affect_unsorted_basis() -> None:
    layout = VariableLayout(
        specs=(
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=0,
                local_space=LocalSpace.from_values([0, 1]),
            ),
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=1,
                local_space=LocalSpace.from_values([10, 20]),
            ),
        )
    )

    solver_a = DFSBasisSolver(
        sort=False,
        variable_order=np.array([0, 1], dtype=np.int64),
    )
    solver_b = DFSBasisSolver(
        sort=False,
        variable_order=np.array([1, 0], dtype=np.int64),
    )

    basis_a = solver_a.solve(layout)
    basis_b = solver_b.solve(layout)

    assert basis_a.n_states == basis_b.n_states
    assert set(map(tuple, basis_a.states)) == set(map(tuple, basis_b.states))
    assert not np.array_equal(basis_a.states, basis_b.states)


def test_dfs_basis_solver_respects_max_states() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    basis = DFSBasisSolver(sort=False).solve(
        layout,
        max_states=1,
    )

    assert basis.n_states == 1


def test_dfs_applies_root_propagation_before_first_branch() -> None:
    layout = _binary_site_layout(3)
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=0,
        name="all_empty",
    )

    basis = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    ).solve(layout, constraints=(constraint,))

    np.testing.assert_array_equal(
        basis.states,
        np.array([[0, 0, 0]], dtype=np.int64),
    )


def test_dfs_solve_with_statistics_reports_core_counters() -> None:
    layout = _binary_site_layout(3)
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=1,
        name="exactly_one",
    )

    basis, statistics = DFSBasisSolver(
        sort=True,
        variable_order=np.array([0, 1, 2], dtype=np.int64),
    ).solve_with_statistics(layout, constraints=(constraint,))

    assert isinstance(statistics, DFSStatistics)
    assert basis.n_states == 3
    assert statistics.solution_count == basis.n_states
    assert statistics.branch_count > 0
    assert statistics.propagated_assignment_count > 0
    assert statistics.propagation_round_count > 0
    assert statistics.max_depth == layout.n_variables


def test_dfs_dynamic_variable_selection_prioritizes_tight_active_support() -> None:
    layout = _binary_site_layout(4)
    conditions = (SupportOnlyConstraint(layout, affected=(0, 1, 2)),)
    solver = DFSBasisSolver(variable_order_strategy="dynamic")
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=conditions,
    )
    supports = tuple(info.affected_variables for info in condition_infos)
    condition_ids_by_variable = solver._build_condition_id_lookup(
        n_variables=layout.n_variables,
        condition_infos=condition_infos,
    )
    base_scores = solver._weighted_degree_scores(
        n_variables=layout.n_variables,
        supports=supports,
    )
    static_ranks = solver._static_ranks(
        variable_order=np.array([0, 1, 2, 3], dtype=np.int64),
        n_variables=layout.n_variables,
    )
    assigned_mask = np.array([True, True, False, False], dtype=bool)

    variable_index = solver._select_dynamic_variable(
        layout=layout,
        assigned_mask=assigned_mask,
        supports=supports,
        condition_ids_by_variable=condition_ids_by_variable,
        base_scores=base_scores,
        static_ranks=static_ranks,
    )

    assert variable_index == 2


def test_dfs_propagation_value_ordering_prefers_values_that_force_more() -> None:
    layout = _binary_site_layout(3)
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=1,
        name="exactly_one",
    )
    solver = DFSBasisSolver(value_order_strategy="propagation")
    condition_infos = solver._build_condition_infos(
        n_variables=layout.n_variables,
        conditions=(constraint,),
    )
    partial_checks_by_variable = solver._build_partial_check_lookup(
        n_variables=layout.n_variables,
        condition_infos=condition_infos,
    )
    propagators_by_variable = solver._build_propagator_lookup(
        n_variables=layout.n_variables,
        condition_infos=condition_infos,
    )
    ordered_values = solver._ordered_values_for_variable(
        variable_index=0,
        local_values_by_variable=solver._local_values_by_variable(layout=layout),
        config=layout.default_config(),
        assigned_mask=np.zeros(layout.n_variables, dtype=bool),
        partial_checks_by_variable=partial_checks_by_variable,
        propagators_by_variable=propagators_by_variable,
        value_order_strategy="propagation",
    )

    np.testing.assert_array_equal(ordered_values, np.array([1, 0], dtype=np.int64))


def test_dfs_dynamic_order_matches_static_sorted_basis() -> None:
    layout = _binary_site_layout(4)
    constraints = (
        BoundedLocalCountConstraint.exact(
            layout=layout,
            variable_indices=np.array([0, 1, 2], dtype=np.int64),
            count=1,
            name="first_three_exactly_one",
        ),
        BoundedLocalCountConstraint.at_most(
            layout=layout,
            variable_indices=np.array([2, 3], dtype=np.int64),
            max_count=1,
            name="tail_at_most_one",
        ),
    )

    static_basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)
    dynamic_basis = DFSBasisSolver(
        sort=True,
        variable_order_strategy="dynamic",
        value_order_strategy="propagation",
    ).solve(layout, constraints=constraints)

    np.testing.assert_array_equal(dynamic_basis.states, static_basis.states)


@dataclass(frozen=True, slots=True)
class PrefixObserver:
    name: str = "prefix_observer"

    def can_continue(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        changed_variables,
    ) -> bool:
        del changed_variables
        # Search only branches that can still begin with 10.
        if assigned_mask[0] and int(config[0]) != 1:
            return False
        if assigned_mask[1] and int(config[1]) != 0:
            return False
        return True

    def accept_solution(self, config: npt.NDArray[np.int64]) -> bool:
        # Filter one complete solution after branch pruning has done most work.
        return int(config[2]) == 1


def test_dfs_observer_prunes_branches_and_filters_solutions() -> None:
    layout = _binary_site_layout(3)
    basis, statistics = DFSBasisSolver(sort=True).solve_with_statistics(
        layout,
        observers=(PrefixObserver(),),
    )

    np.testing.assert_array_equal(basis.states, np.array([[1, 0, 1]], dtype=np.int64))
    assert statistics.observer_call_count > 0
    assert statistics.observer_prune_count > 0
    assert statistics.observer_solution_reject_count == 1
    assert statistics.solution_count == 1


def test_dfs_observer_works_with_propagation() -> None:
    layout = _binary_site_layout(4)
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2], dtype=np.int64),
        count=1,
        name="first_three_exactly_one",
    )

    @dataclass(frozen=True, slots=True)
    class PrefixOnlyObserver:
        name: str = "prefix_only_observer"

        def can_continue(
            self,
            config: npt.NDArray[np.int64],
            assigned_mask: npt.NDArray[np.bool_],
            changed_variables,
        ) -> bool:
            del changed_variables
            if assigned_mask[0] and int(config[0]) != 1:
                return False
            if assigned_mask[1] and int(config[1]) != 0:
                return False
            return True

        def accept_solution(self, config: npt.NDArray[np.int64]) -> bool:
            del config
            return True

    basis = DFSBasisSolver(sort=True).solve(
        layout,
        constraints=(constraint,),
        observers=(PrefixOnlyObserver(),),
    )

    np.testing.assert_array_equal(
        basis.states,
        np.array([[1, 0, 0, 0], [1, 0, 0, 1]], dtype=np.int64),
    )
