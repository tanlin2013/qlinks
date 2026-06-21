from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, ConstraintPropagation, SectorCondition
from qlinks.variables import VariableLayout

ConditionLike = Constraint | SectorCondition
PartialCheck = Callable[[npt.NDArray[np.int64], npt.NDArray[np.bool_]], bool]
Propagator = Callable[[npt.NDArray[np.int64], npt.NDArray[np.bool_]], ConstraintPropagation]
VariableOrderStrategy = Literal[
    "auto",
    "layout",
    "degree",
    "weighted_degree",
    "constraint_closure",
    "dynamic",
]
ValueOrderStrategy = Literal[
    "layout",
    "propagation",
]


class DFSSearchObserver(Protocol):
    """Read-only branch/solution observer for :class:`DFSBasisSolver`.

    Observers are intended for search-specific pruning that is not naturally a
    model constraint.  They may inspect the mutable DFS arrays but must not
    mutate them.  Returning ``False`` from ``can_continue`` prunes the current
    partial branch; returning ``False`` from ``accept_solution`` filters the
    complete configuration.

    Stateful observers may additionally implement any of these optional
    methods, which DFSBasisSolver discovers with ``getattr`` so existing
    read-only observers remain valid::

        reset(config, assigned_mask)
        on_assignments(config, assigned_mask, changed_variables)
        on_unassignments(config, assigned_mask, changed_variables)
        on_assign(config, assigned_mask, variable_index, value, forced_assignment)
        on_unassign(config, assigned_mask, variable_index, value)

    Assignment callbacks run after the DFS arrays have been updated; unassign
    callbacks run just before ``assigned_mask[variable_index]`` is cleared.
    Batched callbacks are preferred for observers that maintain incremental
    state over many variables.
    """

    name: str

    def can_continue(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        changed_variables: Sequence[int],
    ) -> bool: ...

    def accept_solution(
        self,
        config: npt.NDArray[np.int64],
    ) -> bool: ...


@dataclass(slots=True)
class DFSStatistics:
    """Execution counters for :class:`DFSBasisSolver`.

    The counters are intentionally lightweight and solver-centric.  They are
    meant for comparing pruning/order heuristics, not for proving exact search
    tree identities across implementation changes.
    """

    branch_count: int = 0
    solution_count: int = 0
    contradiction_count: int = 0
    propagated_assignment_count: int = 0
    skipped_forced_variable_count: int = 0
    partial_check_count: int = 0
    propagation_round_count: int = 0
    propagator_call_count: int = 0
    dynamic_variable_selection_count: int = 0
    dynamic_value_ordering_count: int = 0
    observer_call_count: int = 0
    observer_update_count: int = 0
    observer_prune_count: int = 0
    observer_solution_reject_count: int = 0
    max_depth: int = 0


@dataclass(frozen=True, slots=True)
class _ConditionInfo:
    condition: ConditionLike
    affected_variables: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DFSBasisSolver:
    """Depth-first constrained-basis enumerator.

    The solver incrementally assigns variables, runs partial checks and
    propagators, and optionally notifies search observers.  It is the default
    basis solver for model builds because it supports early stopping, dynamic
    ordering, propagation, and model-specific observer hooks.

    Attributes:
        sort: Whether to lexicographically sort the returned basis.
        variable_order: Optional explicit variable order.
        variable_order_strategy: Heuristic used when ``variable_order`` is not
            supplied.
        value_order_strategy: Heuristic used to order trial local values.
    """

    sort: bool = True

    # Explicit order. If provided, this overrides variable_order_strategy.
    variable_order: npt.ArrayLike | None = None

    # "auto" currently aliases "constraint_closure".
    # "layout" means [0, 1, 2, ...].
    # "degree" means variables appearing in more constraints/sectors first.
    # "weighted_degree" favors variables in smaller constraint supports.
    # "constraint_closure" greedily clusters variables that complete local constraints early.
    # "dynamic" chooses the next variable at each branch from the current propagated state.
    variable_order_strategy: VariableOrderStrategy = "auto"

    # "layout" uses the local-space value order.  "propagation" is opt-in and
    # tries values that survive the immediate local checks and force more
    # assignments first; it can help max_states/first-solution workflows, but it
    # is deliberately not the default for full enumeration.
    value_order_strategy: ValueOrderStrategy = "layout"

    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
        *,
        max_states: int | None = None,
        observers: Sequence[DFSSearchObserver] = (),
    ) -> Basis:
        basis, _ = self._solve(
            layout=layout,
            constraints=constraints,
            sectors=sectors,
            max_states=max_states,
            observers=observers,
            collect_statistics=False,
        )
        return basis

    def solve_with_statistics(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
        *,
        max_states: int | None = None,
        observers: Sequence[DFSSearchObserver] = (),
    ) -> tuple[Basis, DFSStatistics]:
        """Solve and return lightweight DFS execution statistics."""
        basis, statistics = self._solve(
            layout=layout,
            constraints=constraints,
            sectors=sectors,
            max_states=max_states,
            observers=observers,
            collect_statistics=True,
        )
        assert statistics is not None
        return basis, statistics

    def _solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
        *,
        max_states: int | None = None,
        observers: Sequence[DFSSearchObserver] = (),
        collect_statistics: bool = False,
    ) -> tuple[Basis, DFSStatistics | None]:
        statistics = DFSStatistics() if collect_statistics else None
        observers = tuple(observers)

        if max_states is not None and max_states < 0:
            raise ValueError("max_states must be non-negative or None.")
        if max_states == 0:
            return Basis.empty(layout), statistics

        n = layout.n_variables
        config = layout.default_config()
        assigned_mask = np.zeros(n, dtype=bool)
        states: list[npt.NDArray[np.int64]] = []

        all_conditions: tuple[ConditionLike, ...] = tuple(constraints) + tuple(sectors)

        condition_infos = self._build_condition_infos(
            n_variables=n,
            conditions=all_conditions,
        )
        partial_checks_by_variable = self._build_partial_check_lookup(
            n_variables=n,
            condition_infos=condition_infos,
        )
        propagators_by_variable = self._build_propagator_lookup(
            n_variables=n,
            condition_infos=condition_infos,
        )
        root_propagators = self._build_root_propagators(condition_infos=condition_infos)

        # Conditions with no affected variables are global constants.
        # They should be checked once before DFS starts.
        if not self._zero_variable_conditions_are_satisfied(
            config=config,
            assigned_mask=assigned_mask,
            condition_infos=condition_infos,
            statistics=statistics,
        ):
            if statistics is not None:
                statistics.contradiction_count += 1
            return Basis.empty(layout), statistics

        use_dynamic_order = (
            self.variable_order is None and self.variable_order_strategy == "dynamic"
        )

        if self.variable_order is None:
            variable_order = self._choose_variable_order(
                layout=layout,
                condition_infos=condition_infos,
                strategy=self.variable_order_strategy,
            )
        else:
            variable_order = self._validate_variable_order(
                variable_order=self.variable_order,
                n_variables=n,
            )

        ordered_local_values = self._ordered_local_values(
            layout=layout,
            variable_order=variable_order,
        )
        local_values_by_variable = self._local_values_by_variable(layout=layout)

        supports = tuple(info.affected_variables for info in condition_infos)
        condition_ids_by_variable = self._build_condition_id_lookup(
            n_variables=n,
            condition_infos=condition_infos,
        )
        base_dynamic_scores = self._weighted_degree_scores(
            n_variables=n,
            supports=tuple(support for support in supports if support.size != 0),
        )
        static_ranks = self._static_ranks(variable_order=variable_order, n_variables=n)

        has_propagators = any(propagators_by_variable) or bool(root_propagators)

        observer_reset_callbacks = tuple(
            callback
            for observer in observers
            if callable(callback := getattr(observer, "reset", None))
        )
        observer_assign_callbacks = tuple(
            callback
            for observer in observers
            if callable(callback := getattr(observer, "on_assign", None))
        )
        observer_unassign_callbacks = tuple(
            callback
            for observer in observers
            if callable(callback := getattr(observer, "on_unassign", None))
        )
        observer_assignments_callbacks = tuple(
            callback
            for observer in observers
            if callable(callback := getattr(observer, "on_assignments", None))
        )
        observer_unassignments_callbacks = tuple(
            callback
            for observer in observers
            if callable(callback := getattr(observer, "on_unassignments", None))
        )

        def notify_observers_reset() -> None:
            for reset in observer_reset_callbacks:
                if statistics is not None:
                    statistics.observer_update_count += 1
                reset(config, assigned_mask)

        def notify_observers_assigned(
            changed_assignments: Sequence[tuple[int, int, bool]],
        ) -> None:
            if not changed_assignments:
                return

            changed_variables = tuple(int(variable) for variable, _, _ in changed_assignments)

            for on_assignments in observer_assignments_callbacks:
                if statistics is not None:
                    statistics.observer_update_count += 1
                on_assignments(config, assigned_mask, changed_variables)

            for variable_index, value, forced_assignment in changed_assignments:
                for on_assign in observer_assign_callbacks:
                    if statistics is not None:
                        statistics.observer_update_count += 1
                    on_assign(
                        config,
                        assigned_mask,
                        int(variable_index),
                        int(value),
                        bool(forced_assignment),
                    )

        def notify_observers_unassigned(
            changed_assignments: Sequence[tuple[int, int, bool]],
        ) -> None:
            if not changed_assignments:
                return

            changed_variables = tuple(
                int(variable) for variable, _, _ in reversed(changed_assignments)
            )

            for on_unassignments in observer_unassignments_callbacks:
                if statistics is not None:
                    statistics.observer_update_count += 1
                on_unassignments(config, assigned_mask, changed_variables)

            for variable_index, value, _ in reversed(changed_assignments):
                for on_unassign in observer_unassign_callbacks:
                    if statistics is not None:
                        statistics.observer_update_count += 1
                    on_unassign(
                        config,
                        assigned_mask,
                        int(variable_index),
                        int(value),
                    )

        def observers_can_continue(changed_variables: Sequence[int]) -> bool:
            if not observers:
                return True

            for observer in observers:
                if statistics is not None:
                    statistics.observer_call_count += 1
                if not observer.can_continue(config, assigned_mask, changed_variables):
                    if statistics is not None:
                        statistics.observer_prune_count += 1
                    return False

            return True

        def observers_accept_solution() -> bool:
            if not observers:
                return True

            for observer in observers:
                if statistics is not None:
                    statistics.observer_call_count += 1
                if not observer.accept_solution(config):
                    if statistics is not None:
                        statistics.observer_solution_reject_count += 1
                    return False

            return True

        notify_observers_reset()

        def record_solution() -> None:
            if not observers_accept_solution():
                return
            states.append(config.copy())
            if statistics is not None:
                statistics.solution_count += 1
                statistics.max_depth = max(
                    statistics.max_depth,
                    int(np.count_nonzero(assigned_mask)),
                )

        def record_branch() -> None:
            if statistics is not None:
                statistics.branch_count += 1

        def record_contradiction() -> None:
            if statistics is not None:
                statistics.contradiction_count += 1

        if not has_propagators and not use_dynamic_order:
            partial_checks_by_depth = tuple(
                partial_checks_by_variable[int(variable_index)] for variable_index in variable_order
            )

            def dfs_without_propagation(depth: int) -> None:
                if max_states is not None and len(states) >= max_states:
                    return

                if statistics is not None:
                    statistics.max_depth = max(statistics.max_depth, depth)

                if depth == n:
                    # Every non-empty condition support is checked exactly when the
                    # last variable in that support is assigned.  Zero-variable
                    # conditions are checked once before DFS starts.  Re-running all
                    # full checks at every leaf is therefore redundant and costly for
                    # large bases with many local constraints.
                    record_solution()
                    return

                variable_index = int(variable_order[depth])

                for value in ordered_local_values[depth]:
                    if max_states is not None and len(states) >= max_states:
                        return

                    record_branch()
                    config[variable_index] = int(value)
                    assigned_mask[variable_index] = True
                    changed_assignments = ((variable_index, int(value), False),)

                    partial_ok = self._partial_check_after_assignment(
                        config=config,
                        assigned_mask=assigned_mask,
                        partial_checks=partial_checks_by_depth[depth],
                        statistics=statistics,
                    )
                    if not partial_ok:
                        record_contradiction()
                    else:
                        notify_observers_assigned(changed_assignments)
                        if observers_can_continue((variable_index,)):
                            dfs_without_propagation(depth + 1)
                        notify_observers_unassigned(changed_assignments)

                    assigned_mask[variable_index] = False

            dfs_without_propagation(0)
        else:

            def assign_with_propagation(
                variable_index: int,
                value: int,
                changed_assignments: list[tuple[int, int, bool]],
                *,
                forced_assignment: bool = False,
            ) -> bool:
                """Assign one variable and close all forced local consequences."""
                pending: list[tuple[int, int, bool]] = [
                    (int(variable_index), int(value), bool(forced_assignment))
                ]

                while pending:
                    current_variable, current_value, current_forced = pending.pop()

                    if assigned_mask[current_variable]:
                        if int(config[current_variable]) != current_value:
                            record_contradiction()
                            return False
                        continue

                    config[current_variable] = current_value
                    assigned_mask[current_variable] = True
                    changed_assignments.append((current_variable, current_value, current_forced))
                    if statistics is not None:
                        if current_forced:
                            statistics.propagated_assignment_count += 1
                        statistics.max_depth = max(
                            statistics.max_depth,
                            int(np.count_nonzero(assigned_mask)),
                        )

                    if not self._partial_check_after_assignment(
                        config=config,
                        assigned_mask=assigned_mask,
                        partial_checks=partial_checks_by_variable[current_variable],
                        statistics=statistics,
                    ):
                        record_contradiction()
                        return False

                    propagation = self._propagate_after_assignment(
                        config=config,
                        assigned_mask=assigned_mask,
                        propagators=propagators_by_variable[current_variable],
                        statistics=statistics,
                    )
                    if not propagation.consistent:
                        record_contradiction()
                        return False

                    pending.extend(
                        (int(forced_variable), int(forced_value), True)
                        for forced_variable, forced_value in propagation.forced_assignments
                    )

                return True

            def undo(
                changed_assignments: Sequence[tuple[int, int, bool]],
                *,
                notify_observers: bool,
            ) -> None:
                if notify_observers:
                    notify_observers_unassigned(changed_assignments)

                for variable_index, _, _ in reversed(changed_assignments):
                    assigned_mask[int(variable_index)] = False

            initial_propagation = self._propagate_after_assignment(
                config=config,
                assigned_mask=assigned_mask,
                propagators=root_propagators,
                statistics=statistics,
            )
            if not initial_propagation.consistent:
                record_contradiction()
                return Basis.empty(layout), statistics

            root_changed_assignments: list[tuple[int, int, bool]] = []
            for forced_variable, forced_value in initial_propagation.forced_assignments:
                if not assign_with_propagation(
                    int(forced_variable),
                    int(forced_value),
                    root_changed_assignments,
                    forced_assignment=True,
                ):
                    return Basis.empty(layout), statistics

            root_changed_variables = tuple(
                int(variable_index) for variable_index, _, _ in root_changed_assignments
            )
            notify_observers_assigned(root_changed_assignments)
            if not observers_can_continue(root_changed_variables):
                return Basis.empty(layout), statistics

            def dfs_with_static_order(depth: int) -> None:
                if max_states is not None and len(states) >= max_states:
                    return

                while depth < n and assigned_mask[int(variable_order[depth])]:
                    depth += 1
                    if statistics is not None:
                        statistics.skipped_forced_variable_count += 1

                if depth == n:
                    # Every non-empty condition support is checked exactly when the
                    # last variable in that support is assigned.  Zero-variable
                    # conditions are checked once before DFS starts.  Re-running all
                    # full checks at every leaf is therefore redundant and costly for
                    # large bases with many local constraints.
                    record_solution()
                    return

                variable_index = int(variable_order[depth])

                values = self._ordered_values_for_variable(
                    variable_index=variable_index,
                    local_values_by_variable=local_values_by_variable,
                    config=config,
                    assigned_mask=assigned_mask,
                    partial_checks_by_variable=partial_checks_by_variable,
                    propagators_by_variable=propagators_by_variable,
                    value_order_strategy=self.value_order_strategy,
                    statistics=statistics,
                )

                for value in values:
                    if max_states is not None and len(states) >= max_states:
                        return

                    record_branch()
                    changed_assignments: list[tuple[int, int, bool]] = []
                    assignment_ok = assign_with_propagation(
                        variable_index,
                        int(value),
                        changed_assignments,
                    )
                    if assignment_ok:
                        changed_variables = tuple(
                            int(variable_index) for variable_index, _, _ in changed_assignments
                        )
                        notify_observers_assigned(changed_assignments)
                        if observers_can_continue(changed_variables):
                            dfs_with_static_order(depth + 1)

                    undo(changed_assignments, notify_observers=assignment_ok)

            def dfs_with_dynamic_order() -> None:
                if max_states is not None and len(states) >= max_states:
                    return

                if np.all(assigned_mask):
                    # Every non-empty condition support is checked exactly when the
                    # last variable in that support is assigned.  Zero-variable
                    # conditions are checked once before DFS starts.  Re-running all
                    # full checks at every leaf is therefore redundant and costly for
                    # large bases with many local constraints.
                    record_solution()
                    return

                variable_index = self._select_dynamic_variable(
                    layout=layout,
                    assigned_mask=assigned_mask,
                    supports=supports,
                    condition_ids_by_variable=condition_ids_by_variable,
                    base_scores=base_dynamic_scores,
                    static_ranks=static_ranks,
                    statistics=statistics,
                )

                values = self._ordered_values_for_variable(
                    variable_index=variable_index,
                    local_values_by_variable=local_values_by_variable,
                    config=config,
                    assigned_mask=assigned_mask,
                    partial_checks_by_variable=partial_checks_by_variable,
                    propagators_by_variable=propagators_by_variable,
                    value_order_strategy=self.value_order_strategy,
                    statistics=statistics,
                )

                for value in values:
                    if max_states is not None and len(states) >= max_states:
                        return

                    record_branch()
                    changed_assignments: list[tuple[int, int, bool]] = []
                    assignment_ok = assign_with_propagation(
                        variable_index,
                        int(value),
                        changed_assignments,
                    )
                    if assignment_ok:
                        changed_variables = tuple(
                            int(variable_index) for variable_index, _, _ in changed_assignments
                        )
                        notify_observers_assigned(changed_assignments)
                        if observers_can_continue(changed_variables):
                            dfs_with_dynamic_order()

                    undo(changed_assignments, notify_observers=assignment_ok)

            if use_dynamic_order:
                dfs_with_dynamic_order()
            else:
                dfs_with_static_order(0)

        if len(states) == 0:
            return Basis.empty(layout), statistics

        arr = np.asarray(states, dtype=np.int64)
        if self.sort:
            order = np.lexsort(arr.T[::-1])
            arr = arr[order]

        return Basis.from_states(layout, arr), statistics

    @staticmethod
    def _validate_variable_order(
        *,
        variable_order: npt.ArrayLike,
        n_variables: int,
    ) -> npt.NDArray[np.int64]:
        order = np.asarray(variable_order, dtype=np.int64)

        if order.shape != (n_variables,):
            raise ValueError("variable_order must have shape (layout.n_variables,).")

        if set(order.tolist()) != set(range(n_variables)):
            raise ValueError("variable_order must be a permutation of variable indices.")

        return order

    @classmethod
    def _choose_variable_order(
        cls,
        *,
        layout: VariableLayout,
        condition_infos: Sequence[_ConditionInfo],
        strategy: VariableOrderStrategy,
    ) -> npt.NDArray[np.int64]:
        if strategy == "auto":
            strategy = "constraint_closure"

        # Dynamic ordering still needs a deterministic static order as a
        # tie-breaker and as a fallback when an explicit order is requested.
        if strategy == "dynamic":
            strategy = "constraint_closure"

        if strategy == "layout":
            return np.arange(layout.n_variables, dtype=np.int64)

        if strategy == "degree":
            return cls._degree_variable_order(
                layout=layout,
                condition_infos=condition_infos,
            )

        if strategy == "weighted_degree":
            return cls._weighted_degree_variable_order(
                layout=layout,
                condition_infos=condition_infos,
            )

        if strategy == "constraint_closure":
            return cls._constraint_closure_variable_order(
                layout=layout,
                condition_infos=condition_infos,
            )

        raise ValueError(
            "variable_order_strategy must be one of 'auto', 'layout', 'degree', "
            "'weighted_degree', 'constraint_closure', or 'dynamic'."
        )

    @staticmethod
    def _condition_affected_variables(
        *,
        condition: Constraint | SectorCondition,
        n_variables: int,
    ) -> npt.NDArray[np.int64]:
        affected = np.asarray(condition.affected_variables(), dtype=np.int64)

        if affected.ndim != 1:
            raise ValueError(f"{condition.name}.affected_variables() must return a 1D array.")

        if affected.size == 0:
            return affected

        if np.any(affected < 0) or np.any(affected >= n_variables):
            raise ValueError(
                f"{condition.name}.affected_variables() contains indices outside "
                f"[0, {n_variables})."
            )

        # Deduplicate while preserving deterministic order.
        _, first_indices = np.unique(affected, return_index=True)
        return affected[np.sort(first_indices)].astype(np.int64, copy=False)

    @classmethod
    def _degree_variable_order(
        cls,
        *,
        layout: VariableLayout,
        condition_infos: Sequence[_ConditionInfo],
    ) -> npt.NDArray[np.int64]:
        """
        Static degree heuristic.

        Variables appearing in more constraints/sectors are assigned earlier,
        because they are more likely to trigger early pruning.
        """
        scores = np.zeros(layout.n_variables, dtype=np.int64)

        for info in condition_infos:
            for variable_index in info.affected_variables:
                scores[int(variable_index)] += 1

        # Descending score, deterministic tie-break by variable index.
        return np.lexsort((np.arange(layout.n_variables), -scores)).astype(np.int64)

    @classmethod
    def _weighted_degree_variable_order(
        cls,
        *,
        layout: VariableLayout,
        condition_infos: Sequence[_ConditionInfo],
    ) -> npt.NDArray[np.int64]:
        """Static weighted-degree heuristic.

        Compared with ``degree``, this gives more weight to small local
        constraints.  Completing a small constraint early is usually more useful
        for pruning than touching a very large sector condition early.
        """
        scores = np.zeros(layout.n_variables, dtype=np.float64)

        for info in condition_infos:
            support_size = int(info.affected_variables.size)
            if support_size == 0:
                continue
            weight = 1.0 / float(support_size)
            for variable_index in info.affected_variables:
                scores[int(variable_index)] += weight

        # Descending score, deterministic tie-break by variable index.
        return np.lexsort((np.arange(layout.n_variables), -scores)).astype(np.int64)

    @classmethod
    def _constraint_closure_variable_order(
        cls,
        *,
        layout: VariableLayout,
        condition_infos: Sequence[_ConditionInfo],
    ) -> npt.NDArray[np.int64]:
        """Greedy support-closure heuristic.

        Local constraints prune only after enough of their support is assigned.
        A pure degree heuristic may scatter assignments across the lattice and
        postpone those decisive checks.  This heuristic builds a deterministic
        static order by repeatedly choosing the unassigned variable that most
        strongly reduces the remaining support of nearby conditions, with a
        large bonus for completing a condition support.
        """
        n_variables = int(layout.n_variables)
        remaining_variables = set(range(n_variables))
        order: list[int] = []

        supports = tuple(
            np.asarray(info.affected_variables, dtype=np.int64)
            for info in condition_infos
            if info.affected_variables.size != 0
        )

        if not supports:
            return np.arange(n_variables, dtype=np.int64)

        affected_condition_ids_by_variable: list[list[int]] = [[] for _ in range(n_variables)]
        for condition_id, support in enumerate(supports):
            for variable_index in support:
                affected_condition_ids_by_variable[int(variable_index)].append(condition_id)

        base_scores = cls._weighted_degree_scores(
            n_variables=n_variables,
            supports=supports,
        )

        remaining_support_sizes = np.asarray([support.size for support in supports], dtype=np.int64)

        while remaining_variables:
            best_key: tuple[float, float, float, int, int] | None = None
            best_variable: int | None = None

            for variable_index in remaining_variables:
                completed = 0
                closeness = 0.0
                for condition_id in affected_condition_ids_by_variable[variable_index]:
                    remaining_after = int(remaining_support_sizes[condition_id] - 1)
                    if remaining_after == 0:
                        completed += 1
                    closeness += 1.0 / float(remaining_after + 1)

                # Larger entries are better, except the final variable-index
                # tie-break, where smaller is better.  The local-space term is a
                # fail-first tie-break for heterogeneous layouts.
                local_dimension = int(layout.local_space(variable_index).dim)
                key = (
                    float(completed),
                    closeness,
                    float(base_scores[variable_index]),
                    -local_dimension,
                    -variable_index,
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_variable = variable_index

            assert best_variable is not None

            order.append(best_variable)
            remaining_variables.remove(best_variable)
            for condition_id in affected_condition_ids_by_variable[best_variable]:
                remaining_support_sizes[condition_id] -= 1

        return np.asarray(order, dtype=np.int64)

    @staticmethod
    def _weighted_degree_scores(
        *,
        n_variables: int,
        supports: Sequence[npt.NDArray[np.int64]],
    ) -> npt.NDArray[np.float64]:
        scores = np.zeros(n_variables, dtype=np.float64)

        for support in supports:
            support_size = int(support.size)
            if support_size == 0:
                continue
            weight = 1.0 / float(support_size)
            for variable_index in support:
                scores[int(variable_index)] += weight

        return scores

    @staticmethod
    def _static_ranks(
        *,
        variable_order: npt.ArrayLike,
        n_variables: int,
    ) -> npt.NDArray[np.int64]:
        ranks = np.empty(n_variables, dtype=np.int64)
        order = np.asarray(variable_order, dtype=np.int64)
        for rank, variable_index in enumerate(order):
            ranks[int(variable_index)] = int(rank)
        return ranks

    @staticmethod
    def _build_condition_id_lookup(
        *,
        n_variables: int,
        condition_infos: Sequence[_ConditionInfo],
    ) -> tuple[tuple[int, ...], ...]:
        lookup: list[list[int]] = [[] for _ in range(n_variables)]

        for condition_id, info in enumerate(condition_infos):
            for variable_index in info.affected_variables:
                lookup[int(variable_index)].append(condition_id)

        return tuple(tuple(ids) for ids in lookup)

    @classmethod
    def _select_dynamic_variable(
        cls,
        *,
        layout: VariableLayout,
        assigned_mask: npt.NDArray[np.bool_],
        supports: Sequence[npt.NDArray[np.int64]],
        condition_ids_by_variable: Sequence[Sequence[int]],
        base_scores: npt.NDArray[np.float64],
        static_ranks: npt.NDArray[np.int64],
        statistics: DFSStatistics | None = None,
    ) -> int:
        """Choose the next unassigned variable from the current partial state."""
        if statistics is not None:
            statistics.dynamic_variable_selection_count += 1

        best_key: tuple[float, float, float, float, int, int, int] | None = None
        best_variable: int | None = None

        unassigned_variables = np.flatnonzero(~assigned_mask)
        for variable_index_raw in unassigned_variables:
            variable_index = int(variable_index_raw)
            completed = 0
            closeness = 0.0
            active_constraints = 0

            for condition_id in condition_ids_by_variable[variable_index]:
                support = supports[int(condition_id)]
                if support.size == 0:
                    continue
                remaining = int(np.count_nonzero(~assigned_mask[support]))
                if remaining <= 0:
                    continue
                active_constraints += 1
                remaining_after = remaining - 1
                if remaining_after == 0:
                    completed += 1
                closeness += 1.0 / float(remaining_after + 1)

            local_dimension = int(layout.local_space(variable_index).dim)
            key = (
                float(completed),
                closeness,
                float(active_constraints),
                float(base_scores[variable_index]),
                -local_dimension,
                -int(static_ranks[variable_index]),
                -variable_index,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_variable = variable_index

        assert best_variable is not None
        return best_variable

    @staticmethod
    def _ordered_local_values(
        *,
        layout: VariableLayout,
        variable_order: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.int64], ...]:
        """Return local-space values in DFS traversal order.

        This avoids repeatedly looking up each variable's local space inside the
        recursive DFS hot path. The returned arrays are defensive copies, so
        the solver cannot accidentally mutate layout-owned arrays.
        """
        order = np.asarray(variable_order, dtype=np.int64)

        return tuple(
            np.asarray(
                layout.local_space(int(variable_index)).values,
                dtype=np.int64,
            ).copy()
            for variable_index in order
        )

    @staticmethod
    def _local_values_by_variable(
        *,
        layout: VariableLayout,
    ) -> tuple[npt.NDArray[np.int64], ...]:
        return tuple(
            np.asarray(layout.local_space(variable_index).values, dtype=np.int64).copy()
            for variable_index in range(layout.n_variables)
        )

    @classmethod
    def _ordered_values_for_variable(
        cls,
        *,
        variable_index: int,
        local_values_by_variable: Sequence[npt.NDArray[np.int64]],
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        partial_checks_by_variable: Sequence[Sequence[PartialCheck]],
        propagators_by_variable: Sequence[Sequence[Propagator]],
        value_order_strategy: ValueOrderStrategy,
        statistics: DFSStatistics | None = None,
    ) -> npt.NDArray[np.int64]:
        values = np.asarray(local_values_by_variable[int(variable_index)], dtype=np.int64)

        if value_order_strategy == "layout":
            return values

        if value_order_strategy != "propagation":
            raise ValueError("value_order_strategy must be one of 'layout' or 'propagation'.")

        if statistics is not None:
            statistics.dynamic_value_ordering_count += 1

        keys: list[tuple[int, int, int]] = []
        for value_position, value in enumerate(values):
            trial_config = config.copy()
            trial_assigned = assigned_mask.copy()
            trial_config[int(variable_index)] = int(value)
            trial_assigned[int(variable_index)] = True

            consistent = cls._partial_check_after_assignment(
                config=trial_config,
                assigned_mask=trial_assigned,
                partial_checks=partial_checks_by_variable[int(variable_index)],
                statistics=None,
            )
            forced_count = 0
            if consistent:
                propagation = cls._propagate_after_assignment(
                    config=trial_config,
                    assigned_mask=trial_assigned,
                    propagators=propagators_by_variable[int(variable_index)],
                    statistics=None,
                )
                consistent = propagation.consistent
                if consistent:
                    for forced_variable, forced_value in propagation.forced_assignments:
                        forced_variable = int(forced_variable)
                        forced_value = int(forced_value)
                        if trial_assigned[forced_variable]:
                            if int(trial_config[forced_variable]) != forced_value:
                                consistent = False
                                break
                        else:
                            forced_count += 1

            # Sort by: immediately viable values first, then values that trigger
            # more forced assignments, then the original local-space order.
            keys.append((0 if consistent else 1, -forced_count, value_position))

        order = np.asarray(sorted(range(values.size), key=lambda i: keys[i]), dtype=np.int64)
        return values[order]

    @classmethod
    def _build_condition_infos(
        cls,
        *,
        n_variables: int,
        conditions: Sequence[ConditionLike],
    ) -> tuple[_ConditionInfo, ...]:
        return tuple(
            _ConditionInfo(
                condition=condition,
                affected_variables=cls._condition_affected_variables(
                    n_variables=n_variables,
                    condition=condition,
                ),
            )
            for condition in conditions
        )

    @staticmethod
    def _build_partial_check_lookup(
        *,
        n_variables: int,
        condition_infos: Sequence[_ConditionInfo],
    ) -> tuple[tuple[PartialCheck, ...], ...]:
        lookup: list[list[PartialCheck]] = [[] for _ in range(n_variables)]

        for info in condition_infos:
            # Propagating conditions report both contradictions and forced
            # assignments through propagate().  Do not also call their
            # partial_check(), because in most structured constraints it simply
            # delegates to propagate() and would duplicate the hot-path work.
            if callable(getattr(info.condition, "propagate", None)):
                continue

            partial_check = info.condition.partial_check
            for variable_index in info.affected_variables:
                lookup[int(variable_index)].append(partial_check)

        return tuple(tuple(partial_checks) for partial_checks in lookup)

    @staticmethod
    def _build_propagator_lookup(
        *,
        n_variables: int,
        condition_infos: Sequence[_ConditionInfo],
    ) -> tuple[tuple[Propagator, ...], ...]:
        lookup: list[list[Propagator]] = [[] for _ in range(n_variables)]

        for info in condition_infos:
            propagate = getattr(info.condition, "propagate", None)
            if not callable(propagate):
                continue
            for variable_index in info.affected_variables:
                lookup[int(variable_index)].append(propagate)

        return tuple(tuple(propagators) for propagators in lookup)

    @staticmethod
    def _build_root_propagators(
        *,
        condition_infos: Sequence[_ConditionInfo],
    ) -> tuple[Propagator, ...]:
        """Return every propagator once for root-level propagation.

        Per-variable lookup only runs a propagator after one of its support
        variables has been assigned.  Some structured constraints can already
        force values from the empty partial assignment, such as fixed-value
        constraints or fixed-sum sectors with an extremal target.
        """
        propagators: list[Propagator] = []

        for info in condition_infos:
            propagate = getattr(info.condition, "propagate", None)
            if callable(propagate):
                propagators.append(propagate)

        return tuple(propagators)

    @staticmethod
    def _zero_variable_conditions_are_satisfied(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        condition_infos: Sequence[_ConditionInfo],
        statistics: DFSStatistics | None = None,
    ) -> bool:
        for info in condition_infos:
            if info.affected_variables.size != 0:
                continue

            if statistics is not None:
                statistics.partial_check_count += 1
            if not info.condition.partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _partial_check_after_assignment(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        partial_checks: Sequence[PartialCheck],
        statistics: DFSStatistics | None = None,
    ) -> bool:
        for partial_check in partial_checks:
            if statistics is not None:
                statistics.partial_check_count += 1
            if not partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _propagate_after_assignment(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        propagators: Sequence[Propagator],
        statistics: DFSStatistics | None = None,
    ) -> ConstraintPropagation:
        forced_assignments: list[tuple[int, int]] = []

        if statistics is not None and propagators:
            statistics.propagation_round_count += 1

        for propagate in propagators:
            if statistics is not None:
                statistics.propagator_call_count += 1
            result = propagate(config, assigned_mask)
            if not result.consistent:
                return ConstraintPropagation.contradiction()
            forced_assignments.extend(result.forced_assignments)

        if not forced_assignments:
            return ConstraintPropagation.no_change()

        forced_by_variable: dict[int, int] = {}
        for variable_index, value in forced_assignments:
            variable_index = int(variable_index)
            value = int(value)
            previous = forced_by_variable.get(variable_index)
            if previous is not None and previous != value:
                return ConstraintPropagation.contradiction()
            forced_by_variable[variable_index] = value

        return ConstraintPropagation(forced_assignments=tuple(sorted(forced_by_variable.items())))

    @staticmethod
    def _full_check(
        config: npt.NDArray[np.int64],
        conditions: Sequence[ConditionLike],
    ) -> bool:
        return all(condition.is_satisfied(config) for condition in conditions)
