from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, SectorCondition
from qlinks.variables import VariableLayout

ConditionLike = Constraint | SectorCondition
PartialCheck = Callable[[npt.NDArray[np.int64], npt.NDArray[np.bool_]], bool]
VariableOrderStrategy = Literal[
    "auto",
    "layout",
    "degree",
    "weighted_degree",
    "constraint_closure",
]


@dataclass(frozen=True, slots=True)
class _ConditionInfo:
    condition: ConditionLike
    affected_variables: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DFSBasisSolver:
    sort: bool = True

    # Explicit order. If provided, this overrides variable_order_strategy.
    variable_order: npt.ArrayLike | None = None

    # "auto" currently aliases "constraint_closure".
    # "layout" means [0, 1, 2, ...].
    # "degree" means variables appearing in more constraints/sectors first.
    # "weighted_degree" favors variables in smaller constraint supports.
    # "constraint_closure" greedily clusters variables that complete local constraints early.
    variable_order_strategy: VariableOrderStrategy = "auto"

    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
        *,
        max_states: int | None = None,
    ) -> Basis:
        if max_states is not None and max_states < 0:
            raise ValueError("max_states must be non-negative or None.")
        if max_states == 0:
            return Basis.empty(layout)

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

        # Conditions with no affected variables are global constants.
        # They should be checked once before DFS starts.
        if not self._zero_variable_conditions_are_satisfied(
            config=config,
            assigned_mask=assigned_mask,
            condition_infos=condition_infos,
        ):
            return Basis.empty(layout)

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
        partial_checks_by_depth = tuple(
            partial_checks_by_variable[int(variable_index)] for variable_index in variable_order
        )

        def dfs(depth: int) -> None:
            if max_states is not None and len(states) >= max_states:
                return

            if depth == n:
                # Every non-empty condition support is checked exactly when the
                # last variable in that support is assigned.  Zero-variable
                # conditions are checked once before DFS starts.  Re-running all
                # full checks at every leaf is therefore redundant and costly for
                # large bases with many local constraints.
                states.append(config.copy())
                return

            variable_index = int(variable_order[depth])

            for value in ordered_local_values[depth]:
                if max_states is not None and len(states) >= max_states:
                    return

                config[variable_index] = int(value)
                assigned_mask[variable_index] = True

                if self._partial_check_after_assignment(
                    config=config,
                    assigned_mask=assigned_mask,
                    partial_checks=partial_checks_by_depth[depth],
                ):
                    dfs(depth + 1)

                assigned_mask[variable_index] = False

        dfs(0)

        if len(states) == 0:
            return Basis.empty(layout)

        arr = np.asarray(states, dtype=np.int64)
        if self.sort:
            order = np.lexsort(arr.T[::-1])
            arr = arr[order]

        return Basis.from_states(layout, arr)

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
            "'weighted_degree', or 'constraint_closure'."
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
            partial_check = info.condition.partial_check
            for variable_index in info.affected_variables:
                lookup[int(variable_index)].append(partial_check)

        return tuple(tuple(partial_checks) for partial_checks in lookup)

    @staticmethod
    def _zero_variable_conditions_are_satisfied(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        condition_infos: Sequence[_ConditionInfo],
    ) -> bool:
        for info in condition_infos:
            if info.affected_variables.size != 0:
                continue

            if not info.condition.partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _partial_check_after_assignment(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        partial_checks: Sequence[PartialCheck],
    ) -> bool:
        for partial_check in partial_checks:
            if not partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _full_check(
        config: npt.NDArray[np.int64],
        conditions: Sequence[ConditionLike],
    ) -> bool:
        return all(condition.is_satisfied(config) for condition in conditions)
