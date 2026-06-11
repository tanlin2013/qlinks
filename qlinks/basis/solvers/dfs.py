from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, SectorCondition
from qlinks.variables import VariableLayout

ConditionLike = Constraint | SectorCondition
VariableOrderStrategy = Literal["auto", "layout", "degree"]


@dataclass(frozen=True, slots=True)
class _ConditionInfo:
    condition: ConditionLike
    affected_variables: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DFSBasisSolver:
    sort: bool = True

    # Explicit order. If provided, this overrides variable_order_strategy.
    variable_order: npt.ArrayLike | None = None

    # "auto" currently aliases "degree".
    # "layout" means [0, 1, 2, ...].
    # "degree" means variables appearing in more constraints/sectors first.
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
        condition_indices_by_variable = self._build_condition_lookup(
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

        def dfs(depth: int) -> None:
            if max_states is not None and len(states) >= max_states:
                return

            if depth == n:
                if self._full_check(config, all_conditions):
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
                    variable_index=variable_index,
                    condition_infos=condition_infos,
                    condition_indices_by_variable=condition_indices_by_variable,
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
            strategy = "degree"

        if strategy == "layout":
            return np.arange(layout.n_variables, dtype=np.int64)

        if strategy == "degree":
            return cls._degree_variable_order(
                layout=layout,
                condition_infos=condition_infos,
            )

        raise ValueError("variable_order_strategy must be one of 'auto', 'layout', or 'degree'.")

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
    def _build_condition_lookup(
        *,
        n_variables: int,
        condition_infos: Sequence[_ConditionInfo],
    ) -> list[list[int]]:
        lookup: list[list[int]] = [[] for _ in range(n_variables)]

        for condition_index, info in enumerate(condition_infos):
            for variable_index in info.affected_variables:
                lookup[int(variable_index)].append(condition_index)

        return lookup

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
        variable_index: int,
        condition_infos: Sequence[_ConditionInfo],
        condition_indices_by_variable: list[list[int]],
    ) -> bool:
        for condition_index in condition_indices_by_variable[variable_index]:
            condition = condition_infos[condition_index].condition

            if not condition.partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _full_check(
        config: npt.NDArray[np.int64],
        conditions: Sequence[ConditionLike],
    ) -> bool:
        return all(condition.is_satisfied(config) for condition in conditions)
