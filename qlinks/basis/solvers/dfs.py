from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, SectorCondition
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class DFSBasisSolver:
    sort: bool = True

    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
    ) -> Basis:
        n = layout.n_variables

        config = layout.default_config()
        assigned_mask = np.zeros(n, dtype=bool)

        states: list[npt.NDArray[np.int64]] = []

        all_conditions = tuple(constraints) + tuple(sectors)

        condition_indices_by_variable = self._build_condition_lookup(
            n_variables=n,
            conditions=all_conditions,
        )

        variable_order = self._variable_order(
            layout=layout,
            conditions=all_conditions,
        )

        def dfs(depth: int) -> None:
            if depth == n:
                if self._full_check(config, all_conditions):
                    states.append(config.copy())
                return

            variable_index = int(variable_order[depth])
            local_space = layout.local_space(variable_index)

            for value in local_space.values:
                config[variable_index] = int(value)
                assigned_mask[variable_index] = True

                if self._partial_check_after_assignment(
                    config=config,
                    assigned_mask=assigned_mask,
                    variable_index=variable_index,
                    all_conditions=all_conditions,
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
    def _build_condition_lookup(
        *,
        n_variables: int,
        conditions: Sequence[Constraint | SectorCondition],
    ) -> list[list[int]]:
        lookup: list[list[int]] = [[] for _ in range(n_variables)]

        for condition_index, condition in enumerate(conditions):
            affected = np.asarray(condition.affected_variables(), dtype=np.int64)

            for variable_index in affected:
                lookup[int(variable_index)].append(condition_index)

        return lookup

    @staticmethod
    def _variable_order(
        *,
        layout: VariableLayout,
        conditions: Sequence[Constraint | SectorCondition],
    ) -> npt.NDArray[np.int64]:
        """
        Simple heuristic:
            assign high-degree variables first.

        A variable that appears in many local constraints is more useful for
        early pruning.
        """
        scores = np.zeros(layout.n_variables, dtype=np.int64)

        for condition in conditions:
            affected = np.asarray(condition.affected_variables(), dtype=np.int64)
            for variable_index in affected:
                scores[int(variable_index)] += 1

        # Descending score, stable tie-break by variable index.
        return np.lexsort((np.arange(layout.n_variables), -scores)).astype(np.int64)

    @staticmethod
    def _partial_check_after_assignment(
        *,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        variable_index: int,
        all_conditions: Sequence[Constraint | SectorCondition],
        condition_indices_by_variable: list[list[int]],
    ) -> bool:
        for condition_index in condition_indices_by_variable[variable_index]:
            condition = all_conditions[condition_index]

            if not condition.partial_check(config, assigned_mask):
                return False

        return True

    @staticmethod
    def _full_check(
        config: npt.NDArray[np.int64],
        conditions: Sequence[Constraint | SectorCondition],
    ) -> bool:
        return all(condition.is_satisfied(config) for condition in conditions)
