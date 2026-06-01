from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, SectorCondition, all_satisfied
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class BruteForceBasisSolver:
    """
    Exhaustive product-space basis solver.

    This is simple and useful for tests, but it scales as

        prod_i dim(local_space_i)

    so it should only be used for small systems.
    """

    sort: bool = False

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

        domains = [layout.local_space(i).values.tolist() for i in range(layout.n_variables)]
        states: list[np.ndarray] = []

        for values in product(*domains):
            config = np.asarray(values, dtype=np.int64)
            if all_satisfied(config, constraints=constraints, sectors=sectors):
                states.append(config.copy())

                if max_states is not None and len(states) >= max_states:
                    break

        if len(states) == 0:
            return Basis.empty(layout)

        return Basis.from_states(layout, np.vstack(states), sort=self.sort)
