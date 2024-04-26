from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from ortools.sat.python import cp_model

from qlinks import logger
from qlinks.computation_basis import ComputationBasis


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: Dict):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solutions = []

    def on_solution_callback(self) -> None:
        self.solutions.append([self.Value(v) for v in self.__variables.values()])

    @property
    def n_solutions(self) -> int:
        return len(self.solutions)


@dataclass(slots=True)
class CpModel:
    shape: Tuple[int, int]
    charge_distri: npt.NDArray[np.int64]
    flux_sector: Optional[Tuple[int, int]] = field(default=None)
    _model: cp_model.CpModel = field(init=False, repr=False)
    _solver: cp_model.CpSolver = field(init=False, repr=False)
    _links: Dict = field(default_factory=dict, init=False, repr=False)
    _callback: SolutionCallback = field(init=False, repr=False)

    def __post_init__(self):
        self._model = cp_model.CpModel()
        self._solver = cp_model.CpSolver()
        for j, i, k in product(range(self.shape[1]), range(self.shape[0]), range(2)):
            self._links[(i, j, k)] = self._model.NewIntVar(0, 1, f"link_{i}_{j}_{k}")
        self.constraint_gauss_law()
        self.constraint_flux_sector()
        self._callback = SolutionCallback(self._links)

    def constraint_gauss_law(self) -> None:
        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            self._model.Add(
                self._links[(i, j, 0)]
                + self._links[(i, j, 1)]
                - self._links[((i - 1) % self.shape[0], j, 0)]
                - self._links[(i, (j - 1) % self.shape[1], 1)]
                == self.charge_distri[i, j]
            )

    def constraint_flux_sector(self) -> None:
        if self.flux_sector is not None:
            if (np.array(self.shape) % 2 != 0).any():
                raise ValueError("The shape of the lattice must be even.")
            for i in range(self.shape[0]):
                self._model.Add(
                    sum(self._links[(i, j, 0)] for j in range(self.shape[1])) - self.shape[1] // 2
                    == self.flux_sector[0]
                )
            for j in range(self.shape[1]):
                self._model.Add(
                    sum(self._links[(i, j, 1)] for i in range(self.shape[0])) - self.shape[0] // 2
                    == self.flux_sector[1]
                )

    @property
    def n_solutions(self) -> int:
        return self._callback.n_solutions

    def solve(self, all_solutions: bool = True) -> None:
        self._solver.parameters.enumerate_all_solutions = all_solutions
        status = self._solver.solve(self._model, self._callback)
        if status == cp_model.OPTIMAL:
            logger.info(f"Found {self._callback.n_solutions} optimal solutions.")
            logger.info(self._solver.ResponseStats())

    def to_basis(self) -> ComputationBasis:
        basis = ComputationBasis(np.vstack(self._callback.solutions))
        basis.sort()
        return basis
