from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:  # pragma: no cover - exercised when optional extra is absent.
    cp_model = None

from qlinks.basis.basis import Basis
from qlinks.constraints import (
    Constraint,
    DimerCoveringConstraint,
    FixedValueConstraint,
    GaussLawConstraint,
    LocalSumConstraint,
    NearestNeighborBlockadeConstraint,
    ParitySector,
    SectorCondition,
    SquareWindingSector,
    TotalValueSector,
)
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class CPSATBasisSolver:
    """
    OR-Tools CP-SAT basis solver.

    This solver is useful when the constraints are naturally integer/Boolean
    constraints. It currently supports:

        FixedValueConstraint
        LocalSumConstraint
        GaussLawConstraint
        DimerCoveringConstraint
        NearestNeighborBlockadeConstraint
        TotalValueSector
        ParitySector
        SquareWindingSector

    Unsupported custom constraints raise NotImplementedError.

    OR-Tools is imported lazily so qlinks can still be installed without it.
    """

    max_solutions: int | None = None
    num_workers: int = 1
    log_search_progress: bool = False
    sort: bool = False

    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
        *,
        max_states: int | None = None,
    ) -> Basis:
        if cp_model is None:
            raise ImportError(
                "CPSATBasisSolver requires the optional 'ortools' dependency. "
                "Install qlinks with the cpsat extra or add ortools to your "
                "environment to use this solver."
            )

        if max_states is not None and max_states < 0:
            raise ValueError("max_states must be non-negative or None.")
        if max_states == 0:
            return Basis.empty(layout)

        max_solutions = self.max_solutions
        if max_states is not None:
            if max_solutions is None:
                max_solutions = max_states
            else:
                max_solutions = min(max_solutions, max_states)

        model = cp_model.CpModel()

        variables = []
        for i in range(layout.n_variables):
            values = [int(v) for v in layout.local_space(i).values.tolist()]
            domain = cp_model.Domain.FromValues(values)
            variables.append(model.NewIntVarFromDomain(domain, f"x_{i}"))

        for constraint in constraints:
            self._add_constraint(model, variables, constraint)

        for sector in sectors:
            self._add_sector(model, variables, sector)

        collector = _BasisSolutionCollector(
            variables=variables,
            n_variables=layout.n_variables,
            max_solutions=max_solutions,
        )

        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = self.log_search_progress

        solver.SearchForAllSolutions(model, collector)

        if len(collector.states) == 0:
            return Basis.empty(layout)

        states = np.asarray(collector.states, dtype=np.int64)
        return Basis.from_states(layout, states, sort=self.sort)

    def _add_constraint(self, model, variables, constraint: Constraint) -> None:
        if isinstance(constraint, FixedValueConstraint):
            for variable_index, value in zip(
                constraint.variable_indices,
                constraint.values,
                strict=True,
            ):
                model.Add(variables[int(variable_index)] == int(value))
            return

        if isinstance(constraint, LocalSumConstraint):
            expr = sum(
                int(coeff) * variables[int(variable_index)]
                for coeff, variable_index in zip(
                    constraint.coefficients,
                    constraint.variable_indices,
                    strict=True,
                )
            )
            model.Add(expr == int(constraint.target))
            return

        if isinstance(constraint, GaussLawConstraint):
            var_indices = constraint.affected_variables()
            expr = sum(
                int(sign) * variables[int(variable_index)]
                for sign, variable_index in zip(
                    constraint.signs,
                    var_indices,
                    strict=True,
                )
            )
            model.Add(expr == int(constraint.charge))
            return

        if isinstance(constraint, DimerCoveringConstraint):
            var_indices = constraint.affected_variables()
            expr = sum(variables[int(variable_index)] for variable_index in var_indices)
            model.Add(expr == int(constraint.required_count))
            return

        if isinstance(constraint, NearestNeighborBlockadeConstraint):
            vi, vj = constraint.affected_variables()
            occupied = int(constraint.occupied_value)

            bi = model.NewBoolVar(f"blockade_{int(vi)}_is_occ")
            bj = model.NewBoolVar(f"blockade_{int(vj)}_is_occ")

            model.Add(variables[int(vi)] == occupied).OnlyEnforceIf(bi)
            model.Add(variables[int(vi)] != occupied).OnlyEnforceIf(bi.Not())

            model.Add(variables[int(vj)] == occupied).OnlyEnforceIf(bj)
            model.Add(variables[int(vj)] != occupied).OnlyEnforceIf(bj.Not())

            model.AddBoolOr([bi.Not(), bj.Not()])
            return

        raise NotImplementedError(
            f"CPSATBasisSolver does not support constraint type " f"{type(constraint).__name__}."
        )

    def _add_sector(self, model, variables, sector: SectorCondition) -> None:
        if isinstance(sector, TotalValueSector):
            expr = sum(
                int(coeff) * variables[int(variable_index)]
                for coeff, variable_index in zip(
                    sector.coefficients,
                    sector.variable_indices,
                    strict=True,
                )
            )
            model.Add(expr == int(sector.target))
            return

        if isinstance(sector, ParitySector):
            expr = sum(variables[int(variable_index)] for variable_index in sector.variable_indices)

            total = model.NewIntVar(-(10**9), 10**9, "parity_total")
            model.Add(total == expr)

            remainder = model.NewIntVar(0, 1, "parity_remainder")
            model.AddModuloEquality(remainder, total, 2)
            model.Add(remainder == int(sector.target))
            return

        if isinstance(sector, SquareWindingSector):
            expr = sum(variables[int(variable_index)] for variable_index in sector.variable_indices)
            model.Add(expr == int(sector.target))
            return

        raise NotImplementedError(
            f"CPSATBasisSolver does not support sector type " f"{type(sector).__name__}."
        )


class _BasisSolutionCollector:
    def __init__(
        self,
        variables,
        n_variables: int,
        max_solutions: int | None,
    ) -> None:
        class Callback(cp_model.CpSolverSolutionCallback):
            def __init__(self, outer) -> None:
                super().__init__()
                self.outer = outer

            def OnSolutionCallback(self) -> None:
                config = [int(self.Value(var)) for var in self.outer.variables]
                self.outer.states.append(config)

                if (
                    self.outer.max_solutions is not None
                    and len(self.outer.states) >= self.outer.max_solutions
                ):
                    self.StopSearch()

        self.variables = variables
        self.n_variables = n_variables
        self.max_solutions = max_solutions
        self.states: list[list[int]] = []
        self.callback = Callback(self)

    def __getattr__(self, name: str):
        return getattr(self.callback, name)
