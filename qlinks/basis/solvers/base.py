from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, ConstraintCollection, SectorCondition
from qlinks.variables import VariableLayout


class BasisSolver(Protocol):
    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
    ) -> Basis:
        ...


@dataclass(frozen=True, slots=True)
class SolverInput:
    layout: VariableLayout
    constraints: tuple[Constraint, ...]
    sectors: tuple[SectorCondition, ...]

    @classmethod
    def from_parts(
        cls,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
    ) -> SolverInput:
        return cls(
            layout=layout,
            constraints=tuple(constraints),
            sectors=tuple(sectors),
        )

    @classmethod
    def from_collection(
        cls,
        layout: VariableLayout,
        collection: ConstraintCollection,
    ) -> SolverInput:
        return cls(
            layout=layout,
            constraints=collection.constraints,
            sectors=collection.sectors,
        )
    