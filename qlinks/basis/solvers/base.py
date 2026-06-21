from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from qlinks.basis.basis import Basis
from qlinks.constraints import Constraint, ConstraintCollection, SectorCondition
from qlinks.variables import VariableLayout


class BasisSolver(Protocol):
    """Protocol implemented by constrained-basis solvers.

    Solvers enumerate configurations from a :class:`VariableLayout` subject to
    constraints and sector filters, then return a :class:`Basis`.
    """

    def solve(
        self,
        layout: VariableLayout,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
    ) -> Basis: ...


@dataclass(frozen=True, slots=True)
class SolverInput:
    """Immutable bundle of inputs shared by basis solvers.

    Attributes:
        layout: Variable layout defining local spaces.
        constraints: Constraints all states must satisfy.
        sectors: Sector filters all states must satisfy.
    """

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
