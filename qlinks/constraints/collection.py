from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy.typing as npt

from qlinks.constraints.base import Constraint, ConstraintResult, SectorCondition


@dataclass(frozen=True, slots=True)
class ConstraintCollection:
    """
    Bundle local constraints and diagonal sector conditions.

    The future basis solvers should consume this object or its two lists.
    """

    constraints: tuple[Constraint, ...] = ()
    sectors: tuple[SectorCondition, ...] = ()

    @classmethod
    def from_sequences(
        cls,
        constraints: Sequence[Constraint] = (),
        sectors: Sequence[SectorCondition] = (),
    ) -> ConstraintCollection:
        return cls(tuple(constraints), tuple(sectors))

    def check_constraints(self, config: npt.ArrayLike) -> tuple[ConstraintResult, ...]:
        return tuple(constraint.check(config) for constraint in self.constraints)

    def check_sectors(self, config: npt.ArrayLike) -> tuple[ConstraintResult, ...]:
        return tuple(sector.check(config) for sector in self.sectors)

    def check_all(self, config: npt.ArrayLike) -> tuple[ConstraintResult, ...]:
        return self.check_constraints(config) + self.check_sectors(config)

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return all(result.satisfied for result in self.check_all(config))

    def first_failure(self, config: npt.ArrayLike) -> ConstraintResult | None:
        for result in self.check_all(config):
            if not result.satisfied:
                return result
        return None
    