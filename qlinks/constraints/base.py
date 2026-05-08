from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """
    Result of checking one constraint or sector condition.
    """

    satisfied: bool
    name: str = ""
    residual: object | None = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.satisfied


class Constraint(Protocol):
    """
    General constraint interface.

    A constraint decides whether a raw configuration is allowed.
    """

    name: str

    def affected_variables(self) -> npt.NDArray[np.int64]:
        ...

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        ...

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        ...

    def partial_check(
            self,
            config: npt.ArrayLike,
            assigned_mask: npt.ArrayLike,
    ) -> bool:
        ...


class SectorCondition(Protocol):
    """
    Diagonal symmetry-sector filter.
    """

    name: str

    def value(self, config: npt.ArrayLike) -> object:
        ...

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        ...

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        ...


class BaseConstraint:
    """
    Convenience base class for concrete constraints.

    This is intentionally NOT a dataclass, because dataclass inheritance with
    default fields causes problems when subclasses add required fields.
    """

    layout: VariableLayout
    name: str

    def _as_config(
        self,
        config: npt.ArrayLike,
        *,
        validate: bool = True,
    ) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        return arr

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.arange(self.layout.n_variables, dtype=np.int64)

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        self._as_config(config)
        return ConstraintResult(True, name=self.name)

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.check(config).satisfied

    def partial_check(
            self,
            config: npt.ArrayLike,
            assigned_mask: npt.ArrayLike,
    ) -> bool:
        """
        Default partial check.

        If all affected variables are assigned, perform the exact full check.
        Otherwise, do not prune.

        Subclasses should override this for stronger pruning.
        """
        affected = self.affected_variables()
        assigned = np.asarray(assigned_mask, dtype=bool)

        if np.all(assigned[affected]):
            return self.is_satisfied(config)

        return True


class BaseSectorCondition:
    """
    Convenience base class for diagonal sector filters.

    This is intentionally NOT a dataclass for the same reason as BaseConstraint.
    """

    layout: VariableLayout
    target: object
    name: str

    def _as_config(
        self,
        config: npt.ArrayLike,
        *,
        validate: bool = True,
    ) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        return arr

    def value(self, config: npt.ArrayLike) -> object:
        raise NotImplementedError

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        satisfied = actual == self.target

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=actual,
            message=f"{self.name}: value={actual}, target={self.target}",
        )

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.check(config).satisfied

    def partial_check(
            self,
            config: npt.ArrayLike,
            assigned_mask: npt.ArrayLike,
    ) -> bool:
        affected = self.affected_variables()
        assigned = np.asarray(assigned_mask, dtype=bool)

        if np.all(assigned[affected]):
            return self.is_satisfied(config)

        return True


def all_satisfied(
    config: npt.ArrayLike,
    constraints: Sequence[Constraint] = (),
    sectors: Sequence[SectorCondition] = (),
) -> bool:
    """
    Convenience helper used by simple solvers.
    """
    return all(c.is_satisfied(config) for c in constraints) and all(
        s.is_satisfied(config) for s in sectors
    )
