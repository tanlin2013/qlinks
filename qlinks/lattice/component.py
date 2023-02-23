from __future__ import annotations

from dataclasses import astuple, dataclass, field
from functools import total_ordering
from typing import Iterator, TypeAlias

import numpy as np

Real: TypeAlias = int | float | np.floating


@total_ordering
@dataclass
class Site:
    pos_x: int
    pos_y: int

    def __array__(self) -> np.ndarray:
        return np.array(astuple(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.pos_x}, {self.pos_y})"

    def __hash__(self) -> int:
        return hash((self.pos_x, self.pos_y))

    def __iter__(self) -> Iterator[int]:
        return (pos for pos in (self.pos_x, self.pos_y))

    def __add__(self, other: Site | UnitVector) -> Site:
        return Site(self.pos_x + other.pos_x, self.pos_y + other.pos_y)

    def __sub__(self, other: Site) -> UnitVector:
        return UnitVector(self.pos_x - other.pos_x, self.pos_y - other.pos_y)

    def __getitem__(self, item: int) -> int:
        return {0: self.pos_x, 1: self.pos_y}[item]

    def __lt__(self, other: Site) -> bool:
        return (self.pos_y, self.pos_x) < (other.pos_y, other.pos_x)  # tuple comparison


@total_ordering
@dataclass
class UnitVector:
    pos_x: Real
    pos_y: Real

    def __array__(self) -> np.ndarray:
        return np.array(astuple(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.pos_x}, {self.pos_y})"

    def __hash__(self) -> int:
        return hash((self.pos_x, self.pos_y))

    def __iter__(self) -> Iterator[Real]:
        return (pos for pos in (self.pos_x, self.pos_y))

    def __mul__(self, scalar: Real) -> UnitVector:
        return UnitVector(self.pos_x * scalar, self.pos_y * scalar)

    def __rmul__(self, scalar: Real) -> UnitVector:  # type: ignore[misc]
        # TODO: somehow mypy complains this, dunno why
        # >>> component.py:58: error: Forward operator "__mul__" is not callable  [misc]
        return UnitVector(scalar * self.pos_x, scalar * self.pos_y)

    def __imul__(self, scalar: Real) -> UnitVector:
        return scalar * self

    def __getitem__(self, item: int) -> Real:
        return {0: self.pos_x, 1: self.pos_y}[item]

    def __lt__(self, other: UnitVector) -> bool:
        return (self.pos_y, self.pos_x) < (other.pos_y, other.pos_x)  # tuple comparison

    def __abs__(self) -> Real:
        return np.linalg.norm(self)

    @property
    def sign(self) -> int:
        if all(map(bool, astuple(self))):
            raise ValueError("UnitVector has sign only when aligning with x or y axis.")
        return -1 if np.any(np.array(self) < 0) else 1


@dataclass(frozen=True)
class __UnitVectorCollection:
    upward: UnitVector = field(default_factory=lambda: UnitVector(0, 1))
    downward: UnitVector = field(default_factory=lambda: -1 * UnitVector(0, 1))
    rightward: UnitVector = field(default_factory=lambda: UnitVector(1, 0))
    leftward: UnitVector = field(default_factory=lambda: -1 * UnitVector(1, 0))

    def __iter__(self) -> Iterator[UnitVector]:
        return iter(sorted((self.rightward, self.upward)))

    def iter_all_directions(self) -> Iterator[UnitVector]:
        return iter(sorted((self.downward, self.leftward, self.rightward, self.upward)))


UnitVectors = __UnitVectorCollection()
