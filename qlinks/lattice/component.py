from __future__ import annotations

from dataclasses import astuple, dataclass, field
from functools import total_ordering
from typing import Iterator, Self, TypeAlias

import numpy as np

Real: TypeAlias = int | float | np.floating


@total_ordering
@dataclass(slots=True)
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
        assert isinstance(other.pos_x, int) and isinstance(other.pos_y, int)
        return Site(self.pos_x + other.pos_x, self.pos_y + other.pos_y)

    def __sub__(self, other: Site) -> UnitVector:
        return UnitVector(self.pos_x - other.pos_x, self.pos_y - other.pos_y)

    def __getitem__(self, item: int) -> int:
        return {0: self.pos_x, 1: self.pos_y}[item]

    def __lt__(self, other: Site) -> bool:
        return (self.pos_y, self.pos_x) < (other.pos_y, other.pos_x)  # tuple comparison


@total_ordering
@dataclass(slots=True)
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
        # mypy >>> component.py:65: error: Forward operator "__mul__" is not callable  [misc]
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
        if bool(self.pos_x) and bool(self.pos_y):
            raise ValueError("UnitVector has sign only when aligning with x or y axis.")
        return -1 if any(pos < 0 for pos in (self.pos_x, self.pos_y)) else 1


@dataclass(frozen=True)
class UnitVectors:
    upward: UnitVector = field(default=UnitVector(0, 1))
    downward: UnitVector = field(default=-1 * UnitVector(0, 1))
    rightward: UnitVector = field(default=UnitVector(1, 0))
    leftward: UnitVector = field(default=-1 * UnitVector(1, 0))
    _instance: Self = None

    def __new__(cls, *args, **kwargs):  # singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __iter__(self) -> Iterator[UnitVector]:
        return iter(sorted((self.rightward, self.upward)))
