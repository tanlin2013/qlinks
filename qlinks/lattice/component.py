from __future__ import annotations

from dataclasses import astuple, dataclass, field
from functools import total_ordering
from typing import Iterator

import numpy as np


@total_ordering
@dataclass
class Site:
    coord_x: int
    coord_y: int

    def __post_init__(self):
        if (self.coord_x < 0) or (self.coord_y < 0):
            raise ValueError("Coordinate starts from (0, 0).")

    def __array__(self) -> np.ndarray:
        return np.array(astuple(self))

    def __hash__(self) -> int:
        return hash((self.coord_x, self.coord_y))

    def __add__(self, other: Site | UnitVector) -> Site:
        return Site(*(np.array(self) + np.array(other)))

    def __getitem__(self, item: int) -> int:
        return {0: self.coord_x, 1: self.coord_y}[item]

    def __lt__(self, other: Site) -> bool:
        return (self.coord_y, self.coord_x) < (other.coord_y, other.coord_x)  # tuple comparison


@total_ordering
@dataclass
class UnitVector:
    pos_x: int
    pos_y: int

    def __array__(self) -> np.ndarray:
        return np.array(astuple(self))

    def __hash__(self) -> int:
        return hash((self.pos_x, self.pos_y))

    def __rmul__(self, scalar: int | float) -> UnitVector:
        return UnitVector(scalar * self.pos_x, scalar * self.pos_y)

    def __imul__(self, scalar: int | float) -> UnitVector:
        return scalar * self

    def __lt__(self, other: UnitVector) -> bool:
        return (self.pos_y, self.pos_x) < (other.pos_y, other.pos_x)  # tuple comparison

    @property
    def sign(self) -> int:
        return -1 if np.any(np.array(self) < 0) else 1


@dataclass(frozen=True)
class UnitVectorCollection:
    upward: UnitVector = field(default_factory=lambda: UnitVector(0, 1))
    downward: UnitVector = field(default_factory=lambda: -1 * UnitVector(0, 1))
    rightward: UnitVector = field(default_factory=lambda: UnitVector(1, 0))
    leftward: UnitVector = field(default_factory=lambda: -1 * UnitVector(1, 0))

    def __iter__(self) -> Iterator[UnitVector]:
        return iter((self.rightward, self.upward))
