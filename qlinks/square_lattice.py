from dataclasses import dataclass, astuple, field
from typing import Tuple, Sequence, Self

import numpy as np


@dataclass(frozen=True)
class Spin:
    up: np.ndarray = field(default_factory=lambda: np.array([1, 0]))
    down: np.ndarray = field(default_factory=lambda: np.array([0, 1]))


@dataclass(frozen=True)
class SpinOperators:
    Sp: np.ndarray = field(
        default_factory=lambda: np.ndarray([[0, 1], [0, 0]], dtype=float)
    )
    Sm: np.ndarray = field(
        default_factory=lambda: np.ndarray([[0, 0], [1, 0]], dtype=float)
    )
    I2: np.ndarray = field(default_factory=lambda: np.identity(2, dtype=float))
    O2: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float))


@dataclass
class Site:
    coor_x: int
    coor_y: int

    def __array__(self):
        return np.array(astuple(self))

    def __add__(self, other: "Site" | Tuple[int, int]) -> Self:
        return Site(*(np.array(self) + np.array(other)))


@dataclass(frozen=True)
class UnitVector:
    upward: Tuple[int, int] = field(default_factory=lambda: (0, 1))
    downward: Tuple[int, int] = field(default_factory=lambda: (0, -1))
    leftward: Tuple[int, int] = field(default_factory=lambda: (-1, 0))
    rightward: Tuple[int, int] = field(default_factory=lambda: (1, 0))


class Link:
    site: Site
    unit_vec: UnitVector
    spin: Spin
    operator: SpinOperators


class Plaquette:
    corner_site: Site
    links: list[Link]
    operator: np.ndarray

    def __post__init__(self):
        self.operator = self.kron_operators()

    @staticmethod
    def kron_operators(operators: Sequence[np.ndarray]) -> np.ndarray:
        opt = operators[0]
        for next_opt in operators[1:]:
            opt = np.kron(opt, next_opt)
        return opt


class SquareLattice:
    def __init__(self):
        pass

    def apply_plaqutte(self):
        pass
