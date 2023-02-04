from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from qlinks.coordinate import Site, UnitVector


class Spin(np.ndarray):
    def __new__(cls, data: ArrayLike | Sequence, read_only: bool = False):
        arr = np.asarray(data).view(cls)
        arr.setflags(write=not read_only)
        return arr

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def __eq__(self, other: Spin) -> bool:
        return np.array_equal(self, other)

    def __xor__(self, other: Spin) -> Spin:
        return np.kron(self, other).view(Spin)

    @property
    def magnetization(self) -> int | float:
        return ((self.T @ SpinOperatorCollection().Sz @ self) / (self.T @ self)).item()


@dataclass(frozen=True)
class SpinConfig:
    up: Spin = field(default_factory=lambda: Spin([1, 0], read_only=True))
    down: Spin = field(default_factory=lambda: Spin([0, 1], read_only=True))


class SpinOperator(np.ndarray):
    def __new__(cls, data: ArrayLike | Sequence, read_only: bool = False, **kwargs):
        arr = np.asarray(data, **kwargs).view(cls)
        arr.setflags(write=not read_only)
        return arr

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def __eq__(self, other: SpinOperator) -> bool:
        return np.array_equal(self, other)

    def __xor__(self, other: SpinOperator) -> SpinOperator:
        return np.kron(self, other).view(SpinOperator)

    def __pow__(self, power: int) -> SpinOperator:
        return np.linalg.matrix_power(self, power).view(SpinOperator)

    @property
    def conj(self) -> SpinOperator:
        return self.T.conj if np.iscomplexobj(self) else self.T


@dataclass(frozen=True)
class SpinOperatorCollection:
    Sp: SpinOperator = field(
        default_factory=lambda: SpinOperator([[0, 1], [0, 0]], dtype=float, read_only=True)
    )
    Sm: SpinOperator = field(
        default_factory=lambda: SpinOperator([[0, 0], [1, 0]], dtype=float, read_only=True)
    )
    Sz: SpinOperator = field(
        default_factory=lambda: SpinOperator([[1, 0], [0, -1]], dtype=float, read_only=True)
    )
    I2: SpinOperator = field(
        default_factory=lambda: SpinOperator(np.identity(2), dtype=float, read_only=True)
    )
    O2: SpinOperator = field(
        default_factory=lambda: SpinOperator(np.zeros((2, 2)), dtype=float, read_only=True)
    )


@total_ordering
@dataclass
class Link:
    site: Site
    unit_vector: UnitVector
    operator: SpinOperator = field(default_factory=lambda: SpinOperatorCollection().I2)
    config: Optional[Spin] = None

    def __post_init__(self):
        if self.unit_vector.sign < 0:
            self.site += self.unit_vector
            self.unit_vector *= -1

    def __hash__(self) -> int:
        return hash((self.site, self.unit_vector, self.operator, self.config))

    def __lt__(self, other: Link) -> bool:
        return (self.site, self.unit_vector) < (other.site, other.unit_vector)  # tuple comparison

    def __xor__(self, other: Link) -> SpinOperator:
        fore_link, post_link = sorted([self, other])
        return fore_link.operator ^ post_link.operator

    def conj(self, inplace: bool = False) -> Link | None:
        conj_link = self if inplace else deepcopy(self)
        conj_link.operator = self.operator.conj
        if not inplace:
            return conj_link

    def reset(self, inplace: bool = False) -> Link | None:
        reset_link = self if inplace else deepcopy(self)
        reset_link.operator = SpinOperatorCollection().I2
        if not inplace:
            return reset_link
