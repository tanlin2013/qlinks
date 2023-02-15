from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Iterator, Optional, Sequence, Tuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from qlinks.exceptions import InvalidArgumentError, InvalidOperationError
from qlinks.lattice.component import Site, UnitVector

Real: TypeAlias = int | float | np.floating
LinkIndex: TypeAlias = Tuple[Site, UnitVector]


class Spin(np.ndarray):
    def __new__(cls, data: ArrayLike | Sequence, read_only: bool = False, **kwargs):
        arr = np.asarray(data, **kwargs).view(cls)
        arr.setflags(write=not read_only)
        return arr

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.data.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spin):
            return NotImplemented
        return np.allclose(self, other, atol=1e-12)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Spin):
            return NotImplemented
        return not np.allclose(self, other, atol=1e-12)

    def __xor__(self, other: Spin) -> Spin:  # type: ignore[override]
        return np.kron(self, other).view(Spin)

    @property
    def magnetization(self) -> Real:
        return ((self.T @ SpinOperators.Sz @ self) / (self.T @ self)).item()


@dataclass(frozen=True, slots=True)
class __SpinConfigCollection:
    up: Spin = field(default_factory=lambda: Spin([[1], [0]], dtype=float, read_only=True))
    down: Spin = field(default_factory=lambda: Spin([[0], [1]], dtype=float, read_only=True))

    def __iter__(self) -> Iterator[Spin]:
        return iter((self.up, self.down))


SpinConfigs = __SpinConfigCollection()


class SpinOperator(np.ndarray):
    def __new__(cls, data: ArrayLike | Sequence, read_only: bool = False, **kwargs):
        arr = np.asarray(data, **kwargs).view(cls)
        arr.setflags(write=not read_only)
        return arr

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.data.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpinOperator):
            return NotImplemented
        return np.allclose(self, other, atol=1e-12)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, SpinOperator):
            return NotImplemented
        return not np.allclose(self, other, atol=1e-12)

    def __xor__(self, other: SpinOperator) -> SpinOperator:  # type: ignore[override]
        return np.kron(self, other).view(SpinOperator)

    def __pow__(self, power: int) -> SpinOperator:  # type: ignore[override]
        return np.linalg.matrix_power(self, power).view(SpinOperator)

    @property
    def conj(self) -> SpinOperator:  # type: ignore[override]
        return self.T.conj if np.iscomplexobj(self) else self.T


@dataclass(frozen=True, slots=True)
class __SpinOperatorCollection:
    Sp: SpinOperator = field(
        default_factory=lambda: SpinOperator([[0, 1], [0, 0]], dtype=float, read_only=True)
    )
    Sm: SpinOperator = field(
        default_factory=lambda: SpinOperator([[0, 0], [1, 0]], dtype=float, read_only=True)
    )
    Sz: SpinOperator = field(
        default_factory=lambda: 0.5 * SpinOperator([[1, 0], [0, -1]], dtype=float, read_only=True)
    )
    I2: SpinOperator = field(
        default_factory=lambda: SpinOperator(np.identity(2), dtype=float, read_only=True)
    )
    O2: SpinOperator = field(
        default_factory=lambda: SpinOperator(np.zeros((2, 2)), dtype=float, read_only=True)
    )


SpinOperators = __SpinOperatorCollection()


@total_ordering
@dataclass
class Link:
    site: Site
    unit_vector: UnitVector
    operator: SpinOperator = field(default_factory=lambda: SpinOperators.I2)
    state: Optional[Spin] = None

    def __post_init__(self):
        if abs(self.unit_vector) != 1:
            raise InvalidArgumentError("Link can only accept UnitVector with length 1.")
        if self.unit_vector.sign < 0:
            self.site += self.unit_vector
            self.unit_vector *= -1

    def __hash__(self) -> int:
        return hash((self.site, self.unit_vector, self.operator, self.state))

    def __lt__(self, other: Link) -> bool:
        if (self.site, self.unit_vector) == (other.site, other.unit_vector) and (
            self.operator != other.operator or self.state != other.state
        ):
            raise InvalidOperationError(
                "Links on same position but with different operators or states are not comparable."
            )
        return (self.site, self.unit_vector) < (other.site, other.unit_vector)  # tuple comparison

    def __xor__(self, other: Link) -> SpinOperator:
        fore_link, post_link = sorted([self, other])
        return fore_link.operator ^ post_link.operator

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.site}, {self.unit_vector}, "
            f"operator={self.operator}, state={self.state})"
        ).replace("\n", "")

    def conj(self, inplace: bool = False) -> Link | None:  # type: ignore[return]
        conj_link = self if inplace else deepcopy(self)
        conj_link.operator = self.operator.conj
        if not inplace:
            return conj_link

    def reset(self, inplace: bool = False) -> Link | None:  # type: ignore[return]
        reset_link = self if inplace else deepcopy(self)
        reset_link.operator = SpinOperators.I2
        reset_link.state = None
        if not inplace:
            return reset_link

    @property
    def index(self) -> LinkIndex:
        return self.site, self.unit_vector

    @property
    def flux(self) -> Real:
        if self.state is None:
            raise ValueError("Link has not been set with any state yet, got None.")
        return self.state.magnetization
