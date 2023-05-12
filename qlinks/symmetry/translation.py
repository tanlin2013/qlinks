from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, TypeAlias

import numpy as np

from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import UnitVector
from qlinks.lattice.square_lattice import LatticeState
from qlinks.symmetry.abstract import AbstractSymmetry

Real: TypeAlias = int | float | np.floating


@dataclass(slots=True)
class Translation(AbstractSymmetry):
    shift: UnitVector

    def __post__init(self):
        if bool(self.shift[0]) ^ bool(self.shift[0]) != 1:  # TODO: should allow more than one shift
            raise InvalidArgumentError("Translation can only be performed on x or y directions.")

    @property
    def quantum_numbers(self) -> Tuple[Real, ...]:
        return self.shift.pos_x, self.shift.pos_y

    def __matmul__(self, other: LatticeState) -> LatticeState:
        if not isinstance(other, LatticeState):
            return NotImplemented
        new_link_data = deepcopy(other.links)
        for link in new_link_data.values():
            link.site = other[link.site + self.shift]
        return LatticeState(
            *other.shape, link_data={link.index: link for link in new_link_data.values()}
        )
