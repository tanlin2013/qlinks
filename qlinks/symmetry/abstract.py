import abc
from dataclasses import dataclass
from typing import TypeAlias, Tuple

import numpy as np

from qlinks.lattice.square_lattice import LatticeState

Real: TypeAlias = int | float | np.floating


@dataclass
class AbstractSymmetry(abc.ABC):
    quantum_numbers: Real | Tuple[Real, ...]

    @abc.abstractmethod
    def symmetry_operation(self, state: LatticeState) -> LatticeState:
        ...
