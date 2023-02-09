import abc
from dataclasses import dataclass
from typing import TypeAlias, TypeVar, Tuple

import numpy as np

Real: TypeAlias = int | float | np.floating
AnySymmetry = TypeVar("AnySymmetry", bound="AbstractSymmetry")


@dataclass
class AbstractSymmetry(abc.ABC):
    quantum_numbers: Real | Tuple[Real, ...]

    @abc.abstractmethod
    def symmetry_operation(self, state) -> "state":
        ...
