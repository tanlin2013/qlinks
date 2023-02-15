import abc
from typing import Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

Real: TypeAlias = int | float | np.floating


class AbstractSymmetry(abc.ABC):
    @property
    @abc.abstractmethod
    def quantum_numbers(self) -> Real | Tuple[Real, ...] | NDArray:
        ...
