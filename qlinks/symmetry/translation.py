from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Tuple, Self
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd

from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import UnitVector
from qlinks.lattice.square_lattice import SquareLattice
from qlinks.symmetry.computation_basis import ComputationBasis


@dataclass(slots=True)
class Translation:
    lattice: SquareLattice
    basis: ComputationBasis = field(repr=False)
    _df: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        self._df = pd.DataFrame.from_dict(
            {
                shift: (self >> UnitVector(*shift)).index
                for shift in product(range(self.lattice.length_x), range(self.lattice.length_y))
            },
            orient="index",
            columns=self.basis.index,
        )

    def __rshift__(self, shift: UnitVector) -> ComputationBasis:
        shift_idx = np.array(
            [[index := self.lattice.site_index(site + shift), index + 1] for site in self.lattice]
        ).flatten()
        return ComputationBasis(deepcopy(self.basis.links)[:, shift_idx])

    def __lshift__(self, shift: UnitVector) -> ComputationBasis:
        return self.__rshift__(-1 * shift)

    def __matmul__(self, basis: ComputationBasis) -> ComputationBasis:
        ...

    def periodicity(self) -> pd.Series:
        return self._df.nunique()

    def representative_basis(self) -> pd.Series:
        return self._df.min()

    def momentum(self) -> pd.Series:
        ...
