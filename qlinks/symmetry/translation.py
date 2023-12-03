from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import UnitVector
from qlinks.lattice.square_lattice import LocalOperator, SquareLattice
from qlinks.computation_basis import ComputationBasis


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

    def __getitem__(
        self, item: Tuple[LocalOperator, Tuple[int, int]]
    ) -> npt.NDArray[np.float64 | np.complex128]:
        flipper, (kx, ky) = item
        basis = self.representative_basis
        flipped_states = flipper @ basis
        flippable = flipper.flippable(basis)
        if basis.shape != flipped_states.shape:
            raise InvalidArgumentError("The two bases should have the same shape.")
        row_idx = np.arange(basis.n_states)[flippable]
        col_idx = np.argsort(flipped_states)[flippable]
        return (
            sp.csr_array(
                (np.ones(len(row_idx), dtype=int), (row_idx, col_idx)),
                shape=(basis.n_states, basis.n_states),
            )
            * self.normalization_factor()
            * self.phase_factor(kx, ky)
        ).toarray()

    @property
    def periodicity(self) -> pd.Series:
        return self._df.nunique()

    @property
    def representatives(self) -> pd.Series:
        return self._df.min()

    @property
    def representative_basis(self) -> ComputationBasis:
        basis_idx = np.sort(self.representatives.unique())
        return ComputationBasis(self.basis.dataframe.loc[basis_idx].to_numpy())

    def phase_factor(self, kx: int, ky: int) -> npt.NDArray[np.float64 | np.complex128]:
        if (-self.lattice.length_x // 2 + 1 <= kx <= self.lattice.length_x // 2) or (
            -self.lattice.length_y // 2 + 1 <= ky <= self.lattice.length_y // 2
        ):
            raise InvalidArgumentError("The momentum should be in the first Brillouin zone.")
        momentum = 2 * np.array([kx / self.lattice.length_x, ky / self.lattice.length_y])
        shift = ...
        phase = momentum @ shift
        if phase % 2 == 0:
            return np.ones(shift.shape)
        else:
            return np.exp(1j * momentum @ shift)

    def normalization_factor(self) -> npt.NDArray[np.float64]:
        period = self.periodicity[self.representatives.unique()].to_numpy()
        return np.sqrt(np.outer(period, 1 / period))
