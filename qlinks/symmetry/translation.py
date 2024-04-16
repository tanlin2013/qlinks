# type: ignore
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import UnitVector
from qlinks.lattice.square_lattice import LocalOperator, SquareLattice


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

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        for kx, ky in product(range(self.lattice.length_x), range(self.lattice.length_y)):
            yield kx, ky

    @property
    def periodicity(self) -> pd.Series:
        return self._df.nunique()

    @property
    def representatives(self) -> pd.Series:
        return self._df.min()

    @property
    def representative_basis(self) -> ComputationBasis:
        basis_idx = np.sort(self.representatives.unique())
        return ComputationBasis.from_index(basis_idx, self.lattice.n_links)

    def shift(self, target_basis: pd.Series) -> pd.Series:
        return pd.Series(
            {
                col: idx[0]
                for col, target_value in target_basis.items()
                if (idx := self._df.index[self._df[col] == target_value].tolist())
            }
        )

    def phase_factor(
        self, kx: int, ky: int, shift: pd.Series
    ) -> npt.NDArray[np.float64 | np.complex128]:
        if not (
            (-self.lattice.length_x // 2 + 1 <= kx <= self.lattice.length_x // 2)
            or (-self.lattice.length_y // 2 + 1 <= ky <= self.lattice.length_y // 2)
        ):
            raise InvalidArgumentError("The momentum should be in the first Brillouin zone.")
        momentum = np.array([kx / self.lattice.length_x, ky / self.lattice.length_y])
        phase = momentum @ np.array(shift.tolist()).T
        phase_fac = np.real_if_close(np.exp(1j * 2 * np.pi * phase), tol=1e-12)
        return np.repeat(phase_fac[None, :], len(phase_fac), axis=0)

    def normalization_factor(self) -> npt.NDArray[np.float64]:
        period = self.periodicity[self.representatives.unique()].to_numpy()
        return np.sqrt(np.outer(period, 1 / period))

    def __getitem__(
        self, item: Tuple[LocalOperator, Tuple[int, int]]
    ) -> npt.NDArray[np.float64 | np.complex128]:
        """

        Args:
            item: A tuple of a local operator and momentum:
                - local operator: A local operator that acts on the representative basis.
                - momentum: A tuple of momentum (kx, ky).

        Returns:

        Examples:
            >>> lattice = SquareLattice(2, 2)
            >>> ts = Translation(lattice, basis)
            >>> operator = LocalOperator(lattice, Site(0, 0))
            >>> momentum_basis_mat = ts[operator, (0, 0)]
        """
        flipper, (kx, ky) = item
        basis = self.representative_basis
        flipped_states = flipper @ basis
        flippable = flipper.flippable(basis)
        if len(flipped_states) != basis.n_states:
            raise InvalidArgumentError("The number of basis is not preserved.")
        if not np.all(np.isin(flipped_states, self.basis.index)):
            raise InvalidArgumentError("Basis is not closure under the applied operator.")
        col_idx = rankdata(self.representatives.loc[flipped_states], method="dense") - 1
        col_idx = col_idx[flippable[col_idx]]
        row_idx = np.arange(basis.n_states)[flippable[col_idx]]
        shift = self.shift(pd.Series(flipped_states, index=basis.index))
        if shift.empty:
            return np.zeros((basis.n_states, basis.n_states))
        return (
            self.normalization_factor()
            * self.phase_factor(kx, ky, shift)
            * sp.csr_array(
                (np.ones(len(row_idx), dtype=int), (row_idx, col_idx)),
                shape=(basis.n_states, basis.n_states),
            )
        ).toarray()
