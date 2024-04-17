from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Tuple, Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError, InvalidOperationError
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

    def sort_to_representative(self, target_basis: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        repr_idx = self.representative_basis.index
        insert_pos = np.searchsorted(repr_idx, target_basis)
        sorting_key = np.full_like(target_basis, -1)
        mask = (insert_pos < len(repr_idx)) & (repr_idx[insert_pos] == target_basis)
        sorting_key[mask] = insert_pos[mask]
        return sorting_key

    def get_representatives(self, target_basis: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.array(
            [
                self.representatives.iloc[np.min(idx)]
                for val in target_basis
                if (idx := np.where(self._df == val)[1]).size > 0
            ]
        )

    def shift(self, target_basis: npt.NDArray[np.int64]) -> pd.Series:
        """
        Find the shift that maps the target basis into the representative basis.

        Args:
            target_basis:

        Returns:

        """
        res = {}
        for val in target_basis:
            if (idx := np.where(self._df == val))[1].size > 0:
                matched_col = self._df.iloc[:, np.min(idx[1])]
                res[val] = tuple(
                    np.subtract(matched_col.index[matched_col == val][0], matched_col.idxmin())
                )
        return pd.Series(res)

    def phase_factor(
        self, kx: int, ky: int, shift: pd.Series
    ) -> npt.NDArray[np.float64 | np.complex128]:
        if not ((kx in range(self.lattice.length_x)) or (ky in range(self.lattice.length_y))):
            raise InvalidArgumentError("The momentum should be in the first Brillouin zone.")
        momentum = np.array([kx / self.lattice.length_x, ky / self.lattice.length_y])
        phase = momentum @ np.array(shift.tolist()).T
        phase_fac = np.real_if_close(np.exp(1j * 2 * np.pi * phase), tol=1e-12)
        return np.repeat(phase_fac[None, :], len(phase_fac), axis=0)

    def normalization_factor(self) -> npt.NDArray[np.float64]:
        period = self.periodicity[self.representatives.drop_duplicates().index].to_numpy()
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
        flipped_reprs = self.get_representatives(flipped_states)
        if not np.isin(flipped_reprs, basis.index).all():
            raise InvalidOperationError("Basis is not closure under the applied operator.")
        row_idx = np.arange(basis.n_states)[flippable]
        col_idx = self.sort_to_representative(flipped_reprs)[flippable]
        return (
            self.normalization_factor()
            * self.phase_factor(kx, ky, self.shift(flipped_states))
            * sp.csr_array(
                (np.ones(len(row_idx), dtype=int), (row_idx, col_idx)),
                shape=(basis.n_states, basis.n_states),
            )
        ).toarray()
