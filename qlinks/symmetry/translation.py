from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError, InvalidOperationError
from qlinks.lattice.component import UnitVector
from qlinks.lattice.square_lattice import LocalOperator, SquareLattice


@dataclass
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

    def __hash__(self) -> int:
        return int(
            hashlib.sha256(pd.util.hash_pandas_object(self._df, index=True).values).hexdigest(), 16
        )

    @cached_property
    def periodicity(self) -> pd.Series:
        return self._df.nunique()

    @cached_property
    def representatives(self) -> pd.Series:
        return self._df.min()

    # @cache
    def compatible_representatives(self, momenta: Tuple[int, int]) -> pd.Series:
        momenta = 2 * np.pi * np.array(momenta) / np.array(self.lattice.shape)
        equal_to_min = self._df.eq(self.representatives)
        shift = equal_to_min.apply(lambda x: self._df.index[x].tolist(), result_type="reduce")
        mask = [
            np.linalg.norm(np.sum(np.exp(1j * momenta @ np.array(shift[idx]).T))) > 1e-12
            for idx in shift.index
        ]
        return self.representatives[mask]

    # @cache
    def representative_basis(self, momenta: Tuple[int, int]) -> ComputationBasis:
        basis_idx = np.sort(self.compatible_representatives(momenta).unique())
        return ComputationBasis.from_index(basis_idx, self.lattice.n_links)

    @staticmethod
    def search_sorted(
        repr_idx: npt.NDArray[np.int64], target_idx: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        return np.where(
            np.isin(target_idx, repr_idx),
            np.searchsorted(repr_idx, target_idx),
            -1,
        )

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
            if (idx := np.where(self._df == val)[1]).size > 0:
                matched_col = self._df.iloc[:, np.min(idx)]
                res[val] = tuple(
                    np.subtract(matched_col.index[matched_col == val][0], matched_col.idxmin())
                )
        return pd.Series(res)

    def phase_factor(
        self, kx: int, ky: int, shift: pd.Series
    ) -> npt.NDArray[np.float64 | np.complex128]:
        if not ((kx in range(self.lattice.length_x)) and (ky in range(self.lattice.length_y))):
            raise InvalidArgumentError("The momenta should be in the first Brillouin zone.")
        momenta = 2 * np.pi * np.array([kx / self.lattice.length_x, ky / self.lattice.length_y])
        return np.real_if_close(np.exp(1j * momenta @ np.array(shift.tolist()).T), tol=1e-12)

    def normalization_factor(
        self, repr_idx: npt.NDArray[np.int64], row_idx: List[int], col_idx: List[int]
    ) -> npt.NDArray[np.float64]:
        period = np.array(
            [
                self.periodicity.iloc[np.min(idx)]
                for val in repr_idx
                if (idx := np.where(self._df == val)[1]).size > 0
            ]
        )
        return np.sqrt(period[row_idx] / period[col_idx])

    def __getitem__(
        self, item: Tuple[LocalOperator, Tuple[int, int]]
    ) -> sp.sparray[np.float64 | np.complex128]:
        """

        Args:
            item: A tuple of a local operator and momenta:
                - local operator: A local operator that acts on the representative basis.
                - momenta: A tuple of momenta (kx, ky).

        Returns:

        Examples:
            >>> lattice = SquareLattice(2, 2)
            >>> ts = Translation(lattice, basis)
            >>> operator = LocalOperator(lattice, Site(0, 0))
            >>> momentum_basis_mat = ts[operator, (0, 0)]
        """
        flipper, momenta = item
        basis = self.representative_basis(momenta)
        flipped_states = flipper @ basis
        flippable = flipper.flippable(basis)
        flipped_reprs = self.get_representatives(flipped_states)
        if not np.isin(flipped_reprs, self.representatives).all():
            raise InvalidOperationError("Basis is not closure under the plaquette operator.")
        row_idx = np.arange(basis.n_states)[flippable]
        col_idx = self.search_sorted(basis.index, flipped_reprs)[flippable]
        data = (
            self.normalization_factor(basis.index, row_idx, col_idx)
            * self.phase_factor(*momenta, shift=self.shift(flipped_states))[flippable]
        )
        return sp.csr_array(
            (
                data[col_idx >= 0],
                (row_idx[col_idx >= 0], col_idx[col_idx >= 0]),
            ),
            shape=(basis.n_states, basis.n_states),
        )
