from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from itertools import product
from typing import Optional, Self, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.stats import rankdata

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.square_lattice import SquareLattice

Real: TypeAlias = np.int64 | np.float64


@dataclass(slots=True)
class QuantumLinkModel:
    """

    Args:
        coup_j: The coupling strength of the plaquette flipping kinetic term.
        coup_rk: The Rokhsar-Kivelson coupling for counting the number of flippable plaquettes.
        shape: The shape of the lattice.
        basis: Computation basis that respects the gauss law and other lattice symmetries.
    """

    coup_j: Real | npt.NDArray[Real]
    coup_rk: Real | npt.NDArray[Real]
    shape: Tuple[int, int]
    basis: ComputationBasis = field(repr=False)
    _lattice: SquareLattice = field(init=False, repr=False)
    _hamiltonian: sp.spmatrix[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lattice = SquareLattice(*self.shape)
        if isinstance(self.coup_j, (int, float)):
            self.coup_j = np.asarray([self.coup_j] * self._lattice.size)
        if isinstance(self.coup_rk, (int, float)):
            self.coup_rk = np.asarray([self.coup_rk] * self._lattice.size)
        self._hamiltonian = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        for i, plaquette in enumerate(self._lattice.iter_plaquettes()):
            self._hamiltonian += (
                -self.coup_j[i] * plaquette[self.basis]  # type: ignore[index]
                + self.coup_rk[i] * (plaquette**2)[self.basis]  # type: ignore[index]
            )

    @property
    def kinetic_term(self) -> npt.NDArray[np.float64] | sp.spmatrix[np.float64]:
        return self.hamiltonian - self.potential_term

    @property
    def potential_term(self) -> npt.NDArray[np.float64] | sp.spmatrix[np.float64]:
        return sp.diags(self._hamiltonian.diagonal()).toarray()

    @property
    def hamiltonian(self) -> npt.NDArray[np.float64] | sp.spmatrix[np.float64]:
        return self._hamiltonian.toarray()

    @property
    def sparsity(self) -> float:
        return 1 - self._hamiltonian.count_nonzero() / self._hamiltonian.size

    @classmethod
    def from_whole_basis(cls, coup_j: float, coup_rk: float, shape: Tuple[int, int]) -> Self:
        if 2 * np.prod(shape) > 14:
            raise RuntimeError("The system size is too large for whole basis.")
        basis = ComputationBasis(np.asarray(list(product([0, 1], repeat=int(2 * np.prod(shape))))))
        return cls(coup_j, coup_rk, shape, basis)  # type: ignore[arg-type]

    @classmethod
    def from_gauge_invariant_basis(cls):
        ...

    @classmethod
    def from_momentum_basis(cls):
        ...

    def __hash__(self) -> int:
        return hash((self.coup_j.tobytes(), self.coup_rk.tobytes(), self.shape, self.basis))

    @cache  # noqa: B019
    def _bipartite_sorting_index(
        self, idx: int, axis: Optional[int] = 0
    ) -> Tuple[npt.NDArray, ...]:
        """Compute the index that sorts the eigenstate in the bipartite basis.

        Args:
            idx: The `idx`-th row or column in lattice.
            axis: 0 for x-axis and 1 for y-axis, default 0.

        Returns: A tuple of three arrays:
            - sorting_idx: Index that sorts the eigenstate in the bipartite basis.
            - row_idx: Row index to reshape the eigenstate into a matrix.
            - col_idx: Column index to reshape the eigenstate into a matrix.

        Raises: InvalidArgumentError if the index is out of range.
        """
        if idx > self._lattice.shape[axis] - 2:
            raise InvalidArgumentError("The index is out of range.")
        first_partition, second_partition = (
            self.basis.as_index(self.basis.links[:, partition_idx])
            for partition_idx in self._lattice.bipartite_index(idx, axis)
        )
        sorting_idx = np.lexsort((first_partition, second_partition))
        row_idx, col_idx = (
            rankdata(partition, method="dense") - 1
            for partition in (first_partition[sorting_idx], second_partition[sorting_idx])
        )
        return sorting_idx, row_idx, col_idx

    def entropy(
        self, evec: npt.NDArray[np.float64], idx: int, axis: Optional[int] = 0
    ) -> np.float64:
        """

        Args:
            evec:
            idx:
            axis:

        Returns:

        Notes: https://github.com/tanlin2013/qlinks/issues/25
        """
        sorting_idx, row_idx, col_idx = self._bipartite_sorting_index(idx, axis)
        reshaped_evec = sp.csr_array(
            (evec[sorting_idx], (row_idx, col_idx)),
            (len(np.unique(row_idx)), len(np.unique(col_idx))),
        )
        try:
            s = sp.linalg.svds(
                reshaped_evec,
                k=min(reshaped_evec.shape) - 1,
                return_singular_vectors=False,
            )
        except TypeError:
            s = svd(reshaped_evec.toarray(), compute_uv=False)
        return -np.sum((ss := s[s > 1e-12] ** 2) * np.log(ss))
