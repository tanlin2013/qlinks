from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from itertools import product
from typing import Optional, Self, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.stats import rankdata

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.square_lattice import SquareLattice

Real: TypeAlias = int | float | np.int64 | np.float64


@dataclass(slots=True)
class QuantumLinkModel:
    """

    Args:
        coup_j: The coupling strength of the plaquette flipping kinetic term.
        coup_rk: The Rokhsar-Kivelson coupling for counting the number of flippable plaquettes.
        shape: The shape of the lattice.
        basis: Computation basis that respects the gauss law and other lattice symmetries.
    """

    coup_j: Real | npt.ArrayLike[Real]
    coup_rk: Real | npt.ArrayLike[Real]
    shape: Tuple[int, ...]
    basis: ComputationBasis = field(repr=False)
    _lattice: SquareLattice = field(init=False, repr=False)
    _hamiltonian: npt.NDArray[float] | sp.spmatrix[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lattice = SquareLattice(*self.shape)
        if isinstance(self.coup_j, (int, float)):
            self.coup_j = np.asarray([self.coup_j] * self._lattice.size)
        if isinstance(self.coup_rk, (int, float)):
            self.coup_rk = np.asarray([self.coup_rk] * self._lattice.size)
        self._hamiltonian = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        for i, plaquette in enumerate(self._lattice.iter_plaquettes()):
            self._hamiltonian += (
                -self.coup_j[i] * plaquette[self.basis]
                + self.coup_rk[i] * (plaquette**2)[self.basis]
            )

    @property
    def kinetic_term(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return self.hamiltonian - self.potential_term

    @property
    def potential_term(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return sp.diags(self._hamiltonian.diagonal()).toarray()

    @property
    def hamiltonian(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return self._hamiltonian.toarray()

    @property
    def sparsity(self) -> float:
        return 1 - self._hamiltonian.count_nonzero() / self._hamiltonian.size

    @classmethod
    def from_whole_basis(cls, coup_j: float, coup_rk: float, shape: Tuple[int, int]) -> Self:
        if 2 * np.prod(shape) > 14:
            raise RuntimeError("The system size is too large for whole basis.")
        basis = ComputationBasis(np.asarray(list(product([0, 1], repeat=2 * np.prod(shape)))))
        return cls(coup_j, coup_rk, shape, basis)

    @classmethod
    def from_gauge_invariant_basis(cls):
        ...

    @classmethod
    def from_momentum_basis(cls):
        ...

    def __hash__(self) -> int:
        return hash((self.coup_j.tobytes(), self.coup_rk.tobytes(), self.shape, self.basis))

    @cache
    def _bipartite_sorting_index(
        self, idx: int, axis: Optional[int] = 0
    ) -> Tuple[npt.NDArray, ...]:
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
        sorting_idx, row_idx, col_idx = self._bipartite_sorting_index(idx, axis)
        reshaped_evec = sp.csr_array(
            (evec[sorting_idx], (row_idx, col_idx)),
            (len(np.unique(row_idx)), len(np.unique(col_idx))),
        )
        s = sp.linalg.svds(
            reshaped_evec,
            k=min(*reshaped_evec.shape) - 1,
            return_singular_vectors=False,
        )
        return -np.sum((ss := s[s > 1e-12] ** 2) * np.log(ss))
