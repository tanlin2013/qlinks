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
from qlinks.symmetry.translation import Translation

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
    momenta: Optional[Tuple[int, int]] = None
    _lattice: SquareLattice = field(init=False, repr=False)
    _translation: Translation = field(init=False, repr=False)
    _kinetic_term: sp.sparray[np.float64 | np.complex128] = field(init=False, repr=False)
    _potential_term: sp.sparray[np.float64 | np.complex128] = field(init=False, repr=False)
    _hamiltonian: sp.sparray[np.float64 | np.complex128] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lattice = SquareLattice(*self.shape)
        self._translation = Translation(self._lattice, self.basis)
        if isinstance(self.coup_j, (int, float)):
            self.coup_j = np.asarray([self.coup_j] * self._lattice.size)
        if isinstance(self.coup_rk, (int, float)):
            self.coup_rk = np.asarray([self.coup_rk] * self._lattice.size)
        if self.momenta is None:
            self._build_hamiltonian()
        else:
            self._build_momentum_hamiltonian()

    def _build_hamiltonian(self) -> None:
        self._kinetic_term = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        self._potential_term = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        self._hamiltonian = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        for i, plaquette in enumerate(self._lattice.iter_plaquettes()):
            flipper, flip_counter = plaquette[self.basis], (plaquette**2)[self.basis]
            self._kinetic_term += flipper
            self._potential_term += flip_counter
            self._hamiltonian += (
                -self.coup_j[i] * flipper + self.coup_rk[i] * flip_counter  # type: ignore[index]
            )

    def _build_momentum_hamiltonian(self) -> None:
        dim = self.translation.compatible_representatives(self.momenta).unique().size
        if not dim > 0:
            raise InvalidArgumentError("The momentum is not compatible with the lattice.")
        self._kinetic_term = sp.csr_array((dim, dim), dtype=complex)
        self._potential_term = sp.csr_array((dim, dim), dtype=complex)
        self._hamiltonian = sp.csr_array((dim, dim), dtype=complex)
        for i, plaquette in enumerate(self._lattice.iter_plaquettes()):
            flipper, flip_counter = (
                self.translation[plaquette, self.momenta],
                self.translation[plaquette**2, self.momenta],
            )
            self._kinetic_term += flipper
            self._potential_term += flip_counter
            self._hamiltonian += (
                -self.coup_j[i] * flipper + self.coup_rk[i] * flip_counter  # type: ignore[index]
            )

    @property
    def kinetic_term(self) -> sp.sparray[np.float64 | np.complex128]:
        return self._kinetic_term

    @property
    def potential_term(self) -> sp.sparray[np.float64 | np.complex128]:
        return self._potential_term

    @property
    def hamiltonian(self) -> sp.sparray[np.float64 | np.complex128]:
        return self._hamiltonian

    @property
    def sparsity(self) -> float:
        return 1 - self._hamiltonian.count_nonzero() / self._hamiltonian.size

    @property
    def translation(self) -> Translation:
        return self._translation

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

        Reference: https://github.com/tanlin2013/qlinks/issues/40
        """
        if not 0 <= idx < self._lattice.shape[axis] - 1:
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
