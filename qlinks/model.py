from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Self, Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.lattice.square_lattice import SquareLattice
from qlinks.symmetry.computation_basis import ComputationBasis


@dataclass(slots=True)
class QuantumLinkModel:
    """

    Args:
        coup_j: The coupling strength of the plaquette flipping kinetic term.
        coup_rk: The Rokhsar-Kivelson coupling for counting the number of flippable plaquettes.
        shape: The shape of the lattice.
        basis: Computation basis that respects the gauss law and other lattice symmetries.
    """

    coup_j: float
    coup_rk: float
    shape: Tuple[int, ...]
    basis: ComputationBasis = field(repr=False)
    _lattice: SquareLattice = field(init=False, repr=False)
    _hamiltonian: npt.NDArray[float] | sp.spmatrix[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lattice = SquareLattice(*self.shape)
        self._hamiltonian = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        for plaquette in self._lattice.iter_plaquettes():
            self._hamiltonian += (
                -self.coup_j * plaquette[self.basis] + self.coup_rk * (plaquette**2)[self.basis]
            )

    @property
    def kinetic_term(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return self.hamiltonian - self.potential_term

    @property
    def potential_term(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return np.diagflat(self._hamiltonian.diagonal())

    @property
    def hamiltonian(self) -> npt.NDArray[float] | sp.spmatrix[float]:
        return self._hamiltonian.toarray()

    @classmethod
    def from_whole_basis(cls, coup_j: float, coup_rk: float, shape: Tuple[int, int]) -> Self:
        if 2 * np.prod(shape) > 14:
            raise RuntimeError("The system size is too large for whole basis.")
        basis = ComputationBasis(np.asarray(list(product([0, 1], repeat=2 * np.prod(shape)))))
        return cls(coup_j, coup_rk, shape, basis)
