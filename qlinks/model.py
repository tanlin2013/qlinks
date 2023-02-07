from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlinks.lattice.square_lattice import SquareLattice
from qlinks.spin_object import SpinOperator


@dataclass
class QuantumLinkModel(SquareLattice):
    coup_j: float
    coup_rk: float
    __hamiltonian: SpinOperator = field(init=False, repr=False)

    def __post_init__(self):
        self.__hamiltonian = SpinOperator(np.zeros(self.hilbert_dims))
        for plaquette in self.iter_plaquettes():
            flipper = plaquette + plaquette.conj()
            self.__hamiltonian += -self.coup_j * flipper + self.coup_rk * flipper**2

    @property
    def hamiltonian(self) -> SpinOperator:
        return self.__hamiltonian
