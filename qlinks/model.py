from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlinks.spin_object import SpinOperator
from qlinks.square_lattice import SquareLattice


@dataclass
class QuantumLinkModel(SquareLattice):
    coup_j: float
    coup_rk: float
    _hamiltonian: SpinOperator = field(init=False, repr=False)

    def __post_init__(self):
        self._hamiltonian = SpinOperator(np.zeros(self.hilbert_dims))
        for plaquette in self.iter_plaquettes():
            flipper = plaquette + plaquette.conj()
            self._hamiltonian += -self.coup_j * flipper + self.coup_rk * flipper**2

    @property
    def hamiltonian(self) -> SpinOperator:
        return self._hamiltonian

    def gen_config(self):
        pass
