from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, TypeAlias

import numpy as np

from qlinks.lattice.spin_object import SpinOperator
from qlinks.lattice.square_lattice import LatticeMultiStates, SquareLattice

Real: TypeAlias = int | float | np.floating


@dataclass
class QuantumLinkModel(SquareLattice):
    coup_j: float
    coup_rk: float
    basis: Optional[LatticeMultiStates] = field(default=None)
    _hamiltonian: SpinOperator = field(init=False, repr=False)

    def __post_init__(self):
        if self.basis is None:
            assert self.num_links <= 14
            self._build_full_hamiltonian()
        else:
            self._build_symmetry_sector_hamiltonian()

    def _build_full_hamiltonian(self) -> None:
        self._hamiltonian = SpinOperator(np.zeros(self.hilbert_dims))
        for plaquette in self.iter_plaquettes():
            flipper = plaquette + plaquette.conj()
            self._hamiltonian += (  # type: ignore[misc]
                -self.coup_j * flipper + self.coup_rk * flipper**2
            )

    def _build_symmetry_sector_hamiltonian(self) -> None:
        self._hamiltonian = SpinOperator(np.zeros(self.basis.hilbert_dims))
        for plaquette in self.iter_plaquettes():
            self._hamiltonian += -self.coup_j * (  # type: ignore[misc, operator]
                self.basis.T @ plaquette @ self.basis
            )
            self._hamiltonian += -self.coup_j * (  # type: ignore[misc, operator]
                self.basis.T @ plaquette.conj() @ self.basis
            )
            self._hamiltonian += self.coup_rk * (  # type: ignore[misc, operator]
                self.basis.T @ (plaquette.conj() * plaquette) @ self.basis
            )
            self._hamiltonian += self.coup_rk * (  # type: ignore[misc, operator]
                self.basis.T @ (plaquette * plaquette.conj()) @ self.basis
            )

    @cached_property
    def hamiltonian(self) -> SpinOperator:
        return self._hamiltonian
