from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.lattice.spin_operators import SpinOperators
from qlinks.model.utils import kron


@dataclass(slots=True)
class WesleiModel:
    """
    References: https://link.aps.org/doi/10.1103/PhysRevResearch.1.033144
    """
    n: int
    coup_alphas: npt.NDArray
    coup_beta: float
    coup_h1: float
    coup_h2: float
    periodic: bool = False
    _kinetic_term: sp.sparray = field(init=False, repr=False)
    _potential_term: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)

    def __post_init__(self):
        self._hamiltonian = sp.csr_array((2**self.n, 2**self.n), dtype=float)
        self._build_non_local_term()
        self._build_local_term()
        self._potential_term = sp.diags(self._hamiltonian.diagonal()).tocsr()
        self._kinetic_term = self._hamiltonian - self._potential_term

    def _build_non_local_term(self):
        sop = SpinOperators(0.5)
        s_x, s_z, idty = 2 * sop.s_x, 2 * sop.s_z, sop.idty
        exp_x = np.cosh(self.coup_beta) * idty - np.sinh(self.coup_beta) * s_x
        for site in range(self.n):
            self._hamiltonian += self.coup_alphas[site] * kron(
                [exp_x, exp_x, *[idty] * (self.n - 2)], shift=site
            )
            self._hamiltonian += -self.coup_alphas[site] * kron(
                [s_z, s_z, *[idty] * (self.n - 2)], shift=site
            )
            if not self.periodic and site == self.n - 2:
                break

    def _build_local_term(self):
        sop = SpinOperators(0.5)
        s_x, s_z, idty = 2 * sop.s_x, 2 * sop.s_z, sop.idty
        for site in range(self.n):
            self._hamiltonian += self.coup_h1 * kron([s_x, *[idty] * (self.n - 1)], shift=site)
            self._hamiltonian += self.coup_h2 * kron([s_z, *[idty] * (self.n - 1)], shift=site)

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def kinetic_term(self):
        return self._kinetic_term

    @property
    def potential_term(self):
        return self._potential_term
