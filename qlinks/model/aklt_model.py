from __future__ import annotations

from dataclasses import dataclass, field

import scipy.sparse as sp

from qlinks.lattice.spin_operators import SpinOperators
from qlinks.model.utils import kron


@dataclass(slots=True)
class AKLTModel:
    n: int
    periodic: bool = False
    _kinetic_term: sp.sparray = field(init=False, repr=False)
    _potential_term: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)

    def __post_init__(self):
        sop = SpinOperators(1)
        s_plus, s_minus, s_z = sop.s_plus, sop.s_minus, sop.s_z
        self._hamiltonian = 1 / 3 * self.n * sp.eye(3**self.n)

        for site in range(self.n):
            two_site_spin_dot = 0.5 * kron(
                [s_plus, s_minus, *[sp.eye(3)] * (self.n - 2)], shift=site
            )
            two_site_spin_dot += 0.5 * kron(
                [s_minus, s_plus, *[sp.eye(3)] * (self.n - 2)], shift=site
            )
            two_site_spin_dot += kron([s_z, s_z, *[sp.eye(3)] * (self.n - 2)], shift=site)
            self._hamiltonian += (
                0.5 * two_site_spin_dot + 1 / 6 * two_site_spin_dot @ two_site_spin_dot
            )
            if not self.periodic and site == self.n - 2:
                break

        self._potential_term = sp.diags(self.hamiltonian.diagonal())
        self._kinetic_term = self._hamiltonian - self._potential_term

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def kinetic_term(self):
        return self._kinetic_term

    @property
    def potential_term(self):
        return self._potential_term
