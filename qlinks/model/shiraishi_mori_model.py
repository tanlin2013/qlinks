from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.lattice.spin_operators import SpinOperators
from qlinks.model.utils import kron, sparse_real_if_close


@dataclass(slots=True)
class SpinHalfChain:
    ...


@dataclass(slots=True)
class SpinOneChain:
    n: int
    coup_j1s: npt.NDArray
    coup_h1s: npt.NDArray
    coup_j2s: npt.NDArray
    coup_h2s: npt.NDArray
    coup_d: float
    _kinetic_term: sp.sparray = field(init=False, repr=False)
    _potential_term: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)
    _sm_projector: npt.NDArray = field(init=False, repr=False)

    def __post_init__(self):
        self._hamiltonian = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._sm_projector = np.eye(3**self.n, dtype=int)
        self._build_php_term()
        self._build_h0_term()
        self._potential_term = sp.diags(self._hamiltonian.diagonal())
        self._kinetic_term = self._hamiltonian - self._potential_term
        self._build_sm_projector()

    def _build_php_term(self):
        sop = SpinOperators(1)
        s_x, s_y, s_z, idty = sop.s_x, sop.s_y, sop.s_z, sop.idty
        proj = idty - s_z @ s_z
        for site in range(self.n):
            for i, op in enumerate([s_x, s_y, s_z]):
                self._hamiltonian += self.coup_j1s[i] * kron(
                    [op, proj, op, *[idty] * (self.n - 3)], shift=site
                )
                self._hamiltonian += -self.coup_h1s[i] * kron(
                    [op, proj, idty, *[idty] * (self.n - 3)], shift=site
                )
                self._hamiltonian += -self.coup_h1s[i] * kron(
                    [idty, proj, op, *[idty] * (self.n - 3)], shift=site
                )
        self._hamiltonian += self.n * self.coup_d * sp.eye(3**self.n)

    def _build_h0_term(self):
        sop = SpinOperators(0.5)
        p_x, p_y, p_z, idty = (
            self._insert_zeros(sop.s_x),
            self._insert_zeros(sop.s_y),
            self._insert_zeros(sop.s_z),
            self._insert_zeros(sop.idty),
        )
        for site in range(self.n):
            for i, op in enumerate([p_x, p_y, p_z]):
                self._hamiltonian += self.coup_j2s[i] * kron(
                    [op, op, *[idty] * (self.n - 2)], shift=site
                )
                self._hamiltonian += -self.coup_h2s[i] * kron(
                    [op, idty, *[idty] * (self.n - 2)], shift=site
                )
                self._hamiltonian += -self.coup_h2s[i] * kron(
                    [idty, op, *[idty] * (self.n - 2)], shift=site
                )

    def _build_sm_projector(self):
        sop = SpinOperators(1)
        s_z, idty = sop.s_z, sop.idty
        proj = idty - s_z @ s_z
        for site in range(self.n):
            self._sm_projector @= kron([proj, *[idty] * (self.n - 1)], shift=site)

    @staticmethod
    def _insert_zeros(op):
        op = np.insert(op, 1, 0, axis=0)
        op = np.insert(op, 1, 0, axis=1)
        return op

    @property
    def hamiltonian(self):
        return sparse_real_if_close(self._hamiltonian)

    @property
    def kinetic_term(self):
        return sparse_real_if_close(self._kinetic_term)

    @property
    def potential_term(self):
        return sparse_real_if_close(self._potential_term)

    @property
    def sm_projector(self):
        return self._sm_projector
