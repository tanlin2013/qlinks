from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import expm

from qlinks.lattice.spin_operators import SpinOperators
from qlinks.model.utils import kron


@dataclass(slots=True)
class Spin1XYModel:
    n: int
    coup_j: float
    coup_h: float
    coup_d: float
    coup_j3: float
    periodic: bool = False
    _kinetic_term: sp.sparray = field(init=False, repr=False)
    _potential_term: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)
    _parity: npt.NDArray = field(init=False, repr=False)

    def __post_init__(self):
        if self.coup_j3 == 0 and self.n < 2:
            raise ValueError("n should be greater than or equal to 2.")
        elif self.coup_j3 != 0 and self.n < 4:
            raise ValueError("n should be greater than or equal to 4.")
        self._hamiltonian = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._parity = np.eye(3**self.n, dtype=int)

        self._build_xy_term()
        self._build_sz_term()
        if self.coup_j3 != 0:
            self._build_h3_term()
        self._potential_term = sp.diags(self._hamiltonian.diagonal()).tocsr()
        self._kinetic_term = self._hamiltonian - self._potential_term
        self._build_parity_operator()

    def _build_xy_term(self):
        sop = SpinOperators(1)
        s_plus, s_minus, idty = sop.s_plus, sop.s_minus, sop.idty
        for site in range(self.n):
            self._hamiltonian += (
                0.5 * self.coup_j * kron([s_plus, s_minus, *[idty] * (self.n - 2)], shift=site)
            )
            self._hamiltonian += (
                0.5 * self.coup_j * kron([s_minus, s_plus, *[idty] * (self.n - 2)], shift=site)
            )
            if not self.periodic and site == self.n - 2:
                break

    def _build_sz_term(self):
        sop = SpinOperators(1)
        s_z, idty = sop.s_z, sop.idty
        for site in range(self.n):
            self._hamiltonian += self.coup_h * kron([s_z, *[idty] * (self.n - 1)], shift=site)
            self._hamiltonian += self.coup_d * kron([s_z @ s_z, *[idty] * (self.n - 1)], shift=site)

    def _build_h3_term(self):
        sop = SpinOperators(1)
        s_plus, s_minus, idty = sop.s_plus, sop.s_minus, sop.idty
        for site in range(self.n):
            self._hamiltonian += 0.5 * self.coup_j3 * kron(
                [s_plus, idty, idty, s_minus, *[idty] * (self.n - 4)], shift=site
            )  # fmt: skip
            self._hamiltonian += 0.5 * self.coup_j3 * kron(
                [s_minus, idty, idty, s_plus, *[idty] * (self.n - 4)], shift=site
            )  # fmt: skip
            if not self.periodic and site == self.n - 4:
                break

    def _build_parity_operator(self):
        sop = SpinOperators(1)
        s_z, idty = sop.s_z, sop.idty
        sz_str = np.real_if_close(expm(1j * np.pi * s_z), tol=1e-14)
        for site in range(0, self.n, 2):
            self._parity @= kron([sz_str, *[idty] * (self.n - 1)], shift=site)

    @property
    def kinetic_term(self) -> sp.sparray:
        return self._kinetic_term

    @property
    def potential_term(self) -> sp.sparray:
        return self._potential_term

    @property
    def hamiltonian(self) -> sp.sparray:
        return self._hamiltonian

    @property
    def parity(self) -> npt.NDArray[np.int64]:
        return self._parity

    @property
    def basis(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {i: j for i, j in enumerate(product([1, 0, -1], repeat=self.n))}, orient="index"
        )

    def vacuum_state(self) -> sp.sparray:
        state = sp.csr_array((3**self.n, 1), dtype=int)
        state[-1, 0] = 1
        return state

    def sn_tower(self, m: int) -> sp.sparray:
        def j_raise_operator():
            sop = SpinOperators(1)
            s_plus, idty = sop.s_plus, sop.idty
            _j_op = sp.csr_array((3**self.n, 3**self.n), dtype=float)
            for site in range(self.n):
                _j_op += (-1) ** site * kron([s_plus @ s_plus, *[idty] * (self.n - 1)], shift=site)
            return 0.5 * _j_op

        if m < 0 or m > self.n:
            raise ValueError("The value of m should be m = 0, 1, ..., n.")
        if m == 0:
            return self.vacuum_state()
        j_op = j_raise_operator()
        state = sp.linalg.matrix_power(j_op, m) @ self.vacuum_state()
        return state / sp.linalg.norm(state)

    def sn_prime_tower(self, m: int) -> sp.sparray:
        if not self.periodic and self.n % 2 != 0:
            raise ValueError("The system should be periodic or have even number of sites.")
        if m < 0 or m > self.n:
            raise ValueError("The value of m should be m = 0, 1, ..., n.")

        return

    def sn_dprime_tower(self, m: int, parity: int = 0) -> sp.sparray:
        sop = SpinOperators(1)
        s_plus, idty = sop.s_plus, sop.idty

        def j_raise_operator(idx):
            _op = sp.csr_array((3**self.n, 3**self.n), dtype=float)
            for site in range(self.n):
                if site < idx:
                    _op += (-1) ** site * kron([s_plus @ s_plus, *[idty] * (self.n - 1)], shift=site)
                else:
                    _op += (-1) ** (site - 1) * kron([s_plus @ s_plus, *[idty] * (self.n - 1)], shift=site)
            return _op

        if not ((self.periodic and self.n % 4 == 0) or (not self.periodic and self.n % 2 == 1)):
            raise ValueError("The system size and boundary conditions can not support this state.")
        if parity not in [0, 1]:
            raise ValueError("Can only define on even or odd sites.")
        if m < 0 or m > self.n:
            raise ValueError("The value of m should be m = 0, 1, ..., n.")

        state = sp.csr_array((3**self.n, 1), dtype=float)
        for site in range(parity, self.n, 2):
            j_op = j_raise_operator(site)
            state += (
                np.exp(0.5j * np.pi * (site + parity))
                * kron([s_plus, *[idty] * (self.n - 1)], shift=site)
                @ sp.linalg.matrix_power(j_op, m)
                @ self.vacuum_state()
            )
        return state / sp.linalg.norm(state)
