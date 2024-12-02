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
    _potential_term2: sp.sparray = field(init=False, repr=False)
    _h3_term: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)
    _parity: npt.NDArray = field(init=False, repr=False)

    def __post_init__(self):
        self._kinetic_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term2 = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._h3_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._parity = np.eye(3**self.n, dtype=int)

        self._build_kinetic_term()
        self._build_potential_term()
        self._build_h3_term()
        self._build_parity_operator()
        self._hamiltonian = (
            self.coup_j * self._kinetic_term
            + self.coup_h * self._potential_term
            + self.coup_d * self._potential_term2
            + self.coup_j3 * self._h3_term
        )

    def _build_kinetic_term(self):
        sop = SpinOperators(1)
        s_plus, s_minus, idty = sop.s_plus, sop.s_minus, sop.idty
        for site in range(self.n):
            self._kinetic_term += 0.5 * kron([s_plus, s_minus, *[idty] * (self.n - 2)], shift=site)
            self._kinetic_term += 0.5 * kron([s_minus, s_plus, *[idty] * (self.n - 2)], shift=site)
            if not self.periodic and site == self.n - 2:
                break

    def _build_potential_term(self):
        sop = SpinOperators(1)
        s_z, idty = sop.s_z, sop.idty
        for site in range(self.n):
            self._potential_term += kron([s_z, *[idty] * (self.n - 1)], shift=site)
            self._potential_term2 += kron([s_z**2, *[idty] * (self.n - 1)], shift=site)

    def _build_h3_term(self):
        sop = SpinOperators(1)
        s_x, s_y, idty = sop.s_x, sop.s_y, sop.idty
        for site in range(self.n):
            self._h3_term += kron(
                [s_x, idty, idty, s_x, *[idty] * (self.n - 4)], shift=site
            )
            self._h3_term += kron(
                [s_y, idty, idty, s_y, *[idty] * (self.n - 4)], shift=site
            )
            if not self.periodic and site == self.n - 4:
                break

    def _build_parity_operator(self):
        sop = SpinOperators(1)
        s_z, idty = sop.s_z, sop.idty
        sz_str = np.real_if_close(expm(1j * np.pi * s_z), tol=1e-14)
        for site in range(0, self.n, 2):
            self._parity @= kron(
                [sz_str, *[idty] * (self.n - 1)], shift=site
            )

    @property
    def kinetic_term(self) -> sp.sparray:
        return self._kinetic_term

    @property
    def potential_term(self) -> sp.sparray:
        return self._potential_term

    @property
    def potential_term2(self) -> sp.sparray:
        return self._potential_term2

    @property
    def h3_term(self) -> sp.sparray:
        return self._h3_term

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
