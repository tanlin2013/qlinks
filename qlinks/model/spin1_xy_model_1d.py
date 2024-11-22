from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from qlinks.lattice.spin_operators import SpinOperators
from qlinks.model.utils import kron


@dataclass(slots=True)
class Spin1XYModel:
    n: int
    coup_j: float
    coup_h: float
    coup_d: float
    periodic: bool = False
    _kinetic_term: sp.sparray = field(init=False, repr=False)
    _potential_term: sp.sparray = field(init=False, repr=False)
    _potential_term2: sp.sparray = field(init=False, repr=False)
    _hamiltonian: sp.sparray = field(init=False, repr=False)
    _parity: npt.NDArray = field(init=False, repr=False)

    def __post_init__(self):
        sop = SpinOperators(1)
        s_plus, s_minus, s_z = sop.s_plus, sop.s_minus, sop.s_z
        sz_str = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=int)

        self._kinetic_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term2 = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._parity = np.eye(3**self.n, dtype=int)

        for site in range(self.n):
            self._kinetic_term += 0.5 * kron(
                [s_plus, s_minus, *[np.eye(3)] * (self.n - 2)], shift=site
            )
            self._kinetic_term += 0.5 * kron(
                [s_minus, s_plus, *[np.eye(3)] * (self.n - 2)], shift=site
            )
            if not self.periodic and site == self.n - 2:
                break

        for site in range(self.n):
            self._potential_term += reduce(
                sp.kron, [sp.eye(3**site), s_z, sp.eye(3 ** (self.n - 1 - site))]
            )
            self._potential_term2 += reduce(
                sp.kron, [sp.eye(3**site), s_z**2, sp.eye(3 ** (self.n - 1 - site))]
            )

        for site in range(0, self.n, 2):
            self._parity @= reduce(
                np.kron,
                [np.eye(3**site, dtype=int), sz_str, np.eye(3 ** (self.n - 1 - site), dtype=int)],
            )

        self._hamiltonian = (
            self.coup_j * self._kinetic_term
            + self.coup_h * self._potential_term
            + self.coup_d * self._potential_term2
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
