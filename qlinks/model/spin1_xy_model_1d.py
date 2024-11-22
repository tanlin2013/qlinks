from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sp

from qlinks.lattice.spin_operators import SpinOperators

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

    def __post_init__(self):
        local_hopping = sp.kron(s_up, s_down) + sp.kron(s_down, s_up)
        sop = SpinOperators(1)
        s_plus, s_minus, s_z = sop.s_plus, sop.s_minus, sop.s_z

        self._kinetic_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._potential_term2 = sp.csr_array((3**self.n, 3**self.n), dtype=float)
        self._hamiltonian = sp.csr_array((3**self.n, 3**self.n), dtype=float)

        for site in range(self.n - 1):
            self._kinetic_term += 0.5 * reduce(
                sp.kron, [sp.eye(3**site), local_hopping, sp.eye(3 ** (self.n - 2 - site))]
            )
        if self.periodic:
            self._kinetic_term += 0.5 * reduce(sp.kron, [s_down, sp.eye(3 ** (self.n - 2)), s_up])
            self._kinetic_term += 0.5 * reduce(sp.kron, [s_up, sp.eye(3 ** (self.n - 2)), s_down])

        for site in range(self.n):
            self._potential_term += reduce(
                sp.kron, [sp.eye(3**site), s_z, sp.eye(3 ** (self.n - 1 - site))]
            )
            self._potential_term2 += reduce(
                sp.kron, [sp.eye(3**site), s_z**2, sp.eye(3 ** (self.n - 1 - site))]
            )

        self._hamiltonian += (
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
    def basis(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {i: j for i, j in enumerate(product([1, 0, -1], repeat=self.n))}, orient="index"
        )
