from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Dict, Tuple, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.stats import rankdata
from ortools.sat.python import cp_model

from qlinks import logger
from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError, InvalidOperationError


@cache
def fibonacci(n: int) -> int:
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: Dict):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solutions = []

    def on_solution_callback(self) -> None:
        self.solutions.append([self.Value(v) for v in self.__variables.values()])

    @property
    def n_solutions(self) -> int:
        return len(self.solutions)


@dataclass(slots=True)
class CpModel:
    n: int
    periodic: bool = False
    _model: cp_model.CpModel = field(init=False, repr=False)
    _solver: cp_model.CpSolver = field(init=False, repr=False)
    _vars: Dict = field(default_factory=dict, init=False, repr=False)
    _callback: SolutionCallback = field(init=False, repr=False)

    def __post_init__(self):
        self._model = cp_model.CpModel()
        self._solver = cp_model.CpSolver()
        self._vars = {i: self._model.NewBoolVar(f"q_{i}") for i in range(self.n)}
        self._callback = SolutionCallback(self._vars)
        self.add_constraints()

    def add_constraints(self) -> None:
        for i in range(self.n - 1):
            self._model.Add(self._vars[i] + self._vars[i + 1] <= 1)
        if self.periodic:
            self._model.Add(self._vars[0] + self._vars[self.n - 1] <= 1)

    def solve(self, all_solutions: bool = True, log_search_progress: bool = False) -> None:
        self._solver.parameters.enumerate_all_solutions = all_solutions
        self._solver.parameters.log_search_progress = log_search_progress
        status = self._solver.solve(self._model, self._callback)
        logger.info(self._solver.ResponseStats())
        if status == cp_model.OPTIMAL:
            logger.info(f"Found {self._callback.n_solutions} optimal solutions.")
        else:
            raise RuntimeError(f"Failed to find optimal solutions. Status: {status}")
        if self.periodic:
            assert self._callback.n_solutions == fibonacci(self.n - 1) + fibonacci(self.n + 1)

    def to_basis(self) -> ComputationBasis:
        basis = ComputationBasis(np.vstack(self._callback.solutions))
        basis.sort()
        return basis


@dataclass(slots=True)
class PauliX:
    n: int
    site: int
    periodic: bool = False
    _mask: int = field(default=None, repr=False)

    def __post_init__(self):
        self._mask = int(2 ** self.site)

    def flippable(self, basis: ComputationBasis) -> npt.NDArray[np.bool_]:
        if self.periodic:
            link_idx = [(self.site - 1) % self.n, self.site, (self.site + 1) % self.n]
            b1, b2, b3 = (basis.links[:, self.n - 1 - idx].astype(bool) for idx in link_idx)
            return ~(b1 & ~b2) & ~(~b2 & b3)
        else:
            if self.site == 0:
                b1, b2 = (basis.links[:, idx].astype(bool) for idx in [-2, -1])
                return ~(b1 & ~b2)
            elif self.site == self.n - 1:
                b1, b2 = (basis.links[:, idx].astype(bool) for idx in [1, 0])
                return ~(b1 & ~b2)
            else:
                link_idx = [self.site - 1, self.site, self.site + 1]
                b1, b2, b3 = (basis.links[:, self.n - 1 - idx].astype(bool) for idx in link_idx)
                return ~(b1 & ~b2) & ~(~b2 & b3)

    def __matmul__(self, basis: ComputationBasis) -> npt.NDArray[np.int64]:
        if not isinstance(basis, ComputationBasis):
            return NotImplemented
        flipped_states = basis.index.copy()
        flipped_states[self.flippable(basis)] ^= self._mask
        return flipped_states

    def __getitem__(self, basis: ComputationBasis) -> sp.sparray[np.int64]:
        if not isinstance(basis, ComputationBasis):
            return NotImplemented
        flippable = self.flippable(basis)
        flipped_states = self @ basis
        if not np.array_equal(basis.index, np.sort(flipped_states)):
            raise InvalidOperationError("Basis is not closure under the Pauli X operator.")
        row_idx = np.arange(basis.n_states)[flippable]
        col_idx = np.argsort(flipped_states)[flippable]
        return sp.csr_array(
            (np.ones(len(row_idx), dtype=int), (row_idx, col_idx)),
            shape=(basis.n_states, basis.n_states),
        )


@dataclass(slots=True)
class PXPModel1D:
    n: int
    periodic: bool = False
    _basis: ComputationBasis = field(init=False, repr=False)
    _hamiltonian: sp.sparray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        solver = CpModel(self.n, self.periodic)
        solver.solve()
        self._basis = solver.to_basis()
        self._hamiltonian = sp.csr_array((self.basis.n_states, self.basis.n_states), dtype=float)
        for site in range(self.n):
            sx = PauliX(self.n, site, self.periodic)
            self._hamiltonian += sx[self.basis]

    @property
    def basis(self) -> ComputationBasis:
        return self._basis

    @property
    def hamiltonian(self) -> sp.sparray[np.float64]:
        return self._hamiltonian

    def z2_state(self, start_with: int = 0) -> Tuple[int, npt.NDArray[np.int64]]:
        state = {
            0: np.zeros(self.n, dtype=int),
            1: np.ones(self.n, dtype=int),
        }[start_with]
        state[1::2] ^= 1
        z2_index = ComputationBasis.as_index(state[None, :])
        if z2_index not in self.basis.index:
            raise ValueError(f"The Z2 state {state} is not in the basis.")
        return z2_index, state

    def z2_overlap(self, evecs, start_with: int = 0) -> npt.NDArray[np.float64]:
        z2_index, z2_state = self.z2_state(start_with)
        basis_idx = np.where(self.basis.index == z2_index)[0]
        z2_basis = np.zeros(self.basis.n_states, dtype=int)
        z2_basis[basis_idx] = 1
        return np.log(np.abs(z2_basis[None, :] @ evecs) ** 2)

    def _bipartite_sorting_index(self, idx: int) -> Tuple[npt.NDArray, ...]:
        """
        Reference: https://github.com/tanlin2013/qlinks/issues/40
        """
        if not 0 <= idx < self.n - 1:
            raise InvalidArgumentError("The index is out of range.")
        first_partition, second_partition = (
            self.basis.as_index(self.basis.links[:, partition_idx])
            for partition_idx in [np.arange(idx + 1), np.arange(idx + 1, self.n)]
        )
        sorting_idx = np.lexsort((first_partition, second_partition))
        row_idx, col_idx = (
            rankdata(partition, method="dense") - 1
            for partition in (first_partition[sorting_idx], second_partition[sorting_idx])
        )
        return sorting_idx, row_idx, col_idx

    def entropy(self, evec: npt.NDArray[np.float64], site: int) -> float:
        sorting_idx, row_idx, col_idx = self._bipartite_sorting_index(site)
        reshaped_evec = sp.csr_array(
            (evec[sorting_idx], (row_idx, col_idx)),
            (len(np.unique(row_idx)), len(np.unique(col_idx))),
        )
        try:
            s = sp.linalg.svds(
                reshaped_evec,
                k=min(reshaped_evec.shape) - 1,
                return_singular_vectors=False,
            )
        except TypeError:
            s = svd(reshaped_evec.toarray(), compute_uv=False)
        return -np.sum((ss := s[s > 1e-12] ** 2) * np.log(ss))
