from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Generic, Optional, Tuple, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from qlinks.lattice.square_lattice import LatticeMultiStates
from qlinks.symmetry.gauss_law import GaussLaw, GaugeInvariantSnapshot
from qlinks.symmetry.global_flux import GlobalFlux, FluxSectorSnapshot
from qlinks.solver.deep_first_search import DeepFirstSearch

Real: TypeAlias = int | float | np.floating
AnySnapshot = TypeVar("AnySnapshot", bound="GaugeInvariantSnapshot")


@dataclass
class ComputationBasis(Generic[AnySnapshot]):
    length: int
    width: int
    charge_distri: NDArray[np.int64]
    flux_sector: Optional[Tuple[Real, Real]] = field(default=None)
    momentum: Optional[Tuple[int, int]] = field(default=None)
    max_steps: Optional[int] = field(default=30000)
    n_solutions: Optional[int] = field(default=3000)

    def __post_init__(self):
        if all(
            symmetry is None for symmetry in (self.flux_sector, self.momentum)
        ):
            self._init_snapshot = GaugeInvariantSnapshot(
                self.length, self.width, self.charge_distri
            )
        elif self.momentum is None:
            self._init_snapshot = FluxSectorSnapshot(
                self.length, self.width, self.charge_distri, self.flux_sector
            )
        else:
            raise NotImplementedError
        dfs = DeepFirstSearch(self._init_snapshot, self.max_steps)
        self._snapshots: List[AnySnapshot] = dfs.search(self.n_solutions)

    def get(self) -> LatticeMultiStates:
        return LatticeMultiStates(
            self.length, self.width, states=[snapshot.to_state() for snapshot in self._snapshots]
        )

    @cached_property
    def snapshots(self) -> List[AnySnapshot]:
        return self._snapshots

    @property
    def hilbert_dims(self) -> Tuple[int, int]:
        n_snapshots = len(self._snapshots)
        return n_snapshots, n_snapshots

    @property
    def quantum_numbers(self):
        return {
            symmetry.__name__: symmetry.quantum_numbers
            for symmetry in (self.gauss_law, self.global_flux, self.translation)
        }

    @property
    def gauss_law(self):
        return GaussLaw(self.charge_distri)

    @property
    def global_flux(self):
        return GlobalFlux(*self.flux_sector)

    @property
    def translation(self):
        return NotImplemented
