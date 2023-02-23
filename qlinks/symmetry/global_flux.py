from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Self, Tuple, TypeAlias, TypeVar

import numpy as np

from qlinks.exceptions import InvalidArgumentError, LinkOverridingError
from qlinks.lattice.component import Site, UnitVectors
from qlinks.lattice.square_lattice import SquareLattice
from qlinks.symmetry.abstract import AbstractSymmetry
from qlinks.symmetry.gauss_law import GaugeInvariantSnapshot, GaussLaw

Real: TypeAlias = int | float | np.floating
AnySquareLattice = TypeVar("AnySquareLattice", bound=SquareLattice)


@dataclass
class GlobalFlux(AbstractSymmetry):
    flux_x: Real
    flux_y: Real

    def __getitem__(self, item: int) -> Real:
        return {0: self.flux_x, 1: self.flux_y}[item]

    @property
    def quantum_numbers(self) -> Tuple[Real, Real]:
        return self.flux_x, self.flux_y


@dataclass
class FluxSectorSnapshot(GaugeInvariantSnapshot):
    flux_sector: Tuple[Real, Real] = field(default=(0, 0))

    def __post_init__(self):
        super().__post_init__()
        if self.flux_sector[0] not in np.arange(
            -self.width / 2, self.width / 2 + 1
        ) or self.flux_sector[1] not in np.arange(-self.length / 2, self.length / 2 + 1):
            raise InvalidArgumentError(
                "Flux must be within {-l/2, -l/2 + 1, ..., l/2}, where l is the length (or width)."
            )

    def __hash__(self) -> int:
        return super().__hash__()

    def valid_for_axial_flux(self, site: Site) -> bool:
        slide_left = site + UnitVectors.leftward
        slide_down = site + UnitVectors.downward
        for axis, pairs in zip([0, 1], [(site[0], slide_left[0]), (site[1], slide_down[1])]):
            for idx in pairs:
                try:
                    if self.axial_flux(idx, axis) != self.flux_sector[axis]:
                        return False
                except ValueError:
                    continue
        return True

    def extend_node(self) -> List[Self]:
        site = self.find_first_empty_site()
        new_nodes = []
        if site is not None:
            for config in GaussLaw.possible_configs(charge=self.gauss_law[site]):
                try:
                    new_node = deepcopy(self)
                    new_node.set_vertex_links(site, config)
                    if self.valid_for_axial_flux(site):
                        new_nodes.append(new_node)
                except LinkOverridingError:
                    continue
        return new_nodes

    @property
    def global_flux(self) -> GlobalFlux:
        return GlobalFlux(*self.flux_sector)
