from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple, dataclass, field
from enum import IntEnum
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from qlinks.exceptions import LinkOverridingError
from qlinks.lattice.component import Site, UnitVectors
from qlinks.lattice.spin_object import Spin, SpinConfigs
from qlinks.lattice.square_lattice import SquareLattice
from qlinks.solver.deep_first_search import Node


class Flow(IntEnum):
    outward = 1
    inward = -1


@dataclass
class GaussLaw:
    charge: int = 0
    __hash_table: Dict[int, Spin] = field(init=False, repr=False)

    def __post_init__(self):
        if abs(self.charge) > 2:
            raise ValueError("Charge ranges from -2 to +2.")
        self.__hash_table = {spin.magnetization: spin for spin in SpinConfigs}

    def possible_flows(self) -> List[Tuple[int, ...]]:
        flows = product([flow.value for flow in Flow], repeat=4)
        return [quartet for quartet in flows if sum(quartet) / 2 == self.charge]

    def possible_configs(self) -> List[Tuple[Spin, ...]]:
        local_coord_sys = [unit_vec.sign for unit_vec in UnitVectors.iter_all_directions()]
        return [
            tuple(map(lambda idx: self.__hash_table[idx], np.multiply(quartet, local_coord_sys)))
            for quartet in self.possible_flows()
        ]


@dataclass
class SpinConfigSnapshot(Node, SquareLattice):
    charge_distri: Optional[ArrayLike | np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        if self.charge_distri is None:
            self.charge_distri = np.zeros(self.shape)
        self.charge_distri = np.flipud(self.charge_distri).T
        assert self.charge_distri.shape == self.shape

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpinConfigSnapshot):
            return NotImplemented
        assert self.shape == other.shape
        return set(self.links.values()) == set(other.links.values())

    def __str__(self) -> str:
        return ",\n".join(map(repr, list(self.links.values())))

    def find_first_empty_site(self) -> Site | None:  # type: ignore[return]
        for site in self:
            if np.isnan(self.charge(site)):
                return site

    def extend_node(self) -> List[SpinConfigSnapshot]:
        site = self.find_first_empty_site()
        if site is None:
            return []
        charge = self.charge_distri[*astuple(site)]
        new_nodes = []
        for config in GaussLaw(charge).possible_configs():
            try:
                new_node = deepcopy(self)
                new_node.set_cross_links(site, config)
                new_nodes.append(new_node)
            except LinkOverridingError:
                continue
        return new_nodes

    def is_the_solution(self) -> bool:
        for site in self:
            if np.isnan(self.charge(site)) or \
                    self.charge(site) != self.charge_distri[*astuple(site)]:
                return False
        return True

    @property
    def adjacency_matrix(self):
        return

    def plot(self):
        pass
