from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple, dataclass, field
from enum import IntEnum
from itertools import product
from typing import Dict, List, Optional, Tuple

import networkx as nx
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

    @staticmethod
    def random_charge_distri(
            length: int, width: int, seed: Optional[int] = None, max_iter: Optional[int] = 1000
    ) -> np.ndarray:
        r"""Randomly sample static charges spread on 2d square lattice.

        Charges are uniformly sampled from the interval :math:`[-2, 2]`, with total charge being
        zero (this is always true under periodic boundary conditions).

        Internally, this function uses multinomial distribution to sample possible charge
        distributions. Iteration for the trial sampling will stop as soon as it produces a valid
        outcome with zero total charge. Normally this is at most :math:`\mathcal{O}(1)` of
        lattice size, :math:`\text{length} \times \text{width}`.

        Args:
            length: The length of the lattice.
            width: The width of the lattice.
            seed: If provided, used for random generator. Default `None`.
            max_iter: The maximum number of iterations for trial sampling. Default `1000`.

        Returns:
            The drawn samples of shape (`length` :math:`\times` `width`).

        Raises:
            StopIteration: If no valid distribution can be sampled within `max_iter` iterations.
        """
        sum_val, min_val, max_val = 0, -2, 2
        size = length * width
        prob = np.full(shape=size, fill_value=1/size, dtype=float)
        rng = np.random.default_rng(seed=seed)
        for _ in range(max_iter):
            samples = min_val + rng.multinomial(n=sum_val - size * min_val, pvals=prob).flatten()
            if not np.any(samples > max_val):
                return samples.reshape((length, width))
        raise StopIteration(f"Couldn't sample a valid distribution within {max_iter} steps.")


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

    def find_first_empty_site(self) -> Site | None:
        for site in self:
            if np.isnan(self.charge(site)):
                return site
        return None

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
    def adjacency_matrix(self) -> np.ndarray:
        adj_mat = np.zeros((self.size, self.size))
        hash_table = {site: idx for idx, site in enumerate(self)}
        for site, unit_vector in product(self, UnitVectors.iter_all_directions()):
            inds = (hash_table[site], hash_table[self[site + unit_vector]])  # head to tail
            link = self.get_link((site, unit_vector))
            if unit_vector.sign * link.config.magnetization > 0:
                adj_mat[*inds] += 1
            else:
                adj_mat[*inds[::-1]] += 1
        return (adj_mat / 2).astype(int)

    def as_graph(self) -> nx.MultiDiGraph:
        return nx.from_numpy_array(
            self.adjacency_matrix, parallel_edges=True, create_using=nx.MultiDiGraph
        )

    def plot(self) -> None:
        pass
