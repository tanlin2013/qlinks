from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from itertools import product
from typing import Dict, List, Optional, Self, Tuple

import numpy as np
import numpy.typing as npt

from qlinks.exceptions import (
    InvalidArgumentError,
    InvalidOperationError,
    LinkOverridingError,
)
from qlinks.lattice.component import Site, UnitVectors
from qlinks.lattice.square_lattice import SquareLattice, Vertex, ComputationBasis
from qlinks.solver.deep_first_search import Node


class Flow(IntEnum):
    outward = 1
    inward = -1


@dataclass(slots=True)
class GaussLaw(Node):
    """


    Args:
        charge_distri: The charge distribution on the lattice.
                       The (N, M)-shaped input array is repositioned in the first quadrant,
                       aligning the element at (N - 1, 0) to (0, 0),
                       and the element at (0, M - 1) to (M - 1, N - 1).
        flux_sector: The flux sector of the lattice, optional.
    """

    charge_distri: npt.NDArray[int]
    flux_sector: Optional[Tuple[int, int]] = field(default=None)
    _lattice: SquareLattice = field(init=False, repr=False)

    def __post_init__(self):
        self.charge_distri = np.flipud(self.charge_distri).T
        if not np.all((self.charge_distri >= -2) & (self.charge_distri <= 2)):
            raise InvalidArgumentError("Charge ranges from -2 to +2.")
        self._lattice = SquareLattice(*self.shape)

    def __getitem__(self, site: Site) -> int:
        return self.charge_distri[*site]

    def charge(self, site: Site) -> int:
        return self._lattice.charge(site)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.charge_distri.shape

    @staticmethod
    @cache
    def possible_flows(charge: int) -> List[npt.NDArray[int]]:
        flows = product([flow.value for flow in Flow], repeat=4)
        return [np.asarray(quartet) for quartet in flows if sum(quartet) / 2 == charge]

    @staticmethod
    @cache
    def possible_configs(charge: int) -> List[npt.NDArray[int]]:
        def flow_to_spin(flow: npt.NDArray[int]):
            flow *= Vertex.order()
            return flow * (flow == 1)  # turn -1 to 0

        return [flow_to_spin(quartet) for quartet in GaussLaw.possible_flows(charge)]

    @staticmethod
    def random_charge_distri(
        length: int, width: int, seed: Optional[int] = None, max_iter: int = 1000
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
            The drawn samples of shape (`width` :math:`\times` `length`). Note that we have swapped
            rows and columns to match with the lattice shape.

        Raises:
            StopIteration: If no valid distribution can be sampled within `max_iter` iterations.
        """
        sum_val, min_val, max_val = 0, -2, 2
        size = length * width
        prob = np.full(shape=size, fill_value=1 / size, dtype=float)
        rng = np.random.default_rng(seed=seed)
        for _ in range(max_iter):
            samples = min_val + rng.multinomial(n=sum_val - size * min_val, pvals=prob).flatten()
            if not np.any(samples > max_val):
                return samples.reshape((width, length))
        raise StopIteration(f"Couldn't sample a valid distribution within {max_iter} steps.")

    @staticmethod
    def staggered_charge_distri(length: int, width: int) -> np.ndarray:
        if length % 2 != 0 or width % 2 != 0:
            raise InvalidArgumentError("Length and width must be even number.")
        stagger = np.array([[1, -1], [-1, 1]])
        return np.tile(stagger, (width // 2, length // 2))

    @classmethod
    def from_random_charge_distri(cls, length_x: int, length_y: int) -> Self:
        return cls(cls.random_charge_distri(length_x, length_y))

    @classmethod
    def from_staggered_charge_distri(cls, length_x: int, length_y: int) -> Self:
        return cls(cls.staggered_charge_distri(length_x, length_y))

    @classmethod
    def from_zero_charge_distri(cls, length_x: int, length_y: int) -> Self:
        return cls(np.zeros((length_y, length_x), dtype=int))

    def __deepcopy__(self, memo: Dict) -> Self:
        new_inst = type(self).__new__(self.__class__)
        new_inst.charge_distri = self.charge_distri
        new_inst.flux_sector = self.flux_sector
        new_inst._lattice = SquareLattice(*self.shape)
        new_inst._lattice.links = deepcopy(self._lattice.links)
        return new_inst

    def __hash__(self) -> int:
        return hash(str(self._lattice.links))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussLaw):
            return NotImplemented
        if self.shape != other.shape:
            raise InvalidOperationError(f"Shape of two {type(self).__name__} are mismatched.")
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return "".join(map(str, self._lattice.links))

    def _next_empty_site(self) -> Optional[Site]:
        site_idx = self._lattice.empty_link_index().tolist()
        if site_idx:
            site_idx = site_idx[0] // 2
            return Site(site_idx % self._lattice.length_x, site_idx // self._lattice.length_x)
        return

    def _valid_for_flux(self, site: Site) -> bool:
        _ax_flux = self._lattice.axial_flux  # func alias
        neighbors = [site + UnitVectors().leftward, site + UnitVectors().downward]
        for axis, neighbor in enumerate(neighbors):
            if np.isnan(_ax_flux(site[axis], axis)) or np.isnan(_ax_flux(neighbor[axis], axis)):
                continue
            elif (_ax_flux(site[axis], axis) != self.flux_sector[axis]) or (
                _ax_flux(site[axis], axis) != _ax_flux(neighbor[axis], axis)
            ):
                return False
        return True

    def _fill_node(self, site: Site, config: npt.NDArray[int]) -> Optional[Self]:
        try:
            new_node = deepcopy(self)
            new_node._lattice.set_vertex_links(site, config)
            if self.flux_sector is None or new_node._valid_for_flux(site):
                return new_node
        except LinkOverridingError:
            pass
        return

    def _preconditioned_configs(self, site: Site) -> List[npt.NDArray[int]]:
        vertex_links = self._lattice.links[Vertex(self._lattice, site).link_index()]
        been_set_links = vertex_links != self._lattice._empty_link_value
        if np.any(been_set_links):
            return [
                config
                for config in self.possible_configs(self[site])
                if np.all(config[been_set_links] == vertex_links[been_set_links])
            ]
        return self.possible_configs(self[site])

    def extend_node(self) -> List[Self]:
        site = self._next_empty_site()
        if site is not None:
            return [
                new_node
                for config in self._preconditioned_configs(site)
                if (new_node := self._fill_node(site, config)) is not None
            ]
        return []

    def is_the_solution(self) -> bool:
        for site in reversed(self._lattice):
            if np.isnan(self.charge(site)) or (self.charge(site) != self[site]):
                return False
        return True

    @staticmethod
    def to_basis(nodes: List[GaussLaw]) -> ComputationBasis:
        link_data = [node._lattice.links for node in nodes]
        return ComputationBasis(np.vstack(link_data))
