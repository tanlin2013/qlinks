from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from itertools import product
from typing import Dict, List, Optional, Self, Set, Tuple

import numpy as np
import numpy.typing as npt

from qlinks.computation_basis import ComputationBasis
from qlinks.exceptions import InvalidArgumentError, InvalidOperationError
from qlinks.lattice.component import Site
from qlinks.lattice.square_lattice import SquareLattice, Vertex
from qlinks.solver.constraint_programming import CpModel
from qlinks.solver.deep_first_search import DeepFirstSearch, Node


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

    charge_distri: npt.NDArray[np.int64]
    flux_sector: Optional[Tuple[int, int]] = field(default=None)
    _lattice: SquareLattice = field(init=False, repr=False)

    def __post_init__(self):
        self.charge_distri = np.flipud(self.charge_distri).T
        if not np.all((self.charge_distri >= -2) & (self.charge_distri <= 2)):
            raise InvalidArgumentError("Charge ranges from -2 to +2.")
        self._lattice = SquareLattice(*self.shape)

    def __getitem__(self, site: Site) -> int:
        return self.charge_distri[*site]

    def charge(self, site: Site) -> int | float:
        return self._lattice.charge(site)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.charge_distri.shape  # type: ignore[return-value]

    @staticmethod
    @cache
    def possible_flows(charge: int) -> List[npt.NDArray[np.int64]]:
        flows = product([flow.value for flow in Flow], repeat=4)
        return [np.asarray(quartet) for quartet in flows if sum(quartet) / 2 == charge]

    @staticmethod
    @cache
    def possible_configs(charge: int) -> List[npt.NDArray[np.int64]]:
        def flow_to_spin(flow: npt.NDArray[np.int64]):
            flow *= Vertex.order()
            return flow * (flow == 1)  # turn -1 to 0

        return [flow_to_spin(quartet) for quartet in GaussLaw.possible_flows(charge)]

    @staticmethod
    def random_charge_distri(
        length_x: int, length_y: int, seed: Optional[int] = None, max_iter: int = 1000
    ) -> npt.NDArray[np.int64]:
        r"""Randomly sample static charges spread on 2d square lattice.

        Charges are uniformly sampled from the interval :math:`[-2, 2]`, with total charge being
        zero (this is always true under periodic boundary conditions). However, this does not
        guarantee to provide a valid configuration of links.

        Internally, this function uses multinomial distribution to sample possible charge
        distributions. Iteration for the trial sampling will stop as soon as it produces a valid
        outcome with zero total charge. Normally this is at most :math:`\mathcal{O}(1)` of
        lattice size, :math:`\text{length} \times \text{width}`.

        Args:
            length_x: The length of the lattice.
            length_y: The width of the lattice.
            seed: If provided, used for random generator. Default `None`.
            max_iter: The maximum number of iterations for trial sampling. Default `1000`.

        Returns:
            The drawn samples of shape (`width` :math:`\times` `length`). Note that we have swapped
            rows and columns to match with the lattice shape.

        Raises:
            StopIteration: If no valid distribution can be sampled within `max_iter` iterations.
        """
        sum_val, min_val, max_val = 0, -2, 2
        size = length_x * length_y
        prob = np.full(shape=size, fill_value=1 / size, dtype=float)
        rng = np.random.default_rng(seed=seed)
        for _ in range(max_iter):
            samples = min_val + rng.multinomial(n=sum_val - size * min_val, pvals=prob).flatten()
            if not np.any(samples > max_val):
                return samples.reshape((length_y, length_x)).astype(int)
        raise StopIteration(f"Couldn't sample a valid distribution within {max_iter} steps.")

    @staticmethod
    def staggered_charge_distri(length_x: int, length_y: int) -> npt.NDArray[np.int64]:
        """Generate a staggered charge distribution with +1 and -1 charges.
        The shape of the lattice must be even number, and the bottom-left corner is always +1.
        To make the bottom-left corner -1, simply multiply the output by -1.

        Args:
            length_x: The length of the lattice.
            length_y: The width of the lattice.

        Returns:
            The staggered charge distribution.

        Raises:
            InvalidArgumentError: If the shape of the lattice is not even number.

        Examples:
            >>> GaussLaw.staggered_charge_distri(2, 2)
            array([[-1,  1],
                   [ 1, -1]])
            >>> GaussLaw.staggered_charge_distri(4, 4)
            array([[-1,  1, -1,  1],
                   [ 1, -1,  1, -1],
                   [-1,  1, -1,  1],
                   [ 1, -1,  1, -1]])
        """
        if length_x % 2 != 0 or length_y % 2 != 0:
            raise InvalidArgumentError("Length and width must be even number.")
        stagger = np.array([[-1, 1], [1, -1]])
        return np.tile(stagger, (length_y // 2, length_x // 2))

    @classmethod
    def from_random_charge_distri(
        cls, length_x: int, length_y: int, flux_sector: Optional[Tuple[int, int]] = None
    ) -> Self:
        return cls(cls.random_charge_distri(length_x, length_y), flux_sector=flux_sector)

    @classmethod
    def from_staggered_charge_distri(
        cls, length_x: int, length_y: int, flux_sector: Optional[Tuple[int, int]] = None
    ) -> Self:
        return cls(cls.staggered_charge_distri(length_x, length_y), flux_sector=flux_sector)

    @classmethod
    def from_zero_charge_distri(
        cls, length_x: int, length_y: int, flux_sector: Optional[Tuple[int, int]] = None
    ) -> Self:
        return cls(np.zeros((length_y, length_x), dtype=int), flux_sector=flux_sector)

    def __deepcopy__(self, memo: Dict) -> Self:
        new_inst = type(self).__new__(self.__class__)
        new_inst.charge_distri = self.charge_distri
        new_inst.flux_sector = self.flux_sector
        new_inst._lattice = SquareLattice(*self.shape)
        new_inst._lattice.links = self._lattice.links.copy()
        return new_inst

    def __hash__(self) -> int:
        return hash(self._lattice.links.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussLaw):
            return NotImplemented
        if self.shape != other.shape:
            raise InvalidOperationError(f"Shape of two {type(self).__name__} are mismatched.")
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return "".join(map(str, self._lattice.links))

    def _next_empty_site(self) -> Optional[Site]:
        link_idx = self._lattice.empty_link_index()
        if link_idx.size > 0:
            site_idx = link_idx[0] // 2  # retrieve the first one
            return Site(site_idx % self._lattice.length_x, site_idx // self._lattice.length_x)
        return None

    def _valid_for_flux(self, site: Site) -> bool:
        for axis in range(2):
            if site[axis ^ 1] == (self._lattice.shape[axis ^ 1] - 1):  # have filled a line
                if self._lattice.axial_flux(site[axis], axis) != self.flux_sector[axis]:
                    return False
        return True

    def _fill_node(self, site: Site, config: npt.NDArray[np.int64]) -> Optional[Self]:
        new_node = deepcopy(self)
        new_node._lattice.set_vertex_links(site, config)
        if self.flux_sector is None or new_node._valid_for_flux(site):
            return new_node
        return None

    def _preconditioned_configs(self, site: Site) -> List[npt.NDArray[np.int64]]:
        vertex_links = self._lattice.links[Vertex(self._lattice, site).link_index()]
        been_set_links = vertex_links != self._lattice._empty_link_value
        if np.any(been_set_links):
            return [
                config
                for config in self.possible_configs(self[site])
                if np.all(config[been_set_links] == vertex_links[been_set_links])
            ]
        return self.possible_configs(self[site])

    def extend_node(self) -> Set[Self]:
        site = self._next_empty_site()
        if site is not None:
            return {
                new_node
                for config in self._preconditioned_configs(site)
                if (new_node := self._fill_node(site, config)) is not None
            }
        return set()

    def is_the_solution(self) -> bool:
        if self._lattice.empty_link_index().size > 0:  # not fully filled
            return False
        if not np.all([self.charge(site) == self[site] for site in self._lattice]):
            return False
        return True

    def solve(self, method: str = "cp", **kwargs) -> ComputationBasis:
        if method == "cp":
            cp = CpModel(self.shape, self.charge_distri, self.flux_sector)
            cp.solve(**kwargs)
            return cp.to_basis()
        elif method == "dfs":
            dfs = DeepFirstSearch(self)
            return self.to_basis(dfs.solve(**kwargs))
        else:
            raise NotImplementedError(f"Method {method} is not implemented yet.")

    @staticmethod
    def to_basis(nodes: List[GaussLaw]) -> ComputationBasis:
        link_data = [node._lattice.links for node in nodes]
        basis = ComputationBasis(np.vstack(link_data))
        basis.sort()
        return basis
