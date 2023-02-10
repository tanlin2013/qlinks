from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import astuple, dataclass, field
from functools import reduce, total_ordering
from itertools import product
from typing import Dict, Iterator, List, Optional, Self, Tuple, TypeAlias

import numpy as np

from qlinks.exceptions import (
    InvalidArgumentError,
    InvalidOperationError,
    LinkOverridingError,
)
from qlinks.lattice.component import Site, UnitVector, UnitVectors
from qlinks.lattice.spin_object import Link, Spin, SpinOperator, SpinOperators

Real: TypeAlias = int | float | np.floating
LinkIndex: TypeAlias = Tuple[Site, UnitVector]


@dataclass
class SquareLattice:
    length: int
    width: int
    __links: Dict[LinkIndex, Link] = field(init=False, repr=False)

    def __post_init__(self):
        if any(axis < 2 for axis in self.shape):
            raise InvalidArgumentError("Lattice size should be least 2 by 2.")
        self.__links = {(link.site, link.unit_vector): link for link in self.iter_links()}

    def __getitem__(self, coord: Tuple[int, int] | Site) -> Site:
        coord = astuple(coord) if isinstance(coord, Site) else coord  # type: ignore[assignment]
        return Site(coord[0] % self.length, coord[1] % self.width)  # assume periodic b.c.

    def __iter__(self) -> Iterator[Site]:
        for coord_y, coord_x in product(range(self.width), range(self.length)):
            yield Site(coord_x, coord_y)

    def iter_links(self) -> Iterator[Link]:
        for site, unit_vector in product(self, UnitVectors):
            yield Link(site, unit_vector)

    def iter_plaquettes(self) -> Iterator[Plaquette]:
        for corner_site in self:
            yield Plaquette(self, corner_site)

    def iter_vertices(self) -> Iterator[Vertex]:
        for center_site in self:
            yield Vertex(self, center_site)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.length, self.width

    @property
    def size(self) -> int:
        return self.length * self.width

    @property
    def num_links(self) -> int:
        return 2 * self.size

    @property
    def hilbert_dims(self) -> Tuple[int, int]:
        return 2**self.num_links, 2**self.num_links

    @property
    def links(self) -> Dict[LinkIndex, Link]:
        return self.__links

    def get_link(self, link_index: LinkIndex) -> Link:
        site, unit_vector = link_index
        if unit_vector.sign < 0:
            site += unit_vector
            unit_vector *= -1
        return self.__links[(self[site], unit_vector)]

    def set_link(self, link_index: LinkIndex, state: Spin) -> None:
        link = self.get_link(link_index)
        if link.state is not None and link.state != state:
            raise LinkOverridingError("Link has been set, try .reset_link() if needed.")
        link.state = state

    def reset_link(self, link_index: LinkIndex) -> None:
        self.get_link(link_index).reset(inplace=True)

    def _get_vertex_link_indices(self, site: Site) -> List[LinkIndex]:
        return [(self[site], unit_vector) for unit_vector in UnitVectors.iter_all_directions()]

    def get_vertex_links(self, site: Site) -> List[Link]:
        return [self.get_link(link_index) for link_index in self._get_vertex_link_indices(site)]

    def set_vertex_links(self, site: Site, states: Tuple[Spin, ...]) -> None:
        if len(states) != 4:
            raise InvalidArgumentError(f"Expected 4 Spins in states, got {len(states)}.")
        for idx, link_index in enumerate(self._get_vertex_link_indices(site)):
            self.set_link(link_index, state=states[idx])

    def reset_vertex_links(self, site: Site) -> None:
        for link_index in self._get_vertex_link_indices(site):
            self.reset_link(link_index)

    def charge(self, site: Site) -> Real:
        charge: Real = 0
        for link in self.get_vertex_links(site):
            if link.state is not None:
                flux = link.flux
                charge += flux if link.site == self[site] else -1 * flux
            else:
                return np.nan
        return charge / 2

    def axial_flux(self, idx: int, axis: Optional[int] = 0) -> Real:
        unit_vector = {0: UnitVectors.rightward, 1: UnitVectors.upward}[axis]
        sites = {
            0: [Site(idx, y) for y in range(self.width)],
            1: [Site(x, idx) for x in range(self.length)]
        }[axis]
        return sum([self.get_link((site, unit_vector)).flux for site in sites])


@total_ordering
@dataclass
class LatticeState(SquareLattice):
    link_data: Dict[LinkIndex, Link]

    def __post_init__(self):
        self.__links = deepcopy(self.link_data)

    def __hash__(self) -> int:
        return hash(frozenset(self.links.items()))

    def __lt__(self, other: LatticeState) -> bool:
        return tuple(link.flux for link in self.links.values()) < tuple(
            link.flux for link in other.links.values()
        )

    @classmethod
    def from_snapshot(cls, snapshot) -> Self:
        return cls(snapshot.length, snapshot.width, snapshot.links)

    @property
    def tsp(self):
        return NotImplemented


@dataclass
class QuasiLocalOperator(abc.ABC):
    lattice: SquareLattice
    site: Site
    link_t: Link = field(init=False)
    link_d: Link = field(init=False)
    link_l: Link = field(init=False)
    link_r: Link = field(init=False)

    @abc.abstractmethod
    def __post_init__(self):
        ...

    def __array__(self) -> np.ndarray:
        spin_obj = [link for link in self]
        skip_links = [link.reset() for link in self]
        for identity_link in self.lattice.iter_links():
            if identity_link not in skip_links:
                spin_obj.append(identity_link)
        operators = [link.operator for link in sorted(spin_obj)]
        return reduce((lambda x, y: x ^ y), operators).reshape(self.lattice.hilbert_dims)

    def __add__(self, other: QuasiLocalOperator) -> SpinOperator:
        if self.site != other.site:
            raise InvalidOperationError(
                f"{type(self).__name__} in different positions can not be directly added."
            )
        return (np.array(self) + np.array(other)).view(SpinOperator)

    def __iter__(self) -> Iterator[Link]:
        return iter(sorted((self.link_d, self.link_l, self.link_r, self.link_t)))

    def __matmul__(self, other: LatticeState):
        pass

    def __rmatmul__(self, other: LatticeState):
        pass

    def conj(self, inplace: bool = False) -> Self | None:  # type: ignore[return]
        conj_spin_obj = self if inplace else deepcopy(self)
        _ = [link.conj(inplace=True) for link in conj_spin_obj]
        if not inplace:
            return conj_spin_obj


@dataclass
class Plaquette(QuasiLocalOperator):
    def __post_init__(self):
        self.link_d = Link(
            site=self.lattice[self.corner_site],
            unit_vector=UnitVectors.rightward,
            operator=SpinOperators.Sp,
        )
        self.link_r = Link(
            site=self.lattice[self.corner_site + UnitVectors.rightward],
            unit_vector=UnitVectors.upward,
            operator=SpinOperators.Sp,
        )
        self.link_t = Link(
            site=self.lattice[self.corner_site + UnitVectors.upward],
            unit_vector=UnitVectors.rightward,
            operator=SpinOperators.Sm,
        )
        self.link_l = Link(
            site=self.lattice[self.corner_site],
            unit_vector=UnitVectors.upward,
            operator=SpinOperators.Sm,
        )

    @property
    def corner_site(self) -> Site:
        return self.site


@dataclass
class Vertex(QuasiLocalOperator):
    def __post_init__(self):
        self.link_t = Link(site=self.lattice[self.site], unit_vector=UnitVectors.upward)
        self.link_d = Link(
            site=self.lattice[self.site + UnitVectors.downward], unit_vector=UnitVectors.upward
        )
        self.link_l = Link(
            site=self.lattice[self.site + UnitVectors.leftward], unit_vector=UnitVectors.rightward
        )
        self.link_r = Link(site=self.lattice[self.site], unit_vector=UnitVectors.rightward)
