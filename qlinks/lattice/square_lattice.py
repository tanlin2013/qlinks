from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import astuple, dataclass, field
from functools import reduce
from itertools import product
from typing import Dict, Iterator, Self, Tuple, List, Type

import numpy as np

from qlinks.exceptions import InvalidArgumentError, InvalidOperationError, LinkOverridingError
from qlinks.lattice.component import Site, UnitVector, UnitVectors
from qlinks.spin_object import Link, Spin, SpinOperator, SpinOperators


LinkIndex: Type = Tuple[Site, UnitVector]


@dataclass
class SquareLattice:
    length: int
    width: int

    def __post_init__(self):
        if any(axis < 2 for axis in self.shape):
            raise ValueError("Lattice size should be least 2 by 2.")

    def __getitem__(self, coord: Tuple[int, int] | Site) -> Site:
        coord = astuple(coord) if isinstance(coord, Site) else coord
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

    def iter_crosses(self) -> Iterator[Cross]:
        for center_site in self:
            yield Cross(self, center_site)

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


@dataclass
class QuasiLocalSpinObject(abc.ABC):
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

    def __add__(self, other: QuasiLocalSpinObject) -> SpinOperator:
        if self.site != other.site:
            raise InvalidOperationError(
                f"{type(self).__name__} in different positions can not be directly added.")
        return (np.array(self) + np.array(other)).view(SpinOperator)

    def __iter__(self) -> Iterator[Link]:
        return iter((self.link_t, self.link_d, self.link_l, self.link_r))

    def conj(self, inplace: bool = False) -> Self | None:  # type: ignore[return]
        conj_spin_obj = self if inplace else deepcopy(self)
        _ = [link.conj(inplace=True) for link in conj_spin_obj]
        if not inplace:
            return conj_spin_obj

    def set_config(self, config: Tuple[Spin, ...]) -> None:
        for idx, link in enumerate(self):
            link.config = config[idx]


@dataclass
class Plaquette(QuasiLocalSpinObject):
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
class Cross(QuasiLocalSpinObject):
    def __post_init__(self):
        self.link_t = Link(
            site=self.lattice[self.site],
            unit_vector=UnitVectors.upward
        )
        self.link_d = Link(
            site=self.lattice[self.site + UnitVectors.downward],
            unit_vector=UnitVectors.upward
        )
        self.link_l = Link(
            site=self.lattice[self.site + UnitVectors.leftward],
            unit_vector=UnitVectors.rightward
        )
        self.link_r = Link(
            site=self.lattice[self.site],
            unit_vector=UnitVectors.rightward
        )
