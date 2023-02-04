from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import astuple, dataclass, field
from functools import reduce
from itertools import product
from typing import Generator, Iterator, Sequence, Tuple

import numpy as np

from qlinks.lattice.component import Site, UnitVectorCollection
from qlinks.spin_object import Link, SpinOperator, SpinOperatorCollection


@dataclass
class SquareLattice:
    length: int
    width: int

    def __post_init__(self):
        if any(axis < 2 for axis in self.shape):
            raise ValueError("Lattice size should be least 2 by 2.")

    def __getitem__(self, coord: Sequence[int, int] | Site) -> Site:
        coord = astuple(coord) if isinstance(coord, Site) else coord
        return Site(coord[0] % self.length, coord[1] % self.width)  # assume periodic b.c.

    def __iter__(self) -> Generator[Site]:
        for coord_y, coord_x in product(range(self.width), range(self.length)):
            yield Site(coord_x, coord_y)

    def iter_links(self) -> Generator[Link]:
        for site, unit_vector in product(self, UnitVectorCollection()):
            yield Link(site, unit_vector)

    def iter_plaquettes(self) -> Generator[Plaquette]:
        for corner_site in self:
            yield Plaquette(self, corner_site)

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
    _spin_opts: SpinOperatorCollection = field(
        default_factory=lambda: SpinOperatorCollection(), repr=False
    )
    _unit_vectors: UnitVectorCollection = field(
        default_factory=lambda: UnitVectorCollection(), repr=False
    )

    @abc.abstractmethod
    def __post_init__(self):
        ...


@dataclass
class Plaquette(QuasiLocalSpinObject):
    def __post_init__(self):
        self.link_d = Link(
            site=self.lattice[self.corner_site],
            unit_vector=self._unit_vectors.rightward,
            operator=self._spin_opts.Sp,
        )
        self.link_r = Link(
            site=self.lattice[self.corner_site + self._unit_vectors.rightward],
            unit_vector=self._unit_vectors.upward,
            operator=self._spin_opts.Sp,
        )
        self.link_t = Link(
            site=self.lattice[self.corner_site + self._unit_vectors.upward],
            unit_vector=self._unit_vectors.rightward,
            operator=self._spin_opts.Sm,
        )
        self.link_l = Link(
            site=self.lattice[self.corner_site],
            unit_vector=self._unit_vectors.upward,
            operator=self._spin_opts.Sm,
        )

    def __array__(self) -> np.ndarray:
        plaquette = [link for link in self]
        skip_links = [link.reset() for link in self]
        for identity_link in self.lattice.iter_links():
            if identity_link not in skip_links:
                plaquette.append(identity_link)
        operators = [link.operator for link in sorted(plaquette)]
        return reduce((lambda x, y: x ^ y), operators).reshape(self.lattice.hilbert_dims)

    def __iter__(self) -> Iterator[Link]:
        return iter((self.link_t, self.link_d, self.link_l, self.link_r))

    def __add__(self, other: Plaquette) -> SpinOperator:
        if self.corner_site != other.corner_site:
            raise ValueError("Plaquettes in different positions can not be directly added.")
        return (np.array(self) + np.array(other)).view(SpinOperator)

    @property
    def corner_site(self) -> Site:
        return self.site

    def conj(self, inplace: bool = False) -> Plaquette | None:
        conj_plaquette = self if inplace else deepcopy(self)
        _ = [link.conj(inplace=True) for link in conj_plaquette]
        if not inplace:
            return conj_plaquette


@dataclass
class Cross(QuasiLocalSpinObject):

    def __post_init__(self):
        pass

    @property
    def center_site(self) -> Site:
        return self.site
