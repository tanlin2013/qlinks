from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Self, Tuple, Iterator, Protocol

import networkx as nx
import numpy as np
import numpy.typing as npt

from qlinks.exceptions import InvalidArgumentError, InvalidOperationError, LinkOverridingError
from qlinks.lattice.component import Site, UnitVectors
from qlinks.symmetry.computation_basis import ComputationBasis


@dataclass(slots=True)
class SquareLattice:
    """A square lattice with periodic boundary condition.

    Args:
        length_x: The length of lattice in x direction.
        length_y: The length of lattice in y direction.
        links: The link data in shape (n_links,), optional.
    """

    length_x: int
    length_y: int
    links: Optional[npt.NDArray[int]] = field(default=None, repr=False)
    _empty_link_value: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        if any(axis < 2 for axis in self.shape):
            raise InvalidArgumentError("Lattice size should be least 2 by 2.")
        if self.links is None:
            self.links = np.full(self.n_links, fill_value=self._empty_link_value, dtype=int)
        elif not np.all(np.isin(self.links, [0, 1])):
            raise InvalidArgumentError("Link values should be either 0 or 1.")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.length_x, self.length_y

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def n_links(self) -> int:
        return 2 * np.prod(self.shape)

    @property
    def index(self) -> int:
        """
        Returns:
            The integer representation of the binary link data.

        Raises:
            ValueError: If the link data is not binary or emtpy.
        """
        return int("".join(map(str, self.links)), 2)

    def __getitem__(self, coord: Tuple[int, int] | Site) -> Site:
        return Site(coord[0] % self.length_x, coord[1] % self.length_y)  # assume periodic b.c.

    def site_index(self, site: Site) -> int:
        site = self[site]
        return 2 * (site.pos_x + self.length_x * site.pos_y)

    def __iter__(self) -> Iterator[Site]:
        for coord_y, coord_x in product(range(self.length_y), range(self.length_x)):
            yield Site(coord_x, coord_y)

    def __reversed__(self) -> Iterator[Site]:
        for coord_y, coord_x in product(
            reversed(range(self.length_y)), reversed(range(self.length_x))
        ):
            yield Site(coord_x, coord_y)

    def iter_plaquettes(self) -> Iterator[Plaquette]:
        for corner_site in self:
            yield Plaquette(self, corner_site)

    def empty_link_index(self) -> npt.NDArray[int]:
        return np.where(self.links == self._empty_link_value)[0]

    def set_vertex_links(self, site: Site, states: npt.NDArray[int]) -> None:
        vertex_link_idx = Vertex(self, self[site]).link_index()
        if len(states) != 4:
            raise InvalidArgumentError(f"Expected 4 Spins in states, got {len(states)}.")
        if np.any(
            (self.links[vertex_link_idx] != self._empty_link_value)
            & (self.links[vertex_link_idx] != states)
        ):
            raise LinkOverridingError("Some vertex links have been set.")
        self.links[vertex_link_idx] = states

    def charge(self, site: Site) -> Real:
        vertex = Vertex(self, self[site])
        if np.any(self.links[vertex.link_index()] == self._empty_link_value):
            return np.nan
        return np.sum((self.links[vertex.link_index()] - 0.5) * vertex.order()).astype(int)

    def axial_flux(self, idx: int, axis: Optional[int] = 0) -> float:
        """Compute the electric flux along the axis.

           ▲  |   │
           │  |   ▼         ▲
        ──►o──|──►o──►      │
           ▲  |   │
        ---│------▼----  axis=1
        ◄──o◄─|───o◄──
           ▲  |   │
           │  |   ▼

            axis=0  ──►

        Args:
            idx: The `idx`-th row or column in lattice.
                 Negative values or values greater than the length are permissible.
            axis: 0 for x-axis and 1 for y-axis, default 0.

        Returns:
            The total electric flux flowing along the axis, in unit of e/2.

        Notes:
            The flux cross-section is perpendicular to the axis.
        """
        link_idx = {
            0: [self.site_index(Site(idx, y)) for y in range(self.length_y)],
            1: [self.site_index(Site(x, idx)) + 1 for x in range(self.length_x)],
        }[axis]
        if np.any(self.links[link_idx] == self._empty_link_value):
            return np.nan
        return np.sum(self.links[link_idx] - 0.5)

    def as_adj_mat(self) -> npt.NDArray[int]:
        """

        Returns:

        """
        adj_mat = np.zeros((self.size, self.size), dtype=int)
        hash_table = {site: idx for idx, site in enumerate(self)}
        # for site, unit_vector in product(self, UnitVectors().iter_all_directions()):
        #     inds = (hash_table[site], hash_table[self[site + unit_vector]])  # head to tail
        #     link = self.get_link((site, unit_vector))
        #     if unit_vector.sign * link.flux > 0:
        #         adj_mat[*inds] += 1
        #     else:
        #         adj_mat[*inds[::-1]] += 1
        # return (adj_mat / 2).astype(int)
        ...

    def as_graph(self) -> nx.MultiDiGraph:
        return nx.from_numpy_array(
            self.as_adj_mat(), parallel_edges=True, create_using=nx.MultiDiGraph
        )


@dataclass(slots=True)
class LocalOperator(Protocol):
    lattice: SquareLattice
    site: Site
    _mask: int = field(default=None, repr=False)

    def link_index(self) -> npt.NDArray[int]:
        ...

    def __matmul__(self, basis: ComputationBasis) -> npt.NDArray[int]:
        ...

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[int]:
        ...


@dataclass(slots=True)
class Plaquette(LocalOperator):
    """Plaquette operator on a square lattice.

    Args:
        lattice: The lattice on which the plaquette operator acts.
        site: The site on lower-left corner of the plaquette.
        _mask: The binary mask of the plaquette operator, optional.

    Examples:

    """

    lattice: SquareLattice
    site: Site
    _mask: int = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._mask is None:
            self._mask = np.sum(2 ** (self.lattice.n_links - 1 - self.link_index()))

    def link_index(self) -> npt.NDArray[int]:
        return np.array(
            [
                self.lattice.site_index(self.site),
                self.lattice.site_index(self.site) + 1,
                self.lattice.site_index(self.site + UnitVectors().rightward) + 1,
                self.lattice.site_index(self.site + UnitVectors().upward),
            ]
        )

    def flippable(self, basis: ComputationBasis) -> npt.NDArray[bool]:
        b1, b2, b3, b4 = (basis.links[:, idx] for idx in self.link_index())
        return ((b1 & ~b2 & b3 & ~b4) | (~b1 & b2 & ~b3 & b4)).astype(bool)

    def __matmul__(self, basis: ComputationBasis) -> npt.NDArray[int]:
        """

        Args:
            basis:

        Returns:

        """
        if not isinstance(basis, ComputationBasis):
            return NotImplemented
        flipped_states = basis.index
        flipped_states[self.flippable(basis)] ^= self._mask
        return flipped_states

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[int]:
        """

        Args:
            basis:

        Returns:

        """
        if not isinstance(basis, ComputationBasis):
            return NotImplemented
        flippable = self.flippable(basis)
        flipped_states = self @ basis
        if not np.array_equal(basis.index, np.sort(flipped_states)):
            raise InvalidOperationError("Basis is not closure under the plaquette operator.")
        row_idx = np.arange(basis.n_states)
        col_idx = np.searchsorted(basis.index, flipped_states)
        mat = ((row_idx[:, None] == col_idx) | (col_idx[:, None] == row_idx)).astype(int)
        mat[~(flippable[:, None] | flippable)] = 0
        return mat

    def __pow__(self, power: int) -> Self:
        if power % 2 == 0:
            return Plaquette(self.lattice, self.site, _mask=0)
        return self


@dataclass(slots=True)
class Vertex:  # reserved as LocalOperator for future use
    lattice: SquareLattice
    site: Site
    _mask: int = field(default=None, repr=False)

    @staticmethod
    def order() -> npt.NDArray[int]:
        return np.array([-1, -1, 1, 1])

    def link_index(self) -> npt.NDArray[int]:
        return np.array(
            [
                self.lattice.site_index(self.site + UnitVectors().downward) + 1,
                self.lattice.site_index(self.site + UnitVectors().leftward),
                self.lattice.site_index(self.site),
                self.lattice.site_index(self.site) + 1,
            ]
        )

    def __matmul__(self, basis: ComputationBasis) -> ComputationBasis:
        ...

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[int]:
        ...
