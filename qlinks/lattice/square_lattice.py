from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from itertools import product
from typing import Iterator, Optional, Protocol, Self, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.exceptions import (
    InvalidArgumentError,
    InvalidOperationError,
    LinkOverridingError,
)
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
    links: Optional[npt.NDArray[np.int64]] = field(default=None, repr=False)
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
        return np.prod(self.shape).item()

    @property
    def n_links(self) -> int:
        return 2 * np.prod(self.shape).item()

    @property
    def index(self) -> int:
        """
        Returns:
            The integer representation of the binary link data.

        Raises:
            ValueError: If the link data is not binary or emtpy.
        """
        return int("".join(map(str, self.links)), 2)

    def __getitem__(self, pos: Tuple[int, int] | Site) -> Site:
        return Site(pos[0] % self.length_x, pos[1] % self.length_y)  # assume periodic b.c.

    def site_index(self, site: Site) -> int:
        site = self[site]
        return 2 * (site.pos_x + self.length_x * site.pos_y)

    def __iter__(self) -> Iterator[Site]:
        for pos_y, pos_x in product(range(self.length_y), range(self.length_x)):
            yield Site(pos_x, pos_y)

    def __reversed__(self) -> Iterator[Site]:
        for pos_y, pos_x in product(
            reversed(range(self.length_y)), reversed(range(self.length_x))
        ):
            yield Site(pos_x, pos_y)

    def iter_plaquettes(self) -> Iterator[Plaquette]:
        for corner_site in self:
            yield Plaquette(self, corner_site)

    def empty_link_index(self) -> npt.NDArray[np.int64]:
        return np.where(self.links == self._empty_link_value)[0]

    def set_vertex_links(self, site: Site, states: npt.NDArray[np.int64]) -> None:
        vertex_link_idx = Vertex(self, self[site]).link_index()
        if len(states) != 4:
            raise InvalidArgumentError(f"Expected 4 Spins in states, got {len(states)}.")
        if np.any(
            (self.links[vertex_link_idx] != self._empty_link_value)
            & (self.links[vertex_link_idx] != states)
        ):
            raise LinkOverridingError("Some vertex links have been set.")
        self.links[vertex_link_idx] = states

    def charge(self, site: Site) -> int | float:
        vertex_link_idx = Vertex(self, self[site]).link_index()
        if np.any(self.links[vertex_link_idx] == self._empty_link_value):
            return np.nan
        return np.sum((self.links[vertex_link_idx] - 0.5) * Vertex.order()).astype(int)

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

    def adjacency_matrix(self) -> npt.NDArray[np.int64]:
        """

        Returns:

        """
        adj_mat = np.zeros((self.size, self.size), dtype=int)
        for site, (unit_vector, k) in product(self, zip(UnitVectors(), [0, 1])):
            inds = (self.site_index(site) // 2, self.site_index(site + unit_vector) // 2)
            link_val = self.links[2 * inds[0] + k]
            adj_mat[*inds] += link_val  # head_node to tail_node
            adj_mat[*inds[::-1]] += 1 - link_val  # tail_node to head_node
        return adj_mat

    def as_graph(self) -> nx.MultiDiGraph:
        return nx.from_numpy_array(
            self.adjacency_matrix(), parallel_edges=True, create_using=nx.MultiDiGraph
        )


@dataclass(slots=True)
class LocalOperator(Protocol):
    lattice: SquareLattice
    site: Site
    _mask: int = field(default=None, repr=False)

    def link_index(self) -> npt.NDArray[np.int64]:
        ...

    def __matmul__(self, basis: ComputationBasis) -> npt.NDArray[np.int64]:
        ...

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[np.int64] | sp.spmatrix[np.int64]:
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
            self._mask = int(np.sum(2. ** (self.lattice.n_links - 1 - self.link_index())))

    def link_index(self) -> npt.NDArray[np.int64]:
        return np.array(
            [
                self.lattice.site_index(self.site),
                self.lattice.site_index(self.site) + 1,
                self.lattice.site_index(self.site + UnitVectors().rightward) + 1,
                self.lattice.site_index(self.site + UnitVectors().upward),
            ]
        )

    def flippable(self, basis: ComputationBasis) -> npt.NDArray[np.bool_]:
        b1, b2, b3, b4 = (basis.links[:, idx] for idx in self.link_index())
        return ((b1 & ~b2 & b3 & ~b4) | (~b1 & b2 & ~b3 & b4)).astype(bool)

    def __matmul__(self, basis: ComputationBasis) -> npt.NDArray[np.int64]:
        """

        Args:
            basis:

        Returns:

        """
        if not isinstance(basis, ComputationBasis):
            return NotImplemented
        flipped_states = deepcopy(basis.index)
        flipped_states[self.flippable(basis)] ^= self._mask
        return flipped_states

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[np.int64] | sp.spmatrix[np.int64]:
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
        row_idx = np.arange(basis.n_states)[flippable]
        col_idx = np.argsort(flipped_states)[flippable]
        return sp.csr_array(
            (np.ones(len(row_idx), dtype=int), (row_idx, col_idx)), shape=(basis.n_states, basis.n_states)
        )

    def __pow__(self, power: int) -> Self | Plaquette:
        if power % 2 == 0:
            return Plaquette(self.lattice, self.site, _mask=0)
        return self


@dataclass(slots=True)
class Vertex:  # reserved as LocalOperator for future use
    lattice: SquareLattice
    site: Site
    _mask: int = field(default=None, repr=False)

    @staticmethod
    def order() -> npt.NDArray[np.int64]:
        return np.array([-1, -1, 1, 1])

    def link_index(self) -> npt.NDArray[np.int64]:
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

    def __getitem__(self, basis: ComputationBasis) -> npt.NDArray[np.int64] | sp.spmatrix[np.int64]:
        ...
