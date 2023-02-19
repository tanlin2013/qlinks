from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import astuple, dataclass, field
from functools import total_ordering
from itertools import product
from typing import Dict, Iterator, List, Optional, Self, Sequence, Tuple, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import NDArray

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
    _links: Dict[LinkIndex, Link] = field(init=False, repr=False)

    def __post_init__(self):
        if any(axis < 2 for axis in self.shape):
            raise InvalidArgumentError("Lattice size should be least 2 by 2.")
        self._links = {link.index: link for link in self.iter_links()}

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
        return self._links

    def get_link(self, link_index: LinkIndex) -> Link:
        site, unit_vector = link_index
        if unit_vector.sign < 0:
            site += unit_vector
            unit_vector *= -1
        return self._links[(self[site], unit_vector)]

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
        def flux(link: Link) -> Real:
            if link.state is not None:
                return link.flux if link.site == self[site] else -1 * link.flux
            else:
                return np.nan

        return sum([flux(link) for link in self.get_vertex_links(site)])

    def axial_flux(self, idx: int, axis: Optional[int] = 0) -> Real:
        """Compute the flux along the axis.

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
            axis: 0 for x-axis and 1 for y-axis.

        Returns:
            The total flux flowing along the axis.
        """
        fluxes = {
            0: (
                self.get_link((Site(idx, y), UnitVectors.rightward)).flux for y in range(self.width)
            ),
            1: (self.get_link((Site(x, idx), UnitVectors.upward)).flux for x in range(self.length)),
        }[axis]
        return np.sum(np.fromiter(fluxes, dtype=float))

    @property
    def adjacency_matrix(self) -> np.ndarray:
        adj_mat = np.zeros((self.size, self.size))
        hash_table = {site: idx for idx, site in enumerate(self)}
        for site, unit_vector in product(self, UnitVectors.iter_all_directions()):
            inds = (hash_table[site], hash_table[self[site + unit_vector]])  # head to tail
            link = self.get_link((site, unit_vector))
            if unit_vector.sign * link.flux > 0:
                adj_mat[*inds] += 1
            else:
                adj_mat[*inds[::-1]] += 1
        return (adj_mat / 2).astype(int)

    def as_graph(self) -> nx.MultiDiGraph:
        return nx.from_numpy_array(
            self.adjacency_matrix, parallel_edges=True, create_using=nx.MultiDiGraph
        )


@total_ordering
@dataclass
class LatticeState(SquareLattice):
    link_data: Dict[LinkIndex, Link]

    def __post_init__(self):
        self._links = {idx: self.link_data[idx] for idx in sorted(self.link_data)}
        for link in self.links.values():
            if link.state is None:
                raise InvalidArgumentError("Provided link data has state in None.")

    def toarray(self) -> np.ndarray:
        states = (link.state for link in self.links.values())
        return np.asarray(np.bitwise_xor.reduce(np.fromiter(states, dtype=object)))

    def __hash__(self) -> int:
        return hash(frozenset(self.links.values()))

    def __lt__(self, other: LatticeState) -> bool:
        return tuple(link.flux for link in self.links.values()) < tuple(
            link.flux for link in other.links.values()
        )

    def __matmul__(self, other: LatticeState) -> Real:
        if not isinstance(other, LatticeState):
            return NotImplemented
        iterable = (
            contra_link.state @ co_link.state
            for contra_link, co_link in zip(self.links.values(), other.links.values())
        )
        return np.product(np.fromiter(iterable, dtype=float, count=self.num_links))

    @property
    def T(self) -> Self:  # noqa: N802
        transpose_state = deepcopy(self)
        for idx, link in transpose_state.links.items():
            transpose_state._links[idx].state = link.state.T
        return transpose_state


@dataclass
class LatticeMultiStates(SquareLattice):
    states: Sequence[LatticeState] | NDArray

    def __post_init__(self):
        if not isinstance(self.states, np.ndarray):
            self.states = np.asarray(self.states, dtype=object)

    def __matmul__(
        self, other: QuasiLocalOperator | LatticeMultiStates
    ) -> LatticeMultiStates | SpinOperator:
        if isinstance(other, QuasiLocalOperator):
            new_states = [state @ other for state in self.states]
            return LatticeMultiStates(*self.shape, states=new_states)
        elif isinstance(other, LatticeMultiStates):
            return self | other
        else:
            return NotImplemented

    def __rmatmul__(
        self, other: QuasiLocalOperator | LatticeMultiStates
    ) -> LatticeMultiStates | SpinOperator:
        if isinstance(other, QuasiLocalOperator):
            new_states = [other @ state for state in self.states]
            return LatticeMultiStates(*self.shape, states=new_states)
        elif isinstance(other, LatticeMultiStates):
            return self | other
        else:
            return NotImplemented

    def __or__(self, other: LatticeMultiStates) -> SpinOperator:
        cartesian_prod = np.array(
            np.meshgrid(self.states, other.states)  # type: ignore[arg-type]
        ).T.reshape(-1, 2)
        iterable = (
            fore_state @ post_state for fore_state, post_state in cartesian_prod[:, ]  # fmt: skip
        )
        return (
            np.fromiter(iterable, dtype=float, count=np.prod(self.hilbert_dims))
            .reshape(self.hilbert_dims)
            .view(SpinOperator)
        )

    @property
    def hilbert_dims(self) -> Tuple[int, int]:
        n_states = len(self.states)
        return n_states, n_states

    @property
    def T(self) -> Self:  # noqa: N802
        transpose = deepcopy(self)
        transpose.states = [state.T for state in self.states]
        return transpose


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

    def toarray(self) -> np.ndarray:
        quasi_loc_opt = [link for link in self]
        skip_links = [link.reset() for link in self]
        quasi_loc_opt += [
            identity_link
            for identity_link in self.lattice.iter_links()
            if identity_link not in skip_links
        ]
        operators = (link.operator for link in sorted(quasi_loc_opt))
        return np.asarray(
            np.bitwise_xor.reduce(
                np.fromiter(operators, dtype=object, count=self.lattice.num_links)
            )
        )

    def __iter__(self) -> Iterator[Link]:
        return iter(sorted((self.link_d, self.link_l, self.link_r, self.link_t)))

    def __add__(self, other: QuasiLocalOperator) -> SpinOperator:
        if self.site != other.site:
            raise InvalidOperationError(
                f"{type(self).__name__} in different positions can not be directly added."
            )
        return (self.toarray() + other.toarray()).view(SpinOperator)

    def __mul__(self, other: QuasiLocalOperator) -> QuasiLocalOperator:
        if not isinstance(other, QuasiLocalOperator):
            return NotImplemented
        quasi_loc_opt = deepcopy(self)
        for fore_link, post_link in zip(quasi_loc_opt, other):
            fore_link.operator = (fore_link.operator @ post_link.operator).view(SpinOperator)
        return quasi_loc_opt

    def _get_extended_loc_opt(self) -> Dict[LinkIndex, SpinOperator]:
        quasi_loc_opt = {link.index: link.operator for link in self}
        return {
            link.index: (
                link.operator if link.index not in quasi_loc_opt else quasi_loc_opt[link.index]
            )
            for link in self.lattice.iter_links()
        }

    def __matmul__(self, other: LatticeState) -> LatticeState:
        if not isinstance(other, LatticeState):
            return NotImplemented
        if self.lattice.shape != other.shape:
            raise InvalidOperationError(
                f"Dimension mismatch. Cannot multiply shape {self.lattice.shape} with shape "
                f"{other.shape}."
            )
        link_data = deepcopy(other.links)
        extended_loc_opt = self._get_extended_loc_opt()
        for idx in link_data.keys():
            link_data[idx].state = (extended_loc_opt[idx] @ other.links[idx].state).view(Spin)
        return LatticeState(*other.shape, link_data=link_data)

    def __rmatmul__(self, other: LatticeState) -> LatticeState:
        if not isinstance(other, LatticeState):
            return NotImplemented
        if self.lattice.shape != other.shape:
            raise InvalidOperationError(
                f"Dimension mismatch. Cannot multiply shape {self.lattice.shape} with shape "
                f"{other.shape}."
            )
        link_data = deepcopy(other.links)
        extended_loc_opt = self._get_extended_loc_opt()
        for idx in link_data.keys():
            link_data[idx].state = (other.links[idx].state @ extended_loc_opt[idx]).view(Spin)
        return LatticeState(*other.shape, link_data=link_data)

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
