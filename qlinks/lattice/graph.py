from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.lattice.types import BoundaryCondition, Link, OrientedLink, Plaquette, Site


@dataclass(frozen=True, slots=True)
class LatticeGraph:
    """
    Pure geometry/topology object.

    This class knows only about sites, links, incidence, adjacency, plaquettes,
    and translations. It knows nothing about physical variables, Gauss laws,
    dimers, spins, Hamiltonians, or constraints.
    """

    sites: tuple[Site, ...]
    links: tuple[Link, ...]
    plaquettes: tuple[Plaquette, ...] = ()
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN

    # Optional map:
    #     (site_id, displacement_tuple) -> translated_site_id
    #
    # Example:
    #     translations[(site, (1, 0))] gives T_x site
    translations: Mapping[tuple[int, tuple[int, ...]], int] = field(default_factory=dict)

    _primitive_vectors: ClassVar[tuple[npt.NDArray[np.float64], ...] | None] = None
    _basis_offsets: ClassVar[tuple[npt.NDArray[np.float64], ...] | None] = None

    def __post_init__(self) -> None:
        bc = BoundaryCondition(self.boundary_condition)
        object.__setattr__(self, "boundary_condition", bc)

        self._validate_sites()
        self._validate_links()
        self._validate_plaquettes()
        self._validate_translations()

    @property
    def ndim(self) -> int:
        return len(self.sites[0].cell)

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def num_plaquettes(self) -> int:
        return len(self.plaquettes)

    @property
    def site_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.num_sites, dtype=np.int64)

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.num_links, dtype=np.int64)

    @property
    def plaquette_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.num_plaquettes, dtype=np.int64)

    @property
    def link_endpoints(self) -> npt.NDArray[np.int64]:
        """
        Array of shape (num_links, 2), with columns [source, target].
        """
        return np.asarray([(link.source, link.target) for link in self.links], dtype=np.int64)

    @property
    def site_cells(self) -> npt.NDArray[np.int64]:
        return np.asarray([site.cell for site in self.sites], dtype=np.int64)

    @property
    def site_positions(self) -> npt.NDArray[np.float64]:
        if all(site.position for site in self.sites):
            return np.asarray([site.position for site in self.sites], dtype=np.float64)

        return self.site_cells.astype(np.float64)

    @property
    def primitive_vectors(self) -> tuple[npt.NDArray[np.float64], ...]:
        if self._primitive_vectors is None:
            raise NotImplementedError(f"{type(self).__name__} must define _primitive_vectors.")
        return self._primitive_vectors

    @property
    def basis_offsets(self) -> tuple[npt.NDArray[np.float64], ...]:
        if self._basis_offsets is None:
            raise NotImplementedError(f"{type(self).__name__} must define _basis_offsets.")
        return self._basis_offsets

    def embedded_position(
        self,
        cell: tuple[int, ...],
        sublattice: int = 0,
    ) -> tuple[float, ...]:
        if len(cell) != len(self.primitive_vectors):
            raise ValueError(
                f"cell has dimension {len(cell)}, but lattice has "
                f"{len(self.primitive_vectors)} primitive vectors."
            )

        if sublattice < 0 or sublattice >= len(self.basis_offsets):
            raise IndexError(f"Invalid sublattice index: {sublattice}")

        pos = np.zeros_like(
            np.asarray(self.primitive_vectors[0], dtype=float),
            dtype=float,
        )

        for coordinate, vector in zip(cell, self.primitive_vectors, strict=True):
            pos = pos + int(coordinate) * np.asarray(vector, dtype=float)

        pos = pos + np.asarray(self.basis_offsets[int(sublattice)], dtype=float)

        return tuple(float(x) for x in pos)

    def site_embedded_position(self, site_id: int) -> tuple[float, ...]:
        site = self.sites[int(site_id)]
        return self.embedded_position(
            tuple(int(c) for c in site.cell),
            int(site.sublattice),
        )

    def incidence_matrix(self) -> sp.csr_array:
        """
        Return the oriented site-link incidence matrix B.

        Convention:

            B[source, link] = -1
            B[target, link] = +1

        This is the natural convention for divergence-like constraints.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []

        for link in self.links:
            rows.extend([link.source, link.target])
            cols.extend([link.id, link.id])
            data.extend([-1, +1])

        return sp.coo_array(
            (data, (rows, cols)),
            shape=(self.num_sites, self.num_links),
            dtype=np.int8,
        ).tocsr()

    def unoriented_adjacency_matrix(self) -> sp.csr_array:
        """
        Return symmetric site-site adjacency matrix.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []

        for link in self.links:
            rows.extend([link.source, link.target])
            cols.extend([link.target, link.source])
            data.extend([1, 1])

        return sp.coo_array(
            (data, (rows, cols)),
            shape=(self.num_sites, self.num_sites),
            dtype=np.int8,
        ).tocsr()

    def incident_links(self, site_id: int) -> npt.NDArray[np.int64]:
        self._validate_site_id(site_id)

        out = [link.id for link in self.links if link.source == site_id or link.target == site_id]
        return np.asarray(out, dtype=np.int64)

    def incoming_links(self, site_id: int) -> npt.NDArray[np.int64]:
        self._validate_site_id(site_id)

        out = [link.id for link in self.links if link.target == site_id]
        return np.asarray(out, dtype=np.int64)

    def outgoing_links(self, site_id: int) -> npt.NDArray[np.int64]:
        self._validate_site_id(site_id)

        out = [link.id for link in self.links if link.source == site_id]
        return np.asarray(out, dtype=np.int64)

    def neighbors(self, site_id: int) -> npt.NDArray[np.int64]:
        self._validate_site_id(site_id)

        out: set[int] = set()
        for link in self.links:
            if link.source == site_id:
                out.add(link.target)
            elif link.target == site_id:
                out.add(link.source)

        return np.asarray(sorted(out), dtype=np.int64)

    def plaquette_links(self, plaquette_id: int) -> npt.NDArray[np.int64]:
        self._validate_plaquette_id(plaquette_id)
        return np.asarray(self.plaquettes[plaquette_id].links, dtype=np.int64)

    def plaquette_orientations(self, plaquette_id: int) -> npt.NDArray[np.int64]:
        self._validate_plaquette_id(plaquette_id)
        return np.asarray(self.plaquettes[plaquette_id].orientations, dtype=np.int64)

    def plaquette_boundary(self, plaquette_id: int) -> tuple[OrientedLink, ...]:
        """Return the oriented boundary of a plaquette."""
        return self.plaquettes[int(plaquette_id)].boundary

    def plaquette_sites(self, plaquette_id: int) -> npt.NDArray[np.int64]:
        self._validate_plaquette_id(plaquette_id)
        return np.asarray(self.plaquettes[plaquette_id].sites, dtype=np.int64)

    def plaquette_incidence_matrix(self) -> sp.csr_array:
        """Return oriented link-plaquette incidence matrix.

        The matrix has shape ``(num_links, num_plaquettes)`` and entries

            B[link, plaquette] = +1 or -1

        depending on the orientation of the link in the plaquette boundary.
        """
        row_indices: list[int] = []
        column_indices: list[int] = []
        data_values: list[int] = []

        for plaquette in self.plaquettes:
            for oriented_link in plaquette.boundary:
                row_indices.append(int(oriented_link.link_id))
                column_indices.append(int(plaquette.id))
                data_values.append(int(oriented_link.orientation))

        return sp.coo_array(
            (data_values, (row_indices, column_indices)),
            shape=(self.num_links, self.num_plaquettes),
            dtype=np.int8,
        ).tocsr()

    def plaquette_anchor_cell(self, plaquette_id: int) -> tuple[int, ...]:
        self._validate_plaquette_id(plaquette_id)

        anchor_cell = self.plaquettes[int(plaquette_id)].anchor_cell
        if not anchor_cell:
            raise ValueError(f"Plaquette {plaquette_id} does not define anchor_cell.")

        return tuple(int(c) for c in anchor_cell)

    def plaquette_id_from_anchor(
        self,
        cell: tuple[int, ...],
        *,
        kind: str | None = None,
    ) -> int:
        canonical_cell = self.canonical_cell(cell)

        matches = [
            plaquette.id
            for plaquette in self.plaquettes
            if plaquette.anchor_cell == canonical_cell and (kind is None or plaquette.kind == kind)
        ]

        if len(matches) == 1:
            return int(matches[0])

        if not matches:
            kind_msg = "" if kind is None else f" with kind={kind!r}"
            raise KeyError(f"No plaquette anchored at cell={canonical_cell}{kind_msg}.")

        kinds = [self.plaquettes[int(pid)].kind for pid in matches]
        raise ValueError(
            "Multiple plaquettes match "
            f"cell={canonical_cell}; specify kind. "
            f"Matching kinds: {kinds}"
        )

    def canonical_cell(self, cell: tuple[int, ...]) -> tuple[int, ...]:
        if len(cell) != self.ndim:
            raise ValueError(f"Expected cell dimension {self.ndim}, got {len(cell)}.")

        return tuple(int(c) for c in cell)

    def translate_site(self, site_id: int, displacement: tuple[int, ...]) -> int | None:
        """
        Translate a site by an integer lattice displacement.

        Returns None if the translation is not defined, e.g. for an open-boundary
        site translated out of the system.
        """
        self._validate_site_id(site_id)

        if len(displacement) != self.ndim:
            raise ValueError(
                f"Expected displacement dimension {self.ndim}, got {len(displacement)}."
            )

        return self.translations.get((site_id, tuple(displacement)))

    def oriented_link_between(self, source: int, target: int) -> tuple[int, int]:
        """
        Return (link_id, orientation) for a directed traversal source -> target.

        orientation = +1 means traversal agrees with stored link orientation.
        orientation = -1 means traversal is opposite to stored link orientation.
        """
        self._validate_site_id(source)
        self._validate_site_id(target)

        for link in self.links:
            if link.source == source and link.target == target:
                return link.id, +1
            if link.source == target and link.target == source:
                return link.id, -1

        raise KeyError(f"No link found between site {source} and site {target}.")

    def as_metadata(self) -> dict[str, object]:
        return {
            "ndim": self.ndim,
            "num_sites": self.num_sites,
            "num_links": self.num_links,
            "num_plaquettes": self.num_plaquettes,
            "boundary_condition": self.boundary_condition.value,
            "sites": [
                {
                    "id": site.id,
                    "cell": site.cell,
                    "sublattice": site.sublattice,
                    "position": site.position,
                }
                for site in self.sites
            ],
            "links": [
                {
                    "id": link.id,
                    "source": link.source,
                    "target": link.target,
                    "kind": link.kind,
                    "wrap": link.wrap,
                }
                for link in self.links
            ],
            "plaquettes": [
                {
                    "id": plaq.id,
                    "links": plaq.links,
                    "orientations": plaq.orientations,
                    "sites": plaq.sites,
                    "kind": plaq.kind,
                }
                for plaq in self.plaquettes
            ],
        }

    def _validate_sites(self) -> None:
        if len(self.sites) == 0:
            raise ValueError("LatticeGraph must contain at least one site.")

        ids = [site.id for site in self.sites]
        expected = list(range(len(self.sites)))

        if ids != expected:
            raise ValueError(
                "Site ids must be consecutive and ordered as 0, 1, ..., num_sites - 1."
            )

        ndim = len(self.sites[0].cell)
        if any(len(site.cell) != ndim for site in self.sites):
            raise ValueError("All sites must have the same cell-coordinate dimension.")

    def _validate_links(self) -> None:
        ids = [link.id for link in self.links]
        expected = list(range(len(self.links)))

        if ids != expected:
            raise ValueError(
                "Link ids must be consecutive and ordered as 0, 1, ..., num_links - 1."
            )

        for link in self.links:
            self._validate_site_id(link.source)
            self._validate_site_id(link.target)

    def _validate_plaquettes(self) -> None:
        ids = [plaq.id for plaq in self.plaquettes]
        expected = list(range(len(self.plaquettes)))

        if ids != expected:
            raise ValueError(
                "Plaquette ids must be consecutive and ordered as " "0, 1, ..., num_plaquettes - 1."
            )

        for plaq in self.plaquettes:
            for link_id in plaq.links:
                self._validate_link_id(link_id)

            for site_id in plaq.sites:
                self._validate_site_id(site_id)

    def _validate_translations(self) -> None:
        for (site_id, displacement), translated in self.translations.items():
            self._validate_site_id(site_id)
            self._validate_site_id(translated)

            if len(displacement) != self.ndim:
                raise ValueError(
                    f"Translation displacement {displacement} has dimension "
                    f"{len(displacement)}, expected {self.ndim}."
                )

    def _validate_site_id(self, site_id: int) -> None:
        if site_id < 0 or site_id >= self.num_sites:
            raise IndexError(f"site_id {site_id} outside valid range [0, {self.num_sites}).")

    def _validate_link_id(self, link_id: int) -> None:
        if link_id < 0 or link_id >= self.num_links:
            raise IndexError(f"link_id {link_id} outside valid range [0, {self.num_links}).")

    def _validate_plaquette_id(self, plaquette_id: int) -> None:
        if plaquette_id < 0 or plaquette_id >= self.num_plaquettes:
            raise IndexError(
                f"plaquette_id {plaquette_id} outside valid range [0, {self.num_plaquettes})."
            )
