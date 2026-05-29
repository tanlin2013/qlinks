from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

SiteId: TypeAlias = int
LinkId: TypeAlias = int
PlaquetteId: TypeAlias = int
CellCoord: TypeAlias = tuple[int, ...]
Position: TypeAlias = tuple[float, ...]


class BoundaryCondition(StrEnum):
    OPEN = "open"
    PERIODIC = "periodic"


@dataclass(frozen=True, slots=True)
class Site:
    """
    Geometry-level site metadata.

    id:
        Consecutive integer site id.

    cell:
        Unit-cell coordinate, e.g. (x,), (x, y), etc.

    sublattice:
        Sublattice label. For Bravais lattices, this can be 0.

    position:
        Real-space embedding coordinate. This is for geometry/debugging/plotting.
        It should not be used as the primary index in performance-sensitive code.
    """

    id: SiteId
    cell: CellCoord
    sublattice: int = 0
    position: Position = ()

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Site.id must be non-negative.")

        if len(self.cell) == 0:
            raise ValueError("Site.cell cannot be empty.")

        if self.sublattice < 0:
            raise ValueError("Site.sublattice must be non-negative.")

        if self.position and len(self.position) != len(self.cell):
            raise ValueError("Site.position must have the same dimension as Site.cell.")


@dataclass(frozen=True, slots=True)
class Link:
    """
    Oriented link metadata.

    The link orientation is source -> target.

    This orientation defines the sign convention in the incidence matrix:

        incidence[source, link] = -1
        incidence[target, link] = +1

    kind:
        A geometry-dependent link type, e.g. "x", "y", "diag", etc.

    wrap:
        True if this link crosses a periodic boundary.
    """

    id: LinkId
    source: SiteId
    target: SiteId
    kind: str = ""
    wrap: bool = False

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Link.id must be non-negative.")

        if self.source < 0 or self.target < 0:
            raise ValueError("Link endpoints must be non-negative.")

        if self.source == self.target:
            raise ValueError("Self-links are not allowed.")


@dataclass(frozen=True, slots=True)
class OrientedLink:
    """A link with an orientation relative to a plaquette boundary.

    orientation = +1 means the plaquette traverses the link along its stored
    source -> target direction.

    orientation = -1 means the plaquette traverses the link opposite to its
    stored source -> target direction.
    """

    link_id: LinkId
    orientation: int

    def __post_init__(self) -> None:
        if self.orientation not in (-1, 1):
            raise ValueError("OrientedLink.orientation must be either -1 or +1.")


@dataclass(frozen=True, slots=True)
class Plaquette:
    """
    Oriented elementary loop.

    links:
        Link ids around the loop.

    orientations:
        For each link in the loop:
            +1 if traversed along the stored link orientation,
            -1 if traversed opposite to the stored link orientation.

    sites:
        Site ids around the loop. This is metadata useful for debugging,
        plotting, and later local operator construction.

    kind: Geometry-dependent plaquette type.

    anchor_cell: Unit-cell coordinate used to label this plaquette.
    """

    id: PlaquetteId
    links: tuple[LinkId, ...]
    orientations: tuple[int, ...]
    sites: tuple[SiteId, ...]
    kind: str = ""
    anchor_cell: CellCoord = ()

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Plaquette.id must be non-negative.")

        if len(self.links) == 0:
            raise ValueError("Plaquette.links cannot be empty.")

        if len(self.links) != len(self.orientations):
            raise ValueError("Plaquette.links and Plaquette.orientations must have equal length.")

        if len(self.sites) < 3:
            raise ValueError("Plaquette.sites must contain at least three sites.")

        bad = [ori for ori in self.orientations if ori not in (-1, 1)]
        if bad:
            raise ValueError("Plaquette.orientations must only contain +1 or -1.")

        object.__setattr__(
            self,
            "anchor_cell",
            tuple(int(c) for c in self.anchor_cell),
        )

    @property
    def boundary(self) -> tuple[OrientedLink, ...]:
        """Return the oriented boundary links of this plaquette."""
        return tuple(
            OrientedLink(link_id=link_id, orientation=orientation)
            for link_id, orientation in zip(
                self.links,
                self.orientations,
                strict=True,
            )
        )
