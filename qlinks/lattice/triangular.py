from __future__ import annotations

from typing import ClassVar

import numpy as np

from qlinks.lattice.graph import LatticeGraph
from qlinks.lattice.types import BoundaryCondition, Link, Plaquette, Site


class TriangularLattice(LatticeGraph):
    """
    2D triangular lattice.

    Site convention:
        site_id(x, y) = x * ly + y

    Primitive vectors:
        a1 = (1, 0)
        a2 = (1/2, sqrt(3)/2)

    Link directions:
        a: (x, y) -> (x + 1, y)
        b: (x, y) -> (x, y + 1)
        c: (x, y) -> (x - 1, y + 1)

    Plaquettes:
        triangle_up / triangle_down:
            elementary triangular loops.

        rhombus_ab / rhombus_bc / rhombus_ca:
            four-link lozenge loops. These are the natural QDM resonance
            plaquettes on triangular lattices.
    """

    __slots__ = ("lx", "ly")

    _primitive_vectors: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([1.0, 0.0], dtype=float),
        np.array([0.5, np.sqrt(3.0) / 2.0], dtype=float),
    )

    _basis_offsets: ClassVar[tuple[np.ndarray, ...]] = (np.array([0.0, 0.0], dtype=float),)

    def __init__(
        self,
        lx: int,
        ly: int,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        *,
        include_triangles: bool = True,
        include_rhombi: bool = True,
    ) -> None:
        if lx <= 0:
            raise ValueError("lx must be positive.")
        if ly <= 0:
            raise ValueError("ly must be positive.")

        bc = BoundaryCondition(boundary_condition)

        def site_id(x: int, y: int) -> int:
            return x * ly + y

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < lx and 0 <= y < ly

        def wrap_coord(x: int, y: int) -> tuple[int, int]:
            return x % lx, y % ly

        def maybe_site_id(x: int, y: int) -> int | None:
            if bc == BoundaryCondition.PERIODIC:
                wx, wy = wrap_coord(x, y)
                return site_id(wx, wy)

            if in_bounds(x, y):
                return site_id(x, y)

            return None

        def position(x: int, y: int) -> tuple[float, float]:
            return (float(x) + 0.5 * float(y), (3.0**0.5 / 2.0) * float(y))

        sites = tuple(
            Site(
                id=site_id(x, y),
                cell=(x, y),
                sublattice=0,
                position=position(x, y),
            )
            for x in range(lx)
            for y in range(ly)
        )

        links: list[Link] = []
        directed_link_lookup: dict[tuple[int, int], int] = {}

        def add_link(source: int, target: int, kind: str, wrap: bool) -> None:
            if source == target:
                return

            if (source, target) in directed_link_lookup:
                return

            link_id = len(links)
            links.append(
                Link(
                    id=link_id,
                    source=source,
                    target=target,
                    kind=kind,
                    wrap=wrap,
                )
            )
            directed_link_lookup[(source, target)] = link_id

        # Directions in cell coordinates.
        directions = {
            "a": (1, 0),
            "b": (0, 1),
            "c": (-1, 1),
        }

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y)

                for kind, (dx, dy) in directions.items():
                    tx, ty = x + dx, y + dy
                    target = maybe_site_id(tx, ty)

                    if target is None:
                        continue

                    crosses = not in_bounds(tx, ty)
                    if bc == BoundaryCondition.PERIODIC or not crosses:
                        add_link(source, target, kind=kind, wrap=crosses)

        def oriented_link_between(source: int, target: int) -> tuple[int, int]:
            direct = directed_link_lookup.get((source, target))
            if direct is not None:
                return direct, +1

            reverse = directed_link_lookup.get((target, source))
            if reverse is not None:
                return reverse, -1

            raise KeyError(f"No link between {source} and {target}.")

        plaquettes: list[Plaquette] = []

        def add_loop(kind: str, coords: list[tuple[int, int]]) -> None:
            site_ids: list[int] = []

            for x, y in coords:
                sid = maybe_site_id(x, y)
                if sid is None:
                    return
                site_ids.append(sid)

            if len(set(site_ids)) != len(site_ids):
                return

            loop_links: list[int] = []
            orientations: list[int] = []

            for i in range(len(site_ids)):
                s0 = site_ids[i]
                s1 = site_ids[(i + 1) % len(site_ids)]

                try:
                    link_id, orientation = oriented_link_between(s0, s1)
                except KeyError:
                    return

                loop_links.append(link_id)
                orientations.append(orientation)

            plaquettes.append(
                Plaquette(
                    id=len(plaquettes),
                    links=tuple(loop_links),
                    orientations=tuple(orientations),
                    sites=tuple(site_ids),
                    kind=kind,
                )
            )

        max_x = lx if bc == BoundaryCondition.PERIODIC else lx
        max_y = ly if bc == BoundaryCondition.PERIODIC else ly

        if include_triangles:
            for x in range(max_x):
                for y in range(max_y):
                    # Up triangle:
                    # (x,y) -> (x+1,y) -> (x,y+1)
                    add_loop(
                        "triangle_up",
                        [
                            (x, y),
                            (x + 1, y),
                            (x, y + 1),
                        ],
                    )

                    # Down triangle:
                    # (x,y) -> (x+1,y-1) -> (x+1,y)
                    add_loop(
                        "triangle_down",
                        [
                            (x, y),
                            (x + 1, y - 1),
                            (x + 1, y),
                        ],
                    )

        if include_rhombi:
            rhombus_pairs = {
                "rhombus_ab": ((1, 0), (0, 1)),
                "rhombus_bc": ((0, 1), (-1, 1)),
                "rhombus_ca": ((-1, 1), (-1, 0)),
            }

            for x in range(max_x):
                for y in range(max_y):
                    for kind, (d1, d2) in rhombus_pairs.items():
                        x0, y0 = x, y
                        x1, y1 = x0 + d1[0], y0 + d1[1]
                        x2, y2 = x1 + d2[0], y1 + d2[1]
                        x3, y3 = x0 + d2[0], y0 + d2[1]

                        add_loop(
                            kind,
                            [
                                (x0, y0),
                                (x1, y1),
                                (x2, y2),
                                (x3, y3),
                            ],
                        )

        translations: dict[tuple[int, tuple[int, int]], int] = {}

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y)

                for disp in ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)):
                    target = maybe_site_id(x + disp[0], y + disp[1])
                    if target is not None:
                        translations[(source, disp)] = target

        super().__init__(
            sites=sites,
            links=tuple(links),
            plaquettes=tuple(plaquettes),
            boundary_condition=bc,
            translations=translations,
        )

        object.__setattr__(self, "lx", lx)
        object.__setattr__(self, "ly", ly)

    def site_id(self, x: int, y: int) -> int:
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            x %= self.lx
            y %= self.ly
        elif not (0 <= x < self.lx and 0 <= y < self.ly):
            raise IndexError(f"site coordinate ({x}, {y}) outside lattice.")

        return x * self.ly + y

    def qdm_plaquette_ids(self):
        return [
            plaquette.id for plaquette in self.plaquettes if plaquette.kind.startswith("rhombus_")
        ]

    def qlm_plaquette_ids(self, *, use_triangles: bool = False):
        if use_triangles:
            return [
                plaquette.id
                for plaquette in self.plaquettes
                if plaquette.kind.startswith("triangle_")
            ]

        return self.qdm_plaquette_ids()
