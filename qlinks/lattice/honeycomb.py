from __future__ import annotations

from typing import ClassVar

import numpy as np

from qlinks.lattice.graph import LatticeGraph
from qlinks.lattice.types import BoundaryCondition, Link, Plaquette, Site


class HoneycombLattice(LatticeGraph):
    """
    Honeycomb lattice in a brick-wall representation.

    Unit cell:
        A(x, y), sublattice 0
        B(x, y), sublattice 1

    Site id:
        site_id(x, y, sublattice) = 2 * (x * ly + y) + sublattice

    Links:
        z: A(x, y) -> B(x, y)
        x: A(x, y) -> B(x - 1, y)
        y: A(x, y) -> B(x, y - 1)

    Hexagon plaquette around cell (x, y):
        A(x,y)
        B(x,y)
        A(x+1,y)
        B(x+1,y-1)
        A(x+1,y-1)
        B(x,y-1)

    This representation is convenient for QDM/QLM hexagon ring exchange.
    """

    __slots__ = ("lx", "ly")

    _sqrt3: ClassVar[float] = float(np.sqrt(3.0))

    _primitive_vectors: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([-_sqrt3 / 2.0, 1.5], dtype=float),
        np.array([+_sqrt3 / 2.0, 1.5], dtype=float),
    )

    _basis_offsets: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([0.0, 0.0], dtype=float),  # A
        np.array([0.0, 1.0], dtype=float),  # B
    )

    def __init__(
        self,
        lx: int,
        ly: int,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
    ) -> None:
        if lx <= 0:
            raise ValueError("lx must be positive.")
        if ly <= 0:
            raise ValueError("ly must be positive.")

        bc = BoundaryCondition(boundary_condition)

        def site_id(x: int, y: int, sublattice: int) -> int:
            return 2 * (x * ly + y) + sublattice

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < lx and 0 <= y < ly

        def wrap_coord(x: int, y: int) -> tuple[int, int]:
            return x % lx, y % ly

        def maybe_site_id(x: int, y: int, sublattice: int) -> int | None:
            if bc == BoundaryCondition.PERIODIC:
                wx, wy = wrap_coord(x, y)
                return site_id(wx, wy, sublattice)

            if in_bounds(x, y):
                return site_id(x, y, sublattice)

            return None

        def position(x: int, y: int, sublattice: int) -> tuple[float, float]:
            # Brick-wall embedding.
            base_x = 1.5 * float(x)
            base_y = (3.0**0.5) * (float(y) + 0.5 * float(x))

            if sublattice == 0:
                return (base_x, base_y)

            return (base_x + 0.5, base_y + 0.5 / (3.0**0.5))

        sites = tuple(
            Site(
                id=site_id(x, y, sub),
                cell=(x, y),
                sublattice=sub,
                position=position(x, y, sub),
            )
            for x in range(lx)
            for y in range(ly)
            for sub in (0, 1)
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

        # A to three neighboring B sites.
        bond_targets = {
            "z": (0, 0),
            "x": (-1, 0),
            "y": (0, -1),
        }

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y, 0)

                for kind, (dx, dy) in bond_targets.items():
                    tx, ty = x + dx, y + dy
                    target = maybe_site_id(tx, ty, 1)

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

        def add_hexagon(x: int, y: int) -> None:
            """
            Add one honeycomb hexagon.

            Link convention:
                A(x, y) -> B(x, y)
                A(x, y) -> B(x - 1, y)
                A(x, y) -> B(x, y - 1)

            A valid hexagon is:

                A(x, y)
                B(x, y)
                A(x + 1, y)
                B(x + 1, y - 1)
                A(x + 1, y - 1)
                B(x, y - 1)
            """

            coords = [
                (x, y, 0),
                (x, y, 1),
                (x + 1, y, 0),
                (x + 1, y - 1, 1),
                (x + 1, y - 1, 0),
                (x, y - 1, 1),
            ]

            site_ids: list[int] = []

            for cx, cy, sub in coords:
                sid = maybe_site_id(cx, cy, sub)
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
                    kind="hexagon",
                )
            )

        if lx >= 2 and ly >= 2:
            if bc == BoundaryCondition.PERIODIC:
                for x in range(lx):
                    for y in range(ly):
                        add_hexagon(x, y)
            else:
                for x in range(lx - 1):
                    for y in range(1, ly):
                        add_hexagon(x, y)

        translations: dict[tuple[int, tuple[int, int]], int] = {}

        for x in range(lx):
            for y in range(ly):
                for sub in (0, 1):
                    source = site_id(x, y, sub)

                    for disp in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        target = maybe_site_id(x + disp[0], y + disp[1], sub)
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

    def site_id(self, x: int, y: int, sublattice: int) -> int:
        if sublattice not in (0, 1):
            raise ValueError("sublattice must be 0 or 1.")

        if self.boundary_condition == BoundaryCondition.PERIODIC:
            x %= self.lx
            y %= self.ly
        elif not (0 <= x < self.lx and 0 <= y < self.ly):
            raise IndexError(f"site coordinate ({x}, {y}) outside lattice.")

        return 2 * (x * self.ly + y) + sublattice

    def qdm_plaquette_ids(self):
        return [plaquette.id for plaquette in self.plaquettes if plaquette.kind == "hexagon"]

    def qlm_plaquette_ids(self):
        return self.qdm_plaquette_ids()
