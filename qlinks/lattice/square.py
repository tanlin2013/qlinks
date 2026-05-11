from __future__ import annotations

from typing import ClassVar

import numpy as np

from qlinks.lattice.graph import LatticeGraph
from qlinks.lattice.types import BoundaryCondition, Link, Plaquette, Site


class SquareLattice(LatticeGraph):
    """
    Two-dimensional square lattice.

    Site id convention:

        site_id(x, y) = x * Ly + y

    Link orientation convention:

        x-link: (x, y) -> (x + 1, y)
        y-link: (x, y) -> (x, y + 1)

    Plaquette orientation convention:

        counter-clockwise loop:
            bottom edge: +x
            right edge: +y
            top edge: -x
            left edge: -y
    """

    __slots__ = ("lx", "ly")

    _primitive_vectors: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    )

    _basis_offsets: ClassVar[tuple[np.ndarray, ...]] = (np.array([0.0, 0.0], dtype=float),)

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

        sites = tuple(
            Site(
                id=site_id(x, y),
                cell=(x, y),
                sublattice=0,
                position=(float(x), float(y)),
            )
            for x in range(lx)
            for y in range(ly)
        )

        links: list[Link] = []
        directed_link_lookup: dict[tuple[int, int], int] = {}

        def add_link(source: int, target: int, kind: str, wrap: bool) -> None:
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

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y)

                # +x link
                target_x = maybe_site_id(x + 1, y)
                if target_x is not None and target_x != source:
                    crosses = x + 1 >= lx
                    if bc == BoundaryCondition.PERIODIC or not crosses:
                        add_link(source, target_x, kind="x", wrap=crosses)

                # +y link
                target_y = maybe_site_id(x, y + 1)
                if target_y is not None and target_y != source:
                    crosses = y + 1 >= ly
                    if bc == BoundaryCondition.PERIODIC or not crosses:
                        add_link(source, target_y, kind="y", wrap=crosses)

        def oriented_link_between(source: int, target: int) -> tuple[int, int]:
            direct = directed_link_lookup.get((source, target))
            if direct is not None:
                return direct, +1

            reverse = directed_link_lookup.get((target, source))
            if reverse is not None:
                return reverse, -1

            raise KeyError(f"No link between {source} and {target}.")

        plaquettes: list[Plaquette] = []

        max_x = lx if bc == BoundaryCondition.PERIODIC else lx - 1
        max_y = ly if bc == BoundaryCondition.PERIODIC else ly - 1

        if lx >= 2 and ly >= 2:
            for x in range(max_x):
                for y in range(max_y):
                    s00 = maybe_site_id(x, y)
                    s10 = maybe_site_id(x + 1, y)
                    s11 = maybe_site_id(x + 1, y + 1)
                    s01 = maybe_site_id(x, y + 1)

                    if None in (s00, s10, s11, s01):
                        continue

                    assert s00 is not None
                    assert s10 is not None
                    assert s11 is not None
                    assert s01 is not None

                    loop_sites = (s00, s10, s11, s01)

                    l0, o0 = oriented_link_between(s00, s10)
                    l1, o1 = oriented_link_between(s10, s11)
                    l2, o2 = oriented_link_between(s11, s01)
                    l3, o3 = oriented_link_between(s01, s00)

                    plaquettes.append(
                        Plaquette(
                            id=len(plaquettes),
                            links=(l0, l1, l2, l3),
                            orientations=(o0, o1, o2, o3),
                            sites=loop_sites,
                            kind="square",
                        )
                    )

        translations: dict[tuple[int, tuple[int, int]], int] = {}

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y)

                for disp in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    tx = x + disp[0]
                    ty = y + disp[1]

                    target = maybe_site_id(tx, ty)
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
