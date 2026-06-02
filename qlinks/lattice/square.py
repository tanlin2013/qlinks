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
        cell_link_lookup: dict[tuple[int, int, str], int] = {}

        def add_link(
            source: int,
            target: int,
            *,
            cell: tuple[int, int],
            kind: str,
            wrap: bool,
        ) -> None:
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
            cell_link_lookup[(int(cell[0]), int(cell[1]), kind)] = link_id

        for x in range(lx):
            for y in range(ly):
                source = site_id(x, y)

                # +x link anchored at cell (x, y)
                target_x = maybe_site_id(x + 1, y)
                if target_x is not None and target_x != source:
                    crosses = x + 1 >= lx
                    if bc == BoundaryCondition.PERIODIC or not crosses:
                        add_link(
                            source,
                            target_x,
                            cell=(x, y),
                            kind="x",
                            wrap=crosses,
                        )

                # +y link anchored at cell (x, y)
                target_y = maybe_site_id(x, y + 1)
                if target_y is not None and target_y != source:
                    crosses = y + 1 >= ly
                    if bc == BoundaryCondition.PERIODIC or not crosses:
                        add_link(
                            source,
                            target_y,
                            cell=(x, y),
                            kind="y",
                            wrap=crosses,
                        )

        def oriented_link_between(source: int, target: int) -> tuple[int, int]:
            direct = directed_link_lookup.get((source, target))
            if direct is not None:
                return direct, +1

            reverse = directed_link_lookup.get((target, source))
            if reverse is not None:
                return reverse, -1

            raise KeyError(f"No link between {source} and {target}.")

        def canonical_link_cell(x: int, y: int) -> tuple[int, int]:
            if bc == BoundaryCondition.PERIODIC:
                return x % lx, y % ly
            return x, y

        def link_at(x: int, y: int, kind: str) -> int | None:
            cell = canonical_link_cell(x, y)

            if bc != BoundaryCondition.PERIODIC and not in_bounds(cell[0], cell[1]):
                return None

            return cell_link_lookup.get((cell[0], cell[1], kind))

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

                    bottom = link_at(x, y, "x")
                    right = link_at(x + 1, y, "y")
                    top = link_at(x, y + 1, "x")
                    left = link_at(x, y, "y")

                    if None in (bottom, right, top, left):
                        continue

                    assert bottom is not None
                    assert right is not None
                    assert top is not None
                    assert left is not None

                    plaquettes.append(
                        Plaquette(
                            id=len(plaquettes),
                            links=(
                                int(bottom),
                                int(right),
                                int(top),
                                int(left),
                            ),
                            orientations=(+1, +1, -1, -1),
                            sites=loop_sites,
                            kind="square",
                            anchor_cell=(x, y),
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

    def canonical_cell(self, cell: tuple[int, ...]) -> tuple[int, ...]:
        if len(cell) != 2:
            raise ValueError(f"Expected 2D cell coordinate, got {cell!r}.")

        x, y = (int(cell[0]), int(cell[1]))

        if self.boundary_condition == BoundaryCondition.PERIODIC:
            return x % self.lx, y % self.ly

        return x, y

    def plaquette_id_from_cell(self, x: int, y: int) -> int:
        return self.plaquette_id_from_anchor((x, y), kind="square")

    def square_plaquette_id(self, x: int, y: int) -> int:
        return self.plaquette_id_from_cell(x, y)
