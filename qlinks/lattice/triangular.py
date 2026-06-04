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
        cell_link_lookup: dict[tuple[int, int, str], int] = {}

        def add_link(
            source: int,
            target: int,
            *,
            cell: tuple[int, int],
            kind: str,
            wrap: bool,
        ) -> None:
            if source == target:
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
            cell_link_lookup[(int(cell[0]), int(cell[1]), kind)] = link_id

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
                        add_link(
                            source,
                            target,
                            cell=(x, y),
                            kind=kind,
                            wrap=crosses,
                        )

        def canonical_link_cell(x: int, y: int) -> tuple[int, int]:
            if bc == BoundaryCondition.PERIODIC:
                return x % lx, y % ly
            return x, y

        def link_at(x: int, y: int, kind: str) -> int | None:
            cell = canonical_link_cell(x, y)

            if bc != BoundaryCondition.PERIODIC and not in_bounds(cell[0], cell[1]):
                return None

            return cell_link_lookup.get((cell[0], cell[1], kind))

        def oriented_link_between(source: int, target: int) -> tuple[int, int]:
            direct = directed_link_lookup.get((source, target))
            if direct is not None:
                return direct, +1

            reverse = directed_link_lookup.get((target, source))
            if reverse is not None:
                return reverse, -1

            raise KeyError(f"No link between {source} and {target}.")

        plaquettes: list[Plaquette] = []

        def add_cell_loop(
            kind: str,
            *,
            anchor: tuple[int, int],
            site_coords: list[tuple[int, int]],
            boundary: list[tuple[int, int, str, int]],
        ) -> None:
            site_ids: list[int] = []

            for sx, sy in site_coords:
                sid = maybe_site_id(sx, sy)

                if sid is None:
                    return

                site_ids.append(sid)

            if len(set(site_ids)) != len(site_ids):
                return

            loop_links: list[int] = []
            orientations: list[int] = []

            for lx0, ly0, link_kind, orientation in boundary:
                link_id = link_at(lx0, ly0, link_kind)

                if link_id is None:
                    return

                loop_links.append(int(link_id))
                orientations.append(int(orientation))

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
                    # (x,y) -> (x+1,y) -> (x,y+1) -> (x,y)
                    add_cell_loop(
                        "triangle_up",
                        anchor=(x, y),
                        site_coords=[
                            (x, y),
                            (x + 1, y),
                            (x, y + 1),
                        ],
                        boundary=[
                            (x, y, "a", +1),
                            (x + 1, y, "c", +1),
                            (x, y, "b", -1),
                        ],
                    )

                    # Down triangle:
                    # (x,y) -> (x+1,y-1) -> (x+1,y) -> (x,y)
                    add_cell_loop(
                        "triangle_down",
                        anchor=(x, y),
                        site_coords=[
                            (x, y),
                            (x + 1, y - 1),
                            (x + 1, y),
                        ],
                        boundary=[
                            (x + 1, y - 1, "c", -1),
                            (x + 1, y - 1, "b", +1),
                            (x, y, "a", -1),
                        ],
                    )

        if include_rhombi:
            for x in range(max_x):
                for y in range(max_y):
                    # Rhombus ab:
                    add_cell_loop(
                        "rhombus_ab",
                        anchor=(x, y),
                        site_coords=[
                            (x, y),
                            (x + 1, y),
                            (x + 1, y + 1),
                            (x, y + 1),
                        ],
                        boundary=[
                            (x, y, "a", +1),
                            (x + 1, y, "b", +1),
                            (x, y + 1, "a", -1),
                            (x, y, "b", -1),
                        ],
                    )

                    # Rhombus bc:
                    add_cell_loop(
                        "rhombus_bc",
                        anchor=(x, y),
                        site_coords=[
                            (x, y),
                            (x, y + 1),
                            (x - 1, y + 2),
                            (x - 1, y + 1),
                        ],
                        boundary=[
                            (x, y, "b", +1),
                            (x, y + 1, "c", +1),
                            (x - 1, y + 1, "b", -1),
                            (x, y, "c", -1),
                        ],
                    )

                    # Rhombus ca:
                    add_cell_loop(
                        "rhombus_ca",
                        anchor=(x, y),
                        site_coords=[
                            (x, y),
                            (x - 1, y + 1),
                            (x - 2, y + 1),
                            (x - 1, y),
                        ],
                        boundary=[
                            (x, y, "c", +1),
                            (x - 2, y + 1, "a", -1),
                            (x - 1, y, "c", -1),
                            (x - 1, y, "a", +1),
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

    def triangular_plaquette_id(
        self,
        x: int,
        y: int,
        *,
        kind: str,
    ) -> int:
        return self.plaquette_id_from_anchor((x, y), kind=kind)

    def rhombus_plaquette_id(
        self,
        x: int,
        y: int,
        *,
        kind: str,
    ) -> int:
        if kind not in {"rhombus_ab", "rhombus_bc", "rhombus_ca"}:
            raise ValueError(f"Invalid triangular rhombus kind: {kind!r}")

        return self.plaquette_id_from_anchor((x, y), kind=kind)
