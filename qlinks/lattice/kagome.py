from __future__ import annotations

from typing import ClassVar

import numpy as np

from qlinks.lattice.graph import LatticeGraph
from qlinks.lattice.types import BoundaryCondition, Link, Plaquette, Site


class KagomeLattice(LatticeGraph):
    """Two-dimensional kagome lattice with a three-site triangular unit cell.

    Unit cell:
        A(x, y), sublattice 0
        B(x, y), sublattice 1
        C(x, y), sublattice 2

    Primitive vectors:
        a1 = (1, 0)
        a2 = (1/2, sqrt(3)/2)

    Link convention:
        Each unit cell contributes six nearest-neighbor bonds:

        ab       : A(x, y) -> B(x, y)
        ab_prev  : A(x, y) -> B(x - 1, y)
        ac       : A(x, y) -> C(x, y)
        ac_prev  : A(x, y) -> C(x, y - 1)
        bc       : B(x, y) -> C(x, y)
        bc_next  : B(x, y) -> C(x + 1, y - 1)

    Plaquettes:
        triangle_up / triangle_down are included for visualization and local
        geometry diagnostics.  Kagome QDM/QLM resonance terms use only the
        hexagon plaquettes returned by :meth:`qdm_plaquette_ids` and
        :meth:`qlm_plaquette_ids`.
    """

    __slots__ = ("lx", "ly")

    _sqrt3: ClassVar[float] = float(np.sqrt(3.0))

    _primitive_vectors: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([1.0, 0.0], dtype=float),
        np.array([0.5, _sqrt3 / 2.0], dtype=float),
    )

    _basis_offsets: ClassVar[tuple[np.ndarray, ...]] = (
        np.array([0.0, 0.0], dtype=float),
        np.array([0.5, 0.0], dtype=float),
        np.array([0.25, _sqrt3 / 4.0], dtype=float),
    )

    _link_displacements: ClassVar[dict[str, tuple[int, int, int, int]]] = {
        # kind: (source_sublattice, target_sublattice, dx, dy)
        "ab": (0, 1, 0, 0),
        "ab_prev": (0, 1, -1, 0),
        "ac": (0, 2, 0, 0),
        "ac_prev": (0, 2, 0, -1),
        "bc": (1, 2, 0, 0),
        "bc_next": (1, 2, 1, -1),
    }

    def __init__(
        self,
        lx: int,
        ly: int,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        *,
        include_triangles: bool = True,
        include_hexagons: bool = True,
    ) -> None:
        if lx <= 0:
            raise ValueError("lx must be positive.")
        if ly <= 0:
            raise ValueError("ly must be positive.")

        bc = BoundaryCondition(boundary_condition)

        def site_id(x: int, y: int, sublattice: int) -> int:
            return 3 * (x * ly + y) + sublattice

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
            primitive = (
                float(x) * self._primitive_vectors[0]
                + float(y) * self._primitive_vectors[1]
                + self._basis_offsets[int(sublattice)]
            )
            return (float(primitive[0]), float(primitive[1]))

        sites = tuple(
            Site(
                id=site_id(x, y, sub),
                cell=(x, y),
                sublattice=sub,
                position=position(x, y, sub),
            )
            for x in range(lx)
            for y in range(ly)
            for sub in (0, 1, 2)
        )

        links: list[Link] = []
        directed_link_lookup: dict[tuple[int, int], int] = {}
        cell_link_lookup: dict[tuple[int, int, str], int] = {}
        undirected_seen: set[tuple[int, int]] = set()

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

            undirected_key = tuple(sorted((int(source), int(target))))
            if undirected_key in undirected_seen:
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
            undirected_seen.add(undirected_key)
            cell_link_lookup[(int(cell[0]), int(cell[1]), str(kind))] = link_id

        for x in range(lx):
            for y in range(ly):
                for kind, (source_sub, target_sub, dx, dy) in self._link_displacements.items():
                    source = site_id(x, y, source_sub)
                    tx, ty = x + dx, y + dy
                    target = maybe_site_id(tx, ty, target_sub)

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

        def oriented_link_between(source: int, target: int) -> tuple[int, int]:
            direct = directed_link_lookup.get((source, target))
            if direct is not None:
                return direct, +1

            reverse = directed_link_lookup.get((target, source))
            if reverse is not None:
                return reverse, -1

            raise KeyError(f"No link between {source} and {target}.")

        plaquettes: list[Plaquette] = []
        seen_plaquette_links: set[tuple[int, ...]] = set()

        def add_loop(
            kind: str,
            *,
            anchor: tuple[int, int],
            site_specs: tuple[tuple[int, int, int], ...],
        ) -> None:
            site_ids: list[int] = []
            for sx, sy, sub in site_specs:
                sid = maybe_site_id(sx, sy, sub)
                if sid is None:
                    return
                site_ids.append(int(sid))

            if len(set(site_ids)) != len(site_ids):
                return

            loop_links: list[int] = []
            orientations: list[int] = []
            for current, nxt in zip(site_ids, site_ids[1:] + site_ids[:1], strict=True):
                try:
                    link_id, orientation = oriented_link_between(current, nxt)
                except KeyError:
                    return
                loop_links.append(int(link_id))
                orientations.append(int(orientation))

            if len(set(loop_links)) != len(loop_links):
                return

            plaquette_key = tuple(sorted(loop_links))
            if plaquette_key in seen_plaquette_links:
                return
            seen_plaquette_links.add(plaquette_key)

            plaquettes.append(
                Plaquette(
                    id=len(plaquettes),
                    links=tuple(loop_links),
                    orientations=tuple(orientations),
                    sites=tuple(site_ids),
                    kind=kind,
                    anchor_cell=anchor,
                )
            )

        max_x = lx if bc == BoundaryCondition.PERIODIC else lx
        max_y = ly if bc == BoundaryCondition.PERIODIC else ly

        for x in range(max_x):
            for y in range(max_y):
                if include_triangles:
                    add_loop(
                        "triangle_up",
                        anchor=(x, y),
                        site_specs=((x, y, 0), (x, y, 1), (x, y, 2)),
                    )
                    add_loop(
                        "triangle_down",
                        anchor=(x, y),
                        site_specs=((x, y, 1), (x + 1, y - 1, 2), (x + 1, y, 0)),
                    )

                if include_hexagons:
                    add_loop(
                        "hexagon",
                        anchor=(x, y),
                        site_specs=(
                            (x, y, 1),
                            (x, y, 2),
                            (x, y + 1, 0),
                            (x, y + 1, 1),
                            (x + 1, y, 2),
                            (x + 1, y, 0),
                        ),
                    )

        translations: dict[tuple[int, tuple[int, int]], int] = {}

        for x in range(lx):
            for y in range(ly):
                for sub in (0, 1, 2):
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
        if sublattice not in (0, 1, 2):
            raise ValueError("sublattice must be 0, 1, or 2.")

        if self.boundary_condition == BoundaryCondition.PERIODIC:
            x %= self.lx
            y %= self.ly
        elif not (0 <= x < self.lx and 0 <= y < self.ly):
            raise IndexError(f"site coordinate ({x}, {y}) outside lattice.")

        return 3 * (x * self.ly + y) + sublattice

    @classmethod
    def link_cell_displacement(cls, kind: str) -> tuple[int, int]:
        if kind not in cls._link_displacements:
            raise ValueError(f"Unsupported kagome link kind: {kind!r}")
        _source_sub, _target_sub, dx, dy = cls._link_displacements[str(kind)]
        return int(dx), int(dy)

    def qdm_plaquette_ids(self):
        return [plaquette.id for plaquette in self.plaquettes if plaquette.kind == "hexagon"]

    def qlm_plaquette_ids(self):
        return self.qdm_plaquette_ids()

    def hexagon_plaquette_id(self, x: int, y: int) -> int:
        return self.plaquette_id_from_anchor((x, y), kind="hexagon")
