from __future__ import annotations

from typing import ClassVar

import numpy as np

from qlinks.lattice.graph import LatticeGraph
from qlinks.lattice.types import BoundaryCondition, Link, Site


class ChainLattice(LatticeGraph):
    """
    One-dimensional chain.

    Sites:
        i = 0, ..., L - 1

    Open boundary:
        links i -> i + 1 for i = 0, ..., L - 2

    Periodic boundary:
        same as open, plus L - 1 -> 0
    """

    _primitive_vectors: ClassVar[tuple[np.ndarray, ...]] = (np.array([1.0], dtype=float),)

    _basis_offsets: ClassVar[tuple[np.ndarray, ...]] = (np.array([0.0], dtype=float),)

    def __init__(
        self,
        length: int,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive.")

        bc = BoundaryCondition(boundary_condition)

        sites = tuple(
            Site(
                id=i,
                cell=(i,),
                sublattice=0,
                position=(float(i),),
            )
            for i in range(length)
        )

        links: list[Link] = []
        link_id = 0

        for i in range(length - 1):
            links.append(
                Link(
                    id=link_id,
                    source=i,
                    target=i + 1,
                    kind="x",
                    wrap=False,
                )
            )
            link_id += 1

        if bc == BoundaryCondition.PERIODIC and length > 1:
            links.append(
                Link(
                    id=link_id,
                    source=length - 1,
                    target=0,
                    kind="x",
                    wrap=True,
                )
            )

        translations: dict[tuple[int, tuple[int, ...]], int] = {}

        for i in range(length):
            for dx in (-1, 1):
                j = i + dx

                if bc == BoundaryCondition.PERIODIC:
                    j %= length
                    translations[(i, (dx,))] = j
                elif 0 <= j < length:
                    translations[(i, (dx,))] = j

        super().__init__(
            sites=sites,
            links=tuple(links),
            plaquettes=(),
            boundary_condition=bc,
            translations=translations,
        )
