from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LocalTermKind = Literal["plaquette", "site", "link", "bond"]
LocalOperatorKind = Literal["kinetic", "potential", "hamiltonian"]


@dataclass(frozen=True, slots=True)
class LocalTermDescriptor:
    """Geometry-level descriptor for one local operator term.

    This descriptor is intentionally matrix-free.  It tells us which local
    term we want, where it lives in real space, and which model method should
    assemble it.
    """

    term_id: int
    term_kind: LocalTermKind
    operator_kind: LocalOperatorKind
    support_links: tuple[int, ...]
    support_sites: tuple[int, ...] = ()
    support_plaquettes: tuple[int, ...] = ()
    label: str | None = None

    @property
    def support_link_set(self) -> frozenset[int]:
        return frozenset(self.support_links)

    def is_inside_links(self, links: set[int] | frozenset[int]) -> bool:
        return self.support_link_set <= frozenset(links)

    def is_disjoint_from_links(self, links: set[int] | frozenset[int]) -> bool:
        return self.support_link_set.isdisjoint(frozenset(links))

    def crosses_links(self, links: set[int] | frozenset[int]) -> bool:
        region = frozenset(links)
        return not self.is_inside_links(region) and not self.is_disjoint_from_links(region)
