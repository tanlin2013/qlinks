from __future__ import annotations

from dataclasses import dataclass, field
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
    support_variables: tuple[int, ...] = ()
    label: str | None = None
    _support_link_set: frozenset[int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _support_variable_set: frozenset[int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # ``support_links`` and ``support_variables`` are immutable, so cache
        # the corresponding sets once.  Older plaquette models used
        # ``support_links`` as the variable support because all variables lived
        # on links.  Site/bond models should set ``support_variables``
        # explicitly so generic local-term partitioning can work without
        # assuming a plaquette geometry.
        object.__setattr__(self, "_support_link_set", frozenset(self.support_links))
        variable_support = self.support_variables if self.support_variables else self.support_links
        object.__setattr__(self, "_support_variable_set", frozenset(variable_support))

    @property
    def support_link_set(self) -> frozenset[int]:
        return self._support_link_set

    @property
    def support_variable_set(self) -> frozenset[int]:
        return self._support_variable_set

    def is_inside_links(self, links: set[int] | frozenset[int]) -> bool:
        return self._support_variable_set <= frozenset(links)

    def is_disjoint_from_links(self, links: set[int] | frozenset[int]) -> bool:
        return self._support_variable_set.isdisjoint(frozenset(links))

    def crosses_links(self, links: set[int] | frozenset[int]) -> bool:
        region = frozenset(links)
        return (
            not self._support_variable_set <= region
            and not self._support_variable_set.isdisjoint(region)
        )
