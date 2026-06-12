from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from qlinks.variables.local_space import LocalSpace


class VariableKind(StrEnum):
    SITE = "site"
    LINK = "link"


class HasSiteLinkCounts(Protocol):
    """
    Minimal protocol expected from the future lattice layer.

    A concrete lattice does not need to inherit from this protocol.
    It only needs to expose num_sites and/or num_links.
    """

    num_sites: int
    num_links: int


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """
    One variable in the flattened configuration array.

    kind:
        Whether this variable lives on a site or link.

    geometry_index:
        Site index or link index in the lattice layer.

    local_space:
        Allowed values for this variable.
    """

    kind: VariableKind
    geometry_index: int
    local_space: LocalSpace

    def __post_init__(self) -> None:
        if self.geometry_index < 0:
            raise ValueError("geometry_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class VariableLayout:
    """
    Map physical site/link variables to positions in a compact NumPy config array.

    The computational basis state is represented as

        config[var_index]

    This class tells us which var_index corresponds to which site/link variable.
    """

    specs: tuple[VariableSpec, ...]
    _index_by_key: dict[tuple[VariableKind, int], int] = field(
        init=False, repr=False, compare=False
    )
    _site_variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False, compare=False)
    _link_variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.specs) == 0:
            raise ValueError("VariableLayout must contain at least one variable.")

        index_by_key: dict[tuple[VariableKind, int], int] = {}
        site_variable_indices: list[int] = []
        link_variable_indices: list[int] = []

        for variable_index, spec in enumerate(self.specs):
            key = (spec.kind, spec.geometry_index)
            if key in index_by_key:
                raise ValueError(f"Duplicate variable for {spec.kind}:{spec.geometry_index}.")

            index_by_key[key] = variable_index

            if spec.kind == VariableKind.SITE:
                site_variable_indices.append(variable_index)
            elif spec.kind == VariableKind.LINK:
                link_variable_indices.append(variable_index)

        object.__setattr__(self, "_index_by_key", index_by_key)
        object.__setattr__(
            self,
            "_site_variable_indices",
            np.asarray(site_variable_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_link_variable_indices",
            np.asarray(link_variable_indices, dtype=np.int64),
        )

    @classmethod
    def from_sites(cls, num_sites: int, local_space: LocalSpace) -> VariableLayout:
        if num_sites <= 0:
            raise ValueError("num_sites must be positive.")

        return cls(
            tuple(
                VariableSpec(VariableKind.SITE, site_index, local_space)
                for site_index in range(num_sites)
            )
        )

    @classmethod
    def from_links(cls, num_links: int, local_space: LocalSpace) -> VariableLayout:
        if num_links <= 0:
            raise ValueError("num_links must be positive.")

        return cls(
            tuple(
                VariableSpec(VariableKind.LINK, link_index, local_space)
                for link_index in range(num_links)
            )
        )

    @classmethod
    def from_sites_and_links(
        cls,
        num_sites: int,
        site_space: LocalSpace,
        num_links: int,
        link_space: LocalSpace,
    ) -> VariableLayout:
        if num_sites < 0:
            raise ValueError("num_sites must be non-negative.")
        if num_links < 0:
            raise ValueError("num_links must be non-negative.")
        if num_sites + num_links == 0:
            raise ValueError("At least one site or link variable is required.")

        specs: list[VariableSpec] = []

        specs.extend(
            VariableSpec(VariableKind.SITE, site_index, site_space)
            for site_index in range(num_sites)
        )
        specs.extend(
            VariableSpec(VariableKind.LINK, link_index, link_space)
            for link_index in range(num_links)
        )

        return cls(tuple(specs))

    @classmethod
    def from_lattice_sites(
        cls, lattice: HasSiteLinkCounts, local_space: LocalSpace
    ) -> VariableLayout:
        return cls.from_sites(lattice.num_sites, local_space)

    @classmethod
    def from_lattice_links(
        cls, lattice: HasSiteLinkCounts, local_space: LocalSpace
    ) -> VariableLayout:
        return cls.from_links(lattice.num_links, local_space)

    @classmethod
    def from_lattice_sites_and_links(
        cls,
        lattice: HasSiteLinkCounts,
        site_space: LocalSpace,
        link_space: LocalSpace,
    ) -> VariableLayout:
        return cls.from_sites_and_links(
            num_sites=lattice.num_sites,
            site_space=site_space,
            num_links=lattice.num_links,
            link_space=link_space,
        )

    @property
    def n_variables(self) -> int:
        return len(self.specs)

    @property
    def shape(self) -> tuple[int]:
        return (self.n_variables,)

    def __len__(self) -> int:
        return self.n_variables

    def spec(self, variable_index: int) -> VariableSpec:
        self._validate_variable_index(variable_index)
        return self.specs[variable_index]

    def local_space(self, variable_index: int) -> LocalSpace:
        return self.spec(variable_index).local_space

    def variable_index(self, kind: VariableKind | str, geometry_index: int) -> int:
        kind = VariableKind(kind)
        key = (kind, int(geometry_index))

        try:
            return self._index_by_key[key]
        except KeyError as exc:
            raise KeyError(f"No variable found for {kind}:{geometry_index}.") from exc

    def site_variable_index(self, site_index: int) -> int:
        return self.variable_index(VariableKind.SITE, site_index)

    def link_variable_index(self, link_index: int) -> int:
        return self.variable_index(VariableKind.LINK, link_index)

    def site_variable_indices(self) -> npt.NDArray[np.int64]:
        return self._site_variable_indices.copy()

    def link_variable_indices(self) -> npt.NDArray[np.int64]:
        return self._link_variable_indices.copy()

    def empty_config(self, fill_value: int = 0) -> npt.NDArray[np.int64]:
        config = np.full(self.n_variables, fill_value, dtype=np.int64)
        return config

    def default_config(self) -> npt.NDArray[np.int64]:
        """
        Return a valid default configuration.

        For each variable, choose the first value in its local space.
        """
        return np.asarray([spec.local_space.values[0] for spec in self.specs], dtype=np.int64)

    def validate_config(self, config: npt.ArrayLike) -> None:
        arr = np.asarray(config)

        if arr.shape != self.shape:
            raise ValueError(f"Expected config shape {self.shape}, got {arr.shape}.")

        for variable_index, value in enumerate(arr):
            self.local_space(variable_index).validate_value(int(value))

    def validate_batch(self, configs: npt.ArrayLike) -> None:
        arr = np.asarray(configs)

        if arr.ndim != 2:
            raise ValueError("Expected a two-dimensional array of configurations.")

        if arr.shape[1] != self.n_variables:
            raise ValueError(
                f"Expected configs with {self.n_variables} variables, got {arr.shape[1]}."
            )

        for variable_index in range(self.n_variables):
            self.local_space(variable_index).validate_array(arr[:, variable_index])

    def as_metadata(self) -> dict[str, Any]:
        """
        Small serializable summary useful for debugging or future IO.
        """
        return {
            "n_variables": self.n_variables,
            "variables": [
                {
                    "kind": spec.kind.value,
                    "geometry_index": spec.geometry_index,
                    "values": spec.local_space.values.tolist(),
                }
                for spec in self.specs
            ],
        }

    def _validate_variable_index(self, variable_index: int) -> None:
        if variable_index < 0 or variable_index >= self.n_variables:
            raise IndexError(
                f"variable_index {variable_index} outside valid range [0, {self.n_variables})."
            )
