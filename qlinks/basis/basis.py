from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from qlinks.variables import ConfigEncoder, VariableLayout


def full_basis_from_layout(
    layout: VariableLayout,
    *,
    sort: bool = True,
) -> Basis:
    """
    Generate the full Cartesian-product basis for an unconstrained layout.

    This is faster and simpler than DFS when there are no constraints/sectors.

    The output states have shape:

        (prod_i dim_i, layout.n_variables)

    where dim_i is the local-space dimension of variable i.
    """
    spaces = [
        np.asarray(layout.local_space(i).values, dtype=np.int64) for i in range(layout.n_variables)
    ]

    if layout.n_variables == 0:
        states = np.empty((1, 0), dtype=np.int64)
        return Basis.from_states(layout, states, sort=sort)

    n_states = int(np.prod([space.size for space in spaces], dtype=np.int64))
    states = np.empty((n_states, layout.n_variables), dtype=np.int64)

    # Vectorized Cartesian product.
    # For variable i, values repeat in blocks of size repeat,
    # and the whole pattern is tiled to fill n_states.
    repeat = n_states

    for variable_index, values in enumerate(spaces):
        repeat //= int(values.size)
        tile = n_states // (int(values.size) * repeat)

        states[:, variable_index] = np.tile(
            np.repeat(values, repeat),
            tile,
        )

    return Basis.from_states(
        layout,
        states,
        sort=sort,
    )


@dataclass(frozen=True, slots=True)
class Basis:
    """
    Computational basis represented by explicit configurations.

    states:
        Integer array of shape (n_states, n_variables).

    index:
        Map encoded configuration -> basis index.
    """

    layout: VariableLayout
    states: npt.NDArray[np.int64]
    encoder: ConfigEncoder
    index: dict[bytes, int]

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.int64)

        if states.ndim != 2:
            raise ValueError("Basis.states must be a two-dimensional array.")

        if states.shape[1] != self.layout.n_variables:
            raise ValueError(
                f"Expected states with {self.layout.n_variables} variables, "
                f"got {states.shape[1]}."
            )

        self.layout.validate_batch(states)

        object.__setattr__(self, "states", states)

        if len(self.index) != states.shape[0]:
            raise ValueError("Basis.index size does not match number of states.")

    @classmethod
    def from_states(
        cls,
        layout: VariableLayout,
        states: npt.ArrayLike,
        *,
        sort: bool = False,
    ) -> Basis:
        arr = np.asarray(states, dtype=np.int64)

        if arr.ndim != 2:
            raise ValueError("states must be a two-dimensional array.")

        if arr.shape[1] != layout.n_variables:
            raise ValueError(
                f"Expected states with {layout.n_variables} variables, got {arr.shape[1]}."
            )

        layout.validate_batch(arr)

        if sort:
            arr = arr[np.lexsort(arr.T[::-1])]

        encoder = ConfigEncoder(layout)
        index = encoder.build_index(arr)

        return cls(
            layout=layout,
            states=arr,
            encoder=encoder,
            index=index,
        )

    @classmethod
    def empty(cls, layout: VariableLayout) -> Basis:
        states = np.empty((0, layout.n_variables), dtype=np.int64)
        encoder = ConfigEncoder(layout)
        return cls(layout=layout, states=states, encoder=encoder, index={})

    @property
    def n_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_variables(self) -> int:
        return self.layout.n_variables

    def __len__(self) -> int:
        return self.n_states

    def __contains__(self, config: npt.ArrayLike) -> bool:
        return self.encoder.encode(config) in self.index

    def state(self, basis_index: int, *, copy: bool = True) -> npt.NDArray[np.int64]:
        if basis_index < 0 or basis_index >= self.n_states:
            raise IndexError(f"basis_index {basis_index} outside valid range [0, {self.n_states}).")

        if copy:
            return self.states[basis_index].copy()
        return self.states[basis_index]

    def get_index(self, config: npt.ArrayLike) -> int | None:
        key = self.encoder.encode(config)
        return self.index.get(key)

    def require_index(self, config: npt.ArrayLike) -> int:
        idx = self.get_index(config)
        if idx is None:
            raise KeyError("Configuration is not in the basis.")
        return idx

    def iter_states(self, *, copy: bool = True) -> Iterable[npt.NDArray[np.int64]]:
        for i in range(self.n_states):
            yield self.state(i, copy=copy)
