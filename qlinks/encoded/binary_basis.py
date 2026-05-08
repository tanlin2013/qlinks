from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.variables import VariableLayout


def _require_binary_layout(layout: VariableLayout) -> None:
    for variable_index in range(layout.n_variables):
        values = set(layout.local_space(variable_index).values.tolist())
        if values != {0, 1}:
            raise ValueError("BinaryEncodedBasis requires every local space to be {0, 1}.")


def encode_binary_config(config: npt.ArrayLike) -> int:
    """
    Encode binary config into a Python int.

    bit i stores config[i].
    """
    arr = np.asarray(config, dtype=np.int64)

    if arr.ndim != 1:
        raise ValueError("config must be one-dimensional.")

    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("config must contain only 0 and 1.")

    code = 0
    for i, value in enumerate(arr):
        if int(value) == 1:
            code |= 1 << i

    return code


def decode_binary_code(code: int, n_variables: int) -> npt.NDArray[np.int64]:
    """
    Decode Python int code back to binary config.
    """
    if code < 0:
        raise ValueError("code must be non-negative.")

    if n_variables < 0:
        raise ValueError("n_variables must be non-negative.")

    return np.asarray(
        [(code >> i) & 1 for i in range(n_variables)],
        dtype=np.int64,
    )


def bitmask_from_indices(indices: Iterable[int]) -> int:
    mask = 0
    for i in indices:
        if i < 0:
            raise ValueError("bit index must be non-negative.")
        mask |= 1 << int(i)
    return mask


@dataclass(frozen=True, slots=True)
class BinaryEncodedBasis:
    """
    Basis represented by integer bit patterns.

    This is the production-oriented binary fast path. It is useful for PXP,
    QDM, toric-code qubits, and any model whose variables are {0, 1}.
    """

    layout: VariableLayout
    codes: npt.NDArray[np.object_]
    index: dict[int, int]

    def __post_init__(self) -> None:
        _require_binary_layout(self.layout)

        codes = np.asarray(self.codes, dtype=object)

        if codes.ndim != 1:
            raise ValueError("codes must be one-dimensional.")

        normalized: list[int] = []

        max_code = 1 << self.layout.n_variables

        for code in codes.tolist():
            code_int = int(code)

            if code_int < 0:
                raise ValueError("codes must be non-negative.")

            if code_int >= max_code:
                raise ValueError(
                    f"code {code_int} cannot fit in {self.layout.n_variables} binary variables."
                )

            normalized.append(code_int)

        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate encoded basis states found.")

        if len(self.index) != len(normalized):
            raise ValueError("index size does not match number of codes.")

        for i, code in enumerate(normalized):
            if self.index.get(code) != i:
                raise ValueError("index does not match code ordering.")

        object.__setattr__(self, "codes", np.asarray(normalized, dtype=object))

    @classmethod
    def from_codes(
        cls,
        layout: VariableLayout,
        codes: Iterable[int],
        *,
        sort: bool = False,
    ) -> BinaryEncodedBasis:
        _require_binary_layout(layout)

        code_list = [int(code) for code in codes]

        if sort:
            code_list = sorted(code_list)

        if len(set(code_list)) != len(code_list):
            raise ValueError("Duplicate encoded basis states found.")

        index = {code: i for i, code in enumerate(code_list)}

        return cls(
            layout=layout,
            codes=np.asarray(code_list, dtype=object),
            index=index,
        )

    @classmethod
    def from_configs(
        cls,
        layout: VariableLayout,
        configs: npt.ArrayLike,
        *,
        sort: bool = False,
    ) -> BinaryEncodedBasis:
        _require_binary_layout(layout)

        arr = np.asarray(configs, dtype=np.int64)

        if arr.ndim != 2:
            raise ValueError("configs must be a two-dimensional array.")

        if arr.shape[1] != layout.n_variables:
            raise ValueError(
                f"Expected configs with {layout.n_variables} variables, got {arr.shape[1]}."
            )

        layout.validate_batch(arr)

        codes = [encode_binary_config(config) for config in arr]

        return cls.from_codes(layout, codes, sort=sort)

    @classmethod
    def from_basis(
        cls,
        basis: Basis,
        *,
        sort: bool = False,
    ) -> BinaryEncodedBasis:
        return cls.from_configs(basis.layout, basis.states, sort=sort)

    @classmethod
    def full(
        cls,
        layout: VariableLayout,
        *,
        sort: bool = True,
    ) -> BinaryEncodedBasis:
        _require_binary_layout(layout)
        codes = range(1 << layout.n_variables)
        return cls.from_codes(layout, codes, sort=sort)

    @classmethod
    def empty(cls, layout: VariableLayout) -> BinaryEncodedBasis:
        _require_binary_layout(layout)
        return cls(layout=layout, codes=np.asarray([], dtype=object), index={})

    @property
    def n_states(self) -> int:
        return int(self.codes.size)

    @property
    def n_variables(self) -> int:
        return self.layout.n_variables

    def __len__(self) -> int:
        return self.n_states

    def __contains__(self, code: int) -> bool:
        return int(code) in self.index

    def code(self, basis_index: int) -> int:
        if basis_index < 0 or basis_index >= self.n_states:
            raise IndexError(f"basis_index {basis_index} outside valid range [0, {self.n_states}).")

        return int(self.codes[basis_index])

    def config(self, basis_index: int) -> npt.NDArray[np.int64]:
        return decode_binary_code(self.code(basis_index), self.n_variables)

    def get_index(self, code: int) -> int | None:
        return self.index.get(int(code))

    def require_index(self, code: int) -> int:
        idx = self.get_index(code)
        if idx is None:
            raise KeyError(f"Encoded state {code} is not in the basis.")
        return idx

    def to_array_basis(self) -> Basis:
        if self.n_states == 0:
            return Basis.empty(self.layout)

        states = np.vstack([self.config(i) for i in range(self.n_states)])
        return Basis.from_states(self.layout, states)
