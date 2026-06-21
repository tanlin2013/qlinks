from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class LocalSpace:
    """Finite local Hilbert/configuration space for one variable.

    Attributes:
        values: Allowed integer values for the variable.

    Examples:
        Spin-1/2 site or dimer occupation uses ``[0, 1]``.  A spin-half QLM
        electric field uses ``[-1, 1]``.
    """

    values: npt.NDArray[np.integer]
    _value_set: frozenset[int] = field(init=False, repr=False, compare=False)
    _value_to_code: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.int64)

        if values.ndim != 1:
            raise ValueError("LocalSpace.values must be a one-dimensional array.")

        if values.size == 0:
            raise ValueError("LocalSpace.values cannot be empty.")

        if np.unique(values).size != values.size:
            raise ValueError("LocalSpace.values must not contain duplicates.")

        value_to_code = {int(value): code for code, value in enumerate(values)}

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_value_set", frozenset(value_to_code))
        object.__setattr__(self, "_value_to_code", value_to_code)

    @classmethod
    def from_values(cls, values: Iterable[int]) -> LocalSpace:
        return cls(np.asarray(list(values), dtype=np.int64))

    @classmethod
    def binary(cls) -> LocalSpace:
        """Local two-state variable with values {0, 1}."""
        return cls.from_values([0, 1])

    @classmethod
    def spin_half_flux(cls) -> LocalSpace:
        """
        Spin-1/2 QLM-like electric flux variable.

        This uses integer values {-1, +1}; if you prefer physical values
        {-1/2, +1/2}, keep this integer representation internally and divide
        by 2 only when evaluating physical observables.
        """
        return cls.from_values([-1, 1])

    @classmethod
    def spin_one(cls) -> LocalSpace:
        """Spin-1 local variable with S^z values {-1, 0, +1}."""
        return cls.from_values([-1, 0, 1])

    @property
    def dim(self) -> int:
        return int(self.values.size)

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    def contains(self, value: int) -> bool:
        return int(value) in self._value_set

    def validate_value(self, value: int) -> None:
        if not self.contains(value):
            raise ValueError(f"Value {value} is not allowed in local space {self.values.tolist()}.")

    def validate_array(self, array: npt.ArrayLike) -> None:
        arr = np.asarray(array)
        invalid = ~np.isin(arr, self.values)
        if np.any(invalid):
            bad_values = np.unique(arr[invalid])
            raise ValueError(
                f"Array contains values outside local space {self.values.tolist()}: "
                f"{bad_values.tolist()}"
            )

    def value_to_code(self, value: int) -> int:
        """
        Convert a physical local value to a dense integer code 0, ..., dim - 1.
        """
        value = int(value)

        try:
            return self._value_to_code[value]
        except KeyError as exc:
            raise ValueError(
                f"Value {value} is not allowed in local space {self.values.tolist()}."
            ) from exc

    def code_to_value(self, code: int) -> int:
        """
        Convert a dense integer code 0, ..., dim - 1 back to the physical value.
        """
        if code < 0 or code >= self.dim:
            raise ValueError(f"Code {code} is outside valid range [0, {self.dim}).")
        return int(self.values[code])
