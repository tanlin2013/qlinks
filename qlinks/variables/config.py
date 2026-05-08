from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.variables.layout import VariableKind, VariableLayout


@dataclass(frozen=True, slots=True)
class ConfigView:
    """
    Lightweight read/write wrapper around a NumPy configuration array.

    This is useful for readable operator code:

        cfg.link(3)
        cfg.set_link(3, -cfg.link(3))

    The underlying array is still a normal NumPy array.
    """

    layout: VariableLayout
    array: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        arr = np.asarray(self.array, dtype=np.int64)
        self.layout.validate_config(arr)
        object.__setattr__(self, "array", arr)

    @classmethod
    def from_array(cls, layout: VariableLayout, array: npt.ArrayLike) -> ConfigView:
        return cls(layout=layout, array=np.asarray(array, dtype=np.int64))

    @classmethod
    def default(cls, layout: VariableLayout) -> ConfigView:
        return cls(layout=layout, array=layout.default_config())

    def copy(self) -> ConfigView:
        return ConfigView(self.layout, self.array.copy())

    def value(self, variable_index: int) -> int:
        self.layout.spec(variable_index)
        return int(self.array[variable_index])

    def set_value(self, variable_index: int, value: int) -> None:
        self.layout.local_space(variable_index).validate_value(value)
        self.array[variable_index] = value

    def site(self, site_index: int) -> int:
        variable_index = self.layout.variable_index(VariableKind.SITE, site_index)
        return self.value(variable_index)

    def link(self, link_index: int) -> int:
        variable_index = self.layout.variable_index(VariableKind.LINK, link_index)
        return self.value(variable_index)

    def set_site(self, site_index: int, value: int) -> None:
        variable_index = self.layout.variable_index(VariableKind.SITE, site_index)
        self.set_value(variable_index, value)

    def set_link(self, link_index: int, value: int) -> None:
        variable_index = self.layout.variable_index(VariableKind.LINK, link_index)
        self.set_value(variable_index, value)

    def flipped(self, variable_index: int) -> ConfigView:
        """
        Return a new config with a binary variable flipped.

        This only works for variables with local values {0, 1}.
        For QLM flux variables {-1, +1}, use negated().
        """
        local_space = self.layout.local_space(variable_index)
        if set(local_space.values.tolist()) != {0, 1}:
            raise ValueError("flipped() is only defined for binary variables with values {0, 1}.")

        new = self.copy()
        new.array[variable_index] = 1 - new.array[variable_index]
        return new

    def negated(self, variable_index: int) -> ConfigView:
        """
        Return a new config with one variable multiplied by -1.

        This is useful for spin-1/2 QLM flux variables {-1, +1}.
        """
        value = self.value(variable_index)
        new_value = -value
        self.layout.local_space(variable_index).validate_value(new_value)

        new = self.copy()
        new.array[variable_index] = new_value
        return new

    def as_array(self, copy: bool = True) -> npt.NDArray[np.int64]:
        if copy:
            return self.array.copy()
        return self.array
