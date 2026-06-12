from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from qlinks.variables.layout import VariableLayout


@dataclass(frozen=True, slots=True)
class ConfigEncoder:
    """
    Encode configurations into hashable keys for basis lookup.

    The default key is bytes from a canonical int64 representation. This is not
    always the most compressed representation, but it is robust for mixed
    site/link variables and arbitrary integer local values.

    Later, for spin-1/2-only models, this can be supplemented by a bit-packed
    encoder.
    """

    layout: VariableLayout

    def encode(self, config: npt.ArrayLike, *, validate: bool = True) -> bytes:
        arr = np.asarray(config, dtype=np.int64)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        arr = np.ascontiguousarray(arr, dtype=np.int64)
        return arr.tobytes()

    def decode(self, key: bytes, *, validate: bool = True) -> npt.NDArray[np.int64]:
        arr = np.frombuffer(key, dtype=np.int64).copy()

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Decoded config has shape {arr.shape}, expected {self.layout.shape}.")

        return arr

    def build_index(
        self,
        configs: Iterable[npt.ArrayLike],
        *,
        validate: bool = True,
    ) -> dict[bytes, int]:
        if isinstance(configs, np.ndarray):
            return self._build_index_from_array(configs, validate=validate)

        index: dict[bytes, int] = {}

        for i, config in enumerate(configs):
            key = self.encode(config, validate=validate)
            if key in index:
                raise ValueError(f"Duplicate configuration found at position {i}.")
            index[key] = i

        return index

    def _build_index_from_array(
        self,
        configs: npt.NDArray[np.integer],
        *,
        validate: bool,
    ) -> dict[bytes, int]:
        arr = np.asarray(configs, dtype=np.int64)

        if arr.ndim != 2:
            raise ValueError("Expected a two-dimensional array of configurations.")

        if arr.shape[1] != self.layout.n_variables:
            raise ValueError(
                f"Expected configs with {self.layout.n_variables} variables, "
                f"got {arr.shape[1]}."
            )

        if validate:
            self.layout.validate_batch(arr)

        contiguous = np.ascontiguousarray(arr, dtype=np.int64)
        index: dict[bytes, int] = {}

        # Iterating over rows still creates one bytes key per state, but avoids
        # the per-row ConfigEncoder.encode(...) dispatch and per-row validation.
        for i, row in enumerate(contiguous):
            key = row.tobytes()
            if key in index:
                raise ValueError(f"Duplicate configuration found at position {i}.")
            index[key] = i

        return index


@dataclass(frozen=True, slots=True)
class BitPackedBinaryEncoder:
    """
    Compact encoder for pure binary layouts with local values {0, 1}.

    This is useful for PXP, toric-code qubits, or dimer occupation variables.

    It intentionally does not support {-1, +1} directly. For QLM flux variables,
    either use ConfigEncoder or convert values to binary codes first.
    """

    layout: VariableLayout

    def __post_init__(self) -> None:
        for variable_index in range(self.layout.n_variables):
            values = set(self.layout.local_space(variable_index).values.tolist())
            if values != {0, 1}:
                raise ValueError(
                    "BitPackedBinaryEncoder requires every local space to have values {0, 1}."
                )

    def encode(self, config: npt.ArrayLike, *, validate: bool = True) -> bytes:
        arr = np.asarray(config, dtype=np.uint8)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        packed = np.packbits(arr, bitorder="little")
        return packed.tobytes()

    def decode(self, key: bytes, *, validate: bool = True) -> npt.NDArray[np.int64]:
        packed = np.frombuffer(key, dtype=np.uint8)
        unpacked = np.unpackbits(packed, bitorder="little")[: self.layout.n_variables]
        arr = unpacked.astype(np.int64, copy=True)

        if validate:
            self.layout.validate_config(arr)

        return arr
