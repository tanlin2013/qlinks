"""HDF5 IO for interference-caged eigenstates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import numpy.typing as npt

from qlinks.caging import (
    CageState,
    cage_state_to_full_vector,
    cage_states_to_full_matrix,
)
from qlinks.io.hdf5 import _as_path, _normalize_index, _write_attrs

CAGE_SCHEMA_VERSION = "0.1"


def _padded_supports_and_states(
    cage_states: list[CageState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded supports, local states, and support sizes."""
    if len(cage_states) == 0:
        return (
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.complex128),
            np.zeros((0,), dtype=np.int64),
        )

    support_sizes = np.asarray(
        [cage_state.support.size for cage_state in cage_states],
        dtype=np.int64,
    )
    max_support_size = int(np.max(support_sizes))

    supports = -np.ones(
        (len(cage_states), max_support_size),
        dtype=np.int64,
    )
    local_states = np.zeros(
        (len(cage_states), max_support_size),
        dtype=np.complex128,
    )

    for cage_index, cage_state in enumerate(cage_states):
        support_size = int(support_sizes[cage_index])
        supports[cage_index, :support_size] = np.asarray(
            cage_state.support,
            dtype=np.int64,
        )
        local_states[cage_index, :support_size] = np.asarray(
            cage_state.local_state,
            dtype=np.complex128,
        )

    return supports, local_states, support_sizes


def _metadata_values(
    cage_states: list[CageState],
    key: str,
    *,
    default: complex = np.nan,
) -> np.ndarray:
    """Extract one scalar metadata field from each cage state."""
    values: list[complex] = []

    for cage_state in cage_states:
        metadata = cage_state.metadata or {}
        value = metadata.get(key, default)
        values.append(complex(value))

    return np.asarray(values, dtype=np.complex128)


@dataclass
class CageStateHDF5Writer:
    """Writer for interference-caged eigenstate results."""

    path: str | Path
    mode: str = "w"
    compression: str | None = "gzip"
    compression_opts: int | None = 4
    hilbert_size: int | None = None

    def __post_init__(self) -> None:
        self.path = _as_path(self.path)
        self.file = h5py.File(self.path, self.mode)
        self.file.attrs["schema_version"] = CAGE_SCHEMA_VERSION
        self.file.attrs["format"] = "qlinks_cage_states"

    def __enter__(self) -> CageStateHDF5Writer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()

    def write_metadata(
        self,
        *,
        model_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Write file-level metadata."""
        if model_name is not None:
            self.file.attrs["model_name"] = model_name

        if parameters is not None:
            group = self.file.require_group("metadata/parameters")
            _write_attrs(group, parameters)

        if extra is not None:
            group = self.file.require_group("metadata/extra")
            _write_attrs(group, extra)

    def write_cage_states(
        self,
        cage_states: list[CageState],
        *,
        attrs: Mapping[str, Any] | None = None,
        hilbert_size: int | None = None,
        write_signature_metadata: bool = True,
    ) -> None:
        """Write all cage states to HDF5."""
        group = self.file.require_group("cage_states")

        for dataset_name in list(group.keys()):
            del group[dataset_name]

        supports, local_states, support_sizes = _padded_supports_and_states(
            cage_states,
        )

        energies = np.asarray(
            [cage_state.energy for cage_state in cage_states],
            dtype=np.complex128,
        )
        boundary_residuals = np.asarray(
            [cage_state.boundary_residual for cage_state in cage_states],
            dtype=np.float64,
        )
        eigen_residuals = np.asarray(
            [cage_state.eigen_residual for cage_state in cage_states],
            dtype=np.float64,
        )
        full_residuals = np.asarray(
            [
                np.nan if cage_state.full_residual is None else cage_state.full_residual
                for cage_state in cage_states
            ],
            dtype=np.float64,
        )

        group.create_dataset(
            "energies",
            data=energies,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "supports",
            data=supports,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "support_sizes",
            data=support_sizes,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "local_states",
            data=local_states,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "boundary_residuals",
            data=boundary_residuals,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "eigen_residuals",
            data=eigen_residuals,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        group.create_dataset(
            "full_residuals",
            data=full_residuals,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )

        group.attrs["n_cage_states"] = len(cage_states)
        group.attrs["max_support_size"] = supports.shape[1]
        group.attrs["support_padding"] = -1
        group.attrs["local_state_padding"] = 0

        if hilbert_size is not None:
            group.attrs["hilbert_size"] = int(hilbert_size)

        if write_signature_metadata:
            if any(
                (cage_state.metadata or {}).get("kappa") is not None for cage_state in cage_states
            ):
                group.create_dataset(
                    "kappa_values",
                    data=_metadata_values(cage_states, "kappa"),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    chunks=True,
                )

            if any(
                (cage_state.metadata or {}).get("potential_value") is not None
                for cage_state in cage_states
            ):
                group.create_dataset(
                    "potential_values",
                    data=_metadata_values(cage_states, "potential_value"),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    chunks=True,
                )

        if attrs is not None:
            _write_attrs(group, attrs)

    def flush(self) -> None:
        self.file.flush()


@dataclass
class CageStateHDF5Reader:
    """Reader for interference-caged eigenstate results."""

    path: str | Path
    mode: str = "r"

    def __post_init__(self) -> None:
        self.path = _as_path(self.path)
        self.file = h5py.File(self.path, self.mode)

    def __enter__(self) -> CageStateHDF5Reader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()

    @property
    def model_name(self) -> str | None:
        return self.file.attrs.get("model_name", None)

    @property
    def n_cage_states(self) -> int:
        return int(self.file["cage_states/support_sizes"].shape[0])

    def read_metadata(self) -> dict[str, Any]:
        """Read file-level metadata."""
        metadata: dict[str, Any] = {
            "attrs": dict(self.file.attrs),
        }

        if "metadata/parameters" in self.file:
            metadata["parameters"] = dict(self.file["metadata/parameters"].attrs)

        if "metadata/extra" in self.file:
            metadata["extra"] = dict(self.file["metadata/extra"].attrs)

        return metadata

    def read_energies(
        self,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.ndarray:
        dataset = self.file["cage_states/energies"]

        if index is None:
            return dataset[...]

        return dataset[_normalize_index(index)]

    def read_support(
        self,
        index: int,
    ) -> np.ndarray:
        support_size = int(self.file["cage_states/support_sizes"][int(index)])
        return self.file["cage_states/supports"][int(index), :support_size]

    def read_local_state(
        self,
        index: int,
    ) -> np.ndarray:
        support_size = int(self.file["cage_states/support_sizes"][int(index)])
        return self.file["cage_states/local_states"][int(index), :support_size]

    def read_cage_state(self, index: int) -> CageState:
        """Read one cage state."""
        cage_index = int(index)
        support = self.read_support(cage_index)
        local_state = self.read_local_state(cage_index)

        energy = complex(self.file["cage_states/energies"][cage_index])
        boundary_residual = float(self.file["cage_states/boundary_residuals"][cage_index])
        eigen_residual = float(self.file["cage_states/eigen_residuals"][cage_index])
        full_residual_value = float(self.file["cage_states/full_residuals"][cage_index])
        full_residual = None if np.isnan(full_residual_value) else full_residual_value

        metadata: dict[str, object] = {}

        if "cage_states/kappa_values" in self.file:
            kappa_value = self.file["cage_states/kappa_values"][cage_index]
            if not np.isnan(complex(kappa_value).real):
                metadata["kappa"] = complex(kappa_value)

        if "cage_states/potential_values" in self.file:
            potential_value = self.file["cage_states/potential_values"][cage_index]
            if not np.isnan(complex(potential_value).real):
                metadata["potential_value"] = complex(potential_value)

        return CageState(
            energy=energy,
            local_state=local_state,
            support=support,
            boundary_residual=boundary_residual,
            eigen_residual=eigen_residual,
            full_residual=full_residual,
            metadata=metadata,
        )

    def read_cage_states(
        self,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> list[CageState]:
        """Read cage states."""
        if index is None:
            indices = range(self.n_cage_states)
        elif isinstance(index, int):
            indices = [index]
        elif isinstance(index, slice):
            indices = range(self.n_cage_states)[index]
        else:
            indices = [int(local_index) for local_index in np.asarray(index)]

        return [self.read_cage_state(int(cage_index)) for cage_index in indices]

    def iter_cage_states(
        self,
        indices: range | list[int] | None = None,
    ):
        """Iterate over cage states."""
        if indices is None:
            indices = range(self.n_cage_states)

        for cage_index in indices:
            yield int(cage_index), self.read_cage_state(int(cage_index))

    @property
    def hilbert_size(self) -> int | None:
        group = self.file["cage_states"]

        if "hilbert_size" not in group.attrs:
            return None

        return int(group.attrs["hilbert_size"])

    def read_full_vector(
        self,
        index: int,
        *,
        hilbert_size: int | None = None,
    ) -> np.ndarray:
        """Read one cage state and lift it to the full Hilbert space."""
        if hilbert_size is None:
            hilbert_size = self.hilbert_size

        if hilbert_size is None:
            raise ValueError("hilbert_size was not provided and is not stored in the file.")

        cage_state = self.read_cage_state(index)

        return cage_state_to_full_vector(
            cage_state,
            hilbert_size=hilbert_size,
        )

    def read_full_matrix(
        self,
        *,
        hilbert_size: int | None = None,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.ndarray:
        """Read cage states and lift them to dense full-Hilbert vectors."""
        if hilbert_size is None:
            hilbert_size = self.hilbert_size

        if hilbert_size is None:
            raise ValueError("hilbert_size was not provided and is not stored in the file.")

        cage_states = self.read_cage_states(index)

        return cage_states_to_full_matrix(
            cage_states,
            hilbert_size=hilbert_size,
        )
