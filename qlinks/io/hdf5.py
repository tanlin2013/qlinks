from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = "0.1"


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _write_attrs(group: h5py.Group | h5py.File, attrs: Mapping[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, complex, bool, np.number)):
            group.attrs[key] = value
        else:
            group.attrs[key] = str(value)


def _normalize_index(index: int | slice | list[int] | npt.NDArray[np.integer]):
    if isinstance(index, int):
        return index

    if isinstance(index, slice):
        return index

    return np.asarray(index, dtype=np.int64)


@dataclass
class EigenpairHDF5Writer:
    path: str | Path
    mode: str = "w"
    compression: str | None = "gzip"
    compression_opts: int | None = 4
    chunks: bool | tuple[int, ...] | None = True

    def __post_init__(self) -> None:
        self.path = _as_path(self.path)
        self.file = h5py.File(self.path, self.mode)
        self.file.attrs["schema_version"] = SCHEMA_VERSION
        self.file.attrs["format"] = "qlinks_eigenpairs"

    def __enter__(self) -> EigenpairHDF5Writer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()

    def write_metadata(
        self,
        *,
        model_name: str,
        parameters: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.file.attrs["model_name"] = model_name

        if parameters is not None:
            group = self.file.require_group("metadata/parameters")
            _write_attrs(group, parameters)

        if extra is not None:
            group = self.file.require_group("metadata/extra")
            _write_attrs(group, extra)

    def write_basis_states(
        self,
        states: npt.ArrayLike,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> h5py.Dataset:
        arr = np.asarray(states)

        group = self.file.require_group("basis")
        if "states" in group:
            del group["states"]

        dset = group.create_dataset(
            "states",
            data=arr,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )

        if attrs is not None:
            _write_attrs(dset, attrs)

        return dset

    def write_energies(
        self,
        energies: npt.ArrayLike,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> h5py.Dataset:
        arr = np.asarray(energies)

        group = self.file.require_group("eigenpairs")
        if "energies" in group:
            del group["energies"]

        dset = group.create_dataset(
            "energies",
            data=arr,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )

        if attrs is not None:
            _write_attrs(dset, attrs)

        return dset

    def write_eigenvectors(
        self,
        vectors: npt.ArrayLike,
        *,
        eigen_axis: int = 0,
        attrs: Mapping[str, Any] | None = None,
    ) -> h5py.Dataset:
        arr = np.asarray(vectors)

        if arr.ndim != 2:
            raise ValueError("eigenvectors must be a 2D array.")

        # Store as vectors[eigen_index, basis_index].
        if eigen_axis == 1:
            arr = arr.T
        elif eigen_axis != 0:
            raise ValueError("eigen_axis must be 0 or 1.")

        group = self.file.require_group("eigenpairs")
        if "vectors" in group:
            del group["vectors"]

        # Chunk by eigenvector rows, so reading one or a few eigenvectors is efficient.
        chunk_shape = (1, arr.shape[1])

        dset = group.create_dataset(
            "vectors",
            data=arr,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=chunk_shape,
        )

        dset.attrs["vector_axis"] = "row"
        dset.attrs["meaning"] = "vectors[eigen_index, basis_index]"

        if attrs is not None:
            _write_attrs(dset, attrs)

        return dset

    def create_eigenvector_dataset(
        self,
        *,
        n_eigenvectors: int,
        n_basis: int,
        dtype: npt.DTypeLike = np.complex128,
    ) -> h5py.Dataset:
        """
        Create an empty eigenvector dataset for incremental writing.

        This is useful when eigenvectors are generated one at a time.
        """
        group = self.file.require_group("eigenpairs")
        if "vectors" in group:
            del group["vectors"]

        dset = group.create_dataset(
            "vectors",
            shape=(int(n_eigenvectors), int(n_basis)),
            dtype=dtype,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=(1, int(n_basis)),
        )

        dset.attrs["vector_axis"] = "row"
        dset.attrs["meaning"] = "vectors[eigen_index, basis_index]"

        return dset

    def write_eigenvector(
        self,
        index: int,
        vector: npt.ArrayLike,
    ) -> None:
        dset = self.file["eigenpairs/vectors"]
        arr = np.asarray(vector)

        if arr.shape != dset.shape[1:]:
            raise ValueError(f"Expected vector shape {dset.shape[1:]}, got {arr.shape}.")

        dset[int(index), :] = arr

    def write_observable(
        self,
        name: str,
        values: npt.ArrayLike,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> h5py.Dataset:
        arr = np.asarray(values)

        group = self.file.require_group("observables")
        if name in group:
            del group[name]

        # If first axis is eigen-index, chunk by first axis.
        if arr.ndim == 0:
            chunks = None
        elif arr.ndim == 1:
            chunks = True
        else:
            chunks = (1, *arr.shape[1:])

        dset = group.create_dataset(
            name,
            data=arr,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=chunks,
        )

        dset.attrs["first_axis"] = "eigen_index"

        if attrs is not None:
            _write_attrs(dset, attrs)

        return dset

    def flush(self) -> None:
        self.file.flush()


@dataclass
class EigenpairHDF5Reader:
    path: str | Path
    mode: str = "r"

    def __post_init__(self) -> None:
        self.path = _as_path(self.path)
        self.file = h5py.File(self.path, self.mode)

    def __enter__(self) -> EigenpairHDF5Reader:
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
    def n_eigenvectors(self) -> int:
        return int(self.file["eigenpairs/vectors"].shape[0])

    @property
    def n_basis(self) -> int:
        return int(self.file["eigenpairs/vectors"].shape[1])

    def list_observables(self) -> list[str]:
        if "observables" not in self.file:
            return []
        return sorted(self.file["observables"].keys())

    def read_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "attrs": dict(self.file.attrs),
        }

        if "metadata/parameters" in self.file:
            out["parameters"] = dict(self.file["metadata/parameters"].attrs)

        if "metadata/extra" in self.file:
            out["extra"] = dict(self.file["metadata/extra"].attrs)

        return out

    def read_basis_states(
        self,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.ndarray:
        dset = self.file["basis/states"]

        if index is None:
            return dset[...]

        return dset[_normalize_index(index)]

    def read_energy(self, index: int) -> np.number:
        return self.file["eigenpairs/energies"][int(index)]

    def read_energies(
        self,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.ndarray:
        dset = self.file["eigenpairs/energies"]

        if index is None:
            return dset[...]

        return dset[_normalize_index(index)]

    def read_eigenvector(self, index: int) -> np.ndarray:
        return self.file["eigenpairs/vectors"][int(index), :]

    def read_eigenvectors(
        self,
        index: int | slice | list[int] | npt.NDArray[np.integer],
    ) -> np.ndarray:
        return self.file["eigenpairs/vectors"][_normalize_index(index), :]

    def read_observable(
        self,
        name: str,
        index: int | slice | list[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.ndarray:
        dset = self.file[f"observables/{name}"]

        if index is None:
            return dset[...]

        return dset[_normalize_index(index)]

    def read_observables_for_eigenvector(
        self,
        index: int,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for name in self.list_observables():
            dset = self.file[f"observables/{name}"]
            out[name] = dset[int(index)]

        return out

    def iter_eigenvectors(
        self,
        indices: range | list[int] | None = None,
    ):
        if indices is None:
            indices = range(self.n_eigenvectors)

        for index in indices:
            yield int(index), self.read_energy(int(index)), self.read_eigenvector(int(index))
