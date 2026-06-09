from __future__ import annotations

import itertools

import numpy as np


def binary_product_states(n_variables: int) -> np.ndarray:
    return np.array(
        list(itertools.product([0, 1], repeat=n_variables)),
        dtype=np.int64,
    )


def config_index(
    basis_configs: np.ndarray,
    config: tuple[int, ...],
) -> int:
    target = np.asarray(config, dtype=np.int64)
    matches = np.all(basis_configs == target[None, :], axis=1)
    indices = np.flatnonzero(matches)

    if indices.size != 1:
        raise ValueError(f"Configuration {config} not found uniquely.")

    return int(indices[0])


def empty_int_array() -> np.ndarray:
    return np.array([], dtype=np.int64)


def empty_complex_array() -> np.ndarray:
    return np.array([], dtype=np.complex128)
