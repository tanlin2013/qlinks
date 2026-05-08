from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.encoded import BinaryEncodedBasis
from qlinks.variables import LocalSpace, VariableLayout


def flux_to_bit(value: int) -> int:
    """
    Encode spin-1/2 flux value as a bit.

        -1 -> 0
        +1 -> 1
    """
    if int(value) == -1:
        return 0
    if int(value) == 1:
        return 1
    raise ValueError(f"Flux value must be -1 or +1, got {value}.")


def bit_to_flux(value: int) -> int:
    """
    Decode bit as spin-1/2 flux.

        0 -> -1
        1 -> +1
    """
    if int(value) == 0:
        return -1
    if int(value) == 1:
        return 1
    raise ValueError(f"Bit value must be 0 or 1, got {value}.")


def flux_config_to_binary(config: npt.ArrayLike) -> npt.NDArray[np.int64]:
    arr = np.asarray(config, dtype=np.int64)

    if arr.ndim != 1:
        raise ValueError("config must be one-dimensional.")

    out = np.empty_like(arr, dtype=np.int64)

    for i, value in enumerate(arr):
        out[i] = flux_to_bit(int(value))

    return out


def binary_config_to_flux(config: npt.ArrayLike) -> npt.NDArray[np.int64]:
    arr = np.asarray(config, dtype=np.int64)

    if arr.ndim != 1:
        raise ValueError("config must be one-dimensional.")

    out = np.empty_like(arr, dtype=np.int64)

    for i, value in enumerate(arr):
        out[i] = bit_to_flux(int(value))

    return out


def flux_configs_to_binary(configs: npt.ArrayLike) -> npt.NDArray[np.int64]:
    arr = np.asarray(configs, dtype=np.int64)

    if arr.ndim != 2:
        raise ValueError("configs must be two-dimensional.")

    return np.vstack([flux_config_to_binary(config) for config in arr])


def binary_layout_like_flux_layout(flux_layout: VariableLayout) -> VariableLayout:
    """
    Create a binary link-variable layout with the same number/order of link
    variables as a spin-half-flux layout.

    This assumes the QLM layout is link-only and ordered by link id.
    """

    if flux_layout.site_variable_indices().size != 0:
        raise ValueError("Expected a link-only flux layout.")

    return VariableLayout.from_links(
        num_links=flux_layout.n_variables,
        local_space=LocalSpace.binary(),
    )


def binary_encoded_basis_from_flux_basis(
    flux_basis: Basis,
    *,
    sort: bool = False,
) -> BinaryEncodedBasis:
    """
    Convert a {-1,+1} flux Basis into a BinaryEncodedBasis using

        -1 -> 0
        +1 -> 1
    """

    binary_layout = binary_layout_like_flux_layout(flux_basis.layout)
    binary_states = flux_configs_to_binary(flux_basis.states)

    return BinaryEncodedBasis.from_configs(
        binary_layout,
        binary_states,
        sort=sort,
    )
