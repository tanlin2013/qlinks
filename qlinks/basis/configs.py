from __future__ import annotations

from typing import Any

import numpy as np
from numpy._typing import NDArray


def basis_configs_from_basis(basis: Any) -> NDArray[np.integer]:
    """Return explicit basis configurations for ArrayBasis or BinaryEncodedBasis.

    Sparse/array builders usually expose this as ``basis.states``. Bitmask
    builders may use BinaryEncodedBasis, which stores compact integer codes and
    exposes ``to_array_basis()``.
    """
    if hasattr(basis, "states"):
        configs = np.asarray(basis.states)

    elif hasattr(basis, "configs"):
        configs = np.asarray(basis.configs)

    elif hasattr(basis, "to_array_basis"):
        array_basis = basis.to_array_basis()
        if not hasattr(array_basis, "states"):
            raise TypeError(
                "Unsupported encoded basis type: "
                f"{type(basis)!r}. Expected `.to_array_basis().states`."
            )
        configs = np.asarray(array_basis.states)

    else:
        raise TypeError(
            "Unsupported basis type for classification: "
            f"{type(basis)!r}. Expected an object with `.states`, `.configs`, "
            "or `.to_array_basis().states`."
        )

    if configs.ndim != 2:
        raise ValueError("basis configurations must have shape " "(n_basis, n_variables).")

    return configs


def basis_configs_from_build_result(
    build_result: Any,
) -> NDArray[np.integer]:
    """Return physical basis configurations from a ModelBuildResult.

    Bitmask builders may store basis states as local integer codes, e.g.
    0/1 for a two-state local space. The classifier and visualizers need the
    physical variable values declared by build_result.layout, e.g. -1/+1.
    """
    basis_configs = basis_configs_from_basis(build_result.basis)

    layout = getattr(build_result, "layout", None)

    if layout is None:
        return basis_configs

    return decode_basis_configs_with_layout(
        basis_configs,
        layout,
    )


def decode_basis_configs_with_layout(
    basis_configs: NDArray[np.integer],
    layout: Any,
) -> NDArray[np.integer]:
    """Decode local integer codes into physical values using a VariableLayout.

    If the configs already contain valid physical values, they are returned
    unchanged. Otherwise, values 0, 1, ..., d-1 are interpreted as indices into
    each variable's local space.
    """
    configs = np.asarray(basis_configs)

    if configs.ndim != 2:
        raise ValueError("basis_configs must be a two-dimensional array.")

    if configs.shape[1] != len(layout):
        raise ValueError(
            "basis_configs and layout have incompatible variable counts: "
            f"{configs.shape[1]} != {len(layout)}."
        )

    try:
        layout.validate_batch(configs)
        return configs
    except ValueError:
        pass

    decoded = np.empty_like(configs, dtype=np.int64)

    for variable_index in range(configs.shape[1]):
        local_values = np.asarray(
            layout.local_space(variable_index).values,
            dtype=np.int64,
        )
        codes = configs[:, variable_index].astype(np.int64, copy=False)

        if np.any(codes < 0) or np.any(codes >= local_values.size):
            raise ValueError(
                "basis_configs are neither valid physical values nor valid "
                "local-space integer codes for variable "
                f"{variable_index}. Got codes in range "
                f"[{codes.min()}, {codes.max()}], but local space has "
                f"{local_values.size} values: {local_values.tolist()}."
            )

        decoded[:, variable_index] = local_values[codes]

    layout.validate_batch(decoded)

    return decoded
