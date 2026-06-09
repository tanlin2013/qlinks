from __future__ import annotations

import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging.classification import CageClassificationConfig
from tests.helpers.states import binary_product_states, config_index


def base_classification_config() -> CageClassificationConfig:
    return CageClassificationConfig(
        amplitude_tolerance=1.0e-12,
        cancellation_tolerance=1.0e-12,
        action_tolerance=1.0e-12,
        sector_policy="ignore",
    )


def pairwise_interference_system() -> tuple[
    np.ndarray,
    scipy_sparse.csr_array,
    dict[str, int],
]:
    """
    Build a kinetic matrix with one interference zero.

    The local structure is

        |h>  = |000>
        |v1> = |010>
        |v2> = |001>

    with H0[h, v1] = H0[h, v2] = 1.

    Therefore, a state with amplitudes

        c[v1] = +1/sqrt(2)
        c[v2] = -1/sqrt(2)

    has destructive interference at |h>.
    """
    basis_configs = binary_product_states(n_variables=3)

    h = config_index(basis_configs, (0, 0, 0))
    v1 = config_index(basis_configs, (0, 1, 0))
    v2 = config_index(basis_configs, (0, 0, 1))

    rows = [h, h, v1, v2]
    cols = [v1, v2, h, h]
    data = [1.0, 1.0, 1.0, 1.0]

    kinetic = scipy_sparse.csr_array(
        (data, (rows, cols)),
        shape=(basis_configs.shape[0], basis_configs.shape[0]),
        dtype=np.complex128,
    )

    return basis_configs, kinetic, {"h": h, "v1": v1, "v2": v2}


def two_zero_closed_interference_system() -> tuple[
    np.ndarray,
    scipy_sparse.csr_array,
    dict[str, int],
]:
    """
    Build a kinetic matrix with two related nontrivial zeros.

    First zero:

        h0 = |000>
        v1 = |010>
        v2 = |001>

    Second zero:

        h1 = |100>
        w1 = |110>
        w2 = |101>

    The same local transition pattern on the last two variables
    generates both interference zeros.
    """
    basis_configs = binary_product_states(n_variables=3)

    h0 = config_index(basis_configs, (0, 0, 0))
    v1 = config_index(basis_configs, (0, 1, 0))
    v2 = config_index(basis_configs, (0, 0, 1))

    h1 = config_index(basis_configs, (1, 0, 0))
    w1 = config_index(basis_configs, (1, 1, 0))
    w2 = config_index(basis_configs, (1, 0, 1))

    rows = [h0, h0, v1, v2, h1, h1, w1, w2]
    cols = [v1, v2, h0, h0, w1, w2, h1, h1]
    data = [1.0] * len(rows)

    kinetic = scipy_sparse.csr_array(
        (data, (rows, cols)),
        shape=(basis_configs.shape[0], basis_configs.shape[0]),
        dtype=np.complex128,
    )

    return (
        basis_configs,
        kinetic,
        {
            "h0": h0,
            "v1": v1,
            "v2": v2,
            "h1": h1,
            "w1": w1,
            "w2": w2,
        },
    )
