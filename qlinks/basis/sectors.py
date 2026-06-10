from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy._typing import NDArray

from qlinks.basis import basis_configs_from_build_result


def _sector_is_satisfied(
    sector: Any,
    config: NDArray[np.integer],
) -> bool:
    if hasattr(sector, "is_satisfied"):
        return bool(sector.is_satisfied(config))

    if hasattr(sector, "check"):
        return bool(sector.check(config))

    raise TypeError("Each sector must provide `is_satisfied(config)` or `check(config)`.")


def sector_mask_from_sectors(
    basis_configs: NDArray[np.integer],
    sectors: Sequence[Any],
) -> NDArray[np.bool_]:
    """Return a boolean mask selecting configs satisfying all sectors."""
    configs = np.asarray(basis_configs)

    if configs.ndim != 2:
        raise ValueError("basis_configs must be a two-dimensional array.")

    if len(sectors) == 0:
        return np.ones(configs.shape[0], dtype=np.bool_)

    return np.asarray(
        [all(_sector_is_satisfied(sector, config) for sector in sectors) for config in configs],
        dtype=np.bool_,
    )


def sector_mask_from_build_result(
    build_result: Any,
    sectors: Sequence[Any] | None = None,
) -> NDArray[np.bool_]:
    """Return a boolean mask for configs in a build result.

    If ``sectors`` is provided, use those sectors. Otherwise, try common
    build-result fields and fall back to selecting all basis states.
    """
    basis_configs = basis_configs_from_build_result(build_result)

    if sectors is None:
        sectors = getattr(
            build_result,
            "sectors",
            getattr(build_result, "constraints", ()),
        )

    return sector_mask_from_sectors(
        basis_configs=basis_configs,
        sectors=tuple(sectors),
    )
