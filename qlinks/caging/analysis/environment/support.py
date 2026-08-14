from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import EnvironmentRemovalProbeReport


def support_key_from_mask(local_mask: NDArray[np.bool_]) -> tuple[int, ...]:
    """Return the variable-index support key for a local reduced-IZ mask."""
    return tuple(int(index) for index in np.flatnonzero(local_mask))


def support_key_for_zero_report(
    zero_report: EnvironmentRemovalProbeReport,
) -> tuple[int, ...]:
    """Return the variable-index support key for one reduced-IZ report."""
    if zero_report.local_variable_indices:
        return zero_report.local_variable_indices
    return support_key_from_mask(zero_report.local_mask)
