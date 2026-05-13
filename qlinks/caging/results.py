from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CageState:
    """A validated interference-caged eigenstate."""

    energy: complex
    local_state: NDArray[np.complex128]
    support: NDArray[np.int_]
    boundary_residual: float
    eigen_residual: float
    full_residual: float | None = None
    metadata: dict[str, object] | None = None

    @property
    def support_size(self) -> int:
        """Number of vertices supporting the local state."""
        return int(self.support.size)
