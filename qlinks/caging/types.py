from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CageSolverConfig:
    """Numerical configuration for the interference-caging solver."""

    tolerance: float = 1e-10
    max_power: int | None = None
    stabilization_rounds: int = 1
    validate_full_residual: bool = True
    normalize_states: bool = True
