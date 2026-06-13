from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DegenerateBasisStrategy = Literal["none", "ipr"]


@dataclass(frozen=True)
class CageSolverConfig:
    """Numerical configuration for the interference-caging solver."""

    tolerance: float = 1e-10
    max_power: int | None = None
    stabilization_rounds: int = 1
    validate_full_residual: bool = True
    normalize_states: bool = True

    # Degenerate cage handling.
    degenerate_basis_strategy: DegenerateBasisStrategy = "none"

    # Many-start IPR localization.
    ipr_n_restarts: int = 128
    ipr_max_iter: int = 1000
    ipr_step_size: float = 0.1
    ipr_convergence_tolerance: float = 1e-12
    ipr_candidate_count: int = 64
    ipr_rank_completion_patience: int | None = None
    ipr_batch_size: int = 16

    # Support detection/refinement.
    ipr_support_tolerance_factor: float = 100.0
    ipr_rank_tolerance_factor: float = 100.0
    ipr_random_seed: int | None = None

    # Optional mutable timing sink used by benchmark/diagnostic callers.
    # Keys are internal stage names and values are accumulated seconds.
    timing_collector: dict[str, float] | None = None
