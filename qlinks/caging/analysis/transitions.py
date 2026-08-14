"""Shared local-transition pattern contracts for caging analysis.

This module owns the local transition representation used both by exterior-
environment reduction diagnostics and by higher-level local analysis. Keeping
this small contract independent prevents either analysis path from owning the
other's implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class LocalTransitionPattern:
    """One local transition induced by an active Hamiltonian edge.

    ``source_local`` and ``target_local`` are product-state patterns on the
    reduced local support. ``matrix_element`` is the corresponding kinetic
    matrix element in that local pattern basis.
    """

    source_local: tuple[int, ...]
    target_local: tuple[int, ...]
    matrix_element: complex


def transition_pattern_key(
    transitions: Iterable[LocalTransitionPattern],
    *,
    digits: int = 12,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...]:
    """Return a deterministic key for a collection of local transitions."""
    return tuple(
        sorted(
            (
                transition.source_local,
                transition.target_local,
                _complex_key(transition.matrix_element, digits=digits),
            )
            for transition in transitions
        )
    )


def _complex_key(value: complex, *, digits: int) -> tuple[float, float]:
    return (
        round(float(np.real(value)), digits),
        round(float(np.imag(value)), digits),
    )


def local_transition_pattern_signature(
    variable_indices: Iterable[int],
    transitions: Iterable[LocalTransitionPattern],
    *,
    digits: int = 12,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...],
]:
    """Return a support-aware signature for one local cancellation pattern."""
    return (
        tuple(int(index) for index in variable_indices),
        transition_pattern_key(transitions, digits=digits),
    )
