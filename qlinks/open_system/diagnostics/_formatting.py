from __future__ import annotations

import numpy as np


def _state_ipr(state: np.ndarray) -> float:
    norm = float(np.linalg.norm(state))

    if norm == 0.0:
        return 0.0

    normalized = state / norm
    probabilities = np.abs(normalized) ** 2
    return float(np.sum(probabilities**2))


def _format_float(value: float) -> str:
    return f"{value:.3e}"


def _format_float_or_none(value: float | None) -> str:
    if value is None:
        return "not checked"

    return _format_float(float(value))


def _format_float_tuple(
    values: tuple[float, ...],
    *,
    max_items: int = 8,
) -> str:
    if len(values) == 0:
        return "∅"

    if len(values) <= max_items:
        return ", ".join(_format_float(value) for value in values)

    head = ", ".join(_format_float(value) for value in values[:max_items])
    return f"{head}, ... ({len(values)} total)"


def _format_optional_int(value: int | None, *, lower_bound: bool = False) -> str:
    if value is None:
        return "not checked"
    return f"{value}{'+' if lower_bound else ''}"


def _status_for_residual(
    value: float | None,
    *,
    excellent: float = 1e-12,
    acceptable: float = 1e-8,
) -> str:
    if value is None:
        return "[dim]n/a[/dim]"

    if value <= excellent:
        return "[green]ok[/green]"

    if value <= acceptable:
        return "[yellow]warn[/yellow]"

    return "[red]large[/red]"
