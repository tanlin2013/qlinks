from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def format_float(value: float | None, *, precision: int = 3) -> str:
    """Compact scientific notation for human-readable diagnostic reports."""
    if value is None:
        return "none"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision + 3}g}"


def format_complex(value: complex | None, *, precision: int = 3) -> str:
    """Compact complex-number formatter for report text."""
    if value is None:
        return "none"
    z = complex(value)
    if abs(z.imag) <= 10 ** (-(precision + 4)):
        return format_float(z.real, precision=precision)
    if abs(z.real) <= 10 ** (-(precision + 4)):
        return f"{format_float(z.imag, precision=precision)}j"
    sign = "+" if z.imag >= 0 else "-"
    return (
        f"{format_float(z.real, precision=precision)} "
        f"{sign} {format_float(abs(z.imag), precision=precision)}j"
    )


def format_bool(value: bool) -> str:
    return "yes" if bool(value) else "no"


def format_tuple(value: Sequence[Any] | None) -> str:
    if value is None:
        return "none"
    return "(" + ", ".join(str(item) for item in value) + ")"


def complex_to_summary(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(z.real), "imag": float(z.imag)}


def matrix_to_summary(matrix: npt.ArrayLike) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.complex128)
    return {
        "shape": tuple(int(i) for i in arr.shape),
        "frobenius_norm": (
            float(np.linalg.norm(arr, ord="fro")) if arr.ndim == 2 else float(np.linalg.norm(arr))
        ),
        "spectral_norm": float(np.linalg.norm(arr, ord=2)) if arr.ndim == 2 and arr.size else 0.0,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "trace": complex_to_summary(complex(np.trace(arr))) if arr.ndim == 2 else None,
    }


def truncate_sequence(values: Sequence[Any], limit: int) -> tuple[Any, ...]:
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    return tuple(values[:limit])


def format_key_value_lines(
    title: str,
    rows: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    indent: str = "  ",
) -> str:
    """Build a small aligned key/value text block."""
    items = list(rows.items()) if isinstance(rows, Mapping) else list(rows)
    if not items:
        return title
    width = max(len(str(key)) for key, _value in items)
    lines = [title]
    for key, value in items:
        lines.append(f"{indent}{str(key):<{width}} : {value}")
    return "\n".join(lines)


def require_rich(owner: str):
    """Import rich building blocks with a report-specific error message."""
    try:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError as exc:  # pragma: no cover - rich is an optional import path.
        raise ImportError(
            f"{owner}.to_rich() requires the optional `rich` package. "
            "Install it with `pip install rich`."
        ) from exc
    return Group, Panel, Table, Text


def add_summary_rows(table: Any, rows: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> None:
    items = list(rows.items()) if isinstance(rows, Mapping) else list(rows)
    for key, value in items:
        table.add_row(str(key), str(value))
