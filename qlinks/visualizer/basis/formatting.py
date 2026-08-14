from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from qlinks.visualizer.basis.styles import BasisConfigLabelStyle


def format_basis_config(
    config: npt.ArrayLike,
    *,
    style: BasisConfigLabelStyle = "compact",
    max_length: int = 48,
) -> str:
    """
    Format one basis configuration for subplot labels.

    style="compact":
        binary configs are printed like 010101.
        other configs are printed like 1,-1,1,-1.

    style="array":
        use numpy array formatting.

    style="none":
        return an empty string.
    """
    arr = np.asarray(config, dtype=np.int64)

    if style == "none":
        return ""

    if style == "array":
        text = np.array2string(arr, separator=", ")

    elif style == "compact":
        values = set(arr.tolist())

        if values <= {0, 1}:
            text = "".join(str(int(x)) for x in arr)
        else:
            text = ",".join(str(int(x)) for x in arr)

    else:
        raise ValueError("style must be 'none', 'compact', or 'array'.")

    if len(text) > max_length:
        return text[: max_length - 3] + "..."

    return text


def automatic_grid_shape(
    n_items: int, *, ncols: int | None = None, nrows: int | None = None
) -> tuple[int, int]:
    """
    Decide a reasonable grid shape.

    If both nrows and ncols are given, they must fit n_items.
    If only one is given, the other is inferred.
    If neither is given, use a near-square grid.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")

    if n_items == 0:
        return 0, 0

    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive.")

    if ncols is not None and ncols <= 0:
        raise ValueError("ncols must be positive.")

    if nrows is not None and ncols is not None:
        if nrows * ncols < n_items:
            raise ValueError("nrows * ncols is smaller than the number of states.")
        return nrows, ncols

    if ncols is not None:
        return math.ceil(n_items / ncols), ncols

    if nrows is not None:
        return nrows, math.ceil(n_items / nrows)

    ncols_auto = math.ceil(math.sqrt(n_items))
    nrows_auto = math.ceil(n_items / ncols_auto)
    return nrows_auto, ncols_auto


def _select_cage_record(
    result_or_record,
    *,
    signature: tuple[int, int] | None = None,
    record_index: int = 0,
):
    """Return a CageRecord from either a CageRecord or CageSearchResult.

    This intentionally uses duck typing to avoid making the visualizer module
    depend directly on qlinks.caging.
    """
    if hasattr(result_or_record, "support") and hasattr(
        result_or_record,
        "local_state",
    ):
        return result_or_record

    if signature is None:
        return result_or_record[record_index]

    return result_or_record[signature, record_index]


def _amplitude_label(
    *,
    basis_index: int,
    amplitude: complex,
    digits: int = 3,
) -> str:
    real = float(np.real(amplitude))
    imag = float(np.imag(amplitude))

    if abs(imag) < 10 ** (-digits):
        amp_text = f"{real:.{digits}g}"
    elif abs(real) < 10 ** (-digits):
        amp_text = f"{imag:.{digits}g}j"
    else:
        amp_text = f"{real:.{digits}g}{imag:+.{digits}g}j"

    return f"basis {basis_index}\namp={amp_text}"


def _zero_mechanism_label_map(report) -> dict[int, str]:
    """Map zero index to its zero-level mechanism label."""
    labels: dict[int, str] = {}

    for zero_report in report.zero_reports:
        labels[int(zero_report.zero_index)] = str(zero_report.probe_mechanism_label)

    return labels


def _zero_indices_for_mechanism(
    report,
    mechanism: str,
) -> npt.NDArray[np.int64]:
    """Return source-zero indices selected by environment-removal mechanism."""
    zero_reports = tuple(report.zero_reports)
    if mechanism == "all":
        selected = zero_reports
    elif mechanism in {
        "q_empty",
        "closed_by_same_pattern_zeros",
        "domain_blocked",
        "projector_like",
        "collective_cancellation",
        "unexplained_leakage",
    }:
        selected = tuple(zero for zero in zero_reports if zero.probe_mechanism_label == mechanism)
    elif mechanism in {
        "no_environment_weight",
        "projective_annihilation",
        "same_local_cancellation_pattern",
        "unsafe",
    }:
        selected = tuple(zero for zero in zero_reports if zero.removal_mechanism == mechanism)
    else:
        allowed = (
            "all",
            "q_empty",
            "closed_by_same_pattern_zeros",
            "domain_blocked",
            "projector_like",
            "collective_cancellation",
            "unexplained_leakage",
            "no_environment_weight",
            "projective_annihilation",
            "same_local_cancellation_pattern",
            "unsafe",
        )
        raise ValueError(
            f"Unknown environment-removal mechanism {mechanism!r}. "
            f"Expected one of: {', '.join(allowed)}."
        )

    return np.asarray([int(zero.zero_index) for zero in selected], dtype=np.int64)
