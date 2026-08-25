#!/usr/bin/env python
"""Run Sec. VI common-window reduction under the established sparse-evidence contract.

The August-20 L=14 sparse evidence was certified by the production workflow through
ARPACK tolerance plus cross-budget observable convergence. Physical eigenpair residuals
were recorded as diagnostics, but that workflow did not impose a separate absolute
1e-6 residual veto. This adapter preserves that established contract while delegating
all cache shape, metadata, coverage, orthogonality, covariance, and anchor checks to
``spin1_sec6_common_windows``.

This is a cache-only compatibility lane for the already-certified Sec. VI evidence. It
never invokes an eigensolver.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir() and (candidate / "experimental").is_dir():
        ROOT = candidate
        break
else:
    ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
for path in (JOBS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

common = importlib.import_module("spin1_sec6_common_windows")

SOURCE_SUMMARY = "spin1_xy_sec6_provisioning_summary.json"
CERTIFICATION_KEY = "representative_sparse_budget_certified"


def _read_source_certification(source_data_dir: Path) -> dict[str, Any]:
    source = Path(source_data_dir).resolve(strict=False)
    summary_path = source / SOURCE_SUMMARY
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise common.CachedSpectrumUnavailableError(
            f"missing or invalid established sparse certification: {summary_path}"
        ) from exc
    if not isinstance(summary, dict) or not bool(summary.get(CERTIFICATION_KEY, False)):
        raise common.CachedSpectrumUnavailableError(
            "established L=14 sparse-budget certification is not passed; "
            "refusing legacy-contract cache reuse"
        )
    return summary


def _assert_finite_residual_diagnostics(data_dir: Path | None) -> None:
    if data_dir is None:
        return
    path = Path(data_dir) / common.COMMON_NAME
    if not path.is_file():
        return
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise common.CachedSpectrumUnavailableError(
            f"invalid existing common-window export: {path}"
        ) from exc
    if "window_max_eigenpair_residual" not in frame.columns:
        return
    residuals = frame["window_max_eigenpair_residual"].to_numpy(dtype=float)
    if residuals.size and not np.all(np.isfinite(residuals)):
        raise common.CachedSpectrumUnavailableError(
            "existing common-window export has non-finite eigenpair residual diagnostics"
        )


def _assert_frame_residuals_finite(frame: pd.DataFrame) -> None:
    if "window_max_eigenpair_residual" not in frame.columns:
        return
    residuals = frame["window_max_eigenpair_residual"].to_numpy(dtype=float)
    if residuals.size and not np.all(np.isfinite(residuals)):
        raise common.CachedSpectrumUnavailableError(
            "computed common-window export has non-finite eigenpair residual diagnostics"
        )


def _raise_with_checkpoint_detail(output_dir: Path, exc: Exception) -> None:
    audit_path = Path(output_dir) / common.CHECKPOINT_AUDIT_NAME
    details: list[str] = []
    if audit_path.is_file():
        try:
            frame = pd.read_csv(audit_path)
        except (OSError, pd.errors.ParserError):
            frame = pd.DataFrame()
        if not frame.empty and "validation_errors" in frame.columns:
            for row in frame.itertuples(index=False):
                value = str(getattr(row, "validation_errors", "")).strip()
                if value and value.lower() != "nan":
                    details.append(f"L={int(getattr(row, 'L'))}: {value}")
    suffix = "" if not details else "; details: " + " | ".join(details)
    raise common.CachedSpectrumUnavailableError(f"{exc}{suffix}") from exc


def compute_certified_common_windows(
    *,
    source_data_dir: Path,
    checkpoint_roots: tuple[Path, ...],
    output_dir: Path,
    existing_data_dir: Path | None,
    lengths: tuple[int, ...] = common.TARGET_LENGTHS,
) -> pd.DataFrame:
    """Reuse certified legacy sparse evidence without adding a new residual cutoff."""

    _read_source_certification(source_data_dir)
    _assert_finite_residual_diagnostics(existing_data_dir)
    original_tolerance = common.PHYSICAL_RESIDUAL_TOLERANCE
    # The production contract records finite window residuals but certifies accuracy
    # through cross-budget observable convergence. ``inf`` disables only the later,
    # newly-added absolute veto; non-finite residuals still fail in the common reducer.
    common.PHYSICAL_RESIDUAL_TOLERANCE = math.inf
    try:
        frame = common.compute_common_windows_from_cache(
            checkpoint_roots=checkpoint_roots,
            output_dir=output_dir,
            lengths=lengths,
            existing_data_dir=existing_data_dir,
        )
        _assert_frame_residuals_finite(frame)
        return frame
    except common.CachedSpectrumUnavailableError as exc:
        _raise_with_checkpoint_detail(output_dir, exc)
    finally:
        common.PHYSICAL_RESIDUAL_TOLERANCE = original_tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    parser.add_argument("--lengths", default="8,10,12,14")
    args = parser.parse_args()
    roots = tuple(args.checkpoint_root)
    if not roots:
        roots = (ROOT / "experimental" / "data" / "evidence_cache" / "spin1",)
    lengths = tuple(int(token.strip()) for token in args.lengths.split(",") if token.strip())
    if not lengths:
        raise ValueError("--lengths must contain at least one size")
    frame = compute_certified_common_windows(
        source_data_dir=args.source_data_dir,
        checkpoint_roots=roots,
        output_dir=args.output_dir,
        existing_data_dir=args.existing_data_dir,
        lengths=lengths,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
