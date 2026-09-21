#!/usr/bin/env python
"""Cover the selected Lx=12 fixed-O(1) checkerboard microcanonical window.

This stage owns only the expensive folded-spectrum solves. It reads the exact-ED
pilot recommendation, reuses the stable PRIMME checkpoint cache, writes one
convergence row immediately after every budget, and stops after two independent
budgets fully cover the selected window. It does not compute thermal observables
or stripe covariances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdm_checkerboard_large_strip import folded_spectrum_partial_spectrum
from qdm_resumable_spectrum import make_resumable_folded_solver
from qdm_sec7_fixed_o1 import (
    REPRESENTATIVE_PHASE,
    atomic_write_csv,
    atomic_write_json,
    build_context,
    process_memory_gib,
    recover_reference_geometry,
)

RECOMMENDATION_NAME = "qdm_checkerboard_fixed_O1_window_recommendation.json"
CONVERGENCE_NAME = "qdm_checkerboard_L12_fixed_O1_spectral_convergence.csv"
ACCEPTANCE_NAME = "qdm_checkerboard_L12_fixed_O1_spectral_acceptance.json"
OBSERVABLES_ACCEPTANCE_NAME = "qdm_checkerboard_L12_fixed_O1_observables_acceptance.json"
DEFAULT_TOLERANCE = 1.0e-8
DEFAULT_MAX_BUDGET = 8192
BUDGET_QUANTUM = 256
MIN_PRODUCTION_BUDGET = 1024


def _configure_cache(cache_root: Path) -> None:
    os.environ["QLINKS_EVIDENCE_CACHE_ROOT"] = str(Path(cache_root).resolve(strict=False))
    os.environ["QLINKS_EVIDENCE_CACHE_RESUME"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_WRITE"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE"] = "0"
    os.environ.setdefault("QLINKS_QDM_FOLDED_BACKEND", "primme")
    os.environ.setdefault("QLINKS_QDM_PRIMME_WARM_START_VECTORS", "512")


def _load_recommendation(output_dir: Path) -> tuple[dict[str, Any], float, float | None]:
    path = Path(output_dir) / RECOMMENDATION_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"fixed-O(1) pilot recommendation is missing: {path}. "
            "Run --stage fixed-O1-pilot first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "recommended":
        raise RuntimeError(
            "fixed-O(1) pilot did not select a production window; "
            f"status={payload.get('status')!r}"
        )
    width = payload.get("recommended_half_width")
    if width is None or float(width) <= 0:
        raise RuntimeError("fixed-O(1) recommendation has no positive primary half-width")
    width = float(width)

    estimate = None
    estimates = payload.get("estimated_L12_budgets", {})
    if isinstance(estimates, dict):
        candidates: list[tuple[float, float]] = []
        for key, value in estimates.items():
            if value is None:
                continue
            try:
                candidates.append((abs(float(key) - width), float(value)))
            except (TypeError, ValueError):
                continue
        if candidates:
            estimate = min(candidates)[1]
    if estimate is None:
        for row in payload.get("heuristic_candidates", []):
            try:
                if math.isclose(float(row["window_half_width"]), width, rel_tol=0.0, abs_tol=1e-12):
                    raw = row.get("estimated_L12_eigenpair_budget")
                    estimate = None if raw is None else float(raw)
                    break
            except (KeyError, TypeError, ValueError):
                continue
    return payload, width, estimate


def _snap_budget(value: float, *, maximum: int) -> int:
    snapped = int(math.ceil(float(value) / BUDGET_QUANTUM) * BUDGET_QUANTUM)
    return min(int(maximum), max(MIN_PRODUCTION_BUDGET, snapped))


def _auto_budgets(estimate: float | None, *, maximum: int) -> tuple[int, ...]:
    if maximum < MIN_PRODUCTION_BUDGET:
        raise ValueError(f"maximum budget must be at least {MIN_PRODUCTION_BUDGET}")
    center = float(estimate) if estimate is not None and estimate > 0 else 4096.0
    raw = [0.75 * center, center, 1.25 * center, 1.50 * center, float(maximum)]
    budgets = sorted({_snap_budget(value, maximum=maximum) for value in raw})
    return tuple(value for value in budgets if value <= maximum)


def _parse_budgets(raw: str, *, estimate: float | None, maximum: int) -> tuple[int, ...]:
    if raw.strip().lower() == "auto":
        values = _auto_budgets(estimate, maximum=maximum)
    else:
        values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values or any(value <= 512 for value in values):
        raise ValueError("Lx=12 fixed-window production budgets must all exceed 512")
    if any(value > maximum for value in values):
        raise ValueError(f"budget exceeds configured maximum {maximum}: {values}")
    return values


def coverage_metrics(
    energies: np.ndarray,
    residuals: np.ndarray,
    *,
    target_energy: float,
    half_width: float,
    solver_tolerance: float,
) -> dict[str, Any]:
    values = np.asarray(energies, dtype=float).reshape(-1)
    residual_values = np.asarray(residuals, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("coverage check requires returned eigenvalues")
    lower = float(target_energy) - float(half_width)
    upper = float(target_energy) + float(half_width)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    left_margin = lower - minimum
    right_margin = maximum - upper
    required_margin = max(1.0e-6, 100.0 * float(solver_tolerance))
    covered = bool(left_margin >= required_margin and right_margin >= required_margin)
    indices = np.flatnonzero((values >= lower) & (values <= upper))
    window_residual = (
        float(np.max(residual_values[indices], initial=0.0)) if indices.size else math.inf
    )
    return {
        "window_lower_energy": lower,
        "window_upper_energy": upper,
        "partial_min_energy": minimum,
        "partial_max_energy": maximum,
        "left_coverage_margin": float(left_margin),
        "right_coverage_margin": float(right_margin),
        "required_coverage_margin": float(required_margin),
        "window_coverage_complete": covered,
        "window_state_count": int(indices.size),
        "window_maximum_residual": window_residual,
    }


def _merge_row(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    if path.is_file():
        frame = pd.read_csv(path)
        if "requested_subspace_size" in frame.columns:
            frame = frame[
                frame["requested_subspace_size"].astype(int)
                != int(row["requested_subspace_size"])
            ]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("requested_subspace_size").reset_index(drop=True)
    atomic_write_csv(path, frame)
    return frame


def _acceptance(frame: pd.DataFrame, *, width: float) -> dict[str, Any]:
    if frame.empty:
        covered = frame
    else:
        covered = frame[frame["window_coverage_complete"].astype(bool)].copy()
    covered = covered.sort_values("requested_subspace_size")
    last = covered.tail(2)
    checks = {
        "at_least_two_covered_budgets": len(covered) >= 2,
        "last_two_window_counts_stable": (
            len(last) == 2 and len(set(last["window_state_count"].astype(int))) == 1
        ),
        "last_two_window_residuals_acceptable": (
            len(last) == 2
            and bool(np.all(last["window_maximum_residual"].astype(float) <= 1.0e-6))
        ),
    }
    return {
        "schema_version": 1,
        "closed": all(checks.values()),
        "window_half_width": float(width),
        "checks": checks,
        "covered_budgets": covered["requested_subspace_size"].astype(int).tolist(),
        "last_two_covered_budgets": last["requested_subspace_size"].astype(int).tolist(),
        "claim_boundary": (
            "This acceptance closes only explicit spectral coverage/budget convergence. "
            "Thermal observables and stripe covariance are separate solver-free stages."
        ),
    }


def _observables_request_extension(output_dir: Path) -> bool:
    """Return whether a completed observable pass explicitly requested another budget."""

    path = Path(output_dir) / OBSERVABLES_ACCEPTANCE_NAME
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload.get("closed") is False


def run(
    *,
    output_dir: Path,
    cache_root: Path,
    budgets_raw: str,
    tolerance: float,
    max_budget: int,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, half_width, estimate = _load_recommendation(output)
    budgets = _parse_budgets(budgets_raw, estimate=estimate, maximum=int(max_budget))
    _configure_cache(cache_root)

    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, phase=REPRESENTATIVE_PHASE)
    original = folded_spectrum_partial_spectrum
    solver = (
        original
        if getattr(original, "__name__", "") == "resumable_folded_spectrum_partial_spectrum"
        else make_resumable_folded_solver(original)
    )

    convergence_path = output / CONVERGENCE_NAME
    frame = pd.read_csv(convergence_path) if convergence_path.is_file() else pd.DataFrame()
    if not frame.empty:
        existing_widths = pd.to_numeric(frame["window_half_width"], errors="coerce").dropna()
        if not np.all(np.isclose(existing_widths, half_width, atol=1.0e-12)):
            raise RuntimeError(
                "existing spectral-convergence rows use a different fixed window; "
                "start a new evidence run instead of mixing protocols"
            )
    for budget in budgets:
        existing_budgets = (
            set(frame["requested_subspace_size"].astype(int))
            if not frame.empty and "requested_subspace_size" in frame.columns
            else set()
        )
        if int(budget) in existing_budgets:
            continue

        spectrum_closed = bool(_acceptance(frame, width=half_width)["closed"])
        if spectrum_closed and not _observables_request_extension(output):
            break

        started = time.perf_counter()
        partial = solver(
            context.h_sector,
            target_energy=context.tower_energy,
            subspace_size=int(budget),
            tolerance=float(tolerance),
            maxiter=None,
            ncv_factor=2.05,
            random_seed=20260921 + int(budget),
        )
        elapsed = time.perf_counter() - started
        coverage = coverage_metrics(
            partial.energies,
            partial.residuals,
            target_energy=context.tower_energy,
            half_width=half_width,
            solver_tolerance=tolerance,
        )
        row = {
            "Lx": context.lx,
            "Ly": 4,
            "phase": context.phase,
            "sector_dimension": context.sector.sector_dimension,
            "target_energy": context.tower_energy,
            "window_protocol": "fixed_O1_total_energy",
            "window_half_width": half_width,
            "requested_subspace_size": int(budget),
            "returned_eigenpairs": int(partial.energies.size),
            "solver_tolerance": float(tolerance),
            "spectrum_method": partial.method,
            "runtime_seconds": float(elapsed),
            "peak_rss_gib": float(partial.peak_rss_gib or process_memory_gib()),
            "partial_maximum_residual": float(partial.maximum_residual),
            **coverage,
        }
        frame = _merge_row(convergence_path, row)
        atomic_write_json(
            output / "fixed_O1_spectral_stages" / f"budget_{int(budget):08d}.json",
            row,
        )

    acceptance = _acceptance(frame, width=half_width)
    acceptance.update(
        {
            "requested_budget_schedule": list(map(int, budgets)),
            "pilot_estimated_budget": estimate,
            "maximum_configured_budget": int(max_budget),
            "observables_requested_extension": _observables_request_extension(output),
        }
    )
    atomic_write_json(output / ACCEPTANCE_NAME, acceptance)
    if not acceptance["closed"]:
        raise RuntimeError(
            "Lx=12 fixed-O(1) window is not closed after the configured budgets; "
            f"see {output / ACCEPTANCE_NAME}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--budgets", default="auto")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-budget", type=int, default=DEFAULT_MAX_BUDGET)
    args = parser.parse_args()
    frame = run(
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        budgets_raw=args.budgets,
        tolerance=args.tolerance,
        max_budget=args.max_budget,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
