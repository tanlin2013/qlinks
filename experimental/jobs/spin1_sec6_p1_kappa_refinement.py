#!/usr/bin/env python
"""Resumable midpoint refinement of the Sec. VI positive-kappa deformation grid."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    PRIMARY_WINDOW_PREFACTOR,
    PRIMARY_WINDOW_PROTOCOL,
)

TARGET_LENGTHS = (8, 10, 12)
MIDPOINT_KAPPA = (0.075, 0.125, 0.175)
P0_KAPPA = (0.05, 0.10, 0.15, 0.20)
WINDOW_PROTOCOL = PRIMARY_WINDOW_PROTOCOL
OPERATOR_BASIS_DIMENSION = 19

P0_ROWS_NAME = "spin1_xy_sec6_deformation_grid_rows.csv"
ROWS_NAME = "spin1_xy_sec6_p1_kappa_refinement_rows.csv"
SLOPES_NAME = "spin1_xy_sec6_p1_kappa_refinement_finite_differences.csv"
PROGRESS_NAME = "spin1_xy_sec6_p1_kappa_refinement_progress.json"
WORST_NAME = "spin1_xy_sec6_p1_kappa_refinement_worst_eigenoperators.csv"


class KappaRefinementError(RuntimeError):
    """Raised when the P1 midpoint refinement cannot be validated."""


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _slug(length: int, kappa_over_j: float) -> str:
    value = f"{float(kappa_over_j):.6f}".replace(".", "p")
    return f"spin1_L{int(length)}_kappa_p{value}_quarter_power"


def _checkpoint_dir(cache_root: Path, length: int, kappa_over_j: float) -> Path:
    return Path(cache_root) / "sec6_p1_kappa_refinement" / _slug(length, kappa_over_j)


def _validate_row(row: dict[str, Any], *, length: int, kappa_over_j: float) -> None:
    if int(row.get("L", -1)) != int(length):
        raise KappaRefinementError("checkpoint L mismatch")
    if not math.isclose(
        float(row.get("kappa_over_J", np.nan)),
        float(kappa_over_j),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise KappaRefinementError("checkpoint kappa mismatch")
    if str(row.get("window_protocol", "")) != WINDOW_PROTOCOL:
        raise KappaRefinementError("checkpoint window protocol mismatch")
    if int(row.get("operator_basis_dimension", -1)) != OPERATOR_BASIS_DIMENSION:
        raise KappaRefinementError("checkpoint operator-basis dimension mismatch")
    expected_half_width = PRIMARY_WINDOW_PREFACTOR * float(length) ** 0.25
    if not math.isclose(
        float(row.get("window_half_width", np.nan)),
        expected_half_width,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise KappaRefinementError("checkpoint primary half-width mismatch")
    for field in (
        "w_L_raw",
        "tau_A_mc_raw",
        "tau_Z_mc_raw",
        "tau_Y_mc_raw",
        "window_max_eigenpair_residual",
        "tower_residual",
    ):
        if not math.isfinite(float(row.get(field, np.nan))):
            raise KappaRefinementError(f"checkpoint has non-finite {field}")


def _load_checkpoint(
    cache_root: Path,
    *,
    length: int,
    kappa_over_j: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    directory = _checkpoint_dir(cache_root, length, kappa_over_j)
    row_path = directory / "row.json"
    worst_path = directory / "worst_eigenoperator.csv"
    metadata_path = directory / "metadata.json"
    present = [path.is_file() for path in (row_path, worst_path, metadata_path)]
    if not any(present):
        return None
    if not all(present):
        raise KappaRefinementError(
            f"partial checkpoint must be inspected, not overwritten: {directory}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = json.loads(row_path.read_text(encoding="utf-8"))
        worst = pd.read_csv(worst_path).to_dict(orient="records")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise KappaRefinementError(f"invalid checkpoint: {directory}") from exc
    if metadata.get("status") != "complete":
        raise KappaRefinementError(f"incomplete checkpoint must be inspected: {directory}")
    if metadata.get(EXCHANGE_CONVENTION_METADATA_KEY) != CURRENT_EXCHANGE_CONVENTION:
        raise KappaRefinementError(
            f"legacy midpoint checkpoint requires explicit convention migration: {directory}"
        )
    _validate_row(row, length=length, kappa_over_j=kappa_over_j)
    return row, worst


def _write_checkpoint(
    cache_root: Path,
    *,
    row: dict[str, Any],
    worst_rows: list[dict[str, Any]],
) -> None:
    length = int(row["L"])
    kappa_over_j = float(row["kappa_over_J"])
    directory = _checkpoint_dir(cache_root, length, kappa_over_j)
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    _atomic_write_json(directory / "row.json", row)
    worst_frame = pd.DataFrame(worst_rows)
    worst_frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    _atomic_write_csv(directory / "worst_eigenoperator.csv", worst_frame)
    _atomic_write_json(
        directory / "metadata.json",
        {
            "schema_version": 2,
            "status": "complete",
            "cache_role": "spin1_sec6_p1_midpoint_kappa_refinement",
            "L": length,
            "kappa_over_J": kappa_over_j,
            "window_protocol": WINDOW_PROTOCOL,
            "operator_basis_dimension": OPERATOR_BASIS_DIMENSION,
            "source_role": "p1_midpoint_dense_refinement",
            EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        },
    )


def _load_p0_rows(p0_data_dir: Path) -> pd.DataFrame:
    path = Path(p0_data_dir) / P0_ROWS_NAME
    if not path.is_file():
        raise KappaRefinementError(f"frozen P0 deformation grid is missing: {path}")
    frame = pd.read_csv(path)
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns or set(
        frame[EXCHANGE_CONVENTION_METADATA_KEY].astype(str)
    ) != {CURRENT_EXCHANGE_CONVENTION}:
        raise KappaRefinementError(
            "P0 deformation rows use historical exchange semantics; use the derived "
            "spin-1 convention-migration directory before P1 reuse"
        )
    selected = frame[
        frame["L"].astype(int).isin(TARGET_LENGTHS)
        & frame["kappa_over_J"].astype(float).isin(P0_KAPPA)
    ].copy()
    expected = {(length, kappa) for length in TARGET_LENGTHS for kappa in P0_KAPPA}
    actual = {
        (int(row.L), round(float(row.kappa_over_J), 12)) for row in selected.itertuples(index=False)
    }
    normalized_expected = {(length, round(kappa, 12)) for length, kappa in expected}
    if actual != normalized_expected:
        raise KappaRefinementError(
            "frozen P0 deformation grid is incomplete for the principal points"
        )
    return selected


def _compute_dense_point(length: int, kappa_over_j: float):
    grid = importlib.import_module("spin1_sec6_deformation_grid")
    row, worst_rows = grid._compute_dense_point(
        length=int(length),
        kappa_over_j=float(kappa_over_j),
    )
    row = dict(row)
    row["source_role"] = "p1_midpoint_dense_refinement"
    row["checkpoint_reused"] = False
    row[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    worst_rows = [
        dict(
            value,
            source_role="p1_midpoint_dense_refinement",
            **{EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION},
        )
        for value in worst_rows
    ]
    return row, worst_rows


def _finite_differences(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantities = ("w_L_raw", "tau_A_mc_raw", "tau_Z_mc_raw", "tau_Y_mc_raw")
    for length in TARGET_LENGTHS:
        group = frame[frame["L"].astype(int) == length].sort_values("kappa_over_J")
        kappas = group["kappa_over_J"].to_numpy(dtype=float)
        for quantity in quantities:
            values = group[quantity].to_numpy(dtype=float)
            for left in range(len(group) - 1):
                delta_kappa = float(kappas[left + 1] - kappas[left])
                slope = float((values[left + 1] - values[left]) / delta_kappa)
                rows.append(
                    {
                        "L": length,
                        "quantity": quantity,
                        "kappa_left": float(kappas[left]),
                        "kappa_right": float(kappas[left + 1]),
                        "delta_kappa": delta_kappa,
                        "finite_difference_slope": slope,
                        "absolute_slope": abs(slope),
                        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
                    }
                )
    return pd.DataFrame(rows)


def run(
    *,
    p0_data_dir: Path,
    cache_root: Path,
    output_dir: Path,
    compute_missing: bool,
) -> pd.DataFrame:
    p0 = _load_p0_rows(p0_data_dir)
    cache = Path(cache_root).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    midpoint_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for length in TARGET_LENGTHS:
        for kappa_over_j in MIDPOINT_KAPPA:
            slug = _slug(length, kappa_over_j)
            cached = _load_checkpoint(cache, length=length, kappa_over_j=kappa_over_j)
            if cached is not None:
                row, point_worst = cached
                row = dict(row)
                row["checkpoint_reused"] = True
                midpoint_rows.append(row)
                worst_rows.extend(point_worst)
                statuses[slug] = "reused"
                continue
            if not compute_missing:
                statuses[slug] = "pending"
                continue
            row, point_worst = _compute_dense_point(length, kappa_over_j)
            _validate_row(row, length=length, kappa_over_j=kappa_over_j)
            _write_checkpoint(cache, row=row, worst_rows=point_worst)
            midpoint_rows.append(row)
            worst_rows.extend(point_worst)
            statuses[slug] = "computed"

    midpoint = pd.DataFrame(midpoint_rows)
    combined = pd.concat([p0, midpoint], ignore_index=True, sort=False)
    combined[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    combined = combined.sort_values(["L", "kappa_over_J"]).reset_index(drop=True)
    _atomic_write_csv(output / ROWS_NAME, combined)
    if worst_rows:
        worst = pd.DataFrame(worst_rows).sort_values(["L", "kappa_over_J", "basis_operator"])
        worst[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
        _atomic_write_csv(output / WORST_NAME, worst)

    complete = all(status in {"computed", "reused"} for status in statuses.values())
    slopes = _finite_differences(combined) if complete else pd.DataFrame()
    if not slopes.empty:
        _atomic_write_csv(output / SLOPES_NAME, slopes)
    _atomic_write_json(
        output / PROGRESS_NAME,
        {
            "schema_version": 2,
            EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
            "target_lengths": list(TARGET_LENGTHS),
            "frozen_p0_kappa": list(P0_KAPPA),
            "midpoint_kappa": list(MIDPOINT_KAPPA),
            "full_sampled_kappa": sorted(set(P0_KAPPA + MIDPOINT_KAPPA)),
            "window_protocol": WINDOW_PROTOCOL,
            "window_prefactor": PRIMARY_WINDOW_PREFACTOR,
            "solve_policy": "new full-dense solves allowed only for the nine midpoint L<=12 points",
            "point_status": dict(sorted(statuses.items())),
            "complete": complete,
            "new_dense_target_count": len(TARGET_LENGTHS) * len(MIDPOINT_KAPPA),
            "new_dense_completed_count": sum(
                status in {"computed", "reused"} for status in statuses.values()
            ),
            "max_absolute_finite_difference_slope": (
                None
                if slopes.empty
                else {
                    quantity: float(group["absolute_slope"].max())
                    for quantity, group in slopes.groupby("quantity")
                }
            ),
            "claim_boundary": (
                "denser sampled positive-kappa evidence only; this does not by itself prove "
                "an open thermodynamic interval"
            ),
        },
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-data-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-missing", action="store_true")
    args = parser.parse_args()
    frame = run(
        p0_data_dir=args.p0_data_dir,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        compute_missing=args.compute_missing,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
