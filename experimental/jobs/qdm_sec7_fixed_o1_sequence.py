#!/usr/bin/env python
"""Assemble the homogeneous Lx=4,8,12 fixed-O(1) Sec. VII sequence.

This stage is solver-free. It combines the exact-ED pilot rows with the accepted
Lx=12 primary-window row and writes descriptive three-size fit diagnostics
without selecting a thermodynamic extrapolation or reporting a scaling exponent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdm_sec7_fixed_o1 import atomic_write_csv, atomic_write_json
from qdm_sec7_fixed_o1_l12_observables import (
    ACCEPTANCE_NAME as OBSERVABLES_ACCEPTANCE_NAME,
)
from qdm_sec7_fixed_o1_l12_observables import THERMAL_NAME
from qdm_sec7_fixed_o1_l12_spectrum import RECOMMENDATION_NAME
from qdm_sec7_fixed_o1_pilot import SYSTEMATICS_NAME as PILOT_SYSTEMATICS_NAME

SEQUENCE_NAME = "qdm_checkerboard_fixed_O1_three_size_sequence.csv"
FIT_NAME = "qdm_checkerboard_fixed_O1_three_size_fit_diagnostics.csv"
STATUS_NAME = "qdm_checkerboard_fixed_O1_three_size_status.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _primary_width(output_dir: Path) -> float:
    recommendation = _load_json(Path(output_dir) / RECOMMENDATION_NAME)
    if recommendation.get("status") != "recommended":
        raise RuntimeError("fixed-O(1) pilot recommendation is not accepted")
    raw = recommendation.get("recommended_half_width")
    if raw is None or float(raw) <= 0:
        raise RuntimeError("fixed-O(1) recommendation has no positive primary width")
    return float(raw)


def _pick_pilot_row(frame: pd.DataFrame, *, lx: int, width: float) -> pd.Series:
    rows = frame[
        (frame["Lx"].astype(int) == int(lx))
        & np.isclose(frame["window_half_width"].astype(float), float(width), atol=1.0e-12)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one exact-ED pilot row for Lx={lx}, DeltaE={width}; got {len(rows)}"
        )
    return rows.iloc[0]


def _pick_l12_row(frame: pd.DataFrame, *, width: float) -> pd.Series:
    rows = frame[
        (frame["Lx"].astype(int) == 12)
        & np.isclose(frame["window_half_width"].astype(float), float(width), atol=1.0e-12)
    ].sort_values("requested_subspace_size")
    if rows.empty:
        raise RuntimeError(f"no accepted Lx=12 primary row for DeltaE={width}")
    return rows.iloc[-1]


def _sequence_row(source: pd.Series, *, lx: int, width: float) -> dict[str, Any]:
    required = (
        "phase",
        "raw_window_state_count",
        "joint_dark_removed_rank",
        "removed_fraction",
        "tau_A_mc_raw",
        "tau_Z_mc_raw",
        "matching_distance_raw",
        "w_raw",
    )
    missing = [name for name in required if name not in source.index or pd.isna(source[name])]
    if missing:
        raise RuntimeError(f"Lx={lx} fixed-window row is missing required values: {missing}")
    return {
        "Lx": int(lx),
        "Ly": 4,
        "phase": float(source["phase"]),
        "window_protocol": "fixed_O1_total_energy",
        "window_half_width": float(width),
        "window_energy_density_half_width": float(width) / int(lx),
        "raw_window_state_count": int(source["raw_window_state_count"]),
        "entropy_proxy_log_count_over_Lx": float(
            math.log(max(1, int(source["raw_window_state_count"]))) / int(lx)
        ),
        "joint_dark_removed_rank": int(source["joint_dark_removed_rank"]),
        "removed_fraction": float(source["removed_fraction"]),
        "tau_A_mc_raw": float(source["tau_A_mc_raw"]),
        "tau_Z_mc_raw": float(source["tau_Z_mc_raw"]),
        "tau_A_can": float(
            source["tau_A_can"] if "tau_A_can" in source.index else source["tau_A_can_raw"]
        ),
        "tau_Z_can": float(
            source["tau_Z_can"] if "tau_Z_can" in source.index else source["tau_Z_can_raw"]
        ),
        "matching_distance_raw": float(source["matching_distance_raw"]),
        "w_raw": float(source["w_raw"]),
        "source_requested_subspace_size": (
            int(source["requested_subspace_size"])
            if "requested_subspace_size" in source.index
            and not pd.isna(source["requested_subspace_size"])
            else None
        ),
    }


def _fit_rows(sequence: pd.DataFrame) -> pd.DataFrame:
    sizes = sequence["Lx"].astype(float).to_numpy()
    metrics = ("tau_A_mc_raw", "tau_Z_mc_raw", "matching_distance_raw", "w_raw")
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = sequence[metric].astype(float).to_numpy()
        constant = float(np.mean(values))
        residual = values - constant
        rows.append(
            {
                "metric": metric,
                "model": "constant",
                "intercept": constant,
                "slope": math.nan,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "n_sizes": int(values.size),
                "selected_model": False,
            }
        )
        design = np.column_stack([np.ones_like(sizes), 1.0 / sizes])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coefficients
        rows.append(
            {
                "metric": metric,
                "model": "a_plus_b_over_Lx",
                "intercept": float(coefficients[0]),
                "slope": float(coefficients[1]),
                "rmse": float(np.sqrt(np.mean((values - fitted) ** 2))),
                "n_sizes": int(values.size),
                "selected_model": False,
            }
        )
    return pd.DataFrame(rows)


def run(*, output_dir: Path) -> pd.DataFrame:
    output = Path(output_dir)
    width = _primary_width(output)
    acceptance = _load_json(output / OBSERVABLES_ACCEPTANCE_NAME)
    if not bool(acceptance.get("closed")):
        raise RuntimeError("Lx=12 fixed-O(1) observable acceptance is not closed")

    pilot_path = output / PILOT_SYSTEMATICS_NAME
    l12_path = output / THERMAL_NAME
    if not pilot_path.is_file() or not l12_path.is_file():
        raise FileNotFoundError(
            f"missing P0-F prerequisite: pilot={pilot_path.is_file()}, L12={l12_path.is_file()}"
        )
    pilot = pd.read_csv(pilot_path)
    l12 = pd.read_csv(l12_path)

    rows = [
        _sequence_row(_pick_pilot_row(pilot, lx=4, width=width), lx=4, width=width),
        _sequence_row(_pick_pilot_row(pilot, lx=8, width=width), lx=8, width=width),
        _sequence_row(_pick_l12_row(l12, width=width), lx=12, width=width),
    ]
    sequence = pd.DataFrame(rows).sort_values("Lx").reset_index(drop=True)
    atomic_write_csv(output / SEQUENCE_NAME, sequence)
    diagnostics = _fit_rows(sequence)
    atomic_write_csv(output / FIT_NAME, diagnostics)
    atomic_write_json(
        output / STATUS_NAME,
        {
            "schema_version": 1,
            "closed": True,
            "window_half_width": width,
            "sizes": [4, 8, 12],
            "raw_window_state_counts": sequence["raw_window_state_count"].astype(int).tolist(),
            "claim_boundary": (
                "Three-size fits are descriptive diagnostics only. No model is selected, "
                "no scaling exponent is reported, and no thermodynamic asymptote is claimed."
            ),
        },
    )
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = run(output_dir=args.output_dir)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
