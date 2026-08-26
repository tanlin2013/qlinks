#!/usr/bin/env python
"""Refresh the Spin-1 Sec. VI integration audit after common-window completion.

The original integration audit predates the later common-window covariance pass.
This post-processing-only refresher validates the authoritative Aug-20 evidence
and the completed Aug-25 integration products, then records the remaining P0
deformation-grid/rendering status. It never invokes an eigensolver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import spin1_sec6_integration as integration

REPRESENTATIVE_KAPPA_OVER_J = 0.10
PRIMARY_WINDOW_PROTOCOL = "quarter_power_c1"
FIXED_WINDOW_PROTOCOL = "fixed_width_1"
TARGET_LENGTHS = (8, 10, 12, 14)
GRID_PROGRESS_NAME = "spin1_xy_sec6_deformation_grid_progress.json"

REPRESENTATIVE_PRODUCTS = (
    "spin1_xy_kappa0p1_concentration_common_windows.csv",
    "spin1_xy_kappa0p1_common_window_summary.json",
    "spin1_xy_kappa0p1_common_window_checkpoint_audit.csv",
    "spin1_xy_kappa0p1_common_window_tolerance_audit.csv",
    "spin1_xy_kappa0p1_common_window_worst_eigenoperator.csv",
    "spin1_xy_figure6_panel_a_scatter.csv",
    "spin1_xy_figure6_panel_b_witness_sequence.csv",
    "spin1_xy_appendix_beta0_bridges_data.csv",
    "spin1_xy_appendix_complex_t2_obstruction_data.csv",
)


class IntegrationAuditRefreshError(RuntimeError):
    """Raised when the completed Sec. VI integration evidence is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrationAuditRefreshError(f"required JSON product is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationAuditRefreshError(f"invalid JSON product: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrationAuditRefreshError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise IntegrationAuditRefreshError(f"required CSV product is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise IntegrationAuditRefreshError(f"invalid CSV product: {path}") from exc
    if frame.empty:
        raise IntegrationAuditRefreshError(f"required CSV product is empty: {path}")
    return frame


def _validate_common_windows(data_dir: Path) -> pd.DataFrame:
    frame = _read_csv(data_dir / "spin1_xy_kappa0p1_concentration_common_windows.csv")
    required = {
        "L",
        "kappa_over_J",
        "variant",
        "window_protocol",
        "window_half_width",
        "w_L",
        "window_state_count",
        "covered_spectral_half_width",
        "window_max_eigenpair_residual",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise IntegrationAuditRefreshError(
            "common-window table is missing columns: " + ", ".join(sorted(missing))
        )
    selected = frame[
        np.isclose(
            frame["kappa_over_J"].to_numpy(dtype=float),
            REPRESENTATIVE_KAPPA_OVER_J,
        )
    ].copy()
    for protocol in (PRIMARY_WINDOW_PROTOCOL, FIXED_WINDOW_PROTOCOL):
        group = selected[selected["window_protocol"].astype(str) == protocol]
        for variant in ("raw", "clean"):
            rows = group[group["variant"].astype(str) == variant]
            if set(rows["L"].astype(int)) != set(TARGET_LENGTHS):
                raise IntegrationAuditRefreshError(
                    f"{protocol} {variant} sequence is incomplete for L=8,10,12,14"
                )
    primary_raw = selected[
        (selected["window_protocol"].astype(str) == PRIMARY_WINDOW_PROTOCOL)
        & (selected["variant"].astype(str) == "raw")
    ].sort_values("L")
    expected_half_widths = primary_raw["L"].to_numpy(dtype=float) ** 0.25
    if not np.allclose(
        primary_raw["window_half_width"].to_numpy(dtype=float),
        expected_half_widths,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise IntegrationAuditRefreshError(
            "primary common-window half-widths do not equal L^(1/4)"
        )
    fixed = selected[
        selected["window_protocol"].astype(str) == FIXED_WINDOW_PROTOCOL
    ]
    if not np.allclose(
        fixed["window_half_width"].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise IntegrationAuditRefreshError("fixed common-window half-width is not 1")
    if np.any(
        selected["covered_spectral_half_width"].to_numpy(dtype=float) + 1.0e-10
        < selected["window_half_width"].to_numpy(dtype=float)
    ):
        raise IntegrationAuditRefreshError(
            "a completed common window exceeds validated spectral coverage"
        )
    residuals = selected["window_max_eigenpair_residual"].to_numpy(dtype=float)
    if not np.all(np.isfinite(residuals)):
        raise IntegrationAuditRefreshError(
            "common-window residual diagnostics contain non-finite values"
        )
    if not np.all(np.isfinite(selected["w_L"].to_numpy(dtype=float))):
        raise IntegrationAuditRefreshError("common-window widths contain non-finite values")
    return selected.sort_values(["window_protocol", "L", "variant"]).reset_index(
        drop=True
    )


def _validate_common_summary(data_dir: Path) -> dict[str, Any]:
    summary = _read_json(data_dir / "spin1_xy_kappa0p1_common_window_summary.json")
    if bool(summary.get("power_law_fit_computed", True)):
        raise IntegrationAuditRefreshError(
            "common-window summary unexpectedly reports a fitted power law"
        )
    lengths = tuple(int(value) for value in summary.get("lengths", ()))
    if set(lengths) != set(TARGET_LENGTHS):
        raise IntegrationAuditRefreshError(
            "common-window summary does not cover L=8,10,12,14"
        )
    protocols = set(str(value) for value in summary.get("window_protocols", ()))
    required = {PRIMARY_WINDOW_PROTOCOL, FIXED_WINDOW_PROTOCOL}
    if not required.issubset(protocols):
        raise IntegrationAuditRefreshError(
            "common-window summary does not contain both locked protocols"
        )
    return summary


def _validate_completed_products(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in REPRESENTATIVE_PRODUCTS:
        path = data_dir / name
        if name.endswith(".csv"):
            _read_csv(path)
        else:
            _read_json(path)
        hashes[name] = _sha256(path)
    return hashes


def _grid_status(data_dir: Path) -> dict[str, Any]:
    path = data_dir / GRID_PROGRESS_NAME
    if not path.is_file():
        return {
            "present": False,
            "panel_c_complete": False,
            "panel_d_complete": False,
            "p0_grid_complete": False,
            "pending_points": [],
        }
    progress = _read_json(path)
    return {
        "present": True,
        "panel_c_complete": bool(progress.get("panel_c_complete", False)),
        "panel_d_complete": bool(progress.get("panel_d_complete", False)),
        "p0_grid_complete": bool(progress.get("p0_grid_complete", False)),
        "pending_points": list(progress.get("pending_points", ())),
        "completed_count": int(progress.get("completed_count", 0)),
        "target_count": int(progress.get("target_count", 0)),
        "sha256": _sha256(path),
    }


def refresh_audit(
    *,
    source_data_dir: Path,
    integration_data_dir: Path,
) -> dict[str, Any]:
    source = Path(source_data_dir).resolve(strict=False)
    data = Path(integration_data_dir).resolve(strict=False)
    data.mkdir(parents=True, exist_ok=True)

    validation = integration.validate_established_evidence(source)
    common_frame = _validate_common_windows(data)
    common_summary = _validate_common_summary(data)
    product_hashes = _validate_completed_products(data)
    grid = _grid_status(data)

    primary_raw = common_frame[
        (common_frame["window_protocol"].astype(str) == PRIMARY_WINDOW_PROTOCOL)
        & (common_frame["variant"].astype(str) == "raw")
    ].sort_values("L")
    fixed_raw = common_frame[
        (common_frame["window_protocol"].astype(str) == FIXED_WINDOW_PROTOCOL)
        & (common_frame["variant"].astype(str) == "raw")
    ].sort_values("L")

    remaining: list[str] = []
    if not grid["panel_c_complete"]:
        remaining.append("common_window_deformation_grid_for_figure6c")
    if not grid["panel_d_complete"]:
        remaining.append("positive_kappa_family_band_for_figure6d")
    remaining.append("render_final_prx_and_appendix_d_figures")

    figure_written = [
        "spin1_xy_figure6_panel_a_scatter.csv",
        "spin1_xy_figure6_panel_b_witness_sequence.csv",
        "spin1_xy_appendix_beta0_bridges_data.csv",
        "spin1_xy_appendix_complex_t2_obstruction_data.csv",
        "spin1_xy_kappa0p1_concentration_common_windows.csv",
    ]
    figure_pending: dict[str, str] = {}
    if not grid["panel_c_complete"]:
        figure_pending["spin1_xy_figure6_panel_c_deformation.csv"] = (
            "positive-kappa primary-window L=12 grid is incomplete"
        )
    else:
        figure_written.append("spin1_xy_figure6_panel_c_deformation.csv")
    if not grid["panel_d_complete"]:
        figure_pending["spin1_xy_figure6_panel_d_family_band.csv"] = (
            "positive-kappa primary-window L<=12 covariance grid is incomplete"
        )
    else:
        figure_written.append("spin1_xy_figure6_panel_d_family_band.csv")

    report = {
        "schema_version": 2,
        "source_data_dir": str(source),
        "integration_data_dir": str(data),
        "representative_l14_validated": bool(
            validation["representative_l14_validated"]
        ),
        "sparse_budget_certified": bool(validation["sparse_budget_certified"]),
        "exact_energy_tolerance_stable": bool(
            validation["exact_energy_tolerance_stable"]
        ),
        "beta0_second_bridge_trace_distance": float(
            validation["beta0_second_bridge_trace_distance"]
        ),
        "primary_window_available_sizes": list(TARGET_LENGTHS),
        "missing_primary_concentration_sizes": [],
        "common_window_status": "READY",
        "representative_common_window_closed": True,
        "primary_raw_widths": {
            str(int(row.L)): float(row.w_L)
            for row in primary_raw.itertuples(index=False)
        },
        "fixed_width_raw_widths": {
            str(int(row.L)): float(row.w_L)
            for row in fixed_raw.itertuples(index=False)
        },
        "common_window_power_law_fit_computed": bool(
            common_summary.get("power_law_fit_computed", False)
        ),
        "figure_data_products": {
            "written": sorted(figure_written),
            "pending": figure_pending,
            "common_window_available": True,
        },
        "deformation_grid": grid,
        "source_files": dict(validation["source_files"]),
        "integration_product_sha256": dict(sorted(product_hashes.items())),
        "solve_policy": (
            "representative P0 is closed; no implicit eigensolver fallback. "
            "Only the explicit dense L<=12 positive-kappa deformation-grid stage "
            "may start new solves."
        ),
        "remaining_p0": remaining,
        "next_numerical_action": (
            "none; render final figures from frozen CSVs"
            if grid["p0_grid_complete"]
            else (
                "compute or reuse only missing L<=12 positive-kappa primary-window "
                "deformation-grid points"
            )
        ),
    }
    _atomic_write_json(data / "spin1_xy_sec6_integration_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--integration-data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = refresh_audit(
        source_data_dir=args.source_data_dir,
        integration_data_dir=args.integration_data_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
