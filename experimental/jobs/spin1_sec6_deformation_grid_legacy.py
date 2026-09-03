#!/usr/bin/env python
"""Compute the remaining Spin-1 Sec. VI positive-kappa P0 grid resumably.

This lane is deliberately narrow. It reuses the completed representative
kappa/J=0.1 common-window products, and it permits new eigensolves only for the
missing L<=12 positive-kappa points needed by Fig. 6(c,d). Every completed
(L, kappa) point is checkpointed before aggregate CSVs are refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la

import spin1_sec6_provisioning as core

SCHEMA_VERSION = 1
TARGET_LENGTHS = (8, 10, 12)
KAPPA_GRID = (0.05, 0.10, 0.15, 0.20)
REPRESENTATIVE_KAPPA_OVER_J = 0.10
WINDOW_PROTOCOL = "quarter_power_c1"
WINDOW_EXPONENT = 0.25
WINDOW_PREFACTOR = 1.0
ENERGY_BLOCK_TOLERANCE = 1.0e-10
DENSE_RESIDUAL_TOLERANCE = 1.0e-8
TOWER_RESIDUAL_TOLERANCE = 1.0e-8
OPERATOR_BASIS_DIMENSION = 19
OPERATOR_BASIS_VERSION = "charge_conserving_two_site_hermitian_basis:v1"

COMMON_NAME = "spin1_xy_kappa0p1_concentration_common_windows.csv"
PANEL_B_NAME = "spin1_xy_figure6_panel_b_witness_sequence.csv"
REPRESENTATIVE_WORST_NAME = "spin1_xy_kappa0p1_common_window_worst_eigenoperator.csv"
GRID_ROWS_NAME = "spin1_xy_sec6_deformation_grid_rows.csv"
WORST_NAME = "spin1_xy_sec6_deformation_grid_worst_eigenoperators.csv"
PANEL_C_NAME = "spin1_xy_figure6_panel_c_deformation.csv"
PANEL_D_NAME = "spin1_xy_figure6_panel_d_family_band.csv"
PROGRESS_NAME = "spin1_xy_sec6_deformation_grid_progress.json"


class DeformationGridError(RuntimeError):
    """Raised when the remaining Sec. VI deformation grid cannot be validated."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256_strings(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _point_slug(length: int, kappa_over_j: float) -> str:
    sign = "p" if kappa_over_j >= 0.0 else "m"
    magnitude = f"{abs(float(kappa_over_j)):.6f}".replace(".", "p")
    return f"spin1_L{int(length)}_kappa_{sign}{magnitude}_quarter_power"


def _checkpoint_dir(cache_root: Path, length: int, kappa_over_j: float) -> Path:
    return Path(cache_root) / "sec6_deformation_grid" / _point_slug(length, kappa_over_j)


def _expected_half_width(length: int) -> float:
    return float(int(length) ** WINDOW_EXPONENT)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DeformationGridError(f"required table is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DeformationGridError(f"invalid CSV table: {path}") from exc
    if frame.empty:
        raise DeformationGridError(f"required table is empty: {path}")
    return frame


def _representative_points(
    integration_data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = Path(integration_data_dir)
    concentration = _read_csv(data / COMMON_NAME)
    sequence = _read_csv(data / PANEL_B_NAME)
    worst = _read_csv(data / REPRESENTATIVE_WORST_NAME)

    raw = concentration[
        (concentration["variant"].astype(str) == "raw")
        & (concentration["window_protocol"].astype(str) == WINDOW_PROTOCOL)
        & np.isclose(
            concentration["kappa_over_J"].to_numpy(dtype=float),
            REPRESENTATIVE_KAPPA_OVER_J,
        )
    ].copy()
    if set(raw["L"].astype(int)) != {8, 10, 12, 14}:
        raise DeformationGridError(
            "representative common-window table is incomplete for L=8,10,12,14"
        )

    points: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    for length in TARGET_LENGTHS:
        selected = raw[raw["L"].astype(int) == length]
        if len(selected) != 1:
            raise DeformationGridError(
                f"expected one representative raw common-window row at L={length}"
            )
        row = selected.iloc[0]
        witness_rows = sequence[sequence["L"].astype(int) == length]
        witness_values: dict[str, float] = {}
        for key in ("A", "Z", "Y"):
            witness = witness_rows[witness_rows["witness"].astype(str) == key]
            if len(witness) != 1:
                raise DeformationGridError(
                    f"representative panel-b table has no unique L={length} {key} row"
                )
            witness_values[key] = float(witness.iloc[0]["tau_mc_raw"])

        point_worst = worst[
            (worst["L"].astype(int) == length)
            & np.isclose(
                worst["kappa_over_J"].to_numpy(dtype=float),
                REPRESENTATIVE_KAPPA_OVER_J,
            )
            & (worst["variant"].astype(str) == "raw")
            & (worst["window_protocol"].astype(str) == WINDOW_PROTOCOL)
        ].copy()
        if point_worst.empty:
            raise DeformationGridError(
                f"representative worst-eigenoperator metadata is missing at L={length}"
            )
        coefficient_column = (
            "coefficient" if "coefficient" in point_worst.columns else "coefficient_real"
        )
        coefficients = point_worst[coefficient_column].to_numpy(dtype=float)
        dominant_index = int(np.argmax(np.abs(coefficients)))
        dominant = point_worst.iloc[dominant_index]
        basis_names = point_worst["basis_operator"].astype(str).tolist()

        points.append(
            {
                "schema_version": SCHEMA_VERSION,
                "L": int(length),
                "M": int(row.get("M", core.TOTAL_SZ)),
                "J3_over_J": float(row.get("J3_over_J", core.J3_OVER_J)),
                "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
                "window_protocol": WINDOW_PROTOCOL,
                "window_exponent": WINDOW_EXPONENT,
                "window_prefactor": WINDOW_PREFACTOR,
                "window_half_width": float(row["window_half_width"]),
                "window_state_count": int(row["window_state_count"]),
                "retained_state_count": int(
                    row.get("retained_state_count", row["window_state_count"])
                ),
                "joint_dark_rank": int(
                    row.get("joint_dark_rank", row.get("removed_projector_rank", 0))
                ),
                "joint_dark_fraction": float(row.get("removed_fraction", 0.0)),
                "energy_block_count": int(row["energy_block_count"]),
                "covered_spectral_half_width": float(row["covered_spectral_half_width"]),
                "window_max_eigenpair_residual": float(row["window_max_eigenpair_residual"]),
                "window_median_eigenpair_residual": float(
                    row.get("window_median_eigenpair_residual", np.nan)
                ),
                "tower_residual": float(row["tower_residual"]),
                "w_L_raw": float(row["w_L"]),
                "median_nonidentity_width_raw": float(row.get("median_nonidentity_width", np.nan)),
                "tau_A_mc_raw": witness_values["A"],
                "tau_Z_mc_raw": witness_values["Z"],
                "tau_Y_mc_raw": witness_values["Y"],
                "worst_basis_operator": str(dominant["basis_operator"]),
                "worst_basis_coefficient_abs": float(abs(coefficients[dominant_index])),
                "operator_basis_dimension": int(len(basis_names)),
                "operator_basis_version": OPERATOR_BASIS_VERSION,
                "operator_basis_sha256": _sha256_strings(basis_names),
                "spectrum_method": str(row.get("spectrum_method", "validated_reusable_spectrum")),
                "full_spectrum": True,
                "sector_dimension": int(
                    row.get(
                        "resolved_sector_dimension",
                        row.get("requested_eigenpairs", 0),
                    )
                ),
                "returned_eigenpairs": int(row.get("requested_eigenpairs", 0)),
                "solve_seconds": 0.0,
                "source_role": "reused_representative_common_window",
                "checkpoint_reused": True,
                "checkpoint_path": str(row.get("checkpoint_path", "")),
            }
        )
        for _, worst_row in point_worst.iterrows():
            coefficient = complex(float(worst_row[coefficient_column]))
            worst_rows.append(
                {
                    "L": int(length),
                    "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
                    "window_protocol": WINDOW_PROTOCOL,
                    "window_half_width": float(row["window_half_width"]),
                    "basis_operator": str(worst_row["basis_operator"]),
                    "coefficient_real": float(coefficient.real),
                    "coefficient_imag": float(coefficient.imag),
                    "coefficient_abs": float(abs(coefficient)),
                    "source_role": "reused_representative_common_window",
                }
            )
    return points, worst_rows


def _validate_point_record(
    record: dict[str, Any],
    *,
    length: int,
    kappa_over_j: float,
) -> None:
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise DeformationGridError("checkpoint schema version mismatch")
    if int(record.get("L", -1)) != int(length):
        raise DeformationGridError("checkpoint L mismatch")
    if not math.isclose(
        float(record.get("kappa_over_J", np.nan)),
        float(kappa_over_j),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise DeformationGridError("checkpoint kappa mismatch")
    if str(record.get("window_protocol", "")) != WINDOW_PROTOCOL:
        raise DeformationGridError("checkpoint window protocol mismatch")
    if not math.isclose(
        float(record.get("window_half_width", np.nan)),
        _expected_half_width(length),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise DeformationGridError("checkpoint half-width mismatch")
    if int(record.get("operator_basis_dimension", -1)) != OPERATOR_BASIS_DIMENSION:
        raise DeformationGridError("checkpoint operator-basis dimension mismatch")
    if str(record.get("operator_basis_version", "")) != OPERATOR_BASIS_VERSION:
        raise DeformationGridError("checkpoint operator-basis version mismatch")
    if int(record.get("window_state_count", 0)) <= 0:
        raise DeformationGridError("checkpoint has no microcanonical states")
    coverage = float(record.get("covered_spectral_half_width", 0.0))
    if coverage + 1.0e-10 < _expected_half_width(length):
        raise DeformationGridError("checkpoint does not cover the primary energy window")
    finite_fields = (
        "w_L_raw",
        "tau_A_mc_raw",
        "tau_Z_mc_raw",
        "tau_Y_mc_raw",
        "window_max_eigenpair_residual",
        "tower_residual",
    )
    for field in finite_fields:
        if not math.isfinite(float(record.get(field, np.nan))):
            raise DeformationGridError(f"checkpoint has non-finite {field}")
    if float(record["window_max_eigenpair_residual"]) > DENSE_RESIDUAL_TOLERANCE:
        raise DeformationGridError("checkpoint dense in-window eigenpair residual is too large")
    if float(record["tower_residual"]) > TOWER_RESIDUAL_TOLERANCE:
        raise DeformationGridError("checkpoint tower residual is too large")
    if float(record["w_L_raw"]) < 0.0:
        raise DeformationGridError("checkpoint has a negative covariance width")


def _load_checkpoint(
    cache_root: Path,
    *,
    length: int,
    kappa_over_j: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    directory = _checkpoint_dir(cache_root, length, kappa_over_j)
    metadata_path = directory / "metadata.json"
    row_path = directory / "row.json"
    worst_path = directory / "worst_eigenoperator.csv"
    if not (metadata_path.is_file() and row_path.is_file() and worst_path.is_file()):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = json.loads(row_path.read_text(encoding="utf-8"))
        worst = pd.read_csv(worst_path).to_dict(orient="records")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise DeformationGridError(f"invalid point checkpoint: {directory}") from exc
    if metadata.get("status") != "complete":
        return None
    _validate_point_record(row, length=length, kappa_over_j=kappa_over_j)
    if int(metadata.get("L", -1)) != length:
        raise DeformationGridError("checkpoint metadata L mismatch")
    if not math.isclose(
        float(metadata.get("kappa_over_J", np.nan)),
        float(kappa_over_j),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise DeformationGridError("checkpoint metadata kappa mismatch")
    return row, worst


def _write_checkpoint(
    cache_root: Path,
    row: dict[str, Any],
    worst_rows: list[dict[str, Any]],
) -> None:
    length = int(row["L"])
    kappa_over_j = float(row["kappa_over_J"])
    directory = _checkpoint_dir(cache_root, length, kappa_over_j)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "cache_role": "spin1_sec6_positive_kappa_primary_window",
        "L": length,
        "M": int(row["M"]),
        "J3_over_J": float(row["J3_over_J"]),
        "kappa_over_J": kappa_over_j,
        "solver": str(row["spectrum_method"]),
        "full_spectrum": bool(row["full_spectrum"]),
        "sector_dimension": int(row["sector_dimension"]),
        "returned_eigenpairs": int(row["returned_eigenpairs"]),
        "window_protocol": WINDOW_PROTOCOL,
        "window_half_width": float(row["window_half_width"]),
        "window_state_count": int(row["window_state_count"]),
        "covered_spectral_half_width": float(row["covered_spectral_half_width"]),
        "window_max_eigenpair_residual": float(row["window_max_eigenpair_residual"]),
        "tower_residual": float(row["tower_residual"]),
        "operator_basis_dimension": int(row["operator_basis_dimension"]),
        "operator_basis_version": str(row["operator_basis_version"]),
        "operator_basis_sha256": str(row["operator_basis_sha256"]),
        "source_role": str(row["source_role"]),
    }
    _atomic_write_json(directory / "row.json", row)
    _atomic_write_csv(pd.DataFrame(worst_rows), directory / "worst_eigenoperator.csv")
    _atomic_write_json(directory / "metadata.json", metadata)


def _compute_dense_point(
    *,
    length: int,
    kappa_over_j: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = core._point_context(length=length, kappa_over_j=kappa_over_j)
    solve_started = time.perf_counter()
    energies, vectors = la.eigh(context["h_sector"].toarray(), check_finite=False)
    solve_seconds = time.perf_counter() - solve_started
    half_width = _expected_half_width(length)
    coverage = float(min(abs(float(energies[0])), abs(float(energies[-1]))))
    if coverage + ENERGY_BLOCK_TOLERANCE < half_width:
        raise DeformationGridError(
            f"full dense L={length}, kappa/J={kappa_over_j:g} spectrum does not cover "
            f"the primary half-width {half_width:.6g}"
        )
    indices = core._window_indices(energies, half_width, ENERGY_BLOCK_TOLERANCE)
    maximum_residual, median_residual = core._window_residuals(
        context["h_sector"],
        energies,
        vectors,
        indices,
        chunk_size=64,
    )
    if not math.isfinite(maximum_residual) or maximum_residual > DENSE_RESIDUAL_TOLERANCE:
        raise DeformationGridError(
            f"dense L={length}, kappa/J={kappa_over_j:g} in-window residual "
            f"{maximum_residual:.3e} exceeds {DENSE_RESIDUAL_TOLERANCE:.1e}"
        )

    q_all = core._translated_joint_dark_operator(
        configs=context["configs"],
        sector=context["sector"],
        length=length,
    )
    exceptional, _dark_rows = core._joint_dark_kernel_from_spectrum(
        energies=energies,
        vectors=vectors,
        q_all=q_all,
        tower=context["tower"],
        energy_tolerance=ENERGY_BLOCK_TOLERANCE,
    )
    empty = np.zeros((context["sector"].sector_dimension, 0), dtype=np.complex128)
    sector_ops = context["pair"][4]
    raw_covariance = core.projector_deleted_block_covariance(
        energies,
        vectors,
        empty,
        sector_ops,
        indices,
        energy_tolerance=ENERGY_BLOCK_TOLERANCE,
        vector_tolerance=1.0e-9,
    )
    clean_covariance = core.projector_deleted_block_covariance(
        energies,
        vectors,
        exceptional,
        sector_ops,
        indices,
        energy_tolerance=ENERGY_BLOCK_TOLERANCE,
        vector_tolerance=1.0e-9,
    )
    witness_values = {
        key: float(
            core.spectral_observable_moments(
                operator,
                vectors,
                squared_operator=operator,
                indices=indices,
            ).mean
        )
        for key, operator in context["q_ops"].items()
    }
    tower_residual = float(
        core.diagnose_eigenpair(context["h_sector"], context["tower"]).residual_norm
    )
    if not math.isfinite(tower_residual) or tower_residual > TOWER_RESIDUAL_TOLERANCE:
        raise DeformationGridError(
            f"dense L={length}, kappa/J={kappa_over_j:g} tower residual "
            f"{tower_residual:.3e} exceeds {TOWER_RESIDUAL_TOLERANCE:.1e}"
        )

    basis_names = [str(value) for value in context["pair"][1]]
    if len(basis_names) != OPERATOR_BASIS_DIMENSION:
        raise DeformationGridError(
            f"expected {OPERATOR_BASIS_DIMENSION} two-site operators, found {len(basis_names)}"
        )
    coefficients = np.asarray(raw_covariance["worst_coefficients"], dtype=np.complex128)
    dominant_index = int(np.argmax(np.abs(coefficients)))
    row = {
        "schema_version": SCHEMA_VERSION,
        "L": int(length),
        "M": core.TOTAL_SZ,
        "J3_over_J": float(core.J3_OVER_J),
        "kappa_over_J": float(kappa_over_j),
        "window_protocol": WINDOW_PROTOCOL,
        "window_exponent": WINDOW_EXPONENT,
        "window_prefactor": WINDOW_PREFACTOR,
        "window_half_width": half_width,
        "window_state_count": int(raw_covariance["window_rank"]),
        "retained_state_count": int(clean_covariance["retained_rank"]),
        "joint_dark_rank": int(exceptional.shape[1]),
        "joint_dark_fraction": float(clean_covariance["removed_fraction"]),
        "energy_block_count": int(raw_covariance["energy_block_count"]),
        "covered_spectral_half_width": coverage,
        "window_max_eigenpair_residual": float(maximum_residual),
        "window_median_eigenpair_residual": float(median_residual),
        "tower_residual": tower_residual,
        "w_L_raw": float(raw_covariance["largest_width"]),
        "median_nonidentity_width_raw": float(raw_covariance["median_nonidentity_width"]),
        "tau_A_mc_raw": witness_values["A"],
        "tau_Z_mc_raw": witness_values["Z"],
        "tau_Y_mc_raw": witness_values["Y"],
        "worst_basis_operator": basis_names[dominant_index],
        "worst_basis_coefficient_abs": float(abs(coefficients[dominant_index])),
        "operator_basis_dimension": len(basis_names),
        "operator_basis_version": OPERATOR_BASIS_VERSION,
        "operator_basis_sha256": _sha256_strings(basis_names),
        "spectrum_method": "scipy.linalg.eigh",
        "full_spectrum": True,
        "sector_dimension": int(context["sector"].sector_dimension),
        "returned_eigenpairs": int(energies.size),
        "solve_seconds": float(solve_seconds),
        "source_role": "new_dense_positive_kappa_grid",
        "checkpoint_reused": False,
        "checkpoint_path": "",
    }
    worst_rows = [
        {
            "L": int(length),
            "kappa_over_J": float(kappa_over_j),
            "window_protocol": WINDOW_PROTOCOL,
            "window_half_width": half_width,
            "basis_operator": name,
            "coefficient_real": float(complex(coefficient).real),
            "coefficient_imag": float(complex(coefficient).imag),
            "coefficient_abs": float(abs(coefficient)),
            "source_role": "new_dense_positive_kappa_grid",
        }
        for name, coefficient in zip(basis_names, coefficients, strict=True)
    ]
    _validate_point_record(row, length=length, kappa_over_j=kappa_over_j)
    return row, worst_rows


def _panel_c(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[rows["L"].astype(int) == 12].sort_values("kappa_over_J")
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        for key in ("A", "Z", "Y"):
            records.append(
                {
                    "L": 12,
                    "kappa_over_J": float(row["kappa_over_J"]),
                    "witness": key,
                    "tau_mc_raw": float(row[f"tau_{key}_mc_raw"]),
                    "window_protocol": WINDOW_PROTOCOL,
                    "window_half_width": float(row["window_half_width"]),
                    "window_state_count": int(row["window_state_count"]),
                    "joint_dark_rank": int(row["joint_dark_rank"]),
                    "tower_residual": float(row["tower_residual"]),
                    "covered_spectral_half_width": float(row["covered_spectral_half_width"]),
                    "window_max_eigenpair_residual": float(row["window_max_eigenpair_residual"]),
                    "source_role": str(row["source_role"]),
                }
            )
    return pd.DataFrame(records)


def _panel_d(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for length in TARGET_LENGTHS:
        group = rows[rows["L"].astype(int) == length].sort_values("kappa_over_J")
        if group.empty:
            continue
        widths = group["w_L_raw"].to_numpy(dtype=float)
        kappas = group["kappa_over_J"].to_numpy(dtype=float)
        records.append(
            {
                "L": int(length),
                "w_min": float(np.min(widths)),
                "w_max": float(np.max(widths)),
                "sampled_kappa_count": int(group["kappa_over_J"].nunique()),
                "sampled_kappa_min": float(np.min(kappas)),
                "sampled_kappa_max": float(np.max(kappas)),
                "kappa_at_w_min": float(kappas[int(np.argmin(widths))]),
                "kappa_at_w_max": float(kappas[int(np.argmax(widths))]),
                "complete": bool(group["kappa_over_J"].nunique() == len(KAPPA_GRID)),
                "window_protocol": WINDOW_PROTOCOL,
            }
        )
    return pd.DataFrame(records)


def _refresh_aggregates(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    worst_rows: list[dict[str, Any]],
    statuses: dict[str, str],
) -> None:
    output = Path(output_dir)
    if records:
        rows = pd.DataFrame(records).sort_values(["L", "kappa_over_J"]).reset_index(drop=True)
        _atomic_write_csv(rows, output / GRID_ROWS_NAME)
        _atomic_write_csv(_panel_c(rows), output / PANEL_C_NAME)
        _atomic_write_csv(_panel_d(rows), output / PANEL_D_NAME)
    if worst_rows:
        worst = pd.DataFrame(worst_rows).sort_values(["L", "kappa_over_J", "basis_operator"])
        _atomic_write_csv(worst, output / WORST_NAME)

    completed = [name for name, status in statuses.items() if status in {"computed", "reused"}]
    pending = [name for name, status in statuses.items() if status == "pending"]
    panel_c_complete = all(
        statuses.get(_point_slug(12, kappa)) in {"computed", "reused"} for kappa in KAPPA_GRID
    )
    panel_d_complete = all(
        statuses.get(_point_slug(length, kappa)) in {"computed", "reused"}
        for length in TARGET_LENGTHS
        for kappa in KAPPA_GRID
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "solve_policy": (
            "explicit dense-only L<=12 positive-kappa grid; representative kappa/J=0.1 "
            "is reused; sparse and L>=14 solves are forbidden"
        ),
        "target_lengths": list(TARGET_LENGTHS),
        "kappa_grid": list(KAPPA_GRID),
        "window_protocol": WINDOW_PROTOCOL,
        "window_exponent": WINDOW_EXPONENT,
        "window_prefactor": WINDOW_PREFACTOR,
        "point_status": dict(sorted(statuses.items())),
        "completed_count": len(completed),
        "target_count": len(TARGET_LENGTHS) * len(KAPPA_GRID),
        "pending_points": sorted(pending),
        "panel_c_complete": panel_c_complete,
        "panel_d_complete": panel_d_complete,
        "p0_grid_complete": panel_c_complete and panel_d_complete,
    }
    _atomic_write_json(output / PROGRESS_NAME, manifest)


def run_grid(
    *,
    integration_data_dir: Path,
    cache_root: Path,
    output_dir: Path,
    compute_missing: bool,
) -> pd.DataFrame:
    output = Path(output_dir).resolve(strict=False)
    cache = Path(cache_root).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    representative_records, representative_worst = _representative_points(integration_data_dir)
    representative_by_length = {
        int(record["L"]): (
            record,
            [row for row in representative_worst if int(row["L"]) == int(record["L"])],
        )
        for record in representative_records
    }

    records: list[dict[str, Any]] = []
    all_worst_rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}

    for length in TARGET_LENGTHS:
        for kappa_over_j in KAPPA_GRID:
            slug = _point_slug(length, kappa_over_j)
            try:
                cached = _load_checkpoint(
                    cache,
                    length=length,
                    kappa_over_j=kappa_over_j,
                )
            except DeformationGridError:
                cached = None
            if cached is not None:
                row, worst_rows = cached
                row = dict(row)
                row["checkpoint_reused"] = True
                records.append(row)
                all_worst_rows.extend(worst_rows)
                statuses[slug] = "reused"
                _refresh_aggregates(
                    output_dir=output,
                    records=records,
                    worst_rows=all_worst_rows,
                    statuses=statuses,
                )
                continue

            if math.isclose(
                kappa_over_j,
                REPRESENTATIVE_KAPPA_OVER_J,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                row, worst_rows = representative_by_length[length]
                _validate_point_record(
                    row,
                    length=length,
                    kappa_over_j=kappa_over_j,
                )
                _write_checkpoint(cache, row, worst_rows)
                records.append(dict(row))
                all_worst_rows.extend(worst_rows)
                statuses[slug] = "reused"
                _refresh_aggregates(
                    output_dir=output,
                    records=records,
                    worst_rows=all_worst_rows,
                    statuses=statuses,
                )
                continue

            if not compute_missing:
                statuses[slug] = "pending"
                _refresh_aggregates(
                    output_dir=output,
                    records=records,
                    worst_rows=all_worst_rows,
                    statuses=statuses,
                )
                continue

            row, worst_rows = _compute_dense_point(
                length=length,
                kappa_over_j=kappa_over_j,
            )
            _write_checkpoint(cache, row, worst_rows)
            records.append(row)
            all_worst_rows.extend(worst_rows)
            statuses[slug] = "computed"
            _refresh_aggregates(
                output_dir=output,
                records=records,
                worst_rows=all_worst_rows,
                statuses=statuses,
            )

    if not records:
        raise DeformationGridError("no reusable or computed deformation-grid rows")
    final = pd.DataFrame(records).sort_values(["L", "kappa_over_J"]).reset_index(drop=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-data-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="Explicitly allow only the missing full-dense L<=12 positive-kappa solves.",
    )
    args = parser.parse_args()
    frame = run_grid(
        integration_data_dir=args.integration_data_dir,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        compute_missing=args.compute_missing,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
