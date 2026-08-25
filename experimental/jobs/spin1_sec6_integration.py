#!/usr/bin/env python
"""Validate and integrate established Spin-1 Sec. VI evidence without rerunning solves.

This lane is deliberately post-processing only. It consumes immutable evidence-job
outputs, validates the already-established representative result, and exports stable
figure-data tables only when their numerical protocol is explicit in the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

REPRESENTATIVE_KAPPA_OVER_J = 0.10
PRIMARY_WINDOW_EXPONENT = 0.25
PRIMARY_WINDOW_PREFACTOR = 1.0
FIXED_CONTROL_HALF_WIDTH = 1.0
REFERENCE_L14_RAW_WIDTH = 0.0237316428
REFERENCE_L14_CLEAN_WIDTH = 0.0236713087
REFERENCE_L14_RAW_STATES = 4011
REFERENCE_L14_DARK_RANK = 1
REFERENCE_L14_REMOVED_FRACTION = 2.493e-4
REFERENCE_SECOND_BRIDGE_MAX = 1.0e-4


class EvidenceValidationError(RuntimeError):
    """Raised when established evidence fails its scientific validation contract."""


@dataclass(frozen=True, slots=True)
class IntegrationAudit:
    source_data_dir: str
    representative_l14_validated: bool
    sparse_budget_certified: bool
    exact_energy_tolerance_stable: bool
    beta0_second_bridge_trace_distance: float
    primary_window_available_sizes: tuple[int, ...]
    missing_primary_concentration_sizes: tuple[int, ...]
    common_window_status: str
    source_files: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceValidationError(f"required evidence file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"invalid JSON evidence file: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise EvidenceValidationError(f"required evidence file is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise EvidenceValidationError(f"invalid CSV evidence file: {path}") from exc
    if frame.empty:
        raise EvidenceValidationError(f"evidence table is empty: {path}")
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise EvidenceValidationError(
        f"none of the required columns are present: {', '.join(names)}"
    )


def _primary_window_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select W_L(gamma=1/4,c=1), refusing untagged legacy data."""

    mask = np.ones(len(frame), dtype=bool)
    matched = False
    if "window_exponent" in frame.columns:
        mask &= np.isclose(
            frame["window_exponent"].to_numpy(dtype=float), PRIMARY_WINDOW_EXPONENT
        )
        matched = True
    if "window_prefactor" in frame.columns:
        mask &= np.isclose(
            frame["window_prefactor"].to_numpy(dtype=float), PRIMARY_WINDOW_PREFACTOR
        )
        matched = True
    if not matched and "window_role" in frame.columns:
        roles = frame["window_role"].astype(str)
        mask &= roles.isin(
            {
                "alpha_0.25_c_1",
                "alpha_0p25_c_1",
                "quarter_power",
                "primary_quarter_power",
            }
        ).to_numpy()
        matched = True
    if not matched:
        raise EvidenceValidationError(
            "cannot identify the primary W_L(gamma=1/4,c=1) window in this table"
        )
    return mask


def _representative_mask(frame: pd.DataFrame) -> np.ndarray:
    if "kappa_over_J" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return np.isclose(
        frame["kappa_over_J"].to_numpy(dtype=float), REPRESENTATIVE_KAPPA_OVER_J
    )


def _validate_l14_concentration(source: Path) -> bool:
    frame = _read_csv(source / "spin1_xy_kappa0p1_concentration_L14.csv")
    if "L" in frame.columns:
        frame = frame[frame["L"].astype(int) == 14]
    frame = frame[_representative_mask(frame)]
    if "variant" not in frame.columns:
        raise EvidenceValidationError("L=14 concentration table has no raw/clean variant column")
    raw = frame[frame["variant"].astype(str) == "raw"]
    clean = frame[frame["variant"].astype(str) == "clean"]
    if len(raw) != 1 or len(clean) != 1:
        raise EvidenceValidationError(
            "expected exactly one raw and one clean representative L=14 concentration row"
        )
    width_column = _column(frame, "w_L", "largest_covariance_width")
    count_column = _column(frame, "window_state_count", "raw_window_state_count")
    dark_column = _column(frame, "joint_dark_rank", "removed_projector_rank")
    removed_column = _column(frame, "removed_fraction")
    raw_row = raw.iloc[0]
    clean_row = clean.iloc[0]
    anchors = (
        (float(raw_row[width_column]), REFERENCE_L14_RAW_WIDTH, "raw width"),
        (float(clean_row[width_column]), REFERENCE_L14_CLEAN_WIDTH, "clean width"),
    )
    for actual, expected, label in anchors:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5.0e-8):
            raise EvidenceValidationError(f"unexpected L=14 {label}: {actual:.10g}")
    if int(raw_row[count_column]) != REFERENCE_L14_RAW_STATES:
        raise EvidenceValidationError(
            f"unexpected L=14 raw window state count: {int(raw_row[count_column])}"
        )
    if int(raw_row[dark_column]) != REFERENCE_L14_DARK_RANK:
        raise EvidenceValidationError(
            f"unexpected L=14 joint-dark rank: {int(raw_row[dark_column])}"
        )
    removed_fraction = float(raw_row[removed_column])
    if not math.isclose(
        removed_fraction,
        REFERENCE_L14_REMOVED_FRACTION,
        rel_tol=5.0e-4,
        abs_tol=5.0e-8,
    ):
        raise EvidenceValidationError(
            f"unexpected L=14 removed fraction: {removed_fraction:.10g}"
        )
    if "sparse_convergence_passed" in frame.columns and not bool(
        frame["sparse_convergence_passed"].fillna(False).astype(bool).all()
    ):
        raise EvidenceValidationError("representative L=14 sparse-budget row is not certified")
    return True


def _validate_tolerance_audit(source: Path) -> bool:
    frame = _read_csv(source / "spin1_xy_kappa0p1_concentration_L14_tolerance_audit.csv")
    width_column = _column(frame, "largest_covariance_width", "w_L")
    if "variant" not in frame.columns:
        raise EvidenceValidationError("tolerance audit is missing the variant column")
    for _, group in frame.groupby("variant", sort=False):
        widths = group[width_column].to_numpy(dtype=float)
        blocks = group[_column(group, "energy_block_count")].to_numpy(dtype=int)
        if np.ptp(widths) > 1.0e-8 or not np.all(blocks == blocks[0]):
            raise EvidenceValidationError("exact-energy tolerance audit is not stable")
    return True


def _validate_beta0_bridges(source: Path) -> float:
    frame = _read_csv(source / "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv")
    if "L" not in frame.columns or "bridge" not in frame.columns:
        raise EvidenceValidationError("two-bridge table is missing L or bridge")
    mask = (frame["L"].astype(int) == 14).to_numpy()
    mask &= _representative_mask(frame)
    primary = frame[mask & _primary_window_mask(frame)]
    first = primary[primary["bridge"] == "mc_to_beta0_resolved"]
    second = primary[primary["bridge"] == "beta0_resolved_to_fixedM"]
    if len(first) != 1 or len(second) != 1:
        raise EvidenceValidationError("expected exactly one row for each L=14 beta-zero bridge")
    first_distance = float(first.iloc[0]["trace_distance"])
    second_distance = float(second.iloc[0]["trace_distance"])
    if not math.isfinite(first_distance) or not math.isfinite(second_distance):
        raise EvidenceValidationError("non-finite L=14 beta-zero bridge distance")
    if second_distance > REFERENCE_SECOND_BRIDGE_MAX:
        raise EvidenceValidationError(
            f"resolved-to-fixed-M beta-zero bridge is too large: {second_distance:.3e}"
        )
    if first_distance <= second_distance:
        raise EvidenceValidationError(
            "beta-zero bridge hierarchy changed: the first bridge should dominate at L=14"
        )
    return second_distance


def validate_established_evidence(source_data_dir: Path) -> dict[str, Any]:
    """Validate the completed August-20 representative evidence in place."""

    source = Path(source_data_dir).resolve(strict=False)
    summary_path = source / "spin1_xy_sec6_provisioning_summary.json"
    concentration_path = source / "spin1_xy_kappa0p1_concentration_L14.csv"
    tolerance_path = source / "spin1_xy_kappa0p1_concentration_L14_tolerance_audit.csv"
    bridges_path = source / "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv"
    summary = _read_json(summary_path)
    if not bool(summary.get("representative_sparse_budget_certified", False)):
        raise EvidenceValidationError(
            "representative L=14 sparse-budget certification is not passed"
        )
    return {
        "representative_l14_validated": _validate_l14_concentration(source),
        "sparse_budget_certified": True,
        "exact_energy_tolerance_stable": _validate_tolerance_audit(source),
        "beta0_second_bridge_trace_distance": _validate_beta0_bridges(source),
        "source_files": {
            path.name: _sha256(path)
            for path in (summary_path, concentration_path, tolerance_path, bridges_path)
        },
    }


def _standardize_panel_a(source: Path) -> pd.DataFrame:
    scatter_path = _first_existing(
        source,
        (
            "spin1_xy_kappa0p1_eth_scatter_Lmax.csv",
            "spin1_xy_kappa0p1_eth_scatter_all_sizes.csv",
        ),
    )
    if scatter_path is None:
        raise EvidenceValidationError("no representative ETH scatter table is available")
    scatter = _read_csv(scatter_path)
    if "L" in scatter.columns:
        scatter = scatter[scatter["L"].astype(int) == 12].copy()
    required = {"energy_density", "Q_A", "Q_Z", "Q_Y"}
    missing = required.difference(scatter.columns)
    if scatter.empty or missing:
        raise EvidenceValidationError(
            "representative L=12 scatter is missing data or columns: "
            + ", ".join(sorted(missing))
        )
    if "is_tower_state" not in scatter.columns:
        if "tower_overlap" in scatter.columns:
            scatter["is_tower_state"] = scatter["tower_overlap"].fillna(0.0) > 1.0 - 1.0e-7
        elif "is_exceptional" in scatter.columns:
            scatter["is_tower_state"] = scatter["is_exceptional"].fillna(False).astype(bool)
        else:
            scatter["is_tower_state"] = False
    columns = ["energy_density", "Q_A", "Q_Z", "Q_Y", "is_tower_state"]
    if "is_exceptional" in scatter.columns:
        columns.append("is_exceptional")
    output = scatter[columns].copy()
    output.insert(0, "L", 12)
    return output


def _standardize_primary_microcanonical(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "spin1_xy_kappa0p1_microcanonical_windows_sec6.csv")
    frame = frame[_primary_window_mask(frame) & _representative_mask(frame)].copy()
    frame = frame[frame["L"].astype(int).isin((8, 10, 12, 14))]
    if set(frame["L"].astype(int)) != {8, 10, 12, 14}:
        raise EvidenceValidationError(
            "primary microcanonical sequence is incomplete for L=8,10,12,14"
        )
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("L").iterrows():
        length = int(row["L"])
        for key in ("A", "Z", "Y"):
            rows.append(
                {
                    "L": length,
                    "witness": key,
                    "tau_mc_raw": float(row[_column(frame, f"tau_{key}_mc_raw")]),
                    "window_half_width": float(row["window_half_width"]),
                    "window_energy_density_half_width": float(row["window_half_width"])
                    / length,
                    "window_state_count": int(row["window_state_count"]),
                }
            )
    return pd.DataFrame(rows)


def _standardize_deformation(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "spin1_xy_kappa_matching_grid.csv")
    if "L" not in frame.columns or "kappa_over_J" not in frame.columns:
        raise EvidenceValidationError("deformation table is missing L or kappa_over_J")
    try:
        protocol_mask = _primary_window_mask(frame)
    except EvidenceValidationError as exc:
        raise EvidenceValidationError(
            "deformation grid does not certify the primary L^(1/4), c=1 window; "
            "do not reuse a mixed/legacy window for Fig. 6(c)"
        ) from exc
    frame = frame[
        protocol_mask
        & (frame["L"].astype(int) == 12).to_numpy()
        & (frame["kappa_over_J"].to_numpy(dtype=float) > 0.0)
    ].copy()
    if frame.empty:
        raise EvidenceValidationError("deformation table has no positive-kappa L=12 rows")
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("kappa_over_J").iterrows():
        for key in ("A", "Z", "Y"):
            column = _column(
                frame,
                f"tau_{key}_mc_raw",
                f"tau_{key}_mc_th",
                f"tau_{key}_mc_clean",
            )
            rows.append(
                {
                    "L": 12,
                    "kappa_over_J": float(row["kappa_over_J"]),
                    "witness": key,
                    "tau_mc_raw": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def _standardize_family_concentration_band(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "spin1_xy_kappa_concentration_grid.csv")
    if "L" not in frame.columns or "kappa_over_J" not in frame.columns:
        raise EvidenceValidationError("concentration grid is missing L or kappa_over_J")
    try:
        protocol_mask = _primary_window_mask(frame)
    except EvidenceValidationError as exc:
        raise EvidenceValidationError(
            "family concentration grid does not certify the primary L^(1/4), c=1 window; "
            "do not reuse it for the Fig. 6(d) band"
        ) from exc
    width_column = _column(
        frame, "w_L", "largest_covariance_width", "largest_covariance_width_raw"
    )
    mask = protocol_mask
    mask &= frame["L"].astype(int).isin((8, 10, 12)).to_numpy()
    mask &= frame["kappa_over_J"].to_numpy(dtype=float) > 0.0
    if "variant" in frame.columns:
        mask &= (
            frame["variant"]
            .astype(str)
            .isin(("raw", "raw_primary_with_clean_companion"))
            .to_numpy()
        )
    selected = frame[mask].copy()
    if selected.empty:
        raise EvidenceValidationError("no positive-kappa primary-window family concentration rows")
    rows = []
    for length, group in selected.groupby(selected["L"].astype(int), sort=True):
        widths = group[width_column].to_numpy(dtype=float)
        rows.append(
            {
                "L": int(length),
                "w_min": float(np.min(widths)),
                "w_max": float(np.max(widths)),
                "sampled_kappa_count": int(group["kappa_over_J"].nunique()),
                "sampled_kappa_min": float(group["kappa_over_J"].min()),
                "sampled_kappa_max": float(group["kappa_over_J"].max()),
            }
        )
    result = pd.DataFrame(rows).sort_values("L")
    if set(result["L"].astype(int)) != {8, 10, 12}:
        raise EvidenceValidationError(
            "primary family concentration band is incomplete for L=8,10,12"
        )
    return result


def _standardize_beta0_appendix(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv")
    frame = frame[_representative_mask(frame) & _primary_window_mask(frame)].copy()
    keep = [
        "L",
        "bridge",
        "trace_distance",
        "abs_delta_tau_A",
        "abs_delta_tau_Z",
        "abs_delta_tau_Y",
        "window_half_width",
        "window_state_count",
        "raw_window_state_count",
    ]
    return frame[[name for name in keep if name in frame.columns]].sort_values(
        ["bridge", "L"]
    )


def _standardize_obstruction(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "spin1_xy_complex_t2_obstruction_grid.csv")
    required = ["real_t2_over_J", "imag_t2_over_J", "normalized_tower_residual"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise EvidenceValidationError(
            f"complex-t2 obstruction table is missing columns: {', '.join(missing)}"
        )
    return frame[required].copy()


def _validate_common_window_table(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {
        "L",
        "kappa_over_J",
        "variant",
        "window_protocol",
        "window_half_width",
        "w_L",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise EvidenceValidationError(
            "common-window concentration table is missing columns: "
            + ", ".join(sorted(missing))
        )
    if not np.all(np.isfinite(frame["w_L"].to_numpy(dtype=float))):
        raise EvidenceValidationError("common-window concentration contains non-finite widths")
    if np.any(frame["w_L"].to_numpy(dtype=float) < 0.0):
        raise EvidenceValidationError("common-window concentration contains negative widths")
    for protocol, group in frame.groupby("window_protocol", sort=False):
        lengths = group["L"].to_numpy(dtype=float)
        half_widths = group["window_half_width"].to_numpy(dtype=float)
        if str(protocol) == "quarter_power_c1":
            expected = lengths**PRIMARY_WINDOW_EXPONENT
        elif str(protocol) == "fixed_width_1":
            expected = np.full_like(lengths, FIXED_CONTROL_HALF_WIDTH)
        else:
            raise EvidenceValidationError(f"unknown common-window protocol: {protocol}")
        if not np.allclose(half_widths, expected, rtol=0.0, atol=1.0e-10):
            raise EvidenceValidationError(
                f"common-window half-width does not match protocol {protocol}"
            )
    if "covered_spectral_half_width" in frame.columns and np.any(
        frame["covered_spectral_half_width"].to_numpy(dtype=float) + 1.0e-10
        < frame["window_half_width"].to_numpy(dtype=float)
    ):
        raise EvidenceValidationError("a common window extends beyond validated spectral coverage")
    if "window_max_eigenpair_residual" in frame.columns:
        residuals = frame["window_max_eigenpair_residual"].to_numpy(dtype=float)
        finite = residuals[np.isfinite(residuals)]
        if finite.size and float(np.max(finite)) > 1.0e-6:
            raise EvidenceValidationError(
                "common-window evidence has a physical eigenpair residual above 1e-6"
            )
    return frame


def build_figure_data(source_data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Export supported figure data and record unsupported products as pending."""

    source = Path(source_data_dir).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[Path], pd.DataFrame]] = {
        "spin1_xy_figure6_panel_a_scatter.csv": _standardize_panel_a,
        "spin1_xy_figure6_panel_b_witness_sequence.csv": _standardize_primary_microcanonical,
        "spin1_xy_figure6_panel_c_deformation.csv": _standardize_deformation,
        "spin1_xy_figure6_panel_d_family_band.csv": _standardize_family_concentration_band,
        "spin1_xy_appendix_beta0_bridges_data.csv": _standardize_beta0_appendix,
        "spin1_xy_appendix_complex_t2_obstruction_data.csv": _standardize_obstruction,
    }
    written: list[str] = []
    pending: dict[str, str] = {}
    for name, builder in builders.items():
        try:
            frame = builder(source)
        except (EvidenceValidationError, FileNotFoundError, KeyError, ValueError) as exc:
            pending[name] = str(exc)
            continue
        frame.to_csv(output / name, index=False)
        written.append(name)

    common_name = "spin1_xy_kappa0p1_concentration_common_windows.csv"
    source_common = source / common_name
    if source_common.is_file():
        try:
            common = _validate_common_window_table(source_common)
        except EvidenceValidationError as exc:
            pending[common_name] = str(exc)
        else:
            common.to_csv(output / common_name, index=False)
            written.append(common_name)
    else:
        pending[common_name] = "homogeneous common-window concentration has not been computed"
    return {
        "written": sorted(written),
        "pending": pending,
        "common_window_available": common_name in written,
    }


def _available_primary_concentration_sizes(source: Path) -> tuple[int, ...]:
    common = source / "spin1_xy_kappa0p1_concentration_common_windows.csv"
    if not common.is_file():
        return ()
    frame = _validate_common_window_table(common)
    selected = frame[
        np.isclose(frame["kappa_over_J"].to_numpy(dtype=float), REPRESENTATIVE_KAPPA_OVER_J)
        & (frame["variant"].astype(str) == "raw").to_numpy()
        & (frame["window_protocol"].astype(str) == "quarter_power_c1").to_numpy()
    ]
    return tuple(sorted(set(selected["L"].astype(int))))


def run_integration(source_data_dir: Path, output_dir: Path) -> IntegrationAudit:
    """Validate established evidence and export every figure datum needing no rerun."""

    source = Path(source_data_dir).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    validation = validate_established_evidence(source)
    figure_products = build_figure_data(source, output)
    available = _available_primary_concentration_sizes(source)
    missing = tuple(length for length in (8, 10, 12, 14) if length not in available)
    audit = IntegrationAudit(
        source_data_dir=str(source),
        representative_l14_validated=bool(validation["representative_l14_validated"]),
        sparse_budget_certified=bool(validation["sparse_budget_certified"]),
        exact_energy_tolerance_stable=bool(validation["exact_energy_tolerance_stable"]),
        beta0_second_bridge_trace_distance=float(
            validation["beta0_second_bridge_trace_distance"]
        ),
        primary_window_available_sizes=available,
        missing_primary_concentration_sizes=missing,
        common_window_status="READY" if not missing else "CACHE_ONLY_P0_A_PENDING",
        source_files=dict(validation["source_files"]),
    )
    report = {
        **asdict(audit),
        "figure_data_products": figure_products,
        "solve_policy": "disabled: validate/reuse established evidence; never solve implicitly",
        "next_numerical_action": (
            "none"
            if not missing
            else (
                "compute only missing complete 19-operator covariance from validated "
                "cached eigensystems"
            )
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "spin1_xy_sec6_integration_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = run_integration(args.source_data_dir, args.output_dir)
    print(json.dumps(asdict(audit), indent=2, sort_keys=True), flush=True)
    if audit.missing_primary_concentration_sizes:
        print(
            "P0-A remains pending for sizes "
            + ",".join(str(value) for value in audit.missing_primary_concentration_sizes)
            + "; no eigensolve was started.",
            flush=True,
        )


if __name__ == "__main__":
    main()
