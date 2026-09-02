#!/usr/bin/env python
"""Current-convention Sec. VI evidence integration adapter.

The August integration logic is preserved in ``spin1_sec6_integration_legacy``.
This active entry point requires an explicitly convention-mapped source directory,
selects the permanent c=1/2 windows, and stamps every derived product with the
exchange convention so renderers cannot silently consume historical units.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import spin1_exchange_convention as _convention
import spin1_sec6_integration_legacy as _legacy

_ORIGINAL_VALIDATE_ESTABLISHED_EVIDENCE = _legacy.validate_established_evidence
_ORIGINAL_BUILD_FIGURE_DATA = _legacy.build_figure_data

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

EvidenceValidationError = _legacy.EvidenceValidationError
REPRESENTATIVE_KAPPA_OVER_J = _legacy.REPRESENTATIVE_KAPPA_OVER_J
_read_csv = _legacy._read_csv

PRIMARY_WINDOW_EXPONENT = _convention.PRIMARY_WINDOW_EXPONENT
PRIMARY_WINDOW_PREFACTOR = _convention.PRIMARY_WINDOW_PREFACTOR
FIXED_CONTROL_HALF_WIDTH = _convention.FIXED_CONTROL_HALF_WIDTH
PRIMARY_WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
FIXED_WINDOW_PROTOCOL = _convention.FIXED_WINDOW_PROTOCOL
EXCHANGE_CONVENTION_METADATA_KEY = _convention.EXCHANGE_CONVENTION_METADATA_KEY
CURRENT_EXCHANGE_CONVENTION = _convention.CURRENT_EXCHANGE_CONVENTION


def _primary_window_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select the permanent W_L(gamma=1/4,c=1/2) protocol."""

    mask = np.ones(len(frame), dtype=bool)
    matched = False
    if "window_exponent" in frame.columns:
        mask &= np.isclose(
            frame["window_exponent"].to_numpy(dtype=float),
            PRIMARY_WINDOW_EXPONENT,
        )
        matched = True
    if "window_prefactor" in frame.columns:
        mask &= np.isclose(
            frame["window_prefactor"].to_numpy(dtype=float),
            PRIMARY_WINDOW_PREFACTOR,
        )
        matched = True
    if "window_protocol" in frame.columns:
        mask &= frame["window_protocol"].astype(str).eq(PRIMARY_WINDOW_PROTOCOL).to_numpy()
        matched = True
    if not matched and "window_role" in frame.columns:
        roles = frame["window_role"].astype(str)
        mask &= roles.isin(
            {
                "alpha_0.25_c_0.5",
                "alpha_0p25_c_0p5",
                PRIMARY_WINDOW_PROTOCOL,
                "primary_quarter_power_c0p5",
            }
        ).to_numpy()
        matched = True
    if not matched:
        raise EvidenceValidationError(
            "cannot identify the primary W_L(gamma=1/4,c=1/2) window in this table"
        )
    return mask


def _require_current_source(source_data_dir: Path) -> None:
    source = Path(source_data_dir).resolve(strict=False)
    manifest_path = source / "spin1_exchange_convention_migration_manifest.json"
    if not manifest_path.is_file():
        raise EvidenceValidationError(
            "Sec. VI integration refuses an unstamped historical evidence directory. "
            "Run spin1_exchange_convention_migrate_evidence.py and use the derived directory."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(
            f"invalid convention migration manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get(EXCHANGE_CONVENTION_METADATA_KEY) != (
        CURRENT_EXCHANGE_CONVENTION
    ):
        raise EvidenceValidationError(
            f"migration manifest does not declare {CURRENT_EXCHANGE_CONVENTION!r}"
        )


def _require_current_frame(frame: pd.DataFrame, *, role: str) -> None:
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        raise EvidenceValidationError(f"{role} has no explicit spin-1 exchange convention")
    values = set(frame[EXCHANGE_CONVENTION_METADATA_KEY].dropna().astype(str))
    if values != {CURRENT_EXCHANGE_CONVENTION}:
        raise EvidenceValidationError(
            f"{role} exchange convention mismatch: expected {CURRENT_EXCHANGE_CONVENTION!r}, "
            f"got {sorted(values)!r}"
        )


def _validate_common_window_table(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_current_frame(frame, role=path.name)
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
            "common-window concentration table is missing columns: " + ", ".join(sorted(missing))
        )
    widths = frame["w_L"].to_numpy(dtype=float)
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise EvidenceValidationError("common-window concentration contains invalid widths")
    for protocol, group in frame.groupby("window_protocol", sort=False):
        lengths = group["L"].to_numpy(dtype=float)
        half_widths = group["window_half_width"].to_numpy(dtype=float)
        if str(protocol) == PRIMARY_WINDOW_PROTOCOL:
            expected = PRIMARY_WINDOW_PREFACTOR * lengths**PRIMARY_WINDOW_EXPONENT
        elif str(protocol) == FIXED_WINDOW_PROTOCOL:
            expected = np.full_like(lengths, FIXED_CONTROL_HALF_WIDTH)
        else:
            raise EvidenceValidationError(f"unknown current common-window protocol: {protocol}")
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


def _available_primary_concentration_sizes(source: Path) -> tuple[int, ...]:
    common = Path(source) / "spin1_xy_kappa0p1_concentration_common_windows.csv"
    if not common.is_file():
        return ()
    frame = _validate_common_window_table(common)
    selected = frame[
        np.isclose(frame["kappa_over_J"].to_numpy(dtype=float), REPRESENTATIVE_KAPPA_OVER_J)
        & (frame["variant"].astype(str) == "raw").to_numpy()
        & (frame["window_protocol"].astype(str) == PRIMARY_WINDOW_PROTOCOL).to_numpy()
    ]
    return tuple(sorted(set(selected["L"].astype(int))))


def _stamp_csv(path: Path) -> None:
    frame = pd.read_csv(path)
    frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_figure_data(source_data_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run the preserved formatter and stamp every derived table explicitly."""

    _require_current_source(source_data_dir)
    result = _ORIGINAL_BUILD_FIGURE_DATA(source_data_dir, output_dir)
    output = Path(output_dir).resolve(strict=False)
    for name in result.get("written", ()):  # type: ignore[union-attr]
        path = output / str(name)
        if path.suffix == ".csv" and path.is_file():
            _stamp_csv(path)
    return result


def validate_established_evidence(source_data_dir: Path):
    _require_current_source(source_data_dir)
    return _ORIGINAL_VALIDATE_ESTABLISHED_EVIDENCE(source_data_dir)


_legacy._primary_window_mask = _primary_window_mask
_legacy._validate_common_window_table = _validate_common_window_table
_legacy._available_primary_concentration_sizes = _available_primary_concentration_sizes
_legacy.build_figure_data = build_figure_data
_legacy.validate_established_evidence = validate_established_evidence
_legacy.PRIMARY_WINDOW_EXPONENT = PRIMARY_WINDOW_EXPONENT
_legacy.PRIMARY_WINDOW_PREFACTOR = PRIMARY_WINDOW_PREFACTOR
_legacy.FIXED_CONTROL_HALF_WIDTH = FIXED_CONTROL_HALF_WIDTH


if __name__ == "__main__":
    _legacy.main()
