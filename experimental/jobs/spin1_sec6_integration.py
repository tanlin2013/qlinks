#!/usr/bin/env python
"""Current-convention Sec. VI evidence integration adapter.

The August integration logic is preserved in ``spin1_sec6_integration_legacy``.
This active entry point requires an explicitly convention-mapped source directory
and changes only the primary-window identification from c=1 to c=1/2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import spin1_exchange_convention as _convention
import spin1_sec6_integration_legacy as _legacy

_ORIGINAL_VALIDATE_ESTABLISHED_EVIDENCE = _legacy.validate_established_evidence

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

PRIMARY_WINDOW_EXPONENT = _convention.PRIMARY_WINDOW_EXPONENT
PRIMARY_WINDOW_PREFACTOR = _convention.PRIMARY_WINDOW_PREFACTOR
FIXED_CONTROL_HALF_WIDTH = _convention.FIXED_CONTROL_HALF_WIDTH
PRIMARY_WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
FIXED_WINDOW_PROTOCOL = _convention.FIXED_WINDOW_PROTOCOL


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
                "quarter_power_c0p5",
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
        raise EvidenceValidationError(f"invalid convention migration manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get(
        _convention.EXCHANGE_CONVENTION_METADATA_KEY
    ) != _convention.CURRENT_EXCHANGE_CONVENTION:
        raise EvidenceValidationError(
            f"migration manifest does not declare {_convention.CURRENT_EXCHANGE_CONVENTION!r}"
        )


def validate_established_evidence(source_data_dir: Path):
    _require_current_source(source_data_dir)
    return _ORIGINAL_VALIDATE_ESTABLISHED_EVIDENCE(source_data_dir)


_legacy._primary_window_mask = _primary_window_mask
_legacy.validate_established_evidence = validate_established_evidence
_legacy.PRIMARY_WINDOW_EXPONENT = PRIMARY_WINDOW_EXPONENT
_legacy.PRIMARY_WINDOW_PREFACTOR = PRIMARY_WINDOW_PREFACTOR
_legacy.FIXED_CONTROL_HALF_WIDTH = FIXED_CONTROL_HALF_WIDTH


if __name__ == "__main__":
    _legacy.main()
