#!/usr/bin/env python
"""Derive conventional-J Spin-1 evidence from immutable historical Sec. VI tables.

This converter never edits the source run. It rescales only CSV/JSON products whose
semantics are known to change under the exact mapping from the historical ladder-
prefactor-one wrappers to the permanent ``J/2`` ladder convention. Expensive spectral
arrays are intentionally not copied; their source paths and hashes remain provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    LEGACY_EXCHANGE_CONVENTION,
    LEGACY_TO_CURRENT_BETA_J_SCALE,
    LEGACY_TO_CURRENT_ENERGY_SCALE,
    RESCALED_FROM_METADATA_KEY,
    map_legacy_window_protocol,
)

MANIFEST_NAME = "spin1_exchange_convention_migration_manifest.json"

# These fields carry one power of energy under H_new = H_old / 2.  Keep this
# list deliberately explicit: ranks, dimensions, normalized witnesses, and
# mixed-coordinate Jacobian singular values must not be rescaled heuristically.
_ENERGY_EXACT_KEYS = {
    "d",
    "d_thermal",
    "d_z",
    "d_over_j",
    "h",
    "h_z",
    "h_over_j",
    "shift",
    "window_half_width",
    "window_prefactor",
    "fixed_half_width",
    "concentration_half_width",
    "covered_spectral_half_width",
    "spectral_half_width",
    "delta_e",
    "energy_block_tolerance",
    "window_max_eigenpair_residual",
    "window_median_eigenpair_residual",
    "sample_maximum_physical_residual",
    "tower_residual",
    "eigenpair_residual",
    "interference_gap",
}
_ENERGY_SUBSTRINGS = (
    "energy_density",
    "target_energy",
    "tower_energy",
    "mean_energy",
    "minimum_energy",
    "maximum_energy",
    "min_energy",
    "max_energy",
    "partial_min_energy",
    "partial_max_energy",
    "energy_half_width",
    "spectral_coverage",
)
_ENERGY_EXCLUSIONS = (
    "count",
    "dimension",
    "rank",
    "block_id",
    "block_count",
    "entropy",
    "converged",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_beta_key(key: str) -> bool:
    lower = key.lower()
    return "beta" in lower and "beta0" not in lower and "beta_zero" not in lower


def _is_energy_key(key: str) -> bool:
    lower = key.lower()
    if lower in _ENERGY_EXACT_KEYS:
        return True
    if any(token in lower for token in _ENERGY_EXCLUSIONS):
        return False
    if lower == "energy" or lower.endswith("_energy"):
        return True
    return any(token in lower for token in _ENERGY_SUBSTRINGS)


def _scale_scalar(key: str, value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return value
        if _is_beta_key(key):
            return numeric * LEGACY_TO_CURRENT_BETA_J_SCALE
        if _is_energy_key(key):
            return numeric * LEGACY_TO_CURRENT_ENERGY_SCALE
    return value


def _convert_json_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        converted = {
            str(child_key): _convert_json_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
        if key == "" or EXCHANGE_CONVENTION_METADATA_KEY in converted:
            converted[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
            converted[RESCALED_FROM_METADATA_KEY] = LEGACY_EXCHANGE_CONVENTION
        return converted
    if isinstance(value, list):
        return [_convert_json_value(item, key=key) for item in value]
    if isinstance(value, str) and key in {"window_protocol", "protocol"}:
        return map_legacy_window_protocol(value)
    return _scale_scalar(key, value)


def _convert_csv(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    changed_columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in {"window_protocol", "protocol"}:
            mapped = frame[column].astype(str).map(map_legacy_window_protocol)
            if not mapped.equals(frame[column].astype(str)):
                frame[column] = mapped
                changed_columns.append(name)
            continue
        if _is_beta_key(name) or _is_energy_key(name):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
            if finite.any():
                scale = (
                    LEGACY_TO_CURRENT_BETA_J_SCALE
                    if _is_beta_key(name)
                    else LEGACY_TO_CURRENT_ENERGY_SCALE
                )
                frame.loc[finite, column] = numeric.loc[finite] * scale
                changed_columns.append(name)

    frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    frame[RESCALED_FROM_METADATA_KEY] = LEGACY_EXCHANGE_CONVENTION
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return {
        "kind": "csv",
        "rows": int(len(frame)),
        "changed_columns": sorted(set(changed_columns)),
    }


def _convert_json(source: Path, destination: Path) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    converted = _convert_json_value(value)
    if isinstance(converted, dict):
        converted[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
        converted[RESCALED_FROM_METADATA_KEY] = LEGACY_EXCHANGE_CONVENTION
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(converted, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return {"kind": "json"}


def convert_evidence_directory(
    *,
    source_dir: Path,
    output_dir: Path,
    source_run_id: str | None = None,
    replace_derived: bool = False,
) -> dict[str, Any]:
    """Convert historical CSV/JSON products without mutating or copying heavy arrays."""

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve(strict=False)
    if not source.is_dir():
        raise FileNotFoundError(f"source evidence directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("derived convention output must not be the source or live inside it")
    if output.exists() and any(output.iterdir()):
        if not replace_derived:
            raise FileExistsError(f"derived output is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    skipped_heavy: list[dict[str, str]] = []
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source)
        if source_path.name == MANIFEST_NAME:
            continue
        suffix = source_path.suffix.lower()
        if suffix not in {".csv", ".json"}:
            if suffix in {".npy", ".npz"}:
                skipped_heavy.append(
                    {
                        "path": str(relative),
                        "sha256": _sha256(source_path),
                    }
                )
            continue
        destination = output / relative
        details = (
            _convert_csv(source_path, destination)
            if suffix == ".csv"
            else _convert_json(source_path, destination)
        )
        records.append(
            {
                "path": str(relative),
                "source_sha256": _sha256(source_path),
                "derived_sha256": _sha256(destination),
                **details,
            }
        )

    manifest = {
        "schema_version": 1,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        RESCALED_FROM_METADATA_KEY: LEGACY_EXCHANGE_CONVENTION,
        "source_run_id": source_run_id or source.name,
        "source_directory": str(source),
        "derived_directory": str(output),
        "energy_scale": LEGACY_TO_CURRENT_ENERGY_SCALE,
        "beta_J_scale": LEGACY_TO_CURRENT_BETA_J_SCALE,
        "converted_files": records,
        "skipped_heavy_arrays": skipped_heavy,
        "notes": (
            "Historical source files are immutable. Eigenvectors are unchanged under the exact "
            "uniform h=D=0 mapping and remain in the source evidence/checkpoint directories."
        ),
    }
    manifest_path = output / MANIFEST_NAME
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--replace-derived", action="store_true")
    args = parser.parse_args()
    manifest = convert_evidence_directory(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        source_run_id=args.source_run_id,
        replace_derived=args.replace_derived,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
