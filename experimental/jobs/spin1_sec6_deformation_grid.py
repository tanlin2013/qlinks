#!/usr/bin/env python
"""Current-convention Sec. VI positive-kappa deformation grid.

The pre-migration implementation is preserved in ``spin1_sec6_deformation_grid_legacy``.
This adapter keeps its dense-only L<=12 solver boundary while changing the physical
exchange/window convention and refusing silent reuse of legacy point checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import spin1_exchange_convention as _convention
import spin1_sec6_deformation_grid_legacy as _legacy

_ORIGINAL_REPRESENTATIVE_POINTS = _legacy._representative_points
_ORIGINAL_COMPUTE_DENSE_POINT = _legacy._compute_dense_point
_ORIGINAL_WRITE_CHECKPOINT = _legacy._write_checkpoint
_ORIGINAL_REFRESH_AGGREGATES = _legacy._refresh_aggregates
_ORIGINAL_RUN_GRID = _legacy.run_grid

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

SCHEMA_VERSION = 2
WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
WINDOW_EXPONENT = _convention.PRIMARY_WINDOW_EXPONENT
WINDOW_PREFACTOR = _convention.PRIMARY_WINDOW_PREFACTOR

_legacy.SCHEMA_VERSION = SCHEMA_VERSION
_legacy.WINDOW_PROTOCOL = WINDOW_PROTOCOL
_legacy.WINDOW_EXPONENT = WINDOW_EXPONENT
_legacy.WINDOW_PREFACTOR = WINDOW_PREFACTOR


def _expected_half_width(length: int) -> float:
    return WINDOW_PREFACTOR * float(int(length) ** WINDOW_EXPONENT)


_legacy._expected_half_width = _expected_half_width


def _stamp_record(record: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(record)
    stamped[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
        _convention.CURRENT_EXCHANGE_CONVENTION
    )
    stamped["schema_version"] = SCHEMA_VERSION
    stamped["window_protocol"] = WINDOW_PROTOCOL
    stamped["window_exponent"] = WINDOW_EXPONENT
    stamped["window_prefactor"] = WINDOW_PREFACTOR
    return stamped


def _representative_points(
    integration_data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, worst_rows = _ORIGINAL_REPRESENTATIVE_POINTS(integration_data_dir)
    return (
        [_stamp_record(record) for record in records],
        [
            {
                **row,
                "window_protocol": WINDOW_PROTOCOL,
                _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                    _convention.CURRENT_EXCHANGE_CONVENTION
                ),
            }
            for row in worst_rows
        ],
    )


_legacy._representative_points = _representative_points


def _compute_dense_point(
    *,
    length: int,
    kappa_over_j: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, worst_rows = _ORIGINAL_COMPUTE_DENSE_POINT(
        length=length,
        kappa_over_j=kappa_over_j,
    )
    return (
        _stamp_record(row),
        [
            {
                **item,
                "window_protocol": WINDOW_PROTOCOL,
                _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                    _convention.CURRENT_EXCHANGE_CONVENTION
                ),
            }
            for item in worst_rows
        ],
    )


_legacy._compute_dense_point = _compute_dense_point


def _write_checkpoint(
    cache_root: Path,
    row: dict[str, Any],
    worst_rows: list[dict[str, Any]],
) -> None:
    stamped_row = _stamp_record(row)
    stamped_worst = [
        {
            **item,
            _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                _convention.CURRENT_EXCHANGE_CONVENTION
            ),
        }
        for item in worst_rows
    ]
    _ORIGINAL_WRITE_CHECKPOINT(cache_root, stamped_row, stamped_worst)
    directory = _legacy._checkpoint_dir(
        cache_root,
        int(stamped_row["L"]),
        float(stamped_row["kappa_over_J"]),
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = SCHEMA_VERSION
    metadata[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
        _convention.CURRENT_EXCHANGE_CONVENTION
    )
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)


_legacy._write_checkpoint = _write_checkpoint


def _stamp_output_file(path: Path) -> None:
    if not path.is_file():
        return
    if path.suffix == ".csv":
        # The status lane can legitimately emit an empty optional aggregate.
        # Preserve that sentinel rather than failing while trying to add provenance.
        if path.stat().st_size == 0:
            return
        frame = pd.read_csv(path)
        frame[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
            _convention.CURRENT_EXCHANGE_CONVENTION
        )
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value["schema_version"] = SCHEMA_VERSION
            value[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
                _convention.CURRENT_EXCHANGE_CONVENTION
            )
            value["window_protocol"] = WINDOW_PROTOCOL
            value["window_prefactor"] = WINDOW_PREFACTOR
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)


def _refresh_aggregates(*args, **kwargs) -> None:
    _ORIGINAL_REFRESH_AGGREGATES(*args, **kwargs)
    output_dir = kwargs.get("output_dir")
    if output_dir is None:
        return
    output = Path(output_dir)
    for name in (GRID_ROWS_NAME, WORST_NAME, PANEL_C_NAME, PANEL_D_NAME, PROGRESS_NAME):
        _stamp_output_file(output / name)


_legacy._refresh_aggregates = _refresh_aggregates


def _require_current_integration_source(integration_data_dir: Path) -> None:
    data = Path(integration_data_dir)
    for name in (COMMON_NAME, PANEL_B_NAME, REPRESENTATIVE_WORST_NAME):
        path = data / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        key = _convention.EXCHANGE_CONVENTION_METADATA_KEY
        if key not in frame.columns or set(frame[key].astype(str)) != {
            _convention.CURRENT_EXCHANGE_CONVENTION
        }:
            raise DeformationGridError(
                f"integration source is not convention-mapped: {path}; use the derived "
                "Spin-1 migration/integration products"
            )


def _preflight_checkpoint_conventions(cache_root: Path) -> None:
    root = Path(cache_root) / "sec6_deformation_grid"
    if not root.is_dir():
        return
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DeformationGridError(f"invalid deformation checkpoint: {metadata_path}") from exc
        if metadata.get(_convention.EXCHANGE_CONVENTION_METADATA_KEY) != (
            _convention.CURRENT_EXCHANGE_CONVENTION
        ):
            raise DeformationGridError(
                "legacy deformation-grid checkpoint detected; refusing to treat it as a "
                f"cache miss: {metadata_path}. Convert/segregate the old cache first."
            )


def run_grid(
    *,
    integration_data_dir: Path,
    cache_root: Path,
    output_dir: Path,
    compute_missing: bool,
) -> pd.DataFrame:
    _require_current_integration_source(integration_data_dir)
    _preflight_checkpoint_conventions(cache_root)
    return _ORIGINAL_RUN_GRID(
        integration_data_dir=integration_data_dir,
        cache_root=cache_root,
        output_dir=output_dir,
        compute_missing=compute_missing,
    )


if __name__ == "__main__":
    _legacy.run_grid = run_grid
    _legacy.main()
