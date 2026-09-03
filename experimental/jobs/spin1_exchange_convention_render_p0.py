#!/usr/bin/env python
"""Render convention-mapped Spin-1 Sec. VI P0 figures.

The authoritative P0 migration source is already a completed Sec. VI integration run.
Therefore ``migrate-p0`` maps the standardized figure/appendix tables directly and
``render-p0`` must not rerun the integration formatter. This entry point requires a
completed migration manifest, verifies the renderer inputs against the manifest hashes,
and then renders them. It contains no eigensolver path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import render_spin1_xy_sec6_integration_figures as renderer
import spin1_exchange_convention as convention
import spin1_exchange_convention_migrate_evidence as migration

_REQUIRED_RENDER_INPUTS = (
    "spin1_xy_figure6_panel_a_scatter.csv",
    "spin1_xy_figure6_panel_b_witness_sequence.csv",
    "spin1_xy_figure6_panel_c_deformation.csv",
    "spin1_xy_figure6_panel_d_family_band.csv",
    "spin1_xy_kappa0p1_concentration_common_windows.csv",
    "spin1_xy_appendix_beta0_bridges_data.csv",
    "spin1_xy_appendix_complex_t2_obstruction_data.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_completed_manifest(data: Path, *, source_run_id: str | None) -> dict[str, Any]:
    manifest_path = data / migration.MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            "completed Spin-1 P0 migration manifest is missing; rerun migrate-p0 into a fresh "
            f"run before rendering: {manifest_path}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Spin-1 P0 migration manifest: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Spin-1 P0 migration manifest is not a JSON object: {manifest_path}")
    actual_convention = value.get(convention.EXCHANGE_CONVENTION_METADATA_KEY)
    if actual_convention != convention.CURRENT_EXCHANGE_CONVENTION:
        raise RuntimeError(
            "Spin-1 P0 migration manifest has the wrong exchange convention: "
            f"{actual_convention!r}"
        )
    if source_run_id is not None and value.get("source_run_id") != source_run_id:
        raise RuntimeError(
            "Spin-1 P0 migration manifest source-run mismatch: "
            f"expected {source_run_id!r}, found {value.get('source_run_id')!r}"
        )
    records = value.get("converted_files")
    if not isinstance(records, list):
        raise RuntimeError("Spin-1 P0 migration manifest has no converted_files list")
    return value


def _verify_render_inputs(data: Path, manifest: dict[str, Any]) -> None:
    records = manifest["converted_files"]
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict) and record.get("path") is not None
    }
    problems: list[str] = []
    for name in _REQUIRED_RENDER_INPUTS:
        path = data / name
        record = by_path.get(name)
        if not path.is_file():
            problems.append(f"missing mapped renderer input: {name}")
            continue
        if record is None:
            problems.append(f"renderer input is absent from migration manifest: {name}")
            continue
        expected_hash = record.get("derived_sha256")
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            problems.append(
                f"renderer input hash mismatch: {name} expected={expected_hash!r} "
                f"actual={actual_hash!r}"
            )
    if problems:
        message = "Spin-1 P0 migration/render preflight failed:\n- " + "\n- ".join(problems)
        raise RuntimeError(message)


def prepare_and_render(
    data_dir: Path,
    *,
    use_tex: bool,
    allow_incomplete: bool,
    source_run_id: str | None = None,
) -> dict[str, object]:
    """Verify a completed mapped integration product and render it directly."""

    data = Path(data_dir).resolve(strict=False)
    manifest = _load_completed_manifest(data, source_run_id=source_run_id)
    _verify_render_inputs(data, manifest)
    written = renderer.render(
        data,
        use_tex=use_tex,
        allow_incomplete=allow_incomplete,
    )
    return {
        "data_dir": str(data),
        "source_run_id": manifest.get("source_run_id"),
        "verified_render_inputs": list(_REQUIRED_RENDER_INPUTS),
        "rendered": list(written),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = prepare_and_render(
        args.data_dir,
        use_tex=args.use_tex,
        allow_incomplete=args.allow_incomplete,
        source_run_id=args.source_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
