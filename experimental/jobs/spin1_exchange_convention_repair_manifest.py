#!/usr/bin/env python
"""Repair a missing Spin-1 exchange-convention migration manifest.

The repair is deliberately conservative. It replays the deterministic CSV/JSON
conversion from the immutable historical source into a temporary directory and
requires every migration-relevant source product already present in the derived
directory to match byte-for-byte. Regenerable Sec. VI integration/renderer products
are not provenance requirements because failed or repeated post-processing may replace
or remove them. Only after verification succeeds is the missing migration manifest
reconstructed. Historical sources and mapped evidence products are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import spin1_exchange_convention_migrate_evidence as migration
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    LEGACY_EXCHANGE_CONVENTION,
    LEGACY_TO_CURRENT_BETA_J_SCALE,
    LEGACY_TO_CURRENT_ENERGY_SCALE,
    RESCALED_FROM_METADATA_KEY,
)

_REGENERABLE_OUTPUT_DIRS = {"figures"}
_REGENERABLE_OUTPUT_NAMES = {
    "spin1_xy_figure6_panel_a_scatter.csv",
    "spin1_xy_figure6_panel_b_witness_sequence.csv",
    "spin1_xy_figure6_panel_c_deformation.csv",
    "spin1_xy_figure6_panel_d_family_band.csv",
    "spin1_xy_appendix_beta0_bridges_data.csv",
    "spin1_xy_appendix_complex_t2_obstruction_data.csv",
    "spin1_xy_sec6_integration_audit.json",
}


def _is_regenerable_output(relative: Path) -> bool:
    """Return whether a historical product is deterministic Sec. VI post-processing."""

    if not relative.parts:
        return False
    if relative.parts[0] in _REGENERABLE_OUTPUT_DIRS:
        return True
    return len(relative.parts) == 1 and relative.name in _REGENERABLE_OUTPUT_NAMES


def repair_missing_manifest(
    *,
    source_dir: Path,
    output_dir: Path,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    """Reconstruct a missing manifest only after exact evidence verification."""

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve(strict=False)
    if not source.is_dir():
        raise FileNotFoundError(f"historical source evidence directory does not exist: {source}")
    if not output.is_dir():
        raise FileNotFoundError(f"derived evidence directory does not exist: {output}")

    manifest_path = output / migration.MANIFEST_NAME
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"existing migration manifest is not a JSON object: {manifest_path}")
        if value.get(EXCHANGE_CONVENTION_METADATA_KEY) != CURRENT_EXCHANGE_CONVENTION:
            raise ValueError(
                "existing migration manifest declares the wrong exchange convention: "
                f"{value.get(EXCHANGE_CONVENTION_METADATA_KEY)!r}"
            )
        return value

    records: list[dict[str, Any]] = []
    skipped_heavy: list[dict[str, str]] = []
    skipped_regenerable: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="spin1-convention-manifest-") as temporary_root:
        temporary = Path(temporary_root)
        for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = source_path.relative_to(source)
            if source_path.name == migration.MANIFEST_NAME:
                continue
            if _is_regenerable_output(relative):
                skipped_regenerable.append(
                    {
                        "path": str(relative),
                        "sha256": migration._sha256(source_path),
                    }
                )
                continue
            suffix = source_path.suffix.lower()
            if suffix in {".npy", ".npz"}:
                skipped_heavy.append(
                    {
                        "path": str(relative),
                        "sha256": migration._sha256(source_path),
                    }
                )
                continue
            if suffix not in {".csv", ".json"}:
                continue

            expected_path = temporary / relative
            details = (
                migration._convert_csv(source_path, expected_path)
                if suffix == ".csv"
                else migration._convert_json(source_path, expected_path)
            )
            derived_path = output / relative
            if not derived_path.is_file():
                raise FileNotFoundError(
                    "cannot repair convention manifest because a mapped evidence "
                    f"product is missing: {derived_path}"
                )
            expected_sha = migration._sha256(expected_path)
            actual_sha = migration._sha256(derived_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    "cannot repair convention manifest because a mapped evidence product does not "
                    f"match deterministic conversion: {relative}"
                )
            records.append(
                {
                    "path": str(relative),
                    "source_sha256": migration._sha256(source_path),
                    "derived_sha256": actual_sha,
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
        "skipped_regenerable_postprocessing_products": skipped_regenerable,
        "notes": (
            "Recovered missing migration provenance only after exact deterministic verification "
            "against the immutable historical source. Deterministic Sec. VI integration/renderer "
            "products are recorded but are not provenance requirements. Historical source files "
            "and mapped evidence products were not modified."
        ),
        "manifest_repaired": True,
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", default=None)
    args = parser.parse_args()
    result = repair_missing_manifest(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        source_run_id=args.source_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
