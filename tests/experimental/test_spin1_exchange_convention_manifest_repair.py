from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_exchange_convention as convention  # noqa: E402
import spin1_exchange_convention_migrate_evidence as migration  # noqa: E402
import spin1_exchange_convention_repair_manifest as repair  # noqa: E402


def _make_legacy_source(path: Path) -> None:
    path.mkdir()
    pd.DataFrame(
        [
            {
                "L": 12,
                "window_protocol": "quarter_power_c1",
                "window_half_width": 1.86121,
                "energy_density": 0.2,
                "tau_A_mc_raw": 0.11,
            }
        ]
    ).to_csv(path / "rows.csv", index=False)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "window_protocol": "fixed_width_1",
                "window_half_width": 1.0,
                "beta_J": -0.05,
            }
        ),
        encoding="utf-8",
    )
    figures = path / "figures"
    figures.mkdir()
    (figures / "spin1_xy_figure6_prx_audit.json").write_text(
        json.dumps({"rendered": True, "energy_density": 0.2}),
        encoding="utf-8",
    )


def test_repair_reconstructs_manifest_when_regenerable_render_product_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "derived"
    _make_legacy_source(source)
    migration.convert_evidence_directory(source_dir=source, output_dir=output)
    manifest_path = output / migration.MANIFEST_NAME
    manifest_path.unlink()
    rendered_audit = output / "figures" / "spin1_xy_figure6_prx_audit.json"
    rendered_audit.unlink()
    rendered_audit.parent.rmdir()
    mapped_snapshot = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    manifest = repair.repair_missing_manifest(
        source_dir=source,
        output_dir=output,
        source_run_id="legacy-run",
    )

    assert manifest["manifest_repaired"] is True
    assert manifest["source_run_id"] == "legacy-run"
    assert manifest[convention.EXCHANGE_CONVENTION_METADATA_KEY] == (
        convention.CURRENT_EXCHANGE_CONVENTION
    )
    assert manifest_path.is_file()
    skipped = manifest["skipped_regenerable_render_products"]
    assert [entry["path"] for entry in skipped] == ["figures/spin1_xy_figure6_prx_audit.json"]
    for relative, payload in mapped_snapshot.items():
        assert (output / relative).read_bytes() == payload


def test_repair_rejects_missing_mapped_evidence_product(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "derived"
    _make_legacy_source(source)
    migration.convert_evidence_directory(source_dir=source, output_dir=output)
    (output / migration.MANIFEST_NAME).unlink()
    (output / "rows.csv").unlink()

    with pytest.raises(FileNotFoundError, match="mapped evidence product is missing"):
        repair.repair_missing_manifest(source_dir=source, output_dir=output)

    assert not (output / migration.MANIFEST_NAME).exists()


def test_repair_rejects_tampered_mapped_product(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "derived"
    _make_legacy_source(source)
    migration.convert_evidence_directory(source_dir=source, output_dir=output)
    (output / migration.MANIFEST_NAME).unlink()
    mapped = pd.read_csv(output / "rows.csv")
    mapped.loc[0, "energy_density"] = 999.0
    mapped.to_csv(output / "rows.csv", index=False)

    with pytest.raises(ValueError, match="does not match deterministic conversion"):
        repair.repair_missing_manifest(source_dir=source, output_dir=output)

    assert not (output / migration.MANIFEST_NAME).exists()
