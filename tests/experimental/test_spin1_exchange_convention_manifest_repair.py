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

_REGENERABLE_ROOT_PRODUCTS = {
    "spin1_xy_figure6_panel_a_scatter.csv",
    "spin1_xy_figure6_panel_b_witness_sequence.csv",
    "spin1_xy_figure6_panel_c_deformation.csv",
    "spin1_xy_figure6_panel_d_family_band.csv",
    "spin1_xy_appendix_beta0_bridges_data.csv",
    "spin1_xy_appendix_complex_t2_obstruction_data.csv",
    "spin1_xy_sec6_integration_audit.json",
}


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
    pd.DataFrame([{"L": 14, "bridge": "mc_to_beta0_resolved", "trace_distance": 0.1}]).to_csv(
        path / "spin1_xy_appendix_beta0_bridges_data.csv",
        index=False,
    )
    (path / "spin1_xy_sec6_integration_audit.json").write_text(
        json.dumps({"representative_l14_validated": True}),
        encoding="utf-8",
    )
    figures = path / "figures"
    figures.mkdir()
    (figures / "spin1_xy_figure6_prx_audit.json").write_text(
        json.dumps({"rendered": True, "energy_density": 0.2}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("name", sorted(_REGENERABLE_ROOT_PRODUCTS))
def test_known_sec6_integration_products_are_regenerable(name: str) -> None:
    assert repair._is_regenerable_output(Path(name))


def test_repair_reconstructs_manifest_when_postprocessing_products_are_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "derived"
    _make_legacy_source(source)
    migration.convert_evidence_directory(source_dir=source, output_dir=output)
    manifest_path = output / migration.MANIFEST_NAME
    manifest_path.unlink()

    missing_regenerable = (
        Path("spin1_xy_appendix_beta0_bridges_data.csv"),
        Path("spin1_xy_sec6_integration_audit.json"),
        Path("figures/spin1_xy_figure6_prx_audit.json"),
    )
    for relative in missing_regenerable:
        (output / relative).unlink()
    (output / "figures").rmdir()

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
    skipped = manifest["skipped_regenerable_postprocessing_products"]
    skipped_paths = {entry["path"] for entry in skipped}
    assert {str(path) for path in missing_regenerable}.issubset(skipped_paths)
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
