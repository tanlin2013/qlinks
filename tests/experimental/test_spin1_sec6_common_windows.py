from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_exchange_convention as convention  # noqa: E402
import spin1_sec6_common_windows as common  # noqa: E402


def _write_complete_export(root: Path) -> None:
    rows = []
    for length in (8, 10, 12, 14):
        for protocol, half_width in (
            (convention.PRIMARY_WINDOW_PROTOCOL, 0.5 * length**0.25),
            (convention.FIXED_WINDOW_PROTOCOL, 0.5),
        ):
            for variant in ("raw", "clean"):
                width = 0.1 / length
                if length == 14 and protocol == convention.FIXED_WINDOW_PROTOCOL:
                    width = 0.0237316428 if variant == "raw" else 0.0236713087
                rows.append(
                    {
                        "L": length,
                        "kappa_over_J": 0.1,
                        "variant": variant,
                        "window_protocol": protocol,
                        "window_half_width": half_width,
                        "w_L": width,
                        "median_nonidentity_width": width / 2.0,
                        "energy_block_count": 10,
                        "removed_projector_rank": 1,
                        "removed_fraction": 2.0e-4,
                        "covered_spectral_half_width": 1.5,
                        "window_max_eigenpair_residual": 1.0e-8,
                        convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                            convention.CURRENT_EXCHANGE_CONVENTION
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(root / common.COMMON_NAME, index=False)
    for name in (common.CHECKPOINT_AUDIT_NAME, common.WORST_NAME, common.TOLERANCE_NAME):
        pd.DataFrame(
            [
                {
                    "validated": True,
                    convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                        convention.CURRENT_EXCHANGE_CONVENTION
                    ),
                }
            ]
        ).to_csv(root / name, index=False)


def test_completed_common_window_export_is_reused_before_heavy_kernel_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_complete_export(source)

    def should_not_load_core():
        raise AssertionError("completed derived evidence must be reused before numerical setup")

    monkeypatch.setattr(common, "_load_core", should_not_load_core)
    frame = common.compute_common_windows_from_cache(
        checkpoint_roots=(),
        output_dir=output,
        existing_data_dir=source,
    )

    assert len(frame) == 16
    assert (output / common.COMMON_NAME).is_file()
    assert set(frame[convention.EXCHANGE_CONVENTION_METADATA_KEY]) == {
        convention.CURRENT_EXCHANGE_CONVENTION
    }


def test_completed_common_window_export_checks_established_l14_anchor(
    tmp_path: Path,
) -> None:
    _write_complete_export(tmp_path)
    path = tmp_path / common.COMMON_NAME
    frame = pd.read_csv(path)
    mask = (
        (frame["L"] == 14)
        & (frame["window_protocol"] == convention.FIXED_WINDOW_PROTOCOL)
        & (frame["variant"] == "raw")
    )
    frame.loc[mask, "w_L"] = 0.5
    frame.to_csv(path, index=False)

    with pytest.raises(common.CachedSpectrumUnavailableError, match="established L=14"):
        common.validate_completed_common_window_export(tmp_path)


def test_cache_validation_ignores_inaccurate_outer_shift_invert_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energies = np.asarray([-5.0, -0.5, 0.0, 0.5, 5.0])
    vectors = np.eye(5, dtype=np.complex128)
    metadata = {
        "L": 14,
        "M": common.TOTAL_SZ,
        "J3_over_J": common.J3_OVER_J,
        "kappa_over_J": 0.1,
        "sector_dimension": 5,
        "requested_eigenpairs": 5,
        convention.EXCHANGE_CONVENTION_METADATA_KEY: convention.CURRENT_EXCHANGE_CONVENTION,
    }
    # validate_cached_spectrum is the preserved numerical kernel, so patch the
    # dependency in that kernel's module namespace rather than the adapter alias.
    h_sector = np.diag([-4.5, -0.5, 0.0, 0.5, 4.5]).astype(np.complex128)
    monkeypatch.setattr(
        common._legacy,
        "_load_arrays",
        lambda _directory: (energies, vectors, metadata),
    )

    _, _, checked = common.validate_cached_spectrum(
        Path("unused"),
        length=14,
        kappa_over_j=0.1,
        context={"h_sector": h_sector},
    )

    assert checked["sample_maximum_physical_residual"] == pytest.approx(0.0)
    assert checked["sampled_energy_abs_max"] <= common._required_validation_half_width(14)


def test_cache_validation_still_rejects_bad_vectors_inside_common_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energies = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    vectors = np.eye(5, dtype=np.complex128)
    metadata = {
        "L": 14,
        "M": common.TOTAL_SZ,
        "J3_over_J": common.J3_OVER_J,
        "kappa_over_J": 0.1,
        "sector_dimension": 5,
        "requested_eigenpairs": 5,
        convention.EXCHANGE_CONVENTION_METADATA_KEY: convention.CURRENT_EXCHANGE_CONVENTION,
    }
    h_sector = np.diag([-1.0, -0.25, 0.0, 0.5, 1.0]).astype(np.complex128)
    monkeypatch.setattr(
        common._legacy,
        "_load_arrays",
        lambda _directory: (energies, vectors, metadata),
    )

    with pytest.raises(
        common.CachedSpectrumUnavailableError,
        match="inside required common window",
    ):
        common.validate_cached_spectrum(
            Path("unused"),
            length=14,
            kappa_over_j=0.1,
            context={"h_sector": h_sector},
        )
