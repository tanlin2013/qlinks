from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_sec6_common_windows as common  # noqa: E402
import spin1_sec6_common_windows_certified as certified  # noqa: E402


def _write_certification(root: Path, *, passed: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / certified.SOURCE_SUMMARY).write_text(
        json.dumps({certified.CERTIFICATION_KEY: passed}) + "\n",
        encoding="utf-8",
    )


def test_certified_lane_requires_established_sparse_budget_pass(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_certification(source, passed=False)

    with pytest.raises(common.CachedSpectrumUnavailableError, match="not passed"):
        certified.compute_certified_common_windows(
            source_data_dir=source,
            checkpoint_roots=(),
            output_dir=tmp_path / "output",
            existing_data_dir=None,
        )


def test_certified_lane_temporarily_disables_only_absolute_residual_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_certification(source)
    original = common.PHYSICAL_RESIDUAL_TOLERANCE
    seen: dict[str, float] = {}

    def fake_compute(**_kwargs):
        seen["tolerance"] = common.PHYSICAL_RESIDUAL_TOLERANCE
        return pd.DataFrame([{"L": 14, "window_max_eigenpair_residual": 2.0e-6}])

    monkeypatch.setattr(common, "compute_common_windows_from_cache", fake_compute)
    frame = certified.compute_certified_common_windows(
        source_data_dir=source,
        checkpoint_roots=(),
        output_dir=tmp_path / "output",
        existing_data_dir=None,
    )

    assert len(frame) == 1
    assert math.isinf(seen["tolerance"])
    assert common.PHYSICAL_RESIDUAL_TOLERANCE == original


def test_certified_lane_rejects_nonfinite_existing_residuals(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_certification(source)
    pd.DataFrame([{"L": 14, "window_max_eigenpair_residual": float("nan")}]).to_csv(
        output / common.COMMON_NAME, index=False
    )

    with pytest.raises(common.CachedSpectrumUnavailableError, match="non-finite"):
        certified.compute_certified_common_windows(
            source_data_dir=source,
            checkpoint_roots=(),
            output_dir=output,
            existing_data_dir=output,
        )


def test_certified_lane_surfaces_checkpoint_validation_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_certification(source)
    pd.DataFrame(
        [
            {
                "L": 14,
                "validation_errors": "sample orthogonality residual 2e-6",
            }
        ]
    ).to_csv(output / common.CHECKPOINT_AUDIT_NAME, index=False)

    def fake_compute(**_kwargs):
        raise common.CachedSpectrumUnavailableError(
            "no validated reusable spectrum remained for L=14"
        )

    monkeypatch.setattr(common, "compute_common_windows_from_cache", fake_compute)
    with pytest.raises(common.CachedSpectrumUnavailableError, match="L=14: sample orthogonality"):
        certified.compute_certified_common_windows(
            source_data_dir=source,
            checkpoint_roots=(),
            output_dir=output,
            existing_data_dir=None,
        )
