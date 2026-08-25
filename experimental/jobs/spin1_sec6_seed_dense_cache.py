#!/usr/bin/env python
"""Seed the missing dense Spin-1 Sec. VI spectra into the stable cache.

This is the only solve-capable stage in the Sec. VI integration handoff. It is
strictly limited to the already-established dense sizes L=8,10,12 at kappa/J=0.1.
The expensive L=14 sparse spectrum is never recomputed here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir() and (candidate / "experimental").is_dir():
        ROOT = candidate
        break
else:
    ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
NOTEBOOKS = ROOT / "experimental" / "notebooks"
for path in (JOBS, NOTEBOOKS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import spin1_sec6_common_windows as common
import spin1_sec6_provisioning as core

TARGET_LENGTHS = (8, 10, 12)
KAPPA_OVER_J = 0.10
AUDIT_NAME = "spin1_xy_sec6_dense_cache_seed_audit.csv"
SUMMARY_NAME = "spin1_xy_sec6_dense_cache_seed_summary.json"


def _required_half_width(length: int) -> float:
    return max(float(length) ** common.PRIMARY_WINDOW_EXPONENT, common.FIXED_CONTROL_HALF_WIDTH)


def _covered_half_width(energies: np.ndarray) -> float:
    values = np.asarray(energies, dtype=float)
    if values.size == 0 or values[0] > 0.0 or values[-1] < 0.0:
        return 0.0
    return float(min(abs(values[0]), abs(values[-1])))


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def _staging_directory(directory: Path) -> Path:
    return directory.with_name(f".{directory.name}.tmp")


def _write_checkpoint(
    directory: Path,
    *,
    energies: np.ndarray,
    vectors: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if directory.exists():
        raise FileExistsError(f"checkpoint target already exists: {directory}")
    staging = _staging_directory(directory)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_save_npy(staging / "energies.npy", np.asarray(energies, dtype=np.float64))
        _atomic_save_npy(staging / "vectors.npy", np.asarray(vectors, dtype=np.complex128))
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _checkpoint_directory(cache_root: Path, *, length: int) -> Path:
    return cache_root / "sec6_dense" / f"spin1_L{int(length)}_kappa_p0p100000_dense_full"


def _validated_existing(
    cache_root: Path,
    *,
    length: int,
    context: dict[str, Any],
) -> tuple[Path, np.ndarray, dict[str, Any]] | None:
    required = _required_half_width(length)
    for directory in common.discover_checkpoint_directories(
        (cache_root,), length=length, kappa_over_j=KAPPA_OVER_J
    ):
        try:
            energies, _, metadata = common.validate_cached_spectrum(
                directory,
                length=length,
                kappa_over_j=KAPPA_OVER_J,
                context=context,
            )
        except common.CachedSpectrumUnavailableError:
            continue
        if _covered_half_width(energies) + 1.0e-10 >= required:
            return directory, energies, metadata
    return None


def _audit_row(
    *,
    length: int,
    status: str,
    directory: Path,
    energies: np.ndarray,
    metadata: dict[str, Any],
    solve_seconds: float,
) -> dict[str, Any]:
    return {
        "L": int(length),
        "kappa_over_J": KAPPA_OVER_J,
        "status": status,
        "checkpoint_path": str(directory),
        "sector_dimension": int(metadata["sector_dimension"]),
        "returned_eigenpairs": int(energies.size),
        "covered_spectral_half_width": _covered_half_width(energies),
        "required_spectral_half_width": _required_half_width(length),
        "sample_orthogonality_residual": float(metadata["sample_orthogonality_residual"]),
        "sample_maximum_physical_residual": float(
            metadata["sample_maximum_physical_residual"]
        ),
        "solve_seconds": float(solve_seconds),
    }


def _write_audit(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / AUDIT_NAME, index=False)
    summary = {
        "schema_version": 1,
        "kappa_over_J": KAPPA_OVER_J,
        "target_lengths": list(TARGET_LENGTHS),
        "completed_lengths": [int(row["L"]) for row in rows],
        "computed_lengths": [int(row["L"]) for row in rows if row["status"] == "COMPUTED"],
        "reused_lengths": [int(row["L"]) for row in rows if row["status"] == "REUSED"],
        "large_size_solve_allowed": False,
        "next_action": "rerun common-windows with the same integration run id",
    }
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def seed_dense_cache(*, cache_root: Path, output_dir: Path) -> pd.DataFrame:
    """Populate reusable spectra for exactly L=8,10,12 and validate each one."""

    cache = Path(cache_root).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for length in TARGET_LENGTHS:
        print(f"[sec6-dense-cache] preparing L={length}, kappa/J={KAPPA_OVER_J:.2f}", flush=True)
        directory = _checkpoint_directory(cache, length=length)
        shutil.rmtree(_staging_directory(directory), ignore_errors=True)
        context = core._point_context(length=length, kappa_over_j=KAPPA_OVER_J)
        existing = _validated_existing(cache, length=length, context=context)
        if existing is not None:
            directory, energies, metadata = existing
            rows.append(
                _audit_row(
                    length=length,
                    status="REUSED",
                    directory=directory,
                    energies=energies,
                    metadata=metadata,
                    solve_seconds=0.0,
                )
            )
            _write_audit(output, rows)
            print(f"[sec6-dense-cache] reused validated checkpoint: {directory}", flush=True)
            continue

        if directory.exists():
            raise RuntimeError(
                f"refusing to overwrite an existing but unvalidated dense cache: {directory}"
            )

        h_dense = context["h_sector"].toarray()
        started = time.perf_counter()
        energies, vectors = la.eigh(h_dense, check_finite=False, overwrite_a=True)
        solve_seconds = time.perf_counter() - started
        del h_dense
        dimension = int(vectors.shape[0])
        coverage = _covered_half_width(energies)
        required = _required_half_width(length)
        if coverage + 1.0e-10 < required:
            raise RuntimeError(
                f"full dense spectrum at L={length} covers only {coverage:.6g}, "
                f"below required {required:.6g}"
            )

        metadata = {
            "schema_version": 1,
            "cache_role": "spin1_sec6_dense_full",
            "solver": "scipy.linalg.eigh",
            "full_spectrum": True,
            "L": int(length),
            "M": int(core.TOTAL_SZ),
            "J3_over_J": float(core.J3_OVER_J),
            "kappa_over_J": KAPPA_OVER_J,
            "sector_dimension": dimension,
            "requested_eigenpairs": dimension,
            "returned_eigenpairs": int(energies.size),
            "covered_spectral_half_width": coverage,
            "solve_seconds": float(solve_seconds),
        }
        _write_checkpoint(directory, energies=energies, vectors=vectors, metadata=metadata)
        checked_energies, _, checked = common.validate_cached_spectrum(
            directory,
            length=length,
            kappa_over_j=KAPPA_OVER_J,
            context=context,
        )
        rows.append(
            _audit_row(
                length=length,
                status="COMPUTED",
                directory=directory,
                energies=checked_energies,
                metadata=checked,
                solve_seconds=solve_seconds,
            )
        )
        _write_audit(output, rows)
        print(f"[sec6-dense-cache] wrote and validated checkpoint: {directory}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = seed_dense_cache(cache_root=args.cache_root, output_dir=args.output_dir)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
