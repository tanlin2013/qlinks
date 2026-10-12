#!/usr/bin/env python
"""Compare ARPACK and PRIMME on the same small physical square-QDM problems.

This is a solver-validation job, not production evidence. A mandatory exact
parity preflight checks complete eigensystems, thermal witnesses and joint-dark
covariance on 4x4 and 8x4 at phi=0.05. Timed solves then use the same measured
target irreps, Hamiltonians and budgets in isolated workers. Cache reuse is
disabled. The one-dimensional 4x4 irrep is a dense symmetry check; the 8x4 irrep
exercises ARPACK and PRIMME. No legacy notebook or unresolved parity is used.

The raw eigensolver tolerances are backend-specific because the Python PRIMME
wrapper interprets ``tol`` relative to an estimate of the folded-operator norm,
whereas SciPy/ARPACK uses different convergence semantics. Backend comparison
is therefore gated on the same *physical* postprocessed residual threshold,
not on numerically identical raw tolerance parameters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidence_job_utils import find_repo_root

KEY_COLUMNS = (
    "repeats",
    "requested_subspace_size",
    "window_half_width",
    "Sy",
    "sector_dimension",
    "symmetry_protocol",
)
COMPARISON_COLUMNS = (
    "returned_eigenpairs",
    "partial_min_energy",
    "partial_max_energy",
    "partial_maximum_residual",
    "full_Gram_residual",
    "transformed_maximum_residual",
    "joint_dark_rank",
    "peak_rss_gib",
    "runtime_seconds",
    "window_coverage_complete",
    "tau_A_mc_raw",
    "tau_Z_mc_raw",
    "Delta_physical_target",
)

PHYSICAL_RESIDUAL_TOLERANCE = 1.0e-6
RAW_FOLDED_TOLERANCE = {
    "arpack": 1.0e-8,
    # PRIMME marks a folded eigenpair converged when its residual is smaller
    # than roughly ||A_folded|| * tol. The 8x4 pilot showed that 1e-8 therefore
    # permits ~1e-5 physical-H residuals. Tighten the raw criterion while
    # keeping the scientifically meaningful postprocessed residual gate common.
    "primme": 1.0e-11,
}


def _run_backend(
    backend: str,
    *,
    output_root: Path,
    repeats: str,
    budgets: str,
    timeout: int,
) -> Path:
    repo_root = find_repo_root()
    data_dir = output_root / backend
    cache_root = output_root / "cache" / backend
    folded_tolerance = RAW_FOLDED_TOLERANCE[backend]
    command = [
        sys.executable,
        str(repo_root / "experimental/jobs/qdm_parity_backend_worker.py"),
        "--backend",
        backend,
        "--output",
        str(data_dir),
        "--cache-root",
        str(cache_root),
        "--repeats",
        repeats,
        "--budgets",
        budgets,
        "--tolerance",
        f"{folded_tolerance:.1e}",
    ]
    print(
        {
            "backend": backend,
            "raw_folded_tolerance": folded_tolerance,
            "physical_residual_tolerance": PHYSICAL_RESIDUAL_TOLERANCE,
            "command": command,
        },
        flush=True,
    )
    subprocess.run(command, cwd=repo_root, check=True, timeout=timeout)
    return data_dir


def _load_convergence(data_dir: Path, backend: str) -> pd.DataFrame:
    path = data_dir / "qdm_checkerboard_L12_spectral_convergence.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{backend} produced an empty spectral-convergence table")
    failures = frame.loc[frame["solver_status"] != "completed"]
    if not failures.empty:
        raise RuntimeError(f"{backend} has failed solver rows:\n{failures.to_string(index=False)}")

    allowed_methods = {
        "arpack": {
            "folded_spectrum_lanczos",
            "folded_spectrum_lanczos_partial_arpack",
        },
        "primme": {"folded_spectrum_primme", "folded_spectrum_primme_full_ritz"},
    }
    methods = set(frame["spectral_method"].astype(str))
    if "dense_eigh_small_sector" in methods:
        dense = frame[frame.spectral_method.eq("dense_eigh_small_sector")]
        if not dense.sector_dimension.le(2).all():
            raise RuntimeError("dense fallback is allowed only for the tiny symmetry check")
        methods.remove("dense_eigh_small_sector")
    unexpected = methods - allowed_methods[backend]
    if unexpected:
        raise RuntimeError(
            f"explicit {backend} comparison was not executed by that backend: {sorted(methods)}"
        )
    return frame


def _finite_difference(left: object, right: object) -> float | None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        return None
    return abs(left_value - right_value)


def compare_frames(arpack: pd.DataFrame, primme: pd.DataFrame) -> pd.DataFrame:
    keep = [*KEY_COLUMNS, *[name for name in COMPARISON_COLUMNS if name in arpack.columns]]
    arpack_view = arpack.loc[:, keep].copy()
    primme_view = primme.loc[:, keep].copy()
    merged = arpack_view.merge(
        primme_view,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_arpack", "_primme"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(
            "ARPACK and PRIMME did not produce the same physical comparison rows:\n"
            + merged.loc[merged["_merge"] != "both"].to_string(index=False)
        )

    diagnostics: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        diagnostic: dict[str, object] = {name: row[name] for name in KEY_COLUMNS}
        for name in COMPARISON_COLUMNS:
            left_name = f"{name}_arpack"
            right_name = f"{name}_primme"
            if left_name not in merged.columns or right_name not in merged.columns:
                continue
            diagnostic[f"{name}_arpack"] = row[left_name]
            diagnostic[f"{name}_primme"] = row[right_name]
            difference = _finite_difference(row[left_name], row[right_name])
            if difference is not None:
                diagnostic[f"abs_diff_{name}"] = difference
        diagnostics.append(diagnostic)
    result = pd.DataFrame(diagnostics)

    if "returned_eigenpairs_arpack" in result:
        if not np.array_equal(
            result["returned_eigenpairs_arpack"].to_numpy(),
            result["returned_eigenpairs_primme"].to_numpy(),
        ):
            raise AssertionError("ARPACK and PRIMME returned different eigenpair counts")
    if "joint_dark_rank_arpack" in result:
        if not np.array_equal(
            result["joint_dark_rank_arpack"].to_numpy(),
            result["joint_dark_rank_primme"].to_numpy(),
        ):
            raise AssertionError("ARPACK and PRIMME returned different joint-dark ranks")
    for name in ("partial_min_energy", "partial_max_energy"):
        column = f"abs_diff_{name}"
        if column in result and float(result[column].max()) > 1.0e-6:
            maximum = float(result[column].max())
            raise AssertionError(f"backend spectral-bound mismatch in {name}: {maximum:.3e}")
    for backend in ("arpack", "primme"):
        column = f"partial_maximum_residual_{backend}"
        if column in result and float(result[column].max()) > PHYSICAL_RESIDUAL_TOLERANCE:
            maximum = float(result[column].max())
            raise AssertionError(
                f"{backend} physical residual exceeds "
                f"{PHYSICAL_RESIDUAL_TOLERANCE:.1e}: {maximum:.3e}"
            )
    for name in ("tau_A_mc_raw", "tau_Z_mc_raw", "Delta_physical_target"):
        column = f"abs_diff_{name}"
        if column in result:
            finite = pd.to_numeric(result[column], errors="coerce").dropna()
            if not finite.empty and float(finite.max()) > 1.0e-5:
                raise AssertionError(f"backend observable mismatch in {name}: {finite.max():.3e}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", default="1,2")
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = find_repo_root()
    preflight = output_root / "parity_preflight"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "experimental/jobs/qdm_parity_backend_worker.py"),
            "--backend",
            "preflight",
            "--output",
            str(preflight),
        ],
        cwd=repo_root,
        check=True,
        timeout=args.timeout,
    )
    arpack_dir = _run_backend(
        "arpack",
        output_root=output_root,
        repeats=args.repeats,
        budgets=args.budgets,
        timeout=args.timeout,
    )
    primme_dir = _run_backend(
        "primme",
        output_root=output_root,
        repeats=args.repeats,
        budgets=args.budgets,
        timeout=args.timeout,
    )
    arpack = _load_convergence(arpack_dir, "arpack")
    primme = _load_convergence(primme_dir, "primme")
    comparison = compare_frames(arpack, primme)
    comparison_path = output_root / "qdm_folded_backend_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "status": "passed",
        "parity_preflight_audit": str(preflight / "qdm_parity_exact_audit.json"),
        "four_by_four_role": "one-dimensional dense symmetry check",
        "window_half_width": 0.5,
        "repeats": args.repeats,
        "budgets": args.budgets,
        "rows": int(len(comparison)),
        "raw_folded_tolerances": RAW_FOLDED_TOLERANCE,
        "physical_residual_tolerance": PHYSICAL_RESIDUAL_TOLERANCE,
        "arpack_max_residual": float(arpack["partial_maximum_residual"].max()),
        "primme_max_residual": float(primme["partial_maximum_residual"].max()),
        "max_spectral_bound_difference": float(
            max(
                comparison.get("abs_diff_partial_min_energy", pd.Series([0.0])).max(),
                comparison.get("abs_diff_partial_max_energy", pd.Series([0.0])).max(),
            )
        ),
        "comparison_csv": str(comparison_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
