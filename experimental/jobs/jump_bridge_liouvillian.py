"""Independent near-zero Liouvillian validation for the jump-bridge P0 cases.

This module is intentionally an experimental scientific job rather than a unit
regression.  The theorem-based Hilbert-space attractivity certificate remains
the claim gate; this job independently checks the near-zero/peripheral
Liouvillian spectrum and records eigenpair residuals.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from evidence_job_utils import find_repo_root, git_metadata
from jump_bridge_p0 import (
    MODERN_DESIGN_KWARGS,
    TOLERANCE,
    JumpBridgeCase,
    _model,
    _reconstruct_directed_rows,
    _retargeted_a_family,
    _search,
    _select_certified_retargeted_single,
    _sort_retargeted_by_inflow,
    _unique_directed_rows,
)

from qlinks.open_system import diagnose_attractive_subspace
from qlinks.open_system.constructions import (
    build_cage_lindblad_detector_operators,
    build_cage_lindblad_problem,
)
from qlinks.open_system.operators import build_liouvillian


@dataclass(frozen=True, slots=True)
class PreparedLiouvillianCase:
    """Small 4x4 target plus the three jump families checked independently."""

    case: JumpBridgeCase
    hamiltonian: Any
    target_basis: np.ndarray
    families: dict[str, tuple[sp.csr_array, ...]]
    build_seconds: float
    search_seconds: float
    design_seconds: float


@dataclass(frozen=True, slots=True)
class NearZeroSpectrumResult:
    """One partial near-peripheral Liouvillian spectrum calculation."""

    eigenvalues: np.ndarray
    residuals: np.ndarray
    solver_seconds: float
    method: str
    k: int
    ncv: int | None
    sigma: complex | None
    requested_k: int


def _timed(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def prepare_case(case: JumpBridgeCase) -> PreparedLiouvillianCase:
    """Rebuild one P0 case without running long-time dynamics."""
    model = _model(case.model_name)
    build_result, build_seconds = _timed(
        lambda: model.build(
            basis_solver="dfs",
            builder="bitmask",
            backend="scipy",
            on_missing="raise",
        )
    )
    search_result, search_seconds = _timed(lambda: _search(build_result).value)
    records = tuple(search_result[case.signature, : case.record_count])
    if len(records) != case.record_count:
        raise RuntimeError(
            f"{case.name} requested {case.record_count} records, got {len(records)}."
        )

    problem = build_cage_lindblad_problem(
        build_result=build_result,
        records=records,
        model=model,
        local_term_kind="plaquette",
    )
    detector_operators = build_cage_lindblad_detector_operators(
        model=model,
        build_result=build_result,
        operator_kind="kinetic",
        builder="sparse",
    )
    design, design_seconds = _timed(
        lambda: problem.design_jumps(
            detector_operators=detector_operators,
            **MODERN_DESIGN_KWARGS,
        )
    )

    directed_rows = _unique_directed_rows(
        _reconstruct_directed_rows(
            records=records,
            build_result=build_result,
            search_result=search_result,
        )
    )
    retargeted_candidates = _sort_retargeted_by_inflow(
        _retargeted_a_family(
            rows=directed_rows,
            build_result=build_result,
            target_basis=problem.target_basis,
        ),
        problem.target_basis,
    )
    selected, _ = _select_certified_retargeted_single(
        candidates=retargeted_candidates,
        hamiltonian=build_result.hamiltonian,
        target_basis=problem.target_basis,
    )
    if selected is None:
        raise RuntimeError(f"{case.name} has no certified single retargeted-A jump.")

    return PreparedLiouvillianCase(
        case=case,
        hamiltonian=build_result.hamiltonian,
        target_basis=np.asarray(problem.target_basis, dtype=np.complex128),
        families={
            "A_retargeted_single": (sp.csr_array(selected.operator),),
            "ML": tuple(sp.csr_array(operator) for operator in design.recycled_jumps),
            "final": tuple(sp.csr_array(operator) for operator in design.jumps),
        },
        build_seconds=float(build_seconds),
        search_seconds=float(search_seconds),
        design_seconds=float(design_seconds),
    )


def _resolved_k(*, requested: int | None, target_dim: int, liouvillian_dim: int) -> int:
    if requested is None:
        requested = max(16, min(96, target_dim * target_dim + 16))
    return max(1, min(int(requested), liouvillian_dim - 2))


def _resolved_ncv(*, requested: int | None, k: int, dimension: int) -> int | None:
    if requested is not None:
        return min(max(int(requested), k + 2), dimension)
    return min(max(2 * k + 1, 32), dimension)


def solve_near_zero_spectrum(
    liouvillian: Any,
    *,
    method: str,
    k: int,
    tolerance: float,
    maxiter: int | None,
    ncv: int | None,
    sigma: float,
) -> NearZeroSpectrumResult:
    """Return eigenpairs nearest the physical spectral boundary.

    ``largest-real`` is the memory-scalable default: Lindblad generators should
    have non-positive real parts, so steady/peripheral modes are at the largest
    real part without requiring a sparse-LU factorization.  ``shift-invert`` is
    retained as an explicit server fallback, not as the default.
    """
    matrix = liouvillian.tocsr() if hasattr(liouvillian, "tocsr") else sp.csr_array(liouvillian)
    kwargs: dict[str, Any] = {
        "k": int(k),
        "return_eigenvectors": True,
        "tol": float(tolerance),
    }
    if maxiter is not None:
        kwargs["maxiter"] = int(maxiter)
    if ncv is not None:
        kwargs["ncv"] = int(ncv)

    start = time.perf_counter()
    resolved_sigma: complex | None = None
    try:
        if method == "largest-real":
            values, vectors = spla.eigs(matrix, which="LR", **kwargs)
        elif method == "smallest-magnitude":
            values, vectors = spla.eigs(matrix, which="SM", **kwargs)
        elif method == "shift-invert":
            resolved_sigma = complex(float(sigma), 0.0)
            values, vectors = spla.eigs(
                matrix,
                sigma=resolved_sigma,
                which="LM",
                **kwargs,
            )
        else:
            raise ValueError(
                "method must be 'largest-real', 'smallest-magnitude', or 'shift-invert'."
            )
    except spla.ArpackNoConvergence as exc:
        # ARPACK may return scientifically useful converged Ritz pairs even when
        # the requested count was not reached. Persist those rather than
        # turning a partial spectrum into an all-or-nothing failure.
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            raise
        values = exc.eigenvalues
        vectors = exc.eigenvectors
    solver_seconds = time.perf_counter() - start

    values = np.asarray(values, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    residuals = np.empty(values.size, dtype=np.float64)
    for index, value in enumerate(values):
        vector = vectors[:, index]
        norm = float(np.linalg.norm(vector))
        residuals[index] = float(
            np.linalg.norm(matrix @ vector - value * vector) / max(norm, 1.0e-300)
        )

    order = np.lexsort((np.abs(values), -np.real(values)))
    return NearZeroSpectrumResult(
        eigenvalues=values[order],
        residuals=residuals[order],
        solver_seconds=float(solver_seconds),
        method=method,
        k=int(len(values)),
        ncv=ncv,
        sigma=resolved_sigma,
        requested_k=int(k),
    )


def _families_equal(left: Sequence[Any], right: Sequence[Any], tolerance: float = 1.0e-12) -> bool:
    if len(left) != len(right):
        return False
    return all(float(sp.linalg.norm(a - b)) <= tolerance for a, b in zip(left, right, strict=True))


def _target_hamiltonian_scalar_residual(hamiltonian: Any, target_basis: np.ndarray) -> float:
    target, _ = np.linalg.qr(np.asarray(target_basis, dtype=np.complex128))
    projected = target.conj().T @ (hamiltonian @ target)
    center = np.trace(projected) / projected.shape[0]
    return float(np.linalg.norm(projected - center * np.eye(projected.shape[0])))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_liouvillian_benchmark(
    *,
    cases: Sequence[JumpBridgeCase],
    output_dir: Path,
    family_names: Sequence[str],
    method: str,
    k: int | None,
    eig_tolerance: float,
    zero_tolerance: float,
    peripheral_tolerance: float,
    maxiter: int | None,
    ncv: int | None,
    sigma: float,
    strict: bool,
) -> dict[str, object]:
    """Run the independent P0.3 spectrum check and persist partial failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrum_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case in cases:
        prepared = prepare_case(case)
        family_cache: list[tuple[str, tuple[sp.csr_array, ...], dict[str, object]]] = []
        for family_name in family_names:
            if family_name not in prepared.families:
                raise ValueError(f"Unsupported family {family_name!r}.")
            jumps = prepared.families[family_name]

            reused: dict[str, object] | None = None
            for previous_name, previous_jumps, previous_summary in family_cache:
                if _families_equal(jumps, previous_jumps):
                    reused = dict(previous_summary)
                    reused["family"] = family_name
                    reused["reused_from_family"] = previous_name
                    summary_rows.append(reused)
                    previous_spectrum_rows = [
                        row
                        for row in spectrum_rows
                        if row["case"] == case.name and row["family"] == previous_name
                    ]
                    for row in previous_spectrum_rows:
                        cloned = dict(row)
                        cloned["family"] = family_name
                        cloned["reused_from_family"] = previous_name
                        spectrum_rows.append(cloned)
                    break
            if reused is not None:
                continue

            attractivity = diagnose_attractive_subspace(
                hamiltonian=prepared.hamiltonian,
                jumps=jumps,
                target_basis=prepared.target_basis,
                tolerance=TOLERANCE,
            )
            h_scalar_residual = _target_hamiltonian_scalar_residual(
                prepared.hamiltonian,
                prepared.target_basis,
            )
            build_start = time.perf_counter()
            liouvillian = build_liouvillian(
                prepared.hamiltonian,
                jumps,
                backend="scipy",
                sparse_format="csr",
            )
            liouvillian_build_seconds = time.perf_counter() - build_start
            resolved_k = _resolved_k(
                requested=k,
                target_dim=prepared.target_basis.shape[1],
                liouvillian_dim=int(liouvillian.shape[0]),
            )
            resolved_ncv = _resolved_ncv(
                requested=ncv,
                k=resolved_k,
                dimension=int(liouvillian.shape[0]),
            )

            result: NearZeroSpectrumResult | None = None
            error: str | None = None
            try:
                result = solve_near_zero_spectrum(
                    liouvillian,
                    method=method,
                    k=resolved_k,
                    tolerance=eig_tolerance,
                    maxiter=maxiter,
                    ncv=resolved_ncv,
                    sigma=sigma,
                )
            except Exception as exc:  # scientific failures must be persisted, not hidden
                error = f"{type(exc).__name__}: {exc}"
                if strict:
                    raise

            if result is None:
                summary = {
                    "case": case.name,
                    "family": family_name,
                    "reused_from_family": "",
                    "hilbert_dimension": int(prepared.target_basis.shape[0]),
                    "target_dimension": int(prepared.target_basis.shape[1]),
                    "n_jumps": len(jumps),
                    "liouvillian_dimension": int(liouvillian.shape[0]),
                    "liouvillian_nnz": int(liouvillian.nnz),
                    "method": method,
                    "requested_eigenpairs": int(resolved_k),
                    "returned_eigenpairs": 0,
                    "ncv": resolved_ncv,
                    "solver_success": False,
                    "solver_error": error,
                    "zero_mode_count_in_returned_spectrum": "",
                    "peripheral_mode_count_in_returned_spectrum": "",
                    "leading_nonzero_decay_rate": "",
                    "max_eigenpair_residual": "",
                    "target_hamiltonian_scalar_residual": h_scalar_residual,
                    "expected_target_stationary_modes_if_scalar": (
                        int(prepared.target_basis.shape[1] ** 2)
                        if h_scalar_residual <= TOLERANCE
                        else ""
                    ),
                    "attractivity_certified": attractivity.target_attractive_certified,
                    "invariant_obstruction_dimension": attractivity.invariant_obstruction_dimension,
                    "spectrum_consistency_flag": False,
                    "liouvillian_build_seconds": liouvillian_build_seconds,
                    "solver_seconds": "",
                }
                summary_rows.append(summary)
                family_cache.append((family_name, jumps, summary))
                continue

            values = result.eigenvalues
            residuals = result.residuals
            zero_mask = np.abs(values) <= zero_tolerance
            peripheral_mask = np.abs(np.real(values)) <= peripheral_tolerance
            decay_mask = (~zero_mask) & (np.real(values) < -peripheral_tolerance)
            nonzero_decay = -np.real(values[decay_mask])
            leading_decay = float(np.min(nonzero_decay)) if nonzero_decay.size else None
            expected = (
                int(prepared.target_basis.shape[1] ** 2) if h_scalar_residual <= TOLERANCE else None
            )
            zero_count = int(np.count_nonzero(zero_mask))
            residuals_good = bool(
                np.max(residuals, initial=0.0) <= max(1.0e-7, 100.0 * eig_tolerance)
            )
            if expected is None:
                spectrum_consistent = residuals_good
            elif attractivity.target_attractive_certified:
                spectrum_consistent = bool(
                    residuals_good
                    and zero_count >= min(expected, result.k)
                    and (result.k <= expected or zero_count == expected)
                )
            else:
                spectrum_consistent = bool(
                    residuals_good and result.k > expected and zero_count >= expected + 1
                )

            for eigen_index, (value, residual) in enumerate(zip(values, residuals, strict=True)):
                spectrum_rows.append(
                    {
                        "case": case.name,
                        "family": family_name,
                        "reused_from_family": "",
                        "eigen_index": int(eigen_index),
                        "eigenvalue_real": float(np.real(value)),
                        "eigenvalue_imag": float(np.imag(value)),
                        "eigenvalue_abs": float(abs(value)),
                        "eigenpair_residual": float(residual),
                        "is_zero_mode": bool(abs(value) <= zero_tolerance),
                        "is_peripheral_mode": bool(abs(np.real(value)) <= peripheral_tolerance),
                    }
                )

            summary = {
                "case": case.name,
                "family": family_name,
                "reused_from_family": "",
                "hilbert_dimension": int(prepared.target_basis.shape[0]),
                "target_dimension": int(prepared.target_basis.shape[1]),
                "n_jumps": len(jumps),
                "liouvillian_dimension": int(liouvillian.shape[0]),
                "liouvillian_nnz": int(liouvillian.nnz),
                "method": result.method,
                "requested_eigenpairs": int(result.requested_k),
                "returned_eigenpairs": int(result.k),
                "ncv": result.ncv,
                "solver_success": True,
                "solver_error": "",
                "zero_mode_count_in_returned_spectrum": zero_count,
                "peripheral_mode_count_in_returned_spectrum": int(
                    np.count_nonzero(peripheral_mask)
                ),
                "leading_nonzero_decay_rate": "" if leading_decay is None else leading_decay,
                "max_eigenpair_residual": float(np.max(residuals, initial=0.0)),
                "target_hamiltonian_scalar_residual": h_scalar_residual,
                "expected_target_stationary_modes_if_scalar": "" if expected is None else expected,
                "attractivity_certified": attractivity.target_attractive_certified,
                "invariant_obstruction_dimension": attractivity.invariant_obstruction_dimension,
                "spectrum_consistency_flag": spectrum_consistent,
                "liouvillian_build_seconds": liouvillian_build_seconds,
                "solver_seconds": result.solver_seconds,
            }
            summary_rows.append(summary)
            family_cache.append((family_name, jumps, summary))

    _write_csv(output_dir / "liouvillian_near_zero.csv", spectrum_rows)
    _write_csv(output_dir / "liouvillian_summary.csv", summary_rows)
    manifest = {
        "job": "jump_bridge_p0_independent_liouvillian",
        "argv": sys.argv,
        "method": method,
        "eig_tolerance": eig_tolerance,
        "zero_tolerance": zero_tolerance,
        "peripheral_tolerance": peripheral_tolerance,
        "requested_k": k,
        "requested_ncv": ncv,
        "maxiter": maxiter,
        "sigma": sigma,
        "families": list(family_names),
        "cases": [case.name for case in cases],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git": git_metadata(find_repo_root(Path(__file__))),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary": summary_rows, "spectrum": spectrum_rows, "manifest": manifest}
