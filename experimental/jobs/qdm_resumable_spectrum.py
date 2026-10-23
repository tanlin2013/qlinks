"""Checkpointed folded-spectrum backends for the large square-QDM strip.

The production notebook still owns the scientific budget-convergence logic.
This module wraps each *individual* folded-spectrum solve so that a completed
budget is validated and persisted immediately, before later dark-space,
covariance, pandas, or plotting work can fail. Subsequent timestamped jobs
reuse compatible checkpoints from the stable evidence cache.

PRIMME is optional and imported lazily. The standard notebook image therefore
continues to work with SciPy/ARPACK, while a PRIMME-enabled evidence image can
select the PRIMME backend without changing the notebook.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from evidence_cache import (
    CacheValidationStatus,
    default_cache_root,
    iter_spectral_checkpoints,
    load_spectral_checkpoint,
    save_spectral_checkpoint,
    sparse_matrix_fingerprint,
    spectral_checkpoint_directory,
)
from qdm_checkerboard_large_strip import PartialSpectrum, process_peak_rss_gib
from qdm_ritz_reconstruction import (
    REFINED_BACKEND,
    is_primme_checkpoint_backend,
    reconstruct_hamiltonian_ritz_pairs,
    refinement_backend,
)

CACHE_NAMESPACE = "qdm/checkerboard_large_strip"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _requested_backend() -> str:
    requested = os.environ.get("QLINKS_QDM_FOLDED_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "arpack", "primme"}:
        raise ValueError(
            f"QLINKS_QDM_FOLDED_BACKEND must be one of auto, arpack, primme; got {requested!r}"
        )
    return requested


def _backend() -> str:
    requested = _requested_backend()
    if requested == "auto":
        return "primme" if importlib.util.find_spec("primme") is not None else "arpack"
    if requested == "primme" and importlib.util.find_spec("primme") is None:
        raise RuntimeError(
            "The PRIMME folded-spectrum backend was requested, but 'primme' is not importable. "
            "Build/use the PRIMME evidence image (Python 3.13) or select "
            "--large-strip-folded-backend arpack."
        )
    return requested


def _matrix_fingerprint(matrix: sp.spmatrix | sp.sparray) -> str:
    # Sparse matrices are mutable. Re-hash the numerical payload whenever a
    # problem description is requested rather than caching by object identity.
    return sparse_matrix_fingerprint(sp.csr_array(matrix))


def folded_problem_description(
    hamiltonian: sp.spmatrix | sp.sparray,
    *,
    target_energy: float,
) -> dict[str, Any]:
    """Return the scientific compatibility key for a folded QDM spectrum."""

    h = sp.csr_array(hamiltonian)
    return {
        "problem_family": "square_qdm_checkerboard_folded_spectrum",
        "folded_operator_version": 1,
        "sector_dimension": int(h.shape[0]),
        "matrix_shape": [int(h.shape[0]), int(h.shape[1])],
        "matrix_dtype": str(h.dtype),
        "hamiltonian_fingerprint": _matrix_fingerprint(h),
        "target_energy": float(target_energy),
    }


def _checkpoint_to_partial(checkpoint) -> PartialSpectrum:
    metadata = checkpoint.metadata
    producer_backend = str(metadata.get("backend", "unknown"))
    return PartialSpectrum(
        energies=np.asarray(checkpoint.energies),
        eigenvectors=np.asarray(checkpoint.eigenvectors),
        residuals=np.asarray(checkpoint.residuals),
        sigma=float(metadata["problem"]["target_energy"]),
        target_energy=float(metadata["problem"]["target_energy"]),
        method=f"folded_spectrum_cache_{producer_backend}",
        requested_subspace_size=int(metadata.get("requested_budget", checkpoint.energies.size)),
        transformed_residuals=(
            None
            if checkpoint.transformed_residuals is None
            else np.asarray(checkpoint.transformed_residuals)
        ),
        peak_rss_gib=process_peak_rss_gib(),
        full_orthogonality_residual=metadata.get("solver", {}).get(
            "full_orthogonality_frobenius_residual"
        ),
    )


def _full_orthogonality_acceptable(checkpoint) -> bool:
    try:
        gram = float(
            checkpoint.metadata.get("solver", {}).get(
                "full_orthogonality_frobenius_residual", float("inf")
            )
        )
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(gram) and 0 <= gram <= 1e-6)


def _load_exact_checkpoint(
    *,
    problem: dict[str, Any],
    budget: int,
    tolerance: float,
    hamiltonian: sp.spmatrix | sp.sparray,
    preferred_backend: str,
    allow_cross_backend: bool,
    require_full_orthogonality: bool = False,
):
    candidates = iter_spectral_checkpoints(namespace=CACHE_NAMESPACE, problem=problem)
    exact = [path for path in candidates if path.name == f"budget_{int(budget):08d}"]
    if not allow_cross_backend:
        exact = [
            path
            for path in exact
            if (
                is_primme_checkpoint_backend(path.parent.name)
                if preferred_backend == "primme"
                else path.parent.name == preferred_backend
            )
        ]
    exact.sort(
        key=lambda path: (
            not (
                preferred_backend == "primme"
                and path.parent.name != "primme"
                and is_primme_checkpoint_backend(path.parent.name)
            ),
            path.parent.name != preferred_backend,
            str(path),
        )
    )
    for path in exact:
        checkpoint = load_spectral_checkpoint(
            path,
            expected_problem=problem,
            hamiltonian=hamiltonian,
            requested_solver_tolerance=float(tolerance),
            sample_vectors=8,
        )
        if checkpoint is None or checkpoint.status is not CacheValidationStatus.VALID_FINAL:
            continue
        if require_full_orthogonality and not _full_orthogonality_acceptable(checkpoint):
            continue
        if int(checkpoint.metadata.get("requested_budget", -1)) != int(budget):
            continue
        if int(checkpoint.energies.size) < int(budget):
            continue
        print(
            {
                "evidence_cache": "reused",
                "status": checkpoint.status.value,
                "budget": int(budget),
                "producer_backend": checkpoint.metadata.get("backend"),
                "path": str(path),
                "validation": checkpoint.validation,
            },
            flush=True,
        )
        return checkpoint
    return None


def _load_warm_start(
    *,
    problem: dict[str, Any],
    budget: int,
    tolerance: float,
    hamiltonian: sp.spmatrix | sp.sparray,
    same_budget: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    max_vectors = int(os.environ.get("QLINKS_QDM_PRIMME_WARM_START_VECTORS", "256"))
    if same_budget:
        max_vectors = int(budget)
    if max_vectors <= 0:
        return None, {}
    candidates = iter_spectral_checkpoints(namespace=CACHE_NAMESPACE, problem=problem)
    ranked: list[tuple[int, Any]] = []
    for path in candidates:
        try:
            previous_budget = int(path.name.removeprefix("budget_"))
        except ValueError:
            continue
        eligible = previous_budget == int(budget) if same_budget else previous_budget < int(budget)
        if not eligible:
            continue
        if same_budget and (
            path.parent.name == "primme" or not is_primme_checkpoint_backend(path.parent.name)
        ):
            continue
        checkpoint = load_spectral_checkpoint(
            path,
            expected_problem=problem,
            hamiltonian=hamiltonian,
            requested_solver_tolerance=float(tolerance),
            sample_vectors=4,
        )
        if checkpoint is None or checkpoint.status is CacheValidationStatus.INCOMPATIBLE:
            continue
        if checkpoint.energies.size == 0:
            continue
        if same_budget:
            if (
                checkpoint.energies.size != int(budget)
                or int(checkpoint.metadata.get("requested_budget", -1)) != int(budget)
                or not _full_orthogonality_acceptable(checkpoint)
            ):
                continue
        ranked.append((previous_budget, checkpoint))
    if not ranked:
        return None, {}
    _, checkpoint = max(
        ranked,
        key=lambda item: (
            item[0],
            item[1].status is CacheValidationStatus.VALID_FINAL,
            -float(np.max(item[1].residuals, initial=0.0)),
        ),
    )
    target = float(problem["target_energy"])
    order = np.argsort(np.abs(np.asarray(checkpoint.energies) - target))
    take = order[: min(max_vectors, order.size)]
    source_budget = int(checkpoint.metadata.get("requested_budget", checkpoint.energies.size))
    print(
        {
            "evidence_cache": "warm_start",
            "source_budget": source_budget,
            "vectors": int(take.size),
            "source_backend": checkpoint.metadata.get("backend"),
            "path": str(checkpoint.directory),
        },
        flush=True,
    )
    return np.asarray(checkpoint.eigenvectors[:, take]), {
        "warm_start_source_checkpoint": str(checkpoint.directory),
        "warm_start_source_metadata": checkpoint.metadata,
        "same_budget_warm_start": bool(same_budget),
    }


def _primme_folded_spectrum(
    hamiltonian: sp.spmatrix | sp.sparray,
    *,
    target_energy: float,
    subspace_size: int,
    tolerance: float,
    maxiter: int | None,
    random_seed: int,
    cluster_tolerance: float | None,
    warm_start: np.ndarray | None,
    max_basis_size: int | None = None,
) -> tuple[PartialSpectrum, dict[str, Any]]:
    import primme

    started = time.perf_counter()
    last_report = started
    report_interval = float(os.environ.get("QLINKS_QDM_PRIMME_PROGRESS_SECONDS", "300"))
    if not np.isfinite(report_interval) or report_interval < 0:
        raise ValueError("QLINKS_QDM_PRIMME_PROGRESS_SECONDS must be finite and nonnegative")

    def report_callback_progress():
        # PRIMME 3.2.3 exposes live counters from within matrix callbacks.
        # This is not a background heartbeat: a long native LAPACK phase
        # can remain silent until PRIMME requests another matrix product.
        nonlocal last_report
        now = time.perf_counter()
        if report_interval == 0 or now - last_report < report_interval:
            return
        last_report = now
        report = {"primme": "matrix_callback_progress", "elapsed_seconds": now - started}
        getter = getattr(primme, "get_eigsh_param", None)
        if getter is not None:
            # timeDense covers Num_update_VWXR, not every dense operation.
            # In PRIMME 3.2.3, projected LAPACK eigensolves are not included.
            for field in (
                "stats_numMatvecs",
                "stats_numOuterIterations",
                "stats_timeMatvec",
                "stats_timeOrtho",
                "stats_timeDense",
            ):
                report[field] = getter(field)
        print(report, flush=True)

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    n = int(h.shape[0])
    k = min(int(subspace_size), n - 2)
    shifted = h - float(target_energy) * sp.eye(n, dtype=np.complex128, format="csr")

    def folded_matvec(vector):
        report_callback_progress()
        return shifted @ (shifted @ vector)

    def folded_matmat(matrix):
        report_callback_progress()
        return shifted @ (shifted @ matrix)

    folded = spla.LinearOperator(
        shape=(n, n), matvec=folded_matvec, matmat=folded_matmat, dtype=np.complex128
    )
    if warm_start is None:
        rng = np.random.default_rng(int(random_seed))
        random_vector = rng.normal(size=n) + 1.0j * rng.normal(size=n)
        random_vector = random_vector / np.linalg.norm(random_vector)
        # PRIMME's Python wrapper expects ``v0`` to be an N x i initial
        # subspace, including the single-vector case.
        v0 = np.asarray(random_vector[:, None], dtype=np.complex128)
    else:
        v0 = np.asarray(warm_start, dtype=np.complex128)
        if v0.ndim == 1:
            v0 = v0[:, None]
        if v0.ndim != 2 or v0.shape[0] != n:
            raise ValueError(
                "PRIMME warm start must have shape (sector_dimension, n_vectors); "
                f"got {v0.shape} for sector dimension {n}"
            )

    kwargs: dict[str, Any] = {
        "k": k,
        "which": "SA",
        "tol": float(tolerance),
        "v0": v0,
        "method": os.environ.get("QLINKS_QDM_PRIMME_METHOD", "PRIMME_DYNAMIC"),
        "return_stats": True,
        "raise_for_unconverged": False,
        # Level 2 prints convergence/locking events, including residuals.
        "printLevel": int(os.environ.get("QLINKS_QDM_PRIMME_PRINT_LEVEL", "2")),
    }
    if not 0 <= kwargs["printLevel"] <= 5:
        raise ValueError("QLINKS_QDM_PRIMME_PRINT_LEVEL must be between 0 and 5")
    if max_basis_size is not None:
        kwargs["ncv"] = int(max_basis_size)
    if maxiter is not None:
        kwargs["maxiter"] = int(maxiter)
    max_block_size = int(os.environ.get("QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE", "0"))
    min_restart_size = int(os.environ.get("QLINKS_QDM_PRIMME_MIN_RESTART_SIZE", "0"))
    max_prev_retain = int(os.environ.get("QLINKS_QDM_PRIMME_MAX_PREV_RETAIN", "0"))
    if max_block_size > 0:
        kwargs["maxBlockSize"] = max_block_size
    if min_restart_size > 0:
        kwargs["minRestartSize"] = min_restart_size
    if max_prev_retain > 0:
        kwargs["maxPrevRetain"] = max_prev_retain

    print(
        {
            "primme": "starting",
            "dimension": n,
            "requested_eigenpairs": k,
            "warm_start_vectors": int(v0.shape[1]),
            "raw_tolerance": float(tolerance),
            "max_basis_size": max_basis_size,
            "method": kwargs["method"],
            "max_block_size": kwargs.get("maxBlockSize"),
            "min_restart_size": kwargs.get("minRestartSize"),
            "max_prev_retain": kwargs.get("maxPrevRetain"),
            "print_level": kwargs["printLevel"],
            "matrix_callback_report_seconds": report_interval,
        },
        flush=True,
    )
    folded_values, vectors, stats = primme.eigsh(folded, **kwargs)
    elapsed = time.perf_counter() - started
    print({"primme": "returned", "elapsed_seconds": elapsed}, flush=True)
    if len(folded_values) != np.asarray(vectors).shape[1]:
        raise RuntimeError("PRIMME folded eigenvalue/vector counts do not agree")
    # Scalar clustering is insufficient for near-degenerate folded pairs:
    # PRIMME's stopping test is scaled by the norm of the folded operator.
    # Keep cluster_tolerance in the drop-in signature, but resolve the full span.
    del cluster_tolerance
    partial, reconstruction_metadata = reconstruct_hamiltonian_ritz_pairs(
        h,
        np.asarray(vectors),
        target_energy=target_energy,
        requested_subspace_size=subspace_size,
    )

    def scalar_stat(name: str):
        value = stats.get(name) if isinstance(stats, dict) else None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) if isinstance(value, (float, np.floating)) else int(value)
        return None

    solver_metadata = {
        "elapsed_seconds": float(elapsed),
        "num_matvecs": scalar_stat("numMatvecs"),
        "num_outer_iterations": scalar_stat("numOuterIterations"),
        "primme_elapsed_time": scalar_stat("elapsedTime"),
        "method": kwargs["method"],
        "max_basis_size": max_basis_size,
        "max_block_size": kwargs.get("maxBlockSize"),
        "min_restart_size": kwargs.get("minRestartSize"),
        "max_prev_retain": kwargs.get("maxPrevRetain"),
        "warm_start_vectors": int(v0.shape[1]),
        "returned_eigenpairs": int(partial.energies.size),
        "reconstruction_source_commit": os.environ.get("QLINKS_EVIDENCE_SOURCE_COMMIT"),
        **reconstruction_metadata,
    }
    return partial, solver_metadata


def reconstruct_cached_spectrum(
    hamiltonian: sp.spmatrix | sp.sparray,
    *,
    target_energy: float,
    budget: int,
    cache_root: Path,
) -> tuple[PartialSpectrum, dict[str, Any]]:
    """Reconstruct an original PRIMME checkpoint without an iterative solve.

    Original payloads stay immutable. Reconstructed payloads live under a
    versioned backend beside them, with the source metadata recorded in full.
    Poor-quality results are saved for diagnosis, never promoted by this helper.
    """

    problem = folded_problem_description(hamiltonian, target_energy=target_energy)
    source = spectral_checkpoint_directory(
        namespace=CACHE_NAMESPACE,
        problem=problem,
        budget=budget,
        backend="primme",
        cache_root=cache_root,
    )
    checkpoint = load_spectral_checkpoint(
        source,
        expected_problem=problem,
        hamiltonian=hamiltonian,
        sample_vectors=8,
    )
    if checkpoint is None or checkpoint.status is CacheValidationStatus.INCOMPATIBLE:
        raise RuntimeError(
            f"no compatible original PRIMME checkpoint for budget {budget}: {source}"
        )
    if int(checkpoint.metadata.get("requested_budget", -1)) != budget:
        raise RuntimeError(f"checkpoint requested budget does not match {budget}: {source}")
    destination = spectral_checkpoint_directory(
        namespace=CACHE_NAMESPACE,
        problem=problem,
        budget=budget,
        backend=REFINED_BACKEND,
        cache_root=cache_root,
    )
    existing = load_spectral_checkpoint(
        destination,
        expected_problem=problem,
        hamiltonian=hamiltonian,
        sample_vectors=8,
    )
    if existing is not None and existing.status is not CacheValidationStatus.INCOMPATIBLE:
        details = existing.metadata.get("solver", {})
        if details.get("source_checkpoint_metadata") == checkpoint.metadata:
            print({"reconstruction": "reused", "path": str(destination)}, flush=True)
            return _checkpoint_to_partial(existing), details

    started = time.perf_counter()
    partial, reconstruction_details = reconstruct_hamiltonian_ritz_pairs(
        hamiltonian,
        checkpoint.eigenvectors,
        target_energy=target_energy,
        requested_subspace_size=budget,
    )
    details: dict[str, Any] = dict(reconstruction_details)
    details.update(
        reconstruction_elapsed_seconds=float(time.perf_counter() - started),
        reconstruction_source_commit=os.environ.get("QLINKS_EVIDENCE_SOURCE_COMMIT"),
        source_checkpoint=str(source),
        source_checkpoint_metadata=checkpoint.metadata,
        source_maximum_H_residual=float(np.max(checkpoint.residuals, initial=0.0)),
        source_maximum_folded_residual=checkpoint.metadata.get("maximum_transformed_residual"),
    )
    save_spectral_checkpoint(
        destination,
        energies=partial.energies,
        eigenvectors=partial.eigenvectors,
        residuals=partial.residuals,
        transformed_residuals=partial.transformed_residuals,
        problem=problem,
        backend=REFINED_BACKEND,
        requested_budget=budget,
        solver_tolerance=float(checkpoint.metadata["solver_tolerance"]),
        solver_metadata=details,
    )
    print({"reconstruction": "checkpoint_written", "path": str(destination), **details}, flush=True)
    return partial, details


def make_resumable_folded_solver(original_solver: Callable[..., PartialSpectrum]):
    """Return a drop-in replacement for ``folded_spectrum_partial_spectrum``."""

    def resumable_solver(
        hamiltonian: sp.spmatrix | sp.sparray,
        *,
        target_energy: float,
        subspace_size: int,
        tolerance: float = 1.0e-8,
        maxiter: int | None = None,
        ncv_factor: float = 2.05,
        random_seed: int = 20260811,
        cluster_tolerance: float | None = None,
        refine_existing_budget: bool = False,
    ) -> PartialSpectrum:
        requested_backend = _requested_backend()
        backend = _backend()
        h = sp.csr_array(hamiltonian, dtype=np.complex128)
        n = int(h.shape[0])
        if refine_existing_budget and backend != "primme":
            raise ValueError("same-budget refinement requires the PRIMME backend")
        budget = min(int(subspace_size), n - 2)
        if budget <= 0:
            if refine_existing_budget:
                raise ValueError("same-budget refinement requires a nontrivial sector")
            return original_solver(
                h,
                target_energy=target_energy,
                subspace_size=subspace_size,
                tolerance=tolerance,
                maxiter=maxiter,
                ncv_factor=ncv_factor,
                random_seed=random_seed,
                cluster_tolerance=cluster_tolerance,
            )
        problem = folded_problem_description(h, target_energy=target_energy)
        resume = _bool_env("QLINKS_EVIDENCE_CACHE_RESUME", True)
        force = _bool_env("QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE", False)
        write_cache = _bool_env("QLINKS_EVIDENCE_CACHE_WRITE", True)

        if resume and not force:
            checkpoint = _load_exact_checkpoint(
                problem=problem,
                budget=budget,
                tolerance=tolerance,
                hamiltonian=h,
                preferred_backend=backend,
                allow_cross_backend=requested_backend == "auto",
                require_full_orthogonality=refine_existing_budget,
            )
            if checkpoint is not None:
                return _checkpoint_to_partial(checkpoint)

        solver_metadata: dict[str, Any]
        if backend == "primme":
            warm_start, warm_metadata = (
                _load_warm_start(
                    problem=problem,
                    budget=budget,
                    tolerance=tolerance,
                    hamiltonian=h,
                    same_budget=refine_existing_budget,
                )
                if resume and not force
                else (None, {})
            )
            if refine_existing_budget and warm_start is None:
                raise RuntimeError(
                    "no compatible full, orthonormal reconstructed checkpoint "
                    f"for budget {budget}; "
                    "no cold iterative solve was started"
                )
            partial, solver_metadata = _primme_folded_spectrum(
                h,
                target_energy=target_energy,
                subspace_size=budget,
                tolerance=tolerance,
                maxiter=maxiter,
                random_seed=random_seed,
                cluster_tolerance=cluster_tolerance,
                warm_start=warm_start,
                max_basis_size=(
                    min(n, max(budget + 2, int(np.ceil(ncv_factor * budget))))
                    if refine_existing_budget
                    else None
                ),
            )
            solver_metadata.update(warm_metadata)
        else:
            started = time.perf_counter()
            partial = original_solver(
                h,
                target_energy=target_energy,
                subspace_size=budget,
                tolerance=tolerance,
                maxiter=maxiter,
                ncv_factor=ncv_factor,
                random_seed=random_seed,
                cluster_tolerance=cluster_tolerance,
            )
            solver_metadata = {
                "elapsed_seconds": float(time.perf_counter() - started),
                "method": "scipy_arpack",
                "returned_eigenpairs": int(partial.energies.size),
                "ncv_factor": float(ncv_factor),
            }

        if write_cache:
            cache_backend = (
                refinement_backend(tolerance)
                if refine_existing_budget
                else REFINED_BACKEND
                if backend == "primme"
                else backend
            )
            target = spectral_checkpoint_directory(
                namespace=CACHE_NAMESPACE,
                problem=problem,
                budget=budget,
                backend=cache_backend,
                cache_root=default_cache_root(),
            )
            metadata = save_spectral_checkpoint(
                target,
                energies=partial.energies,
                eigenvectors=partial.eigenvectors,
                residuals=partial.residuals,
                transformed_residuals=partial.transformed_residuals,
                problem=problem,
                backend=cache_backend,
                requested_budget=budget,
                solver_tolerance=tolerance,
                solver_metadata=solver_metadata,
            )
            print(
                {
                    "evidence_cache": "checkpoint_written",
                    "budget": int(budget),
                    "backend": backend,
                    "returned_eigenpairs": int(partial.energies.size),
                    "path": str(target),
                    "maximum_residual": metadata["maximum_residual"],
                },
                flush=True,
            )
        return partial

    resumable_solver.__name__ = "resumable_folded_spectrum_partial_spectrum"
    resumable_solver.__doc__ = (
        "Checkpointed drop-in wrapper around the square-QDM folded-spectrum solver."
    )
    return resumable_solver
