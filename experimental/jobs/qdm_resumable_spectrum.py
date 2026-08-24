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

CACHE_NAMESPACE = "qdm/checkerboard_large_strip"
_FINGERPRINT_CACHE: dict[tuple[int, tuple[int, int], int], str] = {}


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
    csr = sp.csr_array(matrix)
    key = (id(matrix), tuple(int(v) for v in csr.shape), int(csr.nnz))
    cached = _FINGERPRINT_CACHE.get(key)
    if cached is None:
        cached = sparse_matrix_fingerprint(csr)
        _FINGERPRINT_CACHE[key] = cached
    return cached


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
    )


def _load_exact_checkpoint(
    *,
    problem: dict[str, Any],
    budget: int,
    tolerance: float,
    hamiltonian: sp.spmatrix | sp.sparray,
    preferred_backend: str,
    allow_cross_backend: bool,
):
    candidates = iter_spectral_checkpoints(namespace=CACHE_NAMESPACE, problem=problem)
    exact = [path for path in candidates if path.name == f"budget_{int(budget):08d}"]
    if not allow_cross_backend:
        exact = [path for path in exact if path.parent.name == preferred_backend]
    exact.sort(key=lambda path: (path.parent.name != preferred_backend, str(path)))
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
) -> np.ndarray | None:
    max_vectors = int(os.environ.get("QLINKS_QDM_PRIMME_WARM_START_VECTORS", "256"))
    if max_vectors <= 0:
        return None
    candidates = iter_spectral_checkpoints(namespace=CACHE_NAMESPACE, problem=problem)
    ranked: list[tuple[int, Any]] = []
    for path in candidates:
        try:
            previous_budget = int(path.name.removeprefix("budget_"))
        except ValueError:
            continue
        if previous_budget >= int(budget):
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
        ranked.append((previous_budget, checkpoint))
    if not ranked:
        return None
    _, checkpoint = max(ranked, key=lambda item: item[0])
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
    return np.asarray(checkpoint.eigenvectors[:, take])


def _postprocess_folded_vectors(
    hamiltonian: sp.csr_array,
    shifted: sp.csr_array,
    *,
    target_energy: float,
    folded_values: np.ndarray,
    vectors: np.ndarray,
    tolerance: float,
    cluster_tolerance: float | None,
    requested_subspace_size: int,
    method: str,
) -> PartialSpectrum:
    folded_values = np.asarray(np.real(folded_values), dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.complex128)
    if folded_values.size == 0 or vectors.shape[1] == 0:
        raise RuntimeError(f"{method} returned no folded-spectrum eigenpairs")
    order = np.argsort(folded_values)
    folded_values = folded_values[order]
    vectors = vectors[:, order]
    k = int(folded_values.size)

    if cluster_tolerance is None:
        cluster_tolerance = max(1.0e-11, 50.0 * float(tolerance))

    def folded_matmat(matrix: np.ndarray) -> np.ndarray:
        return shifted @ (shifted @ matrix)

    energies_out: list[np.ndarray] = []
    vectors_out: list[np.ndarray] = []
    transformed_residuals_out: list[np.ndarray] = []
    begin = 0
    while begin < k:
        end = begin + 1
        reference = float(folded_values[begin])
        scale = max(1.0, abs(reference))
        while (
            end < k
            and abs(float(folded_values[end]) - reference) <= float(cluster_tolerance) * scale
        ):
            end += 1
        block = vectors[:, begin:end]
        h_block = hamiltonian @ block
        compressed = block.conj().T @ h_block
        compressed = 0.5 * (compressed + compressed.conj().T)
        local_energies, rotation = np.linalg.eigh(compressed)
        rotated = block @ rotation
        folded_residual = np.linalg.norm(
            folded_matmat(block) - block * folded_values[None, begin:end], axis=0
        )
        energies_out.append(np.asarray(local_energies, dtype=np.float64))
        vectors_out.append(np.asarray(rotated, dtype=np.complex128))
        transformed_residuals_out.append(np.asarray(folded_residual, dtype=np.float64))
        begin = end

    energies = np.concatenate(energies_out)
    eigenvectors = np.column_stack(vectors_out)
    transformed_residuals = np.concatenate(transformed_residuals_out)
    energy_order = np.argsort(energies)
    energies = energies[energy_order]
    eigenvectors = eigenvectors[:, energy_order]
    transformed_residuals = transformed_residuals[energy_order]
    residual_matrix = hamiltonian @ eigenvectors - eigenvectors * energies[None, :]
    residuals = np.linalg.norm(residual_matrix, axis=0)
    return PartialSpectrum(
        energies=np.asarray(energies, dtype=np.float64),
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        residuals=np.asarray(residuals, dtype=np.float64),
        sigma=float(target_energy),
        target_energy=float(target_energy),
        method=method,
        requested_subspace_size=int(requested_subspace_size),
        transformed_residuals=np.asarray(transformed_residuals, dtype=np.float64),
        peak_rss_gib=process_peak_rss_gib(),
    )


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
) -> tuple[PartialSpectrum, dict[str, Any]]:
    import primme

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    n = int(h.shape[0])
    k = min(int(subspace_size), n - 2)
    shifted = h - float(target_energy) * sp.eye(n, dtype=np.complex128, format="csr")

    def folded_matvec(vector):
        return shifted @ (shifted @ vector)

    def folded_matmat(matrix):
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
    }
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

    started = time.perf_counter()
    folded_values, vectors, stats = primme.eigsh(folded, **kwargs)
    elapsed = time.perf_counter() - started
    partial = _postprocess_folded_vectors(
        h,
        shifted,
        target_energy=target_energy,
        folded_values=np.asarray(folded_values),
        vectors=np.asarray(vectors),
        tolerance=tolerance,
        cluster_tolerance=cluster_tolerance,
        requested_subspace_size=subspace_size,
        method="folded_spectrum_primme",
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
        "warm_start_vectors": int(v0.shape[1]),
        "returned_eigenpairs": int(partial.energies.size),
    }
    return partial, solver_metadata


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
    ) -> PartialSpectrum:
        requested_backend = _requested_backend()
        backend = _backend()
        h = sp.csr_array(hamiltonian, dtype=np.complex128)
        n = int(h.shape[0])
        budget = min(int(subspace_size), n - 2)
        if budget <= 0:
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
            )
            if checkpoint is not None:
                return _checkpoint_to_partial(checkpoint)

        solver_metadata: dict[str, Any]
        if backend == "primme":
            warm_start = (
                _load_warm_start(
                    problem=problem,
                    budget=budget,
                    tolerance=tolerance,
                    hamiltonian=h,
                )
                if resume and not force
                else None
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
            )
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
            target = spectral_checkpoint_directory(
                namespace=CACHE_NAMESPACE,
                problem=problem,
                budget=budget,
                backend=backend,
                cache_root=default_cache_root(),
            )
            metadata = save_spectral_checkpoint(
                target,
                energies=partial.energies,
                eigenvectors=partial.eigenvectors,
                residuals=partial.residuals,
                transformed_residuals=partial.transformed_residuals,
                problem=problem,
                backend=backend,
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
