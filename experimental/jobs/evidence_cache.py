"""Reusable checkpoint storage for expensive experimental evidence jobs.

Timestamped ``evidence_jobs`` directories are immutable execution records.  This
module provides a separate, stable cache for expensive numerical payloads that
can be validated and reused by later attempts.  The cache key describes the
scientific problem, not the git commit that happened to produce it; provenance
is recorded separately in checkpoint metadata.

This module intentionally lives under ``experimental/jobs``.  It is workflow
infrastructure, not a supported :mod:`qlinks` API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

CHECKPOINT_SCHEMA_VERSION = 1


class CacheValidationStatus(StrEnum):
    """Compatibility state of one cached spectral payload."""

    VALID_FINAL = "VALID_FINAL"
    VALID_WARM_START = "VALID_WARM_START"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True, slots=True)
class SpectralCheckpoint:
    """Memory-mapped spectral checkpoint returned by :func:`load_spectral_checkpoint`."""

    directory: Path
    energies: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    transformed_residuals: np.ndarray | None
    metadata: dict[str, Any]
    status: CacheValidationStatus
    validation: dict[str, Any]


def _repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "qlinks").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not locate qlinks repository root from {here}")


def default_cache_root() -> Path:
    """Return the stable evidence-cache root, honoring ``QLINKS_EVIDENCE_CACHE_ROOT``."""

    raw = os.environ.get("QLINKS_EVIDENCE_CACHE_ROOT")
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _repo_root() / path
        return path.resolve(strict=False)
    return (_repo_root() / "experimental" / "data" / "evidence_cache").resolve()


def default_registry_root() -> Path:
    """Return the runtime evidence registry root."""

    raw = os.environ.get("QLINKS_EVIDENCE_REGISTRY_ROOT")
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _repo_root() / path
        return path.resolve(strict=False)
    return (_repo_root() / "experimental" / "data" / "evidence_registry").resolve()


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a problem description deterministically."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def problem_signature(payload: Mapping[str, Any], *, prefix: str = "p") -> str:
    """Return a compact SHA-256 signature for a scientific problem description."""

    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:24]}"


def sparse_matrix_fingerprint(matrix: sp.spmatrix | sp.sparray) -> str:
    """Hash a sparse matrix exactly enough for checkpoint compatibility.

    The hash includes shape, dtype, CSR structure and numerical values.  It is
    deliberately independent of object identity and run directory.
    """

    csr = sp.csr_array(matrix)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(v) for v in csr.shape)).encode("ascii"))
    hasher.update(str(csr.dtype).encode("ascii"))
    for array in (csr.indptr, csr.indices, csr.data):
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        hasher.update(memoryview(contiguous).cast("B"))
    return hasher.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def spectral_checkpoint_directory(
    *,
    namespace: str,
    problem: Mapping[str, Any],
    budget: int,
    backend: str,
    cache_root: Path | None = None,
) -> Path:
    """Return the canonical cache directory for one spectral checkpoint."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    normalized_namespace = namespace.strip("/")
    if not normalized_namespace or ".." in Path(normalized_namespace).parts:
        raise ValueError(f"invalid cache namespace {namespace!r}")
    signature = problem_signature(problem)
    root = default_cache_root() if cache_root is None else Path(cache_root).resolve(strict=False)
    return root / normalized_namespace / signature / backend / f"budget_{int(budget):08d}"


def save_spectral_checkpoint(
    directory: Path,
    *,
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    residuals: np.ndarray,
    transformed_residuals: np.ndarray | None,
    problem: Mapping[str, Any],
    backend: str,
    requested_budget: int,
    solver_tolerance: float,
    solver_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist one completed spectral solve.

    ``metadata.json`` is written last and acts as the completion marker.  A
    killed process therefore leaves at worst unreferenced ``*.tmp``/array files,
    never a checkpoint that appears complete.
    """

    directory = Path(directory)
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    residual_values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    transformed = (
        None
        if transformed_residuals is None
        else np.asarray(transformed_residuals, dtype=np.float64).reshape(-1)
    )
    if vectors.ndim != 2 or vectors.shape[1] != values.size:
        raise ValueError("eigenvector/eigenvalue shape mismatch")
    if residual_values.size != values.size:
        raise ValueError("residual/eigenvalue shape mismatch")
    if transformed is not None and transformed.size != values.size:
        raise ValueError("transformed residual/eigenvalue shape mismatch")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(residual_values)):
        raise ValueError("spectral checkpoint contains non-finite values")

    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate an older completed checkpoint before replacing any payload
    # array. Otherwise an interrupted force-recompute could leave old metadata
    # pointing at a mixture of old and new arrays and falsely look complete.
    (directory / "metadata.json").unlink(missing_ok=True)
    _atomic_save_npy(directory / "energies.npy", values)
    _atomic_save_npy(directory / "eigenvectors.npy", vectors)
    _atomic_save_npy(directory / "residuals.npy", residual_values)
    if transformed is not None:
        _atomic_save_npy(directory / "transformed_residuals.npy", transformed)
    else:
        (directory / "transformed_residuals.npy").unlink(missing_ok=True)

    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "problem": dict(problem),
        "problem_signature": problem_signature(problem),
        "backend": str(backend),
        "requested_budget": int(requested_budget),
        "returned_eigenpairs": int(values.size),
        "sector_dimension": int(vectors.shape[0]),
        "solver_tolerance": float(solver_tolerance),
        "maximum_residual": float(np.max(residual_values, initial=0.0)),
        "maximum_transformed_residual": (
            float(np.max(transformed, initial=0.0)) if transformed is not None else None
        ),
        "producer_commit": _git_commit(),
        "producer_run_id": os.environ.get("QLINKS_EVIDENCE_RUN_ID"),
        "producer_run_timestamp": os.environ.get("QLINKS_EVIDENCE_RUN_TIMESTAMP"),
    }
    if solver_metadata:
        metadata["solver"] = dict(solver_metadata)
    _atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def _sample_indices(count: int, requested: int) -> np.ndarray:
    if count <= 0 or requested <= 0:
        return np.zeros(0, dtype=np.int64)
    n = min(int(count), int(requested))
    if n == count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, num=n, dtype=np.int64))


def load_spectral_checkpoint(
    directory: Path,
    *,
    expected_problem: Mapping[str, Any],
    hamiltonian: sp.spmatrix | sp.sparray | None = None,
    requested_solver_tolerance: float | None = None,
    residual_tolerance: float | None = None,
    orthogonality_tolerance: float = 1.0e-6,
    sample_vectors: int = 8,
) -> SpectralCheckpoint | None:
    """Load and cheaply validate one spectral checkpoint.

    Scientific incompatibility returns an ``INCOMPATIBLE`` checkpoint when the
    payload is structurally readable.  Corrupt/incomplete directories return
    ``None``.  A scientifically compatible payload that is numerically sane but
    does not satisfy the requested final residual quality is returned as
    ``VALID_WARM_START``.
    """

    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    required = (
        metadata_path,
        directory / "energies.npy",
        directory / "eigenvectors.npy",
        directory / "residuals.npy",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        energies = np.load(directory / "energies.npy", mmap_mode="r", allow_pickle=False)
        vectors = np.load(directory / "eigenvectors.npy", mmap_mode="r", allow_pickle=False)
        residuals = np.load(directory / "residuals.npy", mmap_mode="r", allow_pickle=False)
        transformed_path = directory / "transformed_residuals.npy"
        transformed = (
            np.load(transformed_path, mmap_mode="r", allow_pickle=False)
            if transformed_path.is_file()
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    validation: dict[str, Any] = {}
    expected_signature = problem_signature(expected_problem)
    compatible = bool(
        metadata.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and metadata.get("problem_signature") == expected_signature
        and metadata.get("problem") == dict(expected_problem)
    )
    validation["problem_signature_match"] = compatible

    structural = bool(
        vectors.ndim == 2
        and energies.ndim == 1
        and residuals.ndim == 1
        and vectors.shape[1] == energies.size == residuals.size
        and (transformed is None or transformed.size == energies.size)
        and vectors.shape[0] == int(expected_problem.get("sector_dimension", vectors.shape[0]))
    )
    validation["structural_valid"] = structural
    if not structural:
        return None

    if not compatible:
        return SpectralCheckpoint(
            directory=directory,
            energies=energies,
            eigenvectors=vectors,
            residuals=residuals,
            transformed_residuals=transformed,
            metadata=metadata,
            status=CacheValidationStatus.INCOMPATIBLE,
            validation=validation,
        )

    finite = bool(np.all(np.isfinite(energies)) and np.all(np.isfinite(residuals)))
    validation["finite"] = finite
    if not finite:
        return None

    stored_max = float(np.max(np.asarray(residuals), initial=0.0))
    validation["stored_maximum_residual"] = stored_max
    if residual_tolerance is None:
        base = 1.0e-8 if requested_solver_tolerance is None else float(requested_solver_tolerance)
        residual_tolerance = max(1.0e-7, 100.0 * base)
    final_quality = stored_max <= float(residual_tolerance)

    sample = _sample_indices(energies.size, sample_vectors)
    if hamiltonian is not None and sample.size:
        block = np.asarray(vectors[:, sample])
        values = np.asarray(energies[sample])
        action = hamiltonian @ block
        recomputed = np.linalg.norm(action - block * values[None, :], axis=0)
        gram = block.conj().T @ block
        orthogonality = float(np.linalg.norm(gram - np.eye(sample.size), ord=2))
        validation.update(
            {
                "sample_size": int(sample.size),
                "sample_maximum_residual": float(np.max(recomputed, initial=0.0)),
                "sample_orthogonality_residual": orthogonality,
            }
        )
        final_quality &= bool(
            float(np.max(recomputed, initial=0.0)) <= float(residual_tolerance)
            and orthogonality <= float(orthogonality_tolerance)
        )

    if requested_solver_tolerance is not None:
        produced_tolerance = float(metadata.get("solver_tolerance", math.inf))
        validation["solver_tolerance_sufficient"] = produced_tolerance <= float(
            requested_solver_tolerance
        )
        final_quality &= produced_tolerance <= float(requested_solver_tolerance)

    status = (
        CacheValidationStatus.VALID_FINAL
        if final_quality
        else CacheValidationStatus.VALID_WARM_START
    )
    return SpectralCheckpoint(
        directory=directory,
        energies=energies,
        eigenvectors=vectors,
        residuals=residuals,
        transformed_residuals=transformed,
        metadata=metadata,
        status=status,
        validation=validation,
    )


def iter_spectral_checkpoints(
    *,
    namespace: str,
    problem: Mapping[str, Any],
    cache_root: Path | None = None,
) -> list[Path]:
    """List completed checkpoint directories for one scientific problem."""

    root = default_cache_root() if cache_root is None else Path(cache_root).resolve(strict=False)
    problem_root = root / namespace.strip("/") / problem_signature(problem)
    if not problem_root.is_dir():
        return []
    return sorted(path.parent for path in problem_root.glob("*/budget_*/metadata.json"))


def write_registry_entry(name: str, payload: Mapping[str, Any]) -> Path:
    """Write one runtime evidence-registry entry atomically."""

    if not name or "/" in name or ".." in name:
        raise ValueError(f"invalid registry name {name!r}")
    path = default_registry_root() / f"{name}.json"
    _atomic_write_json(path, payload)
    return path
