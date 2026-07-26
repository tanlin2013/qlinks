from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging import CageSearchConfig, CageSearcher
from qlinks.caging.candidate import (
    BOUNDARY_OVERLAP_MATRIX_METADATA_KEY,
    INTERNAL_KINETIC_MATRIX_METADATA_KEY,
)
from qlinks.models import SpinOneXYChainModel, SquareQDMModel

# Allow direct execution from a source checkout.
for candidate in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]):
    if (candidate / "qlinks").is_dir():
        REPO_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate the qlinks repository root.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _matrix_storage_bytes(matrix) -> int:
    if matrix is None:
        return 0
    if scipy_sparse.issparse(matrix):
        total = int(matrix.data.nbytes)
        if hasattr(matrix, "indices"):
            total += int(matrix.indices.nbytes)
        if hasattr(matrix, "indptr"):
            total += int(matrix.indptr.nbytes)
        return total
    array = np.asarray(matrix)
    return int(array.nbytes)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    # Linux reports KiB.
    return value / 1024.0


def _build_case(case: dict):
    model_kind = case["model"]
    if model_kind == "spin1_xy":
        length = int(case["L"])
        model = SpinOneXYChainModel(
            length=length,
            boundary_condition="periodic",
            j_xy=float(case.get("j_xy", 1.0)),
            d_z=float(case.get("d_z", 1.0)),
            h_z=float(case.get("h_z", 0.0)),
            total_sz=int(case.get("total_sz", -2)),
        )
        return model.build(
            builder="optimized",
            basis_solver="dfs",
            sort_basis=True,
        )

    if model_kind == "square_qdm":
        lx, ly = (int(value) for value in case["size"])
        model = SquareQDMModel(
            lx=lx,
            ly=ly,
            boundary_condition="periodic",
            winding_x=int(case.get("winding_x", 0)),
            winding_y=int(case.get("winding_y", 0)),
            coup_kin=complex(case.get("coup_kin", -1.0)),
            coup_pot=complex(case.get("coup_pot", 1.0)),
        )
        return model.build(
            builder="sparse",
            backend="scipy",
            basis_solver="dfs",
            sort_basis=True,
        )

    raise ValueError(f"Unsupported benchmark model: {model_kind!r}")


def _base_result(case: dict, method: str, build_seconds: float, build) -> dict:
    h = build.hamiltonian
    k = build.kinetic
    v = build.potential
    return {
        "method": method,
        "case": case,
        "case_label": str(case["label"]),
        "model": str(case["model"]),
        "hilbert_dimension": int(h.shape[0]),
        "hamiltonian_nnz": int(h.nnz) if scipy_sparse.issparse(h) else int(np.count_nonzero(h)),
        "build_seconds": float(build_seconds),
        "sparse_hamiltonian_bytes": _matrix_storage_bytes(h),
        "sparse_kinetic_bytes": _matrix_storage_bytes(k),
        "sparse_potential_bytes": _matrix_storage_bytes(v),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "dtype": str(h.dtype),
    }


def run_type1(case: dict, tolerance: float) -> dict:
    t0 = time.perf_counter()
    build = _build_case(case)
    build_seconds = time.perf_counter() - t0
    result = _base_result(case, "Type-I search", build_seconds, build)

    config = CageSearchConfig(
        search_type="type1",
        tolerance=tolerance,
        validate_full_residual=True,
        degenerate_basis_strategy="none",
        store_full_states=False,
    )
    searcher = CageSearcher.from_model_build_result(build, config=config)
    t1 = time.perf_counter()
    search_result = searcher.run()
    method_seconds = time.perf_counter() - t1

    candidate_cache_bytes = 0
    candidate_sizes: list[int] = []
    candidate_summaries: list[dict] = []
    kinetic = build.kinetic
    for candidate_index, candidate in enumerate(search_result.type1_candidates):
        vertices = np.asarray(candidate.vertices, dtype=np.int64)
        candidate_dimension = int(vertices.size)
        candidate_sizes.append(candidate_dimension)
        internal = candidate.metadata.get(INTERNAL_KINETIC_MATRIX_METADATA_KEY)
        gram = candidate.metadata.get(BOUNDARY_OVERLAP_MATRIX_METADATA_KEY)
        candidate_cache_bytes += _matrix_storage_bytes(internal)
        candidate_cache_bytes += _matrix_storage_bytes(gram)

        if gram is None:
            boundary_rank = -1
            boundary_nullity = -1
        else:
            gram_dense = gram.toarray() if scipy_sparse.issparse(gram) else np.asarray(gram)
            gram_eigenvalues = scipy_linalg.eigvalsh(gram_dense, check_finite=False)
            boundary_rank = int(np.sum(np.abs(gram_eigenvalues) > tolerance))
            boundary_nullity = int(candidate_dimension - boundary_rank)

        column_block = kinetic[:, vertices]
        if scipy_sparse.issparse(column_block):
            active_rows = np.unique(column_block.nonzero()[0])
        else:
            active_rows = np.flatnonzero(
                np.any(np.abs(np.asarray(column_block)) > tolerance, axis=1)
            )
        support_set = set(int(value) for value in vertices)
        active_boundary_rows = int(sum(int(row) not in support_set for row in active_rows))
        candidate_signature = candidate.metadata.get("signature")
        if isinstance(candidate_signature, tuple):
            candidate_signature = [
                int(candidate_signature[0]),
                float(np.real(candidate_signature[1])),
            ]

        candidate_summaries.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_signature": candidate_signature,
                "candidate_dimension": candidate_dimension,
                "active_boundary_rows": active_boundary_rows,
                "boundary_shape": [active_boundary_rows, candidate_dimension],
                "boundary_rank": boundary_rank,
                "boundary_nullity": boundary_nullity,
            }
        )

    residuals = [
        float(record.cage_state.full_residual)
        for record in search_result.records
        if record.cage_state.full_residual is not None
    ]
    max_candidate = max(candidate_sizes, default=0)
    complex_bytes = np.dtype(np.complex128).itemsize
    largest_dense_workspace_proxy = 2 * max_candidate * max_candidate * complex_bytes
    sparse_problem_bytes = (
        result["sparse_hamiltonian_bytes"]
        + result["sparse_kinetic_bytes"]
        + result["sparse_potential_bytes"]
    )

    result.update(
        {
            "method_seconds": float(method_seconds),
            "total_seconds": float(build_seconds + method_seconds),
            "peak_rss_mb": _peak_rss_mb(),
            "n_type1_candidates": int(len(search_result.type1_candidates)),
            "candidate_summaries": candidate_summaries,
            "largest_candidate_dimension": int(max_candidate),
            "sum_candidate_dimensions": int(sum(candidate_sizes)),
            "candidate_cache_bytes": int(candidate_cache_bytes),
            "largest_dense_workspace_proxy_bytes": int(largest_dense_workspace_proxy),
            "type1_storage_proxy_bytes": int(
                sparse_problem_bytes + candidate_cache_bytes + largest_dense_workspace_proxy
            ),
            "n_cage_records": int(len(search_result.records)),
            "counts_by_signature": {
                f"{key[0]},{key[1]}": int(value)
                for key, value in search_result.counts_by_signature.items()
            },
            "max_full_residual": max(residuals, default=0.0),
            "search_stage_seconds": {
                key: float(value) for key, value in search_result.search_stage_seconds.items()
            },
        }
    )
    return result


def run_dense_ed(case: dict) -> dict:
    t0 = time.perf_counter()
    build = _build_case(case)
    build_seconds = time.perf_counter() - t0
    result = _base_result(case, "Dense ED", build_seconds, build)

    t1 = time.perf_counter()
    dense = (
        build.hamiltonian.toarray()
        if scipy_sparse.issparse(build.hamiltonian)
        else np.array(build.hamiltonian, copy=True)
    )
    dense_conversion_seconds = time.perf_counter() - t1
    t2 = time.perf_counter()
    eigenvalues, eigenvectors = scipy_linalg.eigh(
        dense,
        check_finite=False,
        overwrite_a=True,
    )
    diagonalization_seconds = time.perf_counter() - t2
    method_seconds = dense_conversion_seconds + diagonalization_seconds

    # Full eigenvectors are retained because a conventional eigenstate search
    # needs eigenvectors, not eigenvalues alone. H + eigenvector storage is a
    # conservative deterministic lower bound; LAPACK may require extra work arrays.
    d = int(dense.shape[0])
    complex_bytes = np.dtype(np.complex128).itemsize
    dense_storage_lower_bound = 2 * d * d * complex_bytes

    # Keep references alive until peak RSS is sampled.
    _ = (eigenvalues, eigenvectors)
    result.update(
        {
            "method_seconds": float(method_seconds),
            "total_seconds": float(build_seconds + method_seconds),
            "dense_conversion_seconds": float(dense_conversion_seconds),
            "diagonalization_seconds": float(diagonalization_seconds),
            "dense_storage_lower_bound_bytes": int(dense_storage_lower_bound),
            "peak_rss_mb": _peak_rss_mb(),
            "n_eigenpairs": d,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("type1", "dense_ed"), required=True)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()

    case = json.loads(args.case_json)
    if args.method == "type1":
        result = run_type1(case, args.tolerance)
    else:
        result = run_dense_ed(case)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
