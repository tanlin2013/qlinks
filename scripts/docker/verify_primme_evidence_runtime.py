#!/usr/bin/env python
"""Numerically smoke-test the optional PRIMME evidence runtime."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

from qdm_resumable_spectrum import make_resumable_folded_solver  # noqa: E402


def _no_arpack_fallback(*args, **kwargs):
    del args, kwargs
    raise AssertionError("explicit PRIMME smoke test must not enter the ARPACK fallback")


def _assert_spectrum(result, expected_distances: np.ndarray) -> None:
    distances = np.sort(np.abs(np.asarray(result.energies) - 5.25))
    np.testing.assert_allclose(distances, np.sort(expected_distances), atol=1.0e-6)
    if result.maximum_residual > 1.0e-6:
        raise AssertionError(f"physical residual too large: {result.maximum_residual:.3e}")


def main() -> None:
    import primme

    with tempfile.TemporaryDirectory(prefix="qlinks-primme-smoke-") as tmp:
        os.environ["QLINKS_EVIDENCE_CACHE_ROOT"] = tmp
        os.environ["QLINKS_EVIDENCE_CACHE_RESUME"] = "1"
        os.environ["QLINKS_EVIDENCE_CACHE_WRITE"] = "1"
        os.environ["QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE"] = "0"
        os.environ["QLINKS_QDM_FOLDED_BACKEND"] = "primme"
        os.environ["QLINKS_QDM_PRIMME_WARM_START_VECTORS"] = "2"

        solver = make_resumable_folded_solver(_no_arpack_fallback)
        matrix = sp.csr_array(np.diag(np.arange(12.0, dtype=float)))

        # The first solve exercises PRIMME's single-vector initial subspace.
        first = solver(matrix, target_energy=5.25, subspace_size=3, tolerance=1.0e-8)
        if first.method != "folded_spectrum_primme":
            raise AssertionError(first.method)
        _assert_spectrum(first, np.array([0.25, 0.75, 1.25]))

        # The larger budget must consume the validated lower-budget checkpoint
        # as a matrix-valued warm start and still recover the correct spectrum.
        second = solver(matrix, target_energy=5.25, subspace_size=4, tolerance=1.0e-8)
        if second.method != "folded_spectrum_primme":
            raise AssertionError(second.method)
        _assert_spectrum(second, np.array([0.25, 0.75, 1.25, 1.75]))

        # A repeated request should be served from the stable evidence cache.
        cached = solver(matrix, target_energy=5.25, subspace_size=4, tolerance=1.0e-8)
        if cached.method != "folded_spectrum_cache_primme":
            raise AssertionError(cached.method)
        _assert_spectrum(cached, np.array([0.25, 0.75, 1.25, 1.75]))

        print(
            {
                "primme_version": getattr(primme, "__version__", "unknown"),
                "first_method": first.method,
                "warm_start_method": second.method,
                "cached_method": cached.method,
                "maximum_residual": float(max(first.maximum_residual, second.maximum_residual)),
            }
        )


if __name__ == "__main__":
    main()
