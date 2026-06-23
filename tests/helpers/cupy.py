from __future__ import annotations

import os
from typing import Any

import pytest


def _unavailable(message: str) -> None:
    """Skip optional CuPy tests, or fail when GPU CI explicitly requires CuPy."""
    if os.environ.get("QLINKS_REQUIRE_CUPY_GPU") == "1":
        pytest.fail(message)
    pytest.skip(message)


def require_functional_cupy() -> Any:
    """Return CuPy only when a CUDA-capable runtime is actually usable.

    CuPy may import successfully on CPU-only CI runners or machines with an
    incompatible NVIDIA driver, but then fail at the first CUDA runtime call.
    Tests that exercise the real CuPy backend should call this helper before
    constructing backend objects or allocating arrays.

    Returns:
        The imported ``cupy`` module.

    Skips:
        If CuPy/CuPyX is not installed or if no functional CUDA runtime is
        available. If ``QLINKS_REQUIRE_CUPY_GPU=1`` is set, the same condition
        is reported as a test failure instead, which is useful for dedicated
        GPU CI jobs.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse  # noqa: F401
        import cupyx.scipy.sparse.linalg  # noqa: F401
    except ImportError as exc:
        _unavailable(f"CuPy backend is not installed: {exc}")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _unavailable("CuPy is installed, but no CUDA device is visible.")

        probe = cp.asarray([1.0])
        cp.cuda.Stream.null.synchronize()
        float(cp.asnumpy(probe)[0])
    except Exception as exc:  # pragma: no cover - depends on local CUDA setup
        _unavailable(
            "CuPy is installed, but no functional CUDA runtime is available: "
            f"{type(exc).__name__}: {exc}"
        )

    return cp
