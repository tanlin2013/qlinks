"""Print and validate the optional tensor-network runtime."""

from __future__ import annotations

import importlib
import pathlib
import platform
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_tn_python() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "The 'tn' extra is currently constrained to Python < 3.14. "
            "Build the TN environment with Python 3.13 or lower."
        )


def _import_required_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"The optional TN dependency {module_name!r} is not importable. "
            "Install qlinks with the 'tn' extra on Python < 3.14."
        ) from exc


def _require_numpy_constraint() -> None:
    if np.lib.NumpyVersion(np.__version__) >= "2.4.0":
        raise RuntimeError(
            "The TN environment requires NumPy < 2.4 for Numba 0.62; "
            f"found NumPy {np.__version__}. Regenerate uv.lock with the "
            "locked TN dependency constraints."
        )


def main() -> None:
    _require_tn_python()
    _require_numpy_constraint()

    autograd = _import_required_module("autograd")
    llvmlite = _import_required_module("llvmlite")
    numba = _import_required_module("numba")
    quimb = _import_required_module("quimb")

    import qlinks

    print(f"Python: {sys.version.split()[0]}")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"qlinks: {qlinks.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"quimb: {quimb.__version__}")
    print(f"autograd: {getattr(autograd, '__version__', 'installed')}")
    print(f"numba: {numba.__version__}")
    print(f"llvmlite: {llvmlite.__version__}")


if __name__ == "__main__":
    main()
