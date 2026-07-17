"""Print and validate the optional tensor-network runtime."""

from __future__ import annotations

import platform
import sys

import numpy as np

if np.lib.NumpyVersion(np.__version__) >= "2.5.0":
    raise RuntimeError(
        "The TN environment requires NumPy < 2.5 for Numba 0.66; "
        f"found NumPy {np.__version__}. Regenerate poetry.lock with the "
        "project constraint numpy>=2.2,<2.5."
    )

import llvmlite
import numba
import quimb

import qlinks


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"qlinks: {qlinks.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"quimb: {quimb.__version__}")
    print(f"numba: {numba.__version__}")
    print(f"llvmlite: {llvmlite.__version__}")


if __name__ == "__main__":
    main()
