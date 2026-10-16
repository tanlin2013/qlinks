"""Report PRIMME's system BLAS/LAPACK providers, independently of NumPy/SciPy."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-openblas", action="store_true")
    args = parser.parse_args()

    import primme

    result = subprocess.run(["ldd", primme.__file__], check=True, capture_output=True, text=True)
    providers = {}
    for line in result.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if not name.startswith(("libblas.", "liblapack.")) and "openblas" not in name.lower():
            continue
        target = line.partition("=>")[2].strip().split()
        if not target or target[0] == "not":
            raise RuntimeError(f"unresolved PRIMME linear algebra dependency: {line.strip()}")
        providers[name] = str(Path(target[0]).resolve(strict=True))
    optimized = bool(providers) and all("openblas" in path.lower() for path in providers.values())
    print(
        json.dumps(
            {
                "primme_extension": primme.__file__,
                "blas_lapack_providers": providers,
                "all_providers_openblas": optimized,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.require_openblas and not optimized:
        raise SystemExit(
            "PRIMME must resolve its system BLAS/LAPACK to OpenBLAS; "
            "NumPy/SciPy's bundled OpenBLAS libraries are not sufficient."
        )


if __name__ == "__main__":
    main()
