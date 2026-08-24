"""Validate optional runtime extras requested during Docker builds."""

from __future__ import annotations

import argparse
import importlib
import pathlib
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_extras(raw: str) -> set[str]:
    return {part.strip() for part in shlex.split(raw) if part.strip()}


def _require_import(module_name: str, *, extra: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Docker optional extra {extra!r} was requested, but module "
            f"{module_name!r} is not importable. Check the selected Docker "
            "feature, Python version, and build dependencies."
        ) from exc


def _verify_tn_extra() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "The 'tn' extra is currently constrained to Python < 3.14. "
            "Build the TN Docker image with, for example, "
            "--build-arg PYTHON_VERSION=3.13 --build-arg QLINKS_EXTRAS=tn."
        )

    for module_name in ("autograd", "llvmlite", "numba", "quimb"):
        _require_import(module_name, extra="tn")

    from verify_tn_environment import main as verify_tn_environment

    verify_tn_environment()


def _verify_primme_feature() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "The PRIMME evidence image is currently certified only on Python < 3.14. "
            "Build it with --build-arg PYTHON_VERSION=3.13 and "
            "--build-arg QLINKS_EXTRAS='notebook primme'."
        )
    _require_import("primme", extra="primme")


def _verify_notebook_feature() -> None:
    # Import nbconvert itself, not just the Jupyter front ends. The current
    # nbconvert/lxml combination imports lxml.html.clean eagerly, so this catches
    # a missing lxml-html-clean package before a notebook image is published.
    for module_name in ("ipykernel", "jupyterlab", "nbconvert", "lxml.html.clean"):
        _require_import(module_name, extra="notebook")


def _verify_import_extra(extra: str, module_names: tuple[str, ...]) -> None:
    for module_name in module_names:
        _require_import(module_name, extra=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        default="",
        help="Whitespace-separated optional Docker features/project extras.",
    )
    args = parser.parse_args()

    extras = _parse_extras(args.extras)

    import qlinks

    print(f"qlinks: {qlinks.__version__}")
    if not extras:
        print("No optional Docker extras requested.")
        return

    print(f"Requested optional Docker features/extras: {', '.join(sorted(extras))}")

    if "automorphism" in extras:
        _verify_import_extra("automorphism", ("pynauty",))
    if "cpsat" in extras:
        _verify_import_extra("cpsat", ("ortools.sat.python.cp_model",))
    if "cupy" in extras:
        _verify_import_extra("cupy", ("cupy", "cupyx.scipy.sparse"))
    if "distributed" in extras:
        _verify_import_extra("distributed", ("ray",))
    if "drawing" in extras:
        _verify_import_extra("drawing", ("cairo", "igraph", "plotly", "pyvis"))
    if "notebook" in extras:
        _verify_notebook_feature()
    if "primme" in extras:
        _verify_primme_feature()
    if "storage" in extras:
        _verify_import_extra("storage", ("h5py", "pyarrow"))
    if "tn" in extras:
        _verify_tn_extra()


if __name__ == "__main__":
    main()
