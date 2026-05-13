#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

from qlinks.models import (
    PXPModel,
    SpinOneXYChainModel,
    ToricCodeModel,
)
from qlinks.models.qdm import SquareQDMModel
from qlinks.models.qlm import SquareQLMModel


@dataclass(frozen=True)
class HamiltonianBenchmarkResult:
    name: str
    model: str
    parameters: dict
    basis_solver: str
    builder: str
    backend: str
    sort_basis: bool
    n_variables: int
    n_states: int
    h_shape: tuple[int, int]
    h_nnz: int
    kinetic_nnz: int | None
    potential_nnz: int | None
    build_total_seconds: float
    basis_seconds: float | None = None
    matrix_seconds: float | None = None


def _time_call(func: Callable):
    gc.collect()
    start = time.perf_counter()
    out = func()
    elapsed = time.perf_counter() - start
    return out, elapsed


def _nnz(matrix) -> int | None:
    if matrix is None:
        return None

    return int(matrix.nnz)


def make_benchmark_cases() -> list[tuple[str, object]]:
    """
    Keep default cases modest. Larger Hamiltonians should be run manually.
    """
    cases: list[tuple[str, object]] = []

    cases.append(
        (
            "pxp_chain_L16_open",
            PXPModel.chain(
                length=16,
                boundary_condition="open",
            ),
        )
    )

    cases.append(
        (
            "spin_one_xy_chain_L8_open",
            SpinOneXYChainModel(
                length=8,
                boundary_condition="open",
                j_xy=1.0,
            ),
        )
    )

    cases.append(
        (
            "toric_code_2x2_pbc",
            ToricCodeModel(
                lx=2,
                ly=2,
                boundary_condition="periodic",
            ),
        )
    )

    cases.append(
        (
            "square_qlm_4x4_pbc_w00",
            SquareQLMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
            ),
        )
    )

    cases.append(
        (
            "square_qdm_4x4_pbc_w00",
            SquareQDMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
            ),
        )
    )

    return cases


def model_parameters(model: object) -> dict:
    if hasattr(model, "__dataclass_fields__"):
        return asdict(model)

    return {key: value for key, value in vars(model).items() if not key.startswith("_")}


def run_hamiltonian_benchmark(
    *,
    name: str,
    model: object,
    basis_solver: str,
    builder: str,
    backend: str,
    sort_basis: bool,
    split_basis_timing: bool,
) -> HamiltonianBenchmarkResult:
    basis_seconds: float | None = None
    matrix_seconds: float | None = None

    if split_basis_timing:
        basis, basis_seconds = _time_call(
            lambda: model.build_basis(
                solver=basis_solver,
                sort=sort_basis,
            )
        )

        result, matrix_seconds = _time_call(
            lambda: model.build(
                basis=basis,
                basis_solver=basis_solver,
                builder=builder,
                backend=backend,
                sort_basis=sort_basis,
            )
        )

        build_total_seconds = basis_seconds + matrix_seconds

    else:
        result, build_total_seconds = _time_call(
            lambda: model.build(
                basis_solver=basis_solver,
                builder=builder,
                backend=backend,
                sort_basis=sort_basis,
            )
        )

    H = result.hamiltonian

    return HamiltonianBenchmarkResult(
        name=name,
        model=type(model).__name__,
        parameters=model_parameters(model),
        basis_solver=basis_solver,
        builder=builder,
        backend=backend,
        sort_basis=sort_basis,
        n_variables=model.layout.n_variables,
        n_states=result.basis.n_states,
        h_shape=tuple(int(x) for x in H.shape),
        h_nnz=int(H.nnz),
        kinetic_nnz=_nnz(result.kinetic),
        potential_nnz=_nnz(result.potential),
        build_total_seconds=build_total_seconds,
        basis_seconds=basis_seconds,
        matrix_seconds=matrix_seconds,
    )


def print_table(results: list[HamiltonianBenchmarkResult]) -> None:
    headers = [
        "name",
        "model",
        "solver",
        "builder",
        "n_states",
        "H.nnz",
        "K.nnz",
        "V.nnz",
        "total_s",
        "basis_s",
        "matrix_s",
    ]

    rows = [
        [
            result.name,
            result.model,
            result.basis_solver,
            result.builder,
            str(result.n_states),
            str(result.h_nnz),
            str(result.kinetic_nnz),
            str(result.potential_nnz),
            f"{result.build_total_seconds:.6f}",
            "" if result.basis_seconds is None else f"{result.basis_seconds:.6f}",
            "" if result.matrix_seconds is None else f"{result.matrix_seconds:.6f}",
        ]
        for result in results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hamiltonian construction.")
    parser.add_argument(
        "--basis-solver",
        default="dfs",
        choices=["dfs", "brute_force", "cpsat"],
    )
    parser.add_argument(
        "--builder",
        default="sparse",
        choices=["sparse", "optimized", "bitmask"],
    )
    parser.add_argument(
        "--backend",
        default="scipy",
    )
    parser.add_argument(
        "--no-sort-basis",
        action="store_true",
    )
    parser.add_argument(
        "--split-basis-timing",
        action="store_true",
        help="Time basis generation separately from matrix construction.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only cases whose name contains this substring.",
    )

    args = parser.parse_args()

    results: list[HamiltonianBenchmarkResult] = []

    for name, model in make_benchmark_cases():
        if args.only is not None and args.only not in name:
            continue

        print(f"Running {name} ...", flush=True)

        try:
            result = run_hamiltonian_benchmark(
                name=name,
                model=model,
                basis_solver=args.basis_solver,
                builder=args.builder,
                backend=args.backend,
                sort_basis=not args.no_sort_basis,
                split_basis_timing=args.split_basis_timing,
            )
        except NotImplementedError as exc:
            print(f"  skipped: {exc}")
            continue
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue

        results.append(result)

    print()
    print_table(results)

    if args.json is not None:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(result) for result in results],
                f,
                indent=2,
                default=str,
            )

        print(f"\nWrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
