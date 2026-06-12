#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from qlinks.models import (
    PXPModel,
    SpinOneXYChainModel,
    ToricCodeModel,
)
from qlinks.models.qdm import SquareQDMModel
from qlinks.models.qlm import SquareQLMModel

BuilderSelection = Literal["recommended", "all", "sparse", "optimized", "bitmask"]


@dataclass(frozen=True)
class HamiltonianBenchmarkCase:
    name: str
    model: object
    recommended_builder: str
    builders: tuple[str, ...]


@dataclass(frozen=True)
class HamiltonianBenchmarkResult:
    name: str
    model: str
    parameters: dict
    basis_solver: str
    builder: str
    recommended_builder: str
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


def _json_key(key: object) -> str:
    """Return a stable JSON object key for benchmark metadata."""
    return str(key)


def _jsonable(value: object) -> object:
    """Recursively convert benchmark metadata to JSON-serializable values.

    ``json.dump(default=str)`` only handles unsupported values, not unsupported
    dictionary keys. Model parameter dictionaries can contain tuple keys, for
    example lattice translation maps, so keys must be normalized before dumping.
    """
    if isinstance(value, dict):
        return {_json_key(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]

    if isinstance(value, set | frozenset):
        return [_jsonable(item) for item in sorted(value, key=str)]

    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}

    if isinstance(value, str | int | float | bool) or value is None:
        return value

    return str(value)


def make_benchmark_cases() -> list[HamiltonianBenchmarkCase]:
    """
    Keep default cases modest. Larger Hamiltonians should be run manually.
    """
    cases: list[HamiltonianBenchmarkCase] = []

    cases.append(
        HamiltonianBenchmarkCase(
            name="pxp_chain_L16_open",
            model=PXPModel.chain(
                length=16,
                boundary_condition="open",
            ),
            recommended_builder="bitmask",
            builders=("sparse", "optimized", "bitmask"),
        )
    )

    cases.append(
        HamiltonianBenchmarkCase(
            name="spin_one_xy_chain_L8_open",
            model=SpinOneXYChainModel(
                length=8,
                boundary_condition="open",
                j_xy=1.0,
            ),
            recommended_builder="optimized",
            builders=("sparse", "optimized"),
        )
    )

    cases.append(
        HamiltonianBenchmarkCase(
            name="toric_code_2x2_pbc",
            model=ToricCodeModel(
                lx=2,
                ly=2,
                boundary_condition="periodic",
            ),
            recommended_builder="sparse",
            builders=("sparse",),
        )
    )

    cases.append(
        HamiltonianBenchmarkCase(
            name="square_qlm_4x4_pbc_w00",
            model=SquareQLMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
            ),
            recommended_builder="bitmask",
            builders=("sparse", "bitmask"),
        )
    )

    cases.append(
        HamiltonianBenchmarkCase(
            name="square_qdm_4x4_pbc_w00",
            model=SquareQDMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
            ),
            recommended_builder="bitmask",
            builders=("sparse", "bitmask"),
        )
    )

    return cases


def selected_builders(
    case: HamiltonianBenchmarkCase,
    builder_selection: BuilderSelection,
) -> tuple[str, ...]:
    if builder_selection == "recommended":
        return (case.recommended_builder,)

    if builder_selection == "all":
        return case.builders

    if builder_selection not in case.builders:
        return ()

    return (builder_selection,)


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
    recommended_builder: str,
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
        recommended_builder=recommended_builder,
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


def print_case_list(cases: list[HamiltonianBenchmarkCase]) -> None:
    headers = ["name", "model", "recommended", "builders"]
    rows = [
        [
            case.name,
            type(case.model).__name__,
            case.recommended_builder,
            ",".join(case.builders),
        ]
        for case in cases
    ]
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_table(results: list[HamiltonianBenchmarkResult]) -> None:
    headers = [
        "name",
        "model",
        "solver",
        "builder",
        "recommended",
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
            "*" if result.builder == result.recommended_builder else "",
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
        default="recommended",
        choices=["recommended", "all", "sparse", "optimized", "bitmask"],
        help=(
            "Builder to benchmark. Use 'recommended' for each case's default fast "
            "path, or 'all' to compare all supported builders per case."
        ),
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
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List benchmark cases and supported builders, then exit.",
    )

    args = parser.parse_args()

    cases = make_benchmark_cases()

    if args.only is not None:
        cases = [case for case in cases if args.only in case.name]

    if args.list_cases:
        print_case_list(cases)
        return

    results: list[HamiltonianBenchmarkResult] = []

    for case in cases:
        builders = selected_builders(case, args.builder)

        if not builders:
            print(
                f"Skipping {case.name}: builder {args.builder!r} is not supported. "
                f"Supported builders are {case.builders}.",
                flush=True,
            )
            continue

        for builder in builders:
            label = f"{case.name} [{builder}]"
            print(f"Running {label} ...", flush=True)

            try:
                result = run_hamiltonian_benchmark(
                    name=case.name,
                    model=case.model,
                    basis_solver=args.basis_solver,
                    builder=builder,
                    recommended_builder=case.recommended_builder,
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
                [_jsonable(asdict(result)) for result in results],
                f,
                indent=2,
                default=str,
            )

        print(f"\nWrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
