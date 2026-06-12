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
class BasisBenchmarkResult:
    name: str
    model: str
    parameters: dict
    solver: str
    sort: bool
    n_variables: int
    n_states: int
    elapsed_seconds: float


def _time_call(func: Callable):
    gc.collect()
    start = time.perf_counter()
    out = func()
    elapsed = time.perf_counter() - start
    return out, elapsed


def _json_key(key: object) -> str:
    return str(key)


def _jsonable(value: object) -> object:
    """Recursively convert benchmark metadata to JSON-serializable values."""
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


def _markdown_escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_seconds(value: float | None) -> str:
    if value is None:
        return ""

    return f"{value:.6f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def format_markdown_report(results: list[BasisBenchmarkResult]) -> str:
    headers = [
        "case",
        "model",
        "solver",
        "sort",
        "n_variables",
        "n_states",
        "basis_time_s",
    ]
    rows = [
        [
            result.name,
            result.model,
            result.solver,
            result.sort,
            result.n_variables,
            result.n_states,
            _format_seconds(result.elapsed_seconds),
        ]
        for result in results
    ]

    return "\n".join(
        [
            "## Basis-generation benchmark",
            "",
            _markdown_table(headers, rows),
            "",
        ]
    )


def make_benchmark_cases() -> list[tuple[str, object]]:
    """
    Keep these cases small enough to run quickly on a laptop.

    Add larger cases manually when profiling locally.
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
    """
    Best-effort dataclass parameter dump.
    """
    if hasattr(model, "__dataclass_fields__"):
        return asdict(model)

    return {key: value for key, value in vars(model).items() if not key.startswith("_")}


def run_basis_benchmark(
    *,
    name: str,
    model: object,
    solver: str,
    sort: bool,
) -> BasisBenchmarkResult:
    basis, elapsed = _time_call(
        lambda: model.build_basis(
            solver=solver,
            sort=sort,
        )
    )

    return BasisBenchmarkResult(
        name=name,
        model=type(model).__name__,
        parameters=model_parameters(model),
        solver=solver,
        sort=sort,
        n_variables=model.layout.n_variables,
        n_states=basis.n_states,
        elapsed_seconds=elapsed,
    )


def print_table(results: list[BasisBenchmarkResult]) -> None:
    headers = [
        "name",
        "model",
        "solver",
        "sort",
        "n_variables",
        "n_states",
        "basis_time_s",
    ]

    rows = [
        [
            result.name,
            result.model,
            result.solver,
            str(result.sort),
            str(result.n_variables),
            str(result.n_states),
            f"{result.elapsed_seconds:.6f}",
        ]
        for result in results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark basis generation.")
    parser.add_argument(
        "--solver",
        default="dfs",
        choices=["dfs", "brute_force", "cpsat"],
        help="Basis solver to request. Unconstrained models may use the full-basis fast path.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable final basis sorting.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write JSON benchmark results.",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Optional path to write a compact GitHub-ready Markdown report.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only cases whose name contains this substring.",
    )

    args = parser.parse_args()

    results: list[BasisBenchmarkResult] = []

    for name, model in make_benchmark_cases():
        if args.only is not None and args.only not in name:
            continue

        print(f"Running {name} ...", flush=True)

        result = run_basis_benchmark(
            name=name,
            model=model,
            solver=args.solver,
            sort=not args.no_sort,
        )

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

    if args.markdown is not None:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(results))

        print(f"\nWrote Markdown results to {args.markdown}")


if __name__ == "__main__":
    main()
