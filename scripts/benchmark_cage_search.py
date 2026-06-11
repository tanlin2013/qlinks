#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
)
from qlinks.models.qdm import SquareQDMModel
from qlinks.models.qlm import SquareQLMModel

BuilderName = Literal["sparse", "optimized", "bitmask"]
SearchTypeName = Literal["type1", "type2", "type1_and_type2", "custom"]


@dataclass(frozen=True)
class CageBenchmarkCase:
    name: str
    model: object
    builder: BuilderName
    search_type: SearchTypeName


@dataclass(frozen=True)
class CageSearchBenchmarkResult:
    name: str
    model: str
    parameters: dict
    basis_solver: str
    builder: str
    backend: str
    sort_basis: bool
    search_type: str
    degenerate_basis_strategy: str
    n_variables: int
    n_states: int
    h_nnz: int
    kinetic_nnz: int | None
    potential_nnz: int | None
    build_total_seconds: float
    basis_seconds: float | None
    matrix_seconds: float | None
    type1_candidate_seconds: float
    type2_candidate_seconds: float
    type1_solve_seconds: float
    type2_solve_seconds: float
    deduplicate_seconds: float
    search_total_seconds: float
    total_seconds: float
    n_type1_candidates: int
    n_type2_candidates: int
    n_type1_raw_records: int
    n_type2_raw_records: int
    n_records: int
    counts_by_signature: dict[str, int]


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


def make_benchmark_cases() -> list[CageBenchmarkCase]:
    """Return modest default cage-search benchmark cases.

    The 4x4 square QDM/QLM cases are intentionally included because they are
    the smallest commonly used systems with known nontrivial cage content.
    Larger cases should be run manually through ``--only`` after adding local
    cases here.
    """
    return [
        CageBenchmarkCase(
            name="square_qdm_4x4_pbc_w00",
            model=SquareQDMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                winding_convention="electric",
                coup_kin=1.0,
                coup_pot=1.0,
            ),
            builder="sparse",
            search_type="type1",
        ),
        CageBenchmarkCase(
            name="square_qlm_4x4_pbc_w00",
            model=SquareQLMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                charges=0,
                coup_kin=1.0,
                coup_pot=1.0,
            ),
            builder="bitmask",
            search_type="type1_and_type2",
        ),
    ]


def model_parameters(model: object) -> dict:
    if hasattr(model, "__dataclass_fields__"):
        return asdict(model)

    return {key: value for key, value in vars(model).items() if not key.startswith("_")}


def _counts_by_signature_json(result: CageSearchResult) -> dict[str, int]:
    return {str(signature): int(count) for signature, count in result.counts_by_signature.items()}


def _timed_cage_search(
    searcher: CageSearcher,
) -> tuple[
    CageSearchResult,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
]:
    """Run cage search with coarse stage timings.

    This uses the searcher's internal stage methods because the purpose of this
    script is performance diagnosis rather than public API demonstration.
    """
    type1_enabled, type2_enabled = searcher._enabled_candidate_types()

    type1_candidates = []
    type2_candidates = []
    type1_candidate_seconds = 0.0
    type2_candidate_seconds = 0.0

    if type1_enabled:
        type1_candidates, type1_candidate_seconds = _time_call(searcher._build_type1_candidates)

    if type2_enabled:
        type2_candidates, type2_candidate_seconds = _time_call(searcher._build_type2_candidates)

    type1_records: list[CageRecord] = []
    type2_records: list[CageRecord] = []
    type1_solve_seconds = 0.0
    type2_solve_seconds = 0.0

    if type1_enabled:
        type1_records, type1_solve_seconds = _time_call(
            lambda: searcher._solve_candidates(
                candidates=type1_candidates,
                allowed_kappas=searcher.config.type1_kappas,
            )
        )

    if type2_enabled:
        type2_records, type2_solve_seconds = _time_call(
            lambda: searcher._solve_candidates(
                candidates=type2_candidates,
                allowed_kappas=searcher.config.type2_kappas,
            )
        )

    raw_records = [*type1_records, *type2_records]
    records, deduplicate_seconds = _time_call(
        lambda: searcher._deduplicate_records_by_signature(raw_records)
    )

    result = CageSearchResult(
        records=records,
        hilbert_size=int(searcher.hamiltonian_matrix.shape[0]),
        config=searcher.config,
        type1_candidates=type1_candidates,
        type2_candidates=type2_candidates,
    )

    search_total_seconds = (
        type1_candidate_seconds
        + type2_candidate_seconds
        + type1_solve_seconds
        + type2_solve_seconds
        + deduplicate_seconds
    )

    return (
        result,
        type1_candidate_seconds,
        type2_candidate_seconds,
        type1_solve_seconds,
        type2_solve_seconds,
        deduplicate_seconds,
        search_total_seconds,
        len(type1_candidates),
        len(type2_candidates),
        len(type1_records),
        len(type2_records),
    )


def run_cage_search_benchmark(
    *,
    case: CageBenchmarkCase,
    basis_solver: str,
    builder: str,
    backend: str,
    sort_basis: bool,
    split_basis_timing: bool,
    search_config: CageSearchConfig,
) -> CageSearchBenchmarkResult:
    basis_seconds: float | None = None
    matrix_seconds: float | None = None

    if split_basis_timing:
        basis, basis_seconds = _time_call(
            lambda: case.model.build_basis(
                solver=basis_solver,
                sort=sort_basis,
            )
        )

        build_result, matrix_seconds = _time_call(
            lambda: case.model.build(
                basis=basis,
                basis_solver=basis_solver,
                builder=builder,
                backend=backend,
                sort_basis=sort_basis,
                on_missing="raise",
            )
        )
        build_total_seconds = basis_seconds + matrix_seconds

    else:
        build_result, build_total_seconds = _time_call(
            lambda: case.model.build(
                basis_solver=basis_solver,
                builder=builder,
                backend=backend,
                sort_basis=sort_basis,
                on_missing="raise",
            )
        )

    searcher = CageSearcher.from_model_build_result(
        build_result,
        config=search_config,
    )

    (
        search_result,
        type1_candidate_seconds,
        type2_candidate_seconds,
        type1_solve_seconds,
        type2_solve_seconds,
        deduplicate_seconds,
        search_total_seconds,
        n_type1_candidates,
        n_type2_candidates,
        n_type1_raw_records,
        n_type2_raw_records,
    ) = _timed_cage_search(searcher)

    return CageSearchBenchmarkResult(
        name=case.name,
        model=type(case.model).__name__,
        parameters=model_parameters(case.model),
        basis_solver=basis_solver,
        builder=builder,
        backend=backend,
        sort_basis=sort_basis,
        search_type=search_config.search_type,
        degenerate_basis_strategy=search_config.degenerate_basis_strategy,
        n_variables=case.model.layout.n_variables,
        n_states=build_result.basis.n_states,
        h_nnz=int(build_result.hamiltonian.nnz),
        kinetic_nnz=_nnz(build_result.kinetic),
        potential_nnz=_nnz(build_result.potential),
        build_total_seconds=build_total_seconds,
        basis_seconds=basis_seconds,
        matrix_seconds=matrix_seconds,
        type1_candidate_seconds=type1_candidate_seconds,
        type2_candidate_seconds=type2_candidate_seconds,
        type1_solve_seconds=type1_solve_seconds,
        type2_solve_seconds=type2_solve_seconds,
        deduplicate_seconds=deduplicate_seconds,
        search_total_seconds=search_total_seconds,
        total_seconds=build_total_seconds + search_total_seconds,
        n_type1_candidates=n_type1_candidates,
        n_type2_candidates=n_type2_candidates,
        n_type1_raw_records=n_type1_raw_records,
        n_type2_raw_records=n_type2_raw_records,
        n_records=len(search_result),
        counts_by_signature=_counts_by_signature_json(search_result),
    )


def print_table(results: list[CageSearchBenchmarkResult]) -> None:
    headers = [
        "name",
        "builder",
        "n_states",
        "H.nnz",
        "c1",
        "c2",
        "raw1",
        "raw2",
        "records",
        "build_s",
        "cand1_s",
        "cand2_s",
        "solve1_s",
        "solve2_s",
        "dedup_s",
        "search_s",
        "total_s",
    ]

    rows = [
        [
            result.name,
            result.builder,
            str(result.n_states),
            str(result.h_nnz),
            str(result.n_type1_candidates),
            str(result.n_type2_candidates),
            str(result.n_type1_raw_records),
            str(result.n_type2_raw_records),
            str(result.n_records),
            f"{result.build_total_seconds:.6f}",
            f"{result.type1_candidate_seconds:.6f}",
            f"{result.type2_candidate_seconds:.6f}",
            f"{result.type1_solve_seconds:.6f}",
            f"{result.type2_solve_seconds:.6f}",
            f"{result.deduplicate_seconds:.6f}",
            f"{result.search_total_seconds:.6f}",
            f"{result.total_seconds:.6f}",
        ]
        for result in results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _search_config_from_args(
    args: argparse.Namespace,
    *,
    search_type: SearchTypeName,
) -> CageSearchConfig:
    return CageSearchConfig(
        search_type=search_type,
        tolerance=args.tolerance,
        min_component_size=args.min_component_size,
        validate_full_residual=not args.no_validate_full_residual,
        degenerate_basis_strategy=args.degenerate_basis_strategy,
        ipr_n_restarts=args.ipr_n_restarts,
        ipr_max_iter=args.ipr_max_iter,
        ipr_step_size=args.ipr_step_size,
        ipr_candidate_count=args.ipr_candidate_count,
        ipr_random_seed=args.ipr_random_seed,
        deduplicate_by_rank=not args.no_deduplicate_by_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cage-search stages.")
    parser.add_argument(
        "--basis-solver",
        default="dfs",
        choices=["dfs", "brute_force", "cpsat"],
    )
    parser.add_argument(
        "--builder",
        default="auto",
        choices=["auto", "sparse", "optimized", "bitmask"],
        help="Hamiltonian builder. 'auto' uses the default builder for each case.",
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
        "--search-type",
        default="auto",
        choices=["auto", "type1", "type2", "type1_and_type2", "custom"],
        help="Cage search type. 'auto' uses the default search type for each case.",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--min-component-size", type=int, default=2)
    parser.add_argument(
        "--no-validate-full-residual",
        action="store_true",
        help="Disable full residual validation in the cage solver.",
    )
    parser.add_argument(
        "--degenerate-basis-strategy",
        default="none",
        choices=["none", "ipr"],
    )
    parser.add_argument("--ipr-n-restarts", type=int, default=128)
    parser.add_argument("--ipr-max-iter", type=int, default=1000)
    parser.add_argument("--ipr-step-size", type=float, default=0.1)
    parser.add_argument("--ipr-candidate-count", type=int, default=64)
    parser.add_argument("--ipr-random-seed", type=int, default=None)
    parser.add_argument(
        "--no-deduplicate-by-rank",
        action="store_true",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write JSON benchmark results.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only cases whose name contains this substring.",
    )

    args = parser.parse_args()

    results: list[CageSearchBenchmarkResult] = []

    for case in make_benchmark_cases():
        if args.only is not None and args.only not in case.name:
            continue

        builder = case.builder if args.builder == "auto" else args.builder
        search_type = case.search_type if args.search_type == "auto" else args.search_type
        search_config = _search_config_from_args(args, search_type=search_type)

        print(f"Running {case.name} ...", flush=True)

        try:
            result = run_cage_search_benchmark(
                case=case,
                basis_solver=args.basis_solver,
                builder=builder,
                backend=args.backend,
                sort_basis=not args.no_sort_basis,
                split_basis_timing=args.split_basis_timing,
                search_config=search_config,
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
