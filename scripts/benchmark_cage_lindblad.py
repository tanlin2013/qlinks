#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging import (
    CageClassificationConfig,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
    classify_full_state,
)
from qlinks.caging.open_system import (
    JumpOperatorDesign,
    JumpPlaquettePolicy,
    MonitorPlaquettePolicy,
    MonitorSource,
    ReducedIZMonitorContent,
    ReducedIZMonitorDecomposition,
    build_type1_cage_lindblad_construction,
)
from qlinks.models import HoneycombQDMModel, SquareQDMModel, SquareQLMModel
from qlinks.open_system import RecyclingJumpSource

BuilderName = Literal["sparse", "optimized", "bitmask"]
SearchTypeName = Literal["type1", "type2", "type1_and_type2", "custom"]


@dataclass(frozen=True)
class CageLindbladBenchmarkCase:
    name: str
    model: object
    builder: BuilderName
    search_type: SearchTypeName
    signature: tuple[int, int] | None = None
    record_index: int = 0


@dataclass(frozen=True)
class CageLindbladBenchmarkResult:
    name: str
    model: str
    parameters: dict
    basis_solver: str
    builder: str
    local_term_builder: str
    backend: str
    monitor_source: str
    reduced_iz_monitor_decomposition: str
    reduced_iz_monitor_content: str
    jump_operator_design: str
    monitor_plaquette_policy: str
    jump_plaquette_policy: str
    check_liouvillian: bool
    compute_jump_residuals: bool
    recycling_jump_source: str
    max_recycling_jumps_per_region: int
    n_variables: int
    n_states: int
    selected_signature: tuple[int, int]
    selected_record_index: int
    n_cages: int
    build_seconds: float
    search_seconds: float
    classification_seconds: float
    construction_seconds: float
    total_seconds: float
    search_stage_seconds: dict[str, float]
    construction_stage_seconds: dict[str, float]
    n_nontrivial_zeros: int
    classification_label: str
    region_size: int
    n_monitor_components: int
    n_monitor_plaquettes: int
    n_jump_plaquettes: int
    n_jumps: int
    n_component_jumps: int
    n_global_jump_terms: int
    n_recycling_jumps: int
    monitor_residual: float
    max_jump_residual: float
    liouvillian_residual: float | None


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


def _format_residual(value: float | None) -> str:
    if value is None:
        return ""

    return f"{value:.3e}"


def _stage_seconds(
    stages: dict[str, float],
    stage_name: str,
) -> float:
    return float(stages.get(stage_name, 0.0))


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def format_markdown_report(results: list[CageLindbladBenchmarkResult]) -> str:
    if len(results) == 0:
        return "\n".join(
            [
                "## Cage-Lindblad construction benchmark",
                "",
                "No successful benchmark results.",
                "",
            ]
        )

    headers = [
        "case",
        "builder",
        "local_builder",
        "monitor",
        "decomp",
        "content",
        "jump_design",
        "jump_resid",
        "recycling",
        "recycling_jumps",
        "n_states",
        "signature",
        "region",
        "components",
        "jumps",
        "build_s",
        "search_s",
        "cand1_s",
        "solve1_s",
        "ipr_s",
        "dedup_s",
        "classify_s",
        "construct_s",
        "monitor_s",
        "jumps_s",
        "recycle_s",
        "diag_s",
        "Lcheck_s",
        "total_s",
        "monitor_resid",
        "jump_resid_norm",
        "L_resid",
    ]
    rows = [
        [
            result.name,
            result.builder,
            result.local_term_builder,
            result.monitor_source,
            result.reduced_iz_monitor_decomposition,
            result.reduced_iz_monitor_content,
            result.jump_operator_design,
            result.compute_jump_residuals,
            result.recycling_jump_source,
            result.n_recycling_jumps,
            result.n_states,
            result.selected_signature,
            result.region_size,
            result.n_monitor_components,
            result.n_jumps,
            _format_seconds(result.build_seconds),
            _format_seconds(result.search_seconds),
            _format_seconds(_stage_seconds(result.search_stage_seconds, "candidate_build_type1")),
            _format_seconds(_stage_seconds(result.search_stage_seconds, "solve_type1")),
            _format_seconds(_stage_seconds(result.search_stage_seconds, "solver.ipr_localization")),
            _format_seconds(_stage_seconds(result.search_stage_seconds, "rank_deduplication")),
            _format_seconds(result.classification_seconds),
            _format_seconds(result.construction_seconds),
            _format_seconds(_stage_seconds(result.construction_stage_seconds, "monitor_assembly")),
            _format_seconds(_stage_seconds(result.construction_stage_seconds, "jump_assembly")),
            _format_seconds(_stage_seconds(result.construction_stage_seconds, "recycling")),
            _format_seconds(_stage_seconds(result.construction_stage_seconds, "diagnostics")),
            _format_seconds(_stage_seconds(result.construction_stage_seconds, "liouvillian_check")),
            _format_seconds(result.total_seconds),
            _format_residual(result.monitor_residual),
            _format_residual(result.max_jump_residual),
            _format_residual(result.liouvillian_residual),
        ]
        for result in results
    ]

    return "\n".join(
        [
            "## Cage-Lindblad construction benchmark",
            "",
            _markdown_table(headers, rows),
            "",
        ]
    )


def make_benchmark_cases() -> list[CageLindbladBenchmarkCase]:
    """Return modest default Cage-Lindblad benchmark cases."""
    return [
        CageLindbladBenchmarkCase(
            name="square_qdm_4x4_pbc_w00",
            model=SquareQDMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                winding_convention="electric",
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            builder="sparse",
            search_type="type1",
            signature=(0, 4),
        ),
        CageLindbladBenchmarkCase(
            name="square_qlm_4x4_pbc_w00",
            model=SquareQLMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                charges=0,
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            builder="sparse",
            search_type="type1",
            signature=(0, 8),
        ),
        CageLindbladBenchmarkCase(
            name="honeycomb_qdm_4x4_pbc_w-20",
            model=HoneycombQDMModel(
                lx=4,
                ly=4,
                boundary_condition="periodic",
                winding_x=-2,
                winding_y=0,
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            builder="sparse",
            search_type="type1",
            signature=(0, 4),
        ),
    ]


def model_parameters(model: object) -> dict:
    if hasattr(model, "__dataclass_fields__"):
        return asdict(model)

    return {key: value for key, value in vars(model).items() if not key.startswith("_")}


def _select_cage_record(
    search_result: CageSearchResult,
    *,
    signature: tuple[int, int] | None,
    record_index: int,
):
    records = (
        list(search_result[signature].records)
        if signature is not None
        else list(search_result.records)
    )

    if len(records) == 0:
        if signature is None:
            raise ValueError("No cage records were found.")
        raise ValueError(f"No cage records were found for signature {signature}.")

    if record_index < 0 or record_index >= len(records):
        raise IndexError(
            f"record_index={record_index} is out of range for "
            f"{len(records)} selected cage records."
        )

    return records[record_index]


def _full_state_for_record(
    search_result: CageSearchResult,
    record,
) -> np.ndarray:
    if record.full_state is not None:
        return np.asarray(record.full_state, dtype=np.complex128)

    full_state = np.zeros(search_result.hilbert_size, dtype=np.complex128)
    full_state[record.support] = record.local_state
    return full_state


def _signature_from_arg(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    parts = value.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "signature must have the form 'kappa,z', for example '0,6'."
        )

    return int(parts[0]), int(parts[1])


def run_cage_lindblad_benchmark(
    *,
    case: CageLindbladBenchmarkCase,
    basis_solver: str,
    builder: str,
    local_term_builder: str,
    backend: str,
    monitor_source: MonitorSource,
    reduced_iz_monitor_decomposition: ReducedIZMonitorDecomposition,
    reduced_iz_monitor_content: ReducedIZMonitorContent,
    monitor_plaquette_policy: MonitorPlaquettePolicy,
    jump_plaquette_policy: JumpPlaquettePolicy,
    jump_operator_design: JumpOperatorDesign,
    check_liouvillian: bool,
    compute_jump_residuals: bool,
    recycling_jump_source: RecyclingJumpSource,
    max_recycling_jumps_per_region: int,
    recycling_rdm_tolerance: float,
    recycling_dark_tolerance: float,
    recycling_inflow_tolerance: float,
    recycling_prefer_sparse: bool,
    recycling_two_pattern_tolerance: float,
    residual_tolerance: float,
    classification_sector_policy: str,
    signature: tuple[int, int] | None,
    record_index: int,
) -> CageLindbladBenchmarkResult:
    build_result, build_seconds = _time_call(
        lambda: case.model.build(
            basis_solver=basis_solver,
            builder=builder,
            backend=backend,
            on_missing="raise",
        )
    )

    search_config = CageSearchConfig(
        search_type=case.search_type,
        tolerance=residual_tolerance,
        degenerate_basis_strategy="ipr",
        ipr_n_restarts=256,
        ipr_candidate_count=128,
        ipr_random_seed=1234,
        store_full_states=True,
    )
    searcher = CageSearcher.from_model_build_result(build_result, config=search_config)
    search_result, search_seconds = _time_call(searcher.run)

    selected_signature = signature if signature is not None else case.signature
    record = _select_cage_record(
        search_result,
        signature=selected_signature,
        record_index=record_index,
    )
    full_state = _full_state_for_record(search_result, record)
    basis_configs = basis_configs_from_build_result(build_result)

    classification_config = CageClassificationConfig(
        amplitude_tolerance=residual_tolerance,
        action_tolerance=max(residual_tolerance, 1e-9),
        sector_policy=classification_sector_policy,  # type: ignore[arg-type]
    )
    classification_report, classification_seconds = _time_call(
        lambda: classify_full_state(
            full_state,
            kinetic_matrix=build_result.kinetic,
            basis_configs=basis_configs,
            config=classification_config,
            metadata={
                "signature": record.signature,
                "record_index": record_index,
            },
        )
    )

    construction_stage_seconds: dict[str, float] = {}
    construction, construction_seconds = _time_call(
        lambda: build_type1_cage_lindblad_construction(
            model=case.model,
            build_result=build_result,
            cage_state=full_state,
            classification_report=classification_report,
            z_value=record.signature[1],
            builder=local_term_builder,
            backend=backend,
            monitor_source=monitor_source,
            reduced_iz_monitor_decomposition=reduced_iz_monitor_decomposition,
            reduced_iz_monitor_content=reduced_iz_monitor_content,
            monitor_plaquette_policy=monitor_plaquette_policy,
            jump_plaquette_policy=jump_plaquette_policy,
            jump_operator_design=jump_operator_design,
            check_liouvillian=check_liouvillian,
            compute_jump_residuals=compute_jump_residuals,
            recycling_jump_source=recycling_jump_source,
            max_recycling_jumps_per_region=max_recycling_jumps_per_region,
            recycling_rdm_tolerance=recycling_rdm_tolerance,
            recycling_dark_tolerance=recycling_dark_tolerance,
            recycling_inflow_tolerance=recycling_inflow_tolerance,
            recycling_prefer_sparse=recycling_prefer_sparse,
            recycling_two_pattern_tolerance=recycling_two_pattern_tolerance,
            timing_collector=construction_stage_seconds,
            residual_tolerance=residual_tolerance,
        )
    )
    summary = construction.to_summary_dict()

    return CageLindbladBenchmarkResult(
        name=case.name,
        model=type(case.model).__name__,
        parameters=model_parameters(case.model),
        basis_solver=basis_solver,
        builder=builder,
        local_term_builder=local_term_builder,
        backend=backend,
        monitor_source=monitor_source,
        reduced_iz_monitor_decomposition=reduced_iz_monitor_decomposition,
        reduced_iz_monitor_content=reduced_iz_monitor_content,
        jump_operator_design=jump_operator_design,
        monitor_plaquette_policy=monitor_plaquette_policy,
        jump_plaquette_policy=jump_plaquette_policy,
        check_liouvillian=check_liouvillian,
        compute_jump_residuals=compute_jump_residuals,
        recycling_jump_source=recycling_jump_source,
        max_recycling_jumps_per_region=max_recycling_jumps_per_region,
        n_variables=case.model.layout.n_variables,
        n_states=build_result.basis.n_states,
        selected_signature=record.signature,
        selected_record_index=record_index,
        n_cages=len(search_result),
        build_seconds=build_seconds,
        search_seconds=search_seconds,
        classification_seconds=classification_seconds,
        construction_seconds=construction_seconds,
        total_seconds=build_seconds
        + search_seconds
        + classification_seconds
        + construction_seconds,
        search_stage_seconds=dict(search_result.search_stage_seconds),
        construction_stage_seconds=dict(construction_stage_seconds),
        n_nontrivial_zeros=classification_report.n_nontrivial_zeros,
        classification_label=str(classification_report.label),
        region_size=int(summary["region_size"]),
        n_monitor_components=int(summary["n_monitor_components"]),
        n_monitor_plaquettes=int(summary["n_monitor_plaquettes"]),
        n_jump_plaquettes=int(summary["n_jump_plaquettes"]),
        n_jumps=int(summary["n_jumps"]),
        n_component_jumps=int(summary["n_component_jumps"]),
        n_global_jump_terms=int(summary["n_global_jump_terms"]),
        n_recycling_jumps=int(summary["n_recycling_jumps"]),
        monitor_residual=float(summary["monitor_residual"]),
        max_jump_residual=float(summary["max_jump_residual"]),
        liouvillian_residual=(
            None
            if summary["liouvillian_residual"] is None
            else float(summary["liouvillian_residual"])
        ),
    )


def print_table(results: list[CageLindbladBenchmarkResult]) -> None:
    if len(results) == 0:
        print("No successful Cage-Lindblad benchmark results.")
        return

    headers = [
        "name",
        "builder",
        "local_builder",
        "monitor",
        "decomp",
        "recycling",
        "recycle_j",
        "n_states",
        "signature",
        "region",
        "components",
        "jumps",
        "build_s",
        "search_s",
        "cand1_s",
        "solve1_s",
        "ipr_s",
        "dedup_s",
        "classify_s",
        "construct_s",
        "monitor_s",
        "jumps_s",
        "recycle_s",
        "diag_s",
        "total_s",
    ]
    rows = [
        [
            result.name,
            result.builder,
            result.local_term_builder,
            result.monitor_source,
            result.reduced_iz_monitor_decomposition,
            result.recycling_jump_source,
            str(result.n_recycling_jumps),
            str(result.n_states),
            str(result.selected_signature),
            str(result.region_size),
            str(result.n_monitor_components),
            str(result.n_jumps),
            f"{result.build_seconds:.6f}",
            f"{result.search_seconds:.6f}",
            f"{_stage_seconds(result.search_stage_seconds, 'candidate_build_type1'):.6f}",
            f"{_stage_seconds(result.search_stage_seconds, 'solve_type1'):.6f}",
            f"{_stage_seconds(result.search_stage_seconds, 'solver.ipr_localization'):.6f}",
            f"{_stage_seconds(result.search_stage_seconds, 'rank_deduplication'):.6f}",
            f"{result.classification_seconds:.6f}",
            f"{result.construction_seconds:.6f}",
            f"{_stage_seconds(result.construction_stage_seconds, 'monitor_assembly'):.6f}",
            f"{_stage_seconds(result.construction_stage_seconds, 'jump_assembly'):.6f}",
            f"{_stage_seconds(result.construction_stage_seconds, 'recycling'):.6f}",
            f"{_stage_seconds(result.construction_stage_seconds, 'diagnostics'):.6f}",
            f"{result.total_seconds:.6f}",
        ]
        for result in results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CageLindbladConstruction assembly stages."
    )
    parser.add_argument("--basis-solver", default="dfs", choices=["dfs", "brute_force", "cpsat"])
    parser.add_argument(
        "--builder",
        default="auto",
        choices=["auto", "sparse", "optimized", "bitmask"],
        help="Model build builder. 'auto' uses the default for each case.",
    )
    parser.add_argument(
        "--local-term-builder",
        default="sparse",
        choices=["sparse", "optimized", "bitmask"],
        help=(
            "Builder requested for local term matrices. Encoded bitmask build "
            "results are promoted to bitmask internally."
        ),
    )
    parser.add_argument("--backend", default="scipy")
    parser.add_argument(
        "--monitor-source",
        default="reduced_iz_operators",
        choices=["local_hamiltonian_terms", "reduced_iz_operators"],
    )
    parser.add_argument(
        "--reduced-iz-monitor-decomposition",
        default="exact_support",
        choices=["single_sum", "exact_support", "connected_support"],
    )
    parser.add_argument(
        "--reduced-iz-monitor-content",
        default="offdiagonal_only",
        choices=["offdiagonal_only", "offdiagonal_plus_potential"],
    )
    parser.add_argument(
        "--monitor-plaquette-policy",
        default="strict_inside",
        choices=["strict_inside", "touching"],
    )
    parser.add_argument(
        "--jump-plaquette-policy",
        default="outside_or_crossing",
        choices=["disjoint_outside", "crossing", "outside_or_crossing", "not_strictly_inside"],
    )
    parser.add_argument(
        "--jump-operator-design",
        default="kinetic_outside_monitor_inside",
        choices=[
            "kinetic_times_monitor",
            "kinetic_outside_monitor_inside",
            "hamiltonian_outside_monitor_inside",
        ],
    )
    parser.add_argument(
        "--classification-sector-policy",
        default="infer_support_component",
        choices=["raise_if_disconnected", "infer_support_component", "ignore"],
    )
    parser.add_argument(
        "--check-liouvillian",
        action="store_true",
        help="Also time the expensive final Liouvillian dark-state check.",
    )
    parser.add_argument(
        "--skip-jump-residuals",
        action="store_true",
        help=(
            "Skip computing ||J psi|| diagnostics after constructing jumps. "
            "This is useful for separating jump materialization from residual checks."
        ),
    )
    parser.add_argument(
        "--recycling-jump-source",
        default="none",
        choices=["none", "local_rdm_rank_one", "local_rdm_two_pattern"],
        help="Build local reduced-density-matrix recycling jumps during construction.",
    )
    parser.add_argument(
        "--max-recycling-jumps-per-region",
        type=int,
        default=1,
        help="Maximum selected recycling jumps per local region.",
    )
    parser.add_argument("--recycling-rdm-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--recycling-dark-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--recycling-inflow-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--no-recycling-prefer-sparse",
        action="store_true",
        help="Do not prefer sparser local recycling jumps when selecting candidates.",
    )
    parser.add_argument("--recycling-two-pattern-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--signature",
        type=_signature_from_arg,
        default=None,
        help="Optional signature selector 'kappa,z', for example '0,6'.",
    )
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only cases whose name contains this substring.",
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

    args = parser.parse_args()
    results: list[CageLindbladBenchmarkResult] = []

    for case in make_benchmark_cases():
        if args.only is not None and args.only not in case.name:
            continue

        builder = case.builder if args.builder == "auto" else args.builder

        print(f"Running {case.name} ...", flush=True)
        try:
            result = run_cage_lindblad_benchmark(
                case=case,
                basis_solver=args.basis_solver,
                builder=builder,
                local_term_builder=args.local_term_builder,
                backend=args.backend,
                monitor_source=args.monitor_source,
                reduced_iz_monitor_decomposition=args.reduced_iz_monitor_decomposition,
                reduced_iz_monitor_content=args.reduced_iz_monitor_content,
                monitor_plaquette_policy=args.monitor_plaquette_policy,
                jump_plaquette_policy=args.jump_plaquette_policy,
                jump_operator_design=args.jump_operator_design,
                check_liouvillian=args.check_liouvillian,
                compute_jump_residuals=not args.skip_jump_residuals,
                recycling_jump_source=args.recycling_jump_source,
                max_recycling_jumps_per_region=args.max_recycling_jumps_per_region,
                recycling_rdm_tolerance=args.recycling_rdm_tolerance,
                recycling_dark_tolerance=args.recycling_dark_tolerance,
                recycling_inflow_tolerance=args.recycling_inflow_tolerance,
                recycling_prefer_sparse=not args.no_recycling_prefer_sparse,
                recycling_two_pattern_tolerance=args.recycling_two_pattern_tolerance,
                residual_tolerance=args.residual_tolerance,
                classification_sector_policy=args.classification_sector_policy,
                signature=args.signature,
                record_index=args.record_index,
            )
        except (IndexError, NotImplementedError, ValueError) as exc:
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

    if args.markdown is not None:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(results))

        print(f"\nWrote Markdown results to {args.markdown}")


if __name__ == "__main__":
    main()
