#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.open_system import (
    LindbladEvolutionOptions,
    LindbladProblem,
    McwfOptions,
    prepare_dense_lindblad_operators,
    prepare_sparse_lindblad_operators,
    pure_density_matrix,
    run_quantum_jump_trajectory,
    sample_lindblad_mcwf,
    solve_lindblad,
)

OpenSystemOperation = Literal[
    "all",
    "prepare_dense",
    "prepare_sparse",
    "liouvillian",
    "rk4_matrix",
    "rk4_liouville",
    "krylov",
    "single_trajectory",
    "mcwf",
]

_OPERATION_ORDER: tuple[str, ...] = (
    "prepare_dense",
    "prepare_sparse",
    "liouvillian",
    "rk4_matrix",
    "rk4_liouville",
    "krylov",
    "single_trajectory",
    "mcwf",
)


@dataclass(frozen=True)
class OpenSystemBenchmarkCase:
    name: str
    hamiltonian: object
    jumps: tuple[object, ...]
    state_initial: np.ndarray
    parameters: dict

    @property
    def dim(self) -> int:
        return int(self.hamiltonian.shape[0])

    @property
    def density_matrix_initial(self) -> np.ndarray:
        return pure_density_matrix(self.state_initial)


@dataclass(frozen=True)
class OpenSystemBenchmarkResult:
    name: str
    operation: str
    backend: str
    dim: int
    n_jumps: int
    n_times: int
    n_trajectories: int | None
    liouvillian_shape: tuple[int, int] | None
    liouvillian_nnz: int | None
    elapsed_seconds: float
    parameters: dict
    details: dict
    stage_seconds: dict[str, float] | None = None


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


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return ""

    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _details_text(details: dict) -> str:
    if not details:
        return ""

    return ", ".join(f"{key}={value}" for key, value in details.items())


def _pauli_and_ladder_operators() -> dict[str, np.ndarray]:
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sigma_plus = sigma_minus.conj().T
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)

    return {
        "sigma_minus": sigma_minus,
        "sigma_plus": sigma_plus,
        "sigma_x": sigma_x,
        "sigma_z": sigma_z,
        "identity": identity,
        "ket0": ket0,
        "ket1": ket1,
    }


def _select_cage_record(search_result, *, signature: tuple[int, int], record_index: int):
    records = list(search_result[signature].records)
    if not records:
        raise ValueError(f"No cage records were found for signature {signature}.")

    if record_index < 0 or record_index >= len(records):
        raise IndexError(
            f"record_index={record_index} is out of range for "
            f"{len(records)} selected cage records."
        )

    return records[record_index]


def _full_state_for_cage_record(search_result, record) -> np.ndarray:
    if record.full_state is not None:
        return np.asarray(record.full_state, dtype=np.complex128)

    full_state = np.zeros(search_result.hilbert_size, dtype=np.complex128)
    full_state[record.support] = record.local_state
    return full_state


def _make_square_qdm_cage_lindblad_mcwf_case(
    *,
    check_liouvillian: bool,
    ipr_candidate_count: int,
    ipr_max_iter: int,
    ipr_batch_size: int,
    ipr_rank_completion_patience: int | None,
    record_index: int,
) -> OpenSystemBenchmarkCase:
    """Build the square-QDM Cage-Lindblad problem used for real MCWF timing."""
    from qlinks.basis import basis_configs_from_build_result
    from qlinks.caging import (
        CageClassificationConfig,
        CageSearchConfig,
        CageSearcher,
        classify_full_state,
    )
    from qlinks.caging.open_system import build_type1_cage_lindblad_construction
    from qlinks.models import SquareQDMModel

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=-1.0,
        coup_pot=1.0,
    )
    signature = (0, 4)

    build_result, build_seconds = _time_call(
        lambda: model.build(
            basis_solver="dfs",
            builder="sparse",
            backend="scipy",
            on_missing="raise",
        )
    )

    search_config = CageSearchConfig(
        search_type="type1",
        tolerance=1.0e-10,
        degenerate_basis_strategy="ipr",
        ipr_n_restarts=256,
        ipr_max_iter=ipr_max_iter,
        ipr_candidate_count=ipr_candidate_count,
        ipr_rank_completion_patience=ipr_rank_completion_patience,
        ipr_batch_size=ipr_batch_size,
        ipr_random_seed=1234,
        store_full_states=True,
    )
    searcher = CageSearcher.from_model_build_result(build_result, config=search_config)
    search_result, search_seconds = _time_call(searcher.run)

    record = _select_cage_record(
        search_result,
        signature=signature,
        record_index=record_index,
    )
    state_vector = _full_state_for_cage_record(search_result, record)
    basis_configs = basis_configs_from_build_result(build_result)

    classification_config = CageClassificationConfig(
        amplitude_tolerance=1.0e-10,
        action_tolerance=1.0e-9,
        sector_policy="infer_support_component",
    )
    report, classification_seconds = _time_call(
        lambda: classify_full_state(
            state_vector,
            kinetic_matrix=build_result.kinetic,
            basis_configs=basis_configs,
            config=classification_config,
            metadata={
                "signature": record.signature,
                "record_index": record_index,
                "benchmark_case": "square_qdm_cage_lindblad_mcwf",
            },
        )
    )

    construction_stage_seconds: dict[str, float] = {}
    construction, construction_seconds = _time_call(
        lambda: build_type1_cage_lindblad_construction(
            model=model,
            build_result=build_result,
            cage_state=state_vector,
            classification_report=report,
            z_value=record.signature[1],
            builder="sparse",
            backend="scipy",
            monitor_source="reduced_iz_operators",
            reduced_iz_monitor_content="offdiagonal_only",
            reduced_iz_monitor_decomposition="exact_support",
            jump_operator_design="kinetic_outside_monitor_inside",
            jump_plaquette_policy="outside_or_crossing",
            recycling_jump_source="local_rdm_two_pattern",
            max_recycling_jumps_per_region=1,
            check_liouvillian=check_liouvillian,
            timing_collector=construction_stage_seconds,
        )
    )
    summary = construction.to_summary_dict()

    return OpenSystemBenchmarkCase(
        name="square_qdm_cage_lindblad_mcwf",
        hamiltonian=build_result.hamiltonian,
        jumps=construction.jumps,
        state_initial=state_vector,
        parameters={
            "model": type(model).__name__,
            "lx": 4,
            "ly": 4,
            "boundary_condition": "periodic",
            "winding_x": 0,
            "winding_y": 0,
            "winding_convention": "electric",
            "signature": record.signature,
            "record_index": record_index,
            "n_states": build_result.basis.n_states,
            "n_jumps": construction.n_jumps,
            "n_component_jumps": construction.n_component_jumps,
            "n_global_jump_terms": construction.n_global_jump_terms,
            "n_recycling_jumps": construction.n_recycling_jumps,
            "region_size": summary["region_size"],
            "monitor_residual": summary["monitor_residual"],
            "max_jump_residual": summary["max_jump_residual"],
            "liouvillian_residual": summary["liouvillian_residual"],
            "build_seconds": build_seconds,
            "search_seconds": search_seconds,
            "search_stage_seconds": dict(search_result.search_stage_seconds),
            "classification_seconds": classification_seconds,
            "construction_seconds": construction_seconds,
            "construction_stage_seconds": dict(construction_stage_seconds),
            "check_liouvillian": check_liouvillian,
            "recycling_jump_source": "local_rdm_two_pattern",
            "ipr_candidate_count": ipr_candidate_count,
            "ipr_max_iter": ipr_max_iter,
            "ipr_batch_size": ipr_batch_size,
            "ipr_rank_completion_patience": ipr_rank_completion_patience,
        },
    )


def make_benchmark_cases(
    *,
    include_sparse_many_jump_case: bool = False,
    include_cage_lindblad_case: bool = False,
    cage_lindblad_check_liouvillian: bool = True,
    cage_lindblad_record_index: int = 0,
    cage_lindblad_ipr_candidate_count: int = 128,
    cage_lindblad_ipr_max_iter: int = 1000,
    cage_lindblad_ipr_batch_size: int = 32,
    cage_lindblad_ipr_rank_completion_patience: int | None = 0,
) -> list[OpenSystemBenchmarkCase]:
    ops = _pauli_and_ladder_operators()
    sigma_minus = ops["sigma_minus"]
    sigma_x = ops["sigma_x"]
    sigma_z = ops["sigma_z"]
    identity = ops["identity"]
    ket1 = ops["ket1"]

    qubit_decay_rate = 0.4
    qubit_drive = 0.5
    qubit_hamiltonian = qubit_drive * sigma_x
    qubit_jump = np.sqrt(qubit_decay_rate) * sigma_minus

    two_qubit_decay_rate = 0.25
    two_qubit_drive = 0.4
    two_qubit_coupling = 0.15
    two_qubit_hamiltonian = two_qubit_drive * (
        np.kron(sigma_x, identity) + np.kron(identity, sigma_x)
    ) + two_qubit_coupling * np.kron(sigma_z, sigma_z)
    two_qubit_jumps = (
        np.sqrt(two_qubit_decay_rate) * np.kron(sigma_minus, identity),
        np.sqrt(two_qubit_decay_rate) * np.kron(identity, sigma_minus),
    )
    two_qubit_state = np.kron(ket1, ket1)

    cases = [
        OpenSystemBenchmarkCase(
            name="qubit_amplitude_damping",
            hamiltonian=qubit_hamiltonian,
            jumps=(qubit_jump,),
            state_initial=ket1,
            parameters={
                "dim": 2,
                "drive": qubit_drive,
                "decay_rate": qubit_decay_rate,
            },
        ),
        OpenSystemBenchmarkCase(
            name="two_qubit_driven_decay",
            hamiltonian=two_qubit_hamiltonian,
            jumps=two_qubit_jumps,
            state_initial=two_qubit_state,
            parameters={
                "dim": 4,
                "drive": two_qubit_drive,
                "coupling": two_qubit_coupling,
                "decay_rate": two_qubit_decay_rate,
                "n_jumps": len(two_qubit_jumps),
            },
        ),
    ]

    if include_sparse_many_jump_case:
        dim = 512
        n_jumps = 128
        entries_per_jump = 2
        rng = np.random.default_rng(2026)
        sparse_jumps = []
        for _ in range(n_jumps):
            rows = rng.choice(dim, size=entries_per_jump, replace=False)
            columns = rng.choice(dim, size=entries_per_jump, replace=False)
            values = 0.01 * (
                rng.normal(size=entries_per_jump) + 1j * rng.normal(size=entries_per_jump)
            )
            sparse_jumps.append(
                scipy_sparse.csr_array(
                    (values, (rows, columns)),
                    shape=(dim, dim),
                    dtype=np.complex128,
                )
            )

        sparse_state = np.zeros(dim, dtype=np.complex128)
        sparse_state[0] = 1.0
        cases.append(
            OpenSystemBenchmarkCase(
                name="sparse_many_jump_mcwf",
                hamiltonian=scipy_sparse.csr_array((dim, dim), dtype=np.complex128),
                jumps=tuple(sparse_jumps),
                state_initial=sparse_state,
                parameters={
                    "dim": dim,
                    "n_jumps": n_jumps,
                    "entries_per_jump": entries_per_jump,
                },
            )
        )

    if include_cage_lindblad_case:
        cases.append(
            _make_square_qdm_cage_lindblad_mcwf_case(
                check_liouvillian=cage_lindblad_check_liouvillian,
                ipr_candidate_count=cage_lindblad_ipr_candidate_count,
                ipr_max_iter=cage_lindblad_ipr_max_iter,
                ipr_batch_size=cage_lindblad_ipr_batch_size,
                ipr_rank_completion_patience=cage_lindblad_ipr_rank_completion_patience,
                record_index=cage_lindblad_record_index,
            )
        )

    return cases


def selected_operations(operation: OpenSystemOperation) -> tuple[str, ...]:
    if operation == "all":
        return _OPERATION_ORDER

    if operation == "cage_compare":
        return ("liouvillian", "krylov", "rk4_liouville", "rk4_matrix", "mcwf")

    return (operation,)


def _solver_options(
    *,
    method: str,
    backend: str,
    rk4_step_policy: str,
) -> LindbladEvolutionOptions:
    return LindbladEvolutionOptions(
        method=method,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        rk4_step_policy=rk4_step_policy,  # type: ignore[arg-type]
        check_density_matrix=False,
        enforce_hermiticity=True,
        renormalize_trace=True,
    )


def _run_solver_operation(
    *,
    case: OpenSystemBenchmarkCase,
    operation: str,
    times: np.ndarray,
    backend: str,
    rk4_step_policy: str,
):
    options = _solver_options(
        method=operation,
        backend=backend,
        rk4_step_policy=rk4_step_policy,
    )
    return solve_lindblad(
        hamiltonian=case.hamiltonian,
        jumps=case.jumps,
        density_matrix_initial=case.density_matrix_initial,
        times=times,
        method=operation,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        options=options,
    )


def run_open_system_benchmark(
    *,
    case: OpenSystemBenchmarkCase,
    operation: str,
    times: np.ndarray,
    backend: str,
    sparse_format: str,
    n_trajectories: int,
    seed: int | None,
    rk4_step_policy: str,
    mcwf_adaptive_time_step: bool,
    mcwf_max_jump_probability: float,
    mcwf_prefer_sparse_operators: bool,
    mcwf_prefer_sparse_rate_evaluator: bool,
    mcwf_use_total_rate_first: bool,
    mcwf_use_event_driven_jumps: bool,
    mcwf_store_density_matrices: bool,
    mcwf_store_state_snapshots: bool,
    mcwf_trajectory_chunk_size: int | None,
    mcwf_trajectory_chunk_workers: int | None,
) -> OpenSystemBenchmarkResult:
    liouvillian_shape: tuple[int, int] | None = None
    liouvillian_nnz: int | None = None
    details: dict[str, object] = {}
    stage_seconds: dict[str, float] | None = None
    n_trajectories_result: int | None = None

    if operation == "prepare_dense":
        prepared, elapsed = _time_call(
            lambda: prepare_dense_lindblad_operators(
                hamiltonian=case.hamiltonian,
                jumps=case.jumps,
                backend=backend,
            )
        )
        details = {"prepared_jumps": len(prepared.jumps)}

    elif operation == "prepare_sparse":
        prepared, elapsed = _time_call(
            lambda: prepare_sparse_lindblad_operators(
                hamiltonian=case.hamiltonian,
                jumps=case.jumps,
                backend=backend,
                sparse_format=sparse_format,
            )
        )
        details = {
            "format": sparse_format,
            "prepared_jumps": len(prepared.jumps),
        }

    elif operation == "liouvillian":
        problem = LindbladProblem(
            hamiltonian=case.hamiltonian,
            jumps=case.jumps,
            backend=backend,  # type: ignore[arg-type]
        )
        liouvillian, elapsed = _time_call(
            lambda: problem.build_liouvillian(sparse_format=sparse_format)
        )
        liouvillian_shape = tuple(int(x) for x in liouvillian.shape)
        liouvillian_nnz = int(liouvillian.nnz)
        details = {"format": sparse_format}

    elif operation in {"rk4_matrix", "rk4_liouville", "krylov"}:
        result, elapsed = _time_call(
            lambda: _run_solver_operation(
                case=case,
                operation=operation,
                times=times,
                backend=backend,
                rk4_step_policy=rk4_step_policy,
            )
        )
        details = {
            "method": result.method,
            "n_substeps_total": int(sum(result.n_substeps_per_interval)),
        }

    elif operation == "single_trajectory":
        trajectory, elapsed = _time_call(
            lambda: run_quantum_jump_trajectory(
                hamiltonian=case.hamiltonian,
                jumps=case.jumps,
                state_initial=case.state_initial,
                times=times,
                rng=seed,
                backend=backend,  # type: ignore[arg-type]
                store_states=True,
                adaptive_time_step=mcwf_adaptive_time_step,
                max_jump_probability=mcwf_max_jump_probability,
                prefer_sparse_operators=mcwf_prefer_sparse_operators,
                prefer_sparse_rate_evaluator=mcwf_prefer_sparse_rate_evaluator,
                use_total_rate_first=mcwf_use_total_rate_first,
            )
        )
        n_trajectories_result = 1
        details = {
            "stored_states": len(trajectory.states),
            "observed_jumps": int(trajectory.jump_indices.size),
            "adaptive": mcwf_adaptive_time_step,
            "prefer_sparse_operators": mcwf_prefer_sparse_operators,
            "prefer_sparse_rate_evaluator": mcwf_prefer_sparse_rate_evaluator,
            "use_total_rate_first": mcwf_use_total_rate_first,
        }

    elif operation == "mcwf":
        stage_seconds = {}
        options = McwfOptions(
            backend=backend,  # type: ignore[arg-type]
            n_trajectories=n_trajectories,
            seed=seed,
            store_trajectories=False,
            store_states=False,
            adaptive_time_step=mcwf_adaptive_time_step,
            max_jump_probability=mcwf_max_jump_probability,
            prefer_sparse_operators=mcwf_prefer_sparse_operators,
            prefer_sparse_rate_evaluator=mcwf_prefer_sparse_rate_evaluator,
            use_total_rate_first=mcwf_use_total_rate_first,
            use_event_driven_jumps=mcwf_use_event_driven_jumps,
            store_density_matrices=mcwf_store_density_matrices,
            store_state_snapshots=mcwf_store_state_snapshots,
            trajectory_chunk_size=mcwf_trajectory_chunk_size,
            trajectory_chunk_workers=mcwf_trajectory_chunk_workers,
            timing_collector=stage_seconds,
        )
        result, elapsed = _time_call(
            lambda: sample_lindblad_mcwf(
                hamiltonian=case.hamiltonian,
                jumps=case.jumps,
                state_initial=case.state_initial,
                times=times,
                options=options,
            )
        )
        n_trajectories_result = n_trajectories
        details = {
            "n_rho": len(result.rho_t),
            "n_state_snapshots": (
                0 if result.state_snapshots is None else len(result.state_snapshots)
            ),
            "store_density_matrices": mcwf_store_density_matrices,
            "store_state_snapshots": mcwf_store_state_snapshots,
            "trajectory_chunk_size": mcwf_trajectory_chunk_size,
            "trajectory_chunk_workers": mcwf_trajectory_chunk_workers,
            "adaptive": mcwf_adaptive_time_step,
            "prefer_sparse_operators": mcwf_prefer_sparse_operators,
            "prefer_sparse_rate_evaluator": mcwf_prefer_sparse_rate_evaluator,
            "use_total_rate_first": mcwf_use_total_rate_first,
            "use_event_driven_jumps": mcwf_use_event_driven_jumps,
        }

    else:
        raise ValueError(f"Unsupported open-system benchmark operation: {operation!r}")

    return OpenSystemBenchmarkResult(
        name=case.name,
        operation=operation,
        backend=backend,
        dim=case.dim,
        n_jumps=len(case.jumps),
        n_times=int(times.size),
        n_trajectories=n_trajectories_result,
        liouvillian_shape=liouvillian_shape,
        liouvillian_nnz=liouvillian_nnz,
        elapsed_seconds=elapsed,
        parameters=case.parameters,
        details=details,
        stage_seconds=stage_seconds,
    )


def print_table(results: list[OpenSystemBenchmarkResult]) -> None:
    if not results:
        print("No benchmark results.")
        return

    headers = [
        "name",
        "operation",
        "backend",
        "dim",
        "jumps",
        "times",
        "traj",
        "L.nnz",
        "elapsed_s",
        "prep_s",
        "rate_s",
        "prop_s",
        "chunk_s",
        "parallel_s",
        "rho_s",
        "details",
    ]
    rows = [
        [
            result.name,
            result.operation,
            result.backend,
            str(result.dim),
            str(result.n_jumps),
            str(result.n_times),
            "" if result.n_trajectories is None else str(result.n_trajectories),
            "" if result.liouvillian_nnz is None else str(result.liouvillian_nnz),
            f"{result.elapsed_seconds:.6f}",
            _format_seconds((result.stage_seconds or {}).get("mcwf.operator_preparation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.rate_evaluation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.no_jump_propagation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.chunk_merge")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.chunk_parallel_wall")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.density_accumulation")),
            _details_text(result.details),
        ]
        for result in results
    ]

    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def format_markdown_report(results: list[OpenSystemBenchmarkResult]) -> str:
    headers = [
        "case",
        "operation",
        "backend",
        "dim",
        "jumps",
        "times",
        "traj",
        "L.nnz",
        "elapsed_s",
        "prep_s",
        "rate_s",
        "prop_s",
        "chunk_s",
        "parallel_s",
        "rho_s",
        "details",
    ]
    rows = [
        [
            result.name,
            result.operation,
            result.backend,
            result.dim,
            result.n_jumps,
            result.n_times,
            _format_optional_int(result.n_trajectories),
            _format_optional_int(result.liouvillian_nnz),
            _format_seconds(result.elapsed_seconds),
            _format_seconds((result.stage_seconds or {}).get("mcwf.operator_preparation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.rate_evaluation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.no_jump_propagation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.chunk_merge")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.chunk_parallel_wall")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.density_accumulation")),
            _details_text(result.details),
        ]
        for result in results
    ]

    return "\n".join(
        [
            "## Open-system benchmark",
            "",
            _markdown_table(headers, rows),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark open-system solvers and MCWF sampling.")
    parser.add_argument(
        "--operation",
        default="all",
        choices=["all", "cage_compare", *_OPERATION_ORDER],
        help=(
            "Operation to benchmark. Use 'all' for all open-system operations. "
            "Use 'cage_compare' to run the square-QDM Cage-Lindblad exact-vs-MCWF suite."
        ),
    )
    parser.add_argument(
        "--backend",
        default="scipy",
        help="Open-system backend to use.",
    )
    parser.add_argument(
        "--sparse-format",
        default="csc",
        choices=["csc", "csr"],
        help="Sparse format used for sparse preparation and Liouvillian construction.",
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=11,
        help="Number of output times in the benchmark grid.",
    )
    parser.add_argument(
        "--time-stop",
        type=float,
        default=0.2,
        help="Final time for the benchmark grid.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=128,
        help="Number of MCWF trajectories for operation='mcwf' or operation='all'.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rk4-step-policy",
        default="ignore",
        choices=["raise", "warn", "adaptive", "ignore"],
        help="RK4 step-size policy for deterministic Lindblad solvers.",
    )
    parser.add_argument(
        "--mcwf-adaptive-time-step",
        action="store_true",
        help="Enable adaptive substepping for MCWF trajectories.",
    )
    parser.add_argument(
        "--mcwf-max-jump-probability",
        type=float,
        default=0.1,
        help="Maximum first-order jump probability per MCWF step.",
    )
    parser.add_argument(
        "--mcwf-dense-operators",
        action="store_true",
        help=(
            "Materialize MCWF Hamiltonian and jump operators as dense arrays. "
            "By default, scipy sparse/lazy inputs stay sparse."
        ),
    )
    parser.add_argument(
        "--mcwf-disable-sparse-rate-evaluator",
        action="store_true",
        help=(
            "Disable the sparse row-based MCWF jump-rate evaluator. "
            "By default, very row-sparse scipy jumps avoid full dense J|psi> "
            "rate buffers."
        ),
    )
    parser.add_argument(
        "--mcwf-disable-total-rate-first",
        action="store_true",
        help=(
            "Disable total-rate-first MCWF sampling. By default, MCWF evaluates "
            "the total rate <psi|sum J†J|psi> every step and only evaluates "
            "per-channel rates for trajectories that actually jump."
        ),
    )
    parser.add_argument(
        "--mcwf-event-driven-jumps",
        action="store_true",
        help=(
            "Use piecewise-constant event-driven jump-time sampling in the "
            "vectorized MCWF ensemble path. This is experimental and is aimed "
            "at long time-stop runs with coarse output grids."
        ),
    )
    parser.add_argument(
        "--mcwf-skip-density-matrices",
        action="store_true",
        help="Skip full ensemble density-matrix accumulation for MCWF benchmarks.",
    )
    parser.add_argument(
        "--mcwf-store-state-snapshots",
        action="store_true",
        help=(
            "Store low-rank MCWF ensemble state snapshots with one trajectory "
            "state per column at each output time."
        ),
    )
    parser.add_argument(
        "--mcwf-trajectory-chunk-size",
        type=int,
        default=None,
        help=(
            "Split vectorized MCWF into trajectory chunks of this size. "
            "This lowers memory use for large n_trajectories."
        ),
    )
    parser.add_argument(
        "--mcwf-trajectory-chunk-workers",
        type=int,
        default=None,
        help=(
            "Run vectorized MCWF trajectory chunks in this many worker "
            "processes. Requires --mcwf-trajectory-chunk-size; values <=1 "
            "use the serial chunk path."
        ),
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
    parser.add_argument(
        "--include-sparse-many-jump-case",
        action="store_true",
        help=(
            "Include a synthetic row-sparse many-jump MCWF case. "
            "This is intended for MCWF-only benchmarks; deterministic "
            "Liouvillian solvers may be expensive for this case."
        ),
    )
    parser.add_argument(
        "--include-cage-lindblad-case",
        action="store_true",
        help=(
            "Include the real square_qdm_4x4_pbc_w00 Cage-Lindblad MCWF case "
            "using the (0, 4) type-1 cage, exact-support reduced-IZ monitor, "
            "kinetic-outside/monitor-inside jumps, and local_rdm_two_pattern "
            "recycling jumps."
        ),
    )
    parser.add_argument(
        "--cage-lindblad-skip-liouvillian-check",
        action="store_true",
        help="Skip check_liouvillian when building the optional Cage-Lindblad MCWF case.",
    )
    parser.add_argument("--cage-lindblad-record-index", type=int, default=0)
    parser.add_argument("--cage-lindblad-ipr-candidate-count", type=int, default=128)
    parser.add_argument("--cage-lindblad-ipr-max-iter", type=int, default=1000)
    parser.add_argument("--cage-lindblad-ipr-batch-size", type=int, default=32)
    parser.add_argument(
        "--cage-lindblad-ipr-rank-completion-patience",
        type=int,
        default=0,
        help=(
            "IPR early-stop patience for the optional Cage-Lindblad setup. "
            "Use a negative value to disable early stopping."
        ),
    )

    args = parser.parse_args()

    times = np.linspace(0.0, args.time_stop, args.n_times, dtype=np.float64)
    results: list[OpenSystemBenchmarkResult] = []

    cage_lindblad_patience = (
        None
        if args.cage_lindblad_ipr_rank_completion_patience < 0
        else args.cage_lindblad_ipr_rank_completion_patience
    )
    include_cage_lindblad_case = args.include_cage_lindblad_case or args.operation == "cage_compare"

    for case in make_benchmark_cases(
        include_sparse_many_jump_case=args.include_sparse_many_jump_case,
        include_cage_lindblad_case=include_cage_lindblad_case,
        cage_lindblad_check_liouvillian=(not args.cage_lindblad_skip_liouvillian_check),
        cage_lindblad_record_index=args.cage_lindblad_record_index,
        cage_lindblad_ipr_candidate_count=args.cage_lindblad_ipr_candidate_count,
        cage_lindblad_ipr_max_iter=args.cage_lindblad_ipr_max_iter,
        cage_lindblad_ipr_batch_size=args.cage_lindblad_ipr_batch_size,
        cage_lindblad_ipr_rank_completion_patience=cage_lindblad_patience,
    ):
        if args.only is not None and args.only not in case.name:
            continue

        for operation in selected_operations(args.operation):
            print(f"Running {case.name} [{operation}] ...", flush=True)
            try:
                result = run_open_system_benchmark(
                    case=case,
                    operation=operation,
                    times=times,
                    backend=args.backend,
                    sparse_format=args.sparse_format,
                    n_trajectories=args.n_trajectories,
                    seed=args.seed,
                    rk4_step_policy=args.rk4_step_policy,
                    mcwf_adaptive_time_step=args.mcwf_adaptive_time_step,
                    mcwf_max_jump_probability=args.mcwf_max_jump_probability,
                    mcwf_prefer_sparse_operators=not args.mcwf_dense_operators,
                    mcwf_prefer_sparse_rate_evaluator=(not args.mcwf_disable_sparse_rate_evaluator),
                    mcwf_use_total_rate_first=(not args.mcwf_disable_total_rate_first),
                    mcwf_use_event_driven_jumps=args.mcwf_event_driven_jumps,
                    mcwf_store_density_matrices=(not args.mcwf_skip_density_matrices),
                    mcwf_store_state_snapshots=args.mcwf_store_state_snapshots,
                    mcwf_trajectory_chunk_size=args.mcwf_trajectory_chunk_size,
                    mcwf_trajectory_chunk_workers=args.mcwf_trajectory_chunk_workers,
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

    if args.markdown is not None:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(results))

        print(f"\nWrote Markdown results to {args.markdown}")


if __name__ == "__main__":
    main()
