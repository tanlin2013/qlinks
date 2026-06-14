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


def make_benchmark_cases(
    *,
    include_sparse_many_jump_case: bool = False,
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

    return cases


def selected_operations(operation: OpenSystemOperation) -> tuple[str, ...]:
    if operation == "all":
        return _OPERATION_ORDER

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
            )
        )
        n_trajectories_result = 1
        details = {
            "stored_states": len(trajectory.states),
            "observed_jumps": int(trajectory.jump_indices.size),
            "adaptive": mcwf_adaptive_time_step,
            "prefer_sparse_operators": mcwf_prefer_sparse_operators,
            "prefer_sparse_rate_evaluator": mcwf_prefer_sparse_rate_evaluator,
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
            "adaptive": mcwf_adaptive_time_step,
            "prefer_sparse_operators": mcwf_prefer_sparse_operators,
            "prefer_sparse_rate_evaluator": mcwf_prefer_sparse_rate_evaluator,
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
        "rate_s",
        "prop_s",
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
            _format_seconds((result.stage_seconds or {}).get("mcwf.rate_evaluation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.no_jump_propagation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.density_accumulation")),
            _details_text(result.details),
        ]
        for result in results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

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
        "rate_s",
        "prop_s",
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
            _format_seconds((result.stage_seconds or {}).get("mcwf.rate_evaluation")),
            _format_seconds((result.stage_seconds or {}).get("mcwf.no_jump_propagation")),
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
        choices=["all", *_OPERATION_ORDER],
        help="Operation to benchmark. Use 'all' for all open-system operations.",
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

    args = parser.parse_args()

    times = np.linspace(0.0, args.time_stop, args.n_times, dtype=np.float64)
    results: list[OpenSystemBenchmarkResult] = []

    for case in make_benchmark_cases(
        include_sparse_many_jump_case=args.include_sparse_many_jump_case,
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
