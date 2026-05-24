#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from qlinks.caging import CageSearchConfig, CageSearcher
from qlinks.caging.classification import (
    CageClassificationConfig,
    classify_cage_state,
)
from qlinks.distributed import DistributedConfig, TaskFailure, map_tasks
from qlinks.models.base import BasisSolverName, HamiltonianBuilderName
from qlinks.models.qdm import (
    HoneycombQDMModel,
    SquareQDMModel,
    TriangularQDMModel,
)
from qlinks.models.qlm import (
    HoneycombQLMModel,
    SquareQLMModel,
    TriangularQLMModel,
)

try:
    from qlinks.basis import Basis
    from qlinks.encoded import BinaryEncodedBasis
except Exception:  # pragma: no cover - keeps script robust to import reshuffle
    Basis = None
    BinaryEncodedBasis = None


ModelKind = Literal["qdm", "qlm"]
Geometry = Literal["square", "triangular", "honeycomb"]
BoundaryCondition = Literal["periodic"]
JobStatus = Literal[
    "queued",
    "running",
    "completed",
    "skipped_too_large",
    "skipped_too_small",
    "failed",
]


@dataclass(frozen=True, slots=True)
class CageSweepJob:
    """One independently executable cage-search job."""

    job_id: str
    model_kind: ModelKind
    geometry: Geometry
    lx: int
    ly: int
    boundary_condition: BoundaryCondition = "periodic"

    sector: dict[str, Any] = field(default_factory=dict)

    kinetic: float = -1.0
    potential: float = 1.0
    builder: HamiltonianBuilderName = "sparse"
    basis_solver: BasisSolverName = "dfs"
    sort_basis: bool = True

    # QLM charge metadata.
    charges: int | str | None = None
    charge_magnitude: int | None = None
    charge_convention: str | None = None
    charge_normalization: str | None = None

    # Search metadata.
    search_type: Literal["qdm", "qlm", "type1", "type2", "custom"] = "custom"
    ipr_random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class CageSweepSettings:
    output_root: str
    min_states: int = 1
    max_states: int = 2**18
    overwrite: bool = False

    search_tolerance: float = 1e-10
    ipr_n_restarts: int = 128
    ipr_max_iter: int = 3000
    ipr_candidate_count: int = 64

    classify: bool = True
    classification_amplitude_tolerance: float = 1e-10
    classification_cancellation_tolerance: float = 1e-9
    classification_action_tolerance: float = 1e-9


@dataclass(frozen=True, slots=True)
class CageSweepTask:
    job: CageSweepJob
    settings: CageSweepSettings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataclass_to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            field.name: dataclass_to_jsonable(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }

    if isinstance(obj, dict):
        return {
            str(dataclass_to_jsonable(key)): dataclass_to_jsonable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(value) for value in obj]

    if isinstance(obj, Fraction):
        if obj.denominator == 1:
            return int(obj.numerator)
        return {
            "numerator": int(obj.numerator),
            "denominator": int(obj.denominator),
            "as_string": str(obj),
        }

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.complexfloating):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}

    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}

    return obj


def safe_label_value(value: Any) -> str:
    """Return a path-safe label value."""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}over{value.denominator}"

    text = str(value)
    return (
        text.replace("/", "over")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dataclass_to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def job_dir(output_root: str | Path, job: CageSweepJob) -> Path:
    return Path(output_root).expanduser().resolve() / "jobs" / job.job_id


def status_path(output_root: str | Path, job: CageSweepJob) -> Path:
    return job_dir(output_root, job) / "status.json"


def hdf5_path(output_root: str | Path, job: CageSweepJob) -> Path:
    return job_dir(output_root, job) / "cages.h5"


def update_status(
    output_root: str | Path,
    job: CageSweepJob,
    status: JobStatus,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": status,
        "updated_at": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    write_json(status_path(output_root, job), payload)


def should_skip_completed(task: CageSweepTask) -> bool:
    if task.settings.overwrite:
        return False

    status = read_json(status_path(task.settings.output_root, task.job))
    if status is None:
        return False

    return (
        status.get("status") == "completed"
        and hdf5_path(
            task.settings.output_root,
            task.job,
        ).exists()
    )


def make_model(job: CageSweepJob):
    kwargs: dict[str, Any] = {
        "lx": job.lx,
        "ly": job.ly,
        "boundary_condition": job.boundary_condition,
        "kinetic": job.kinetic,
        "potential": job.potential,
        **job.sector,
    }

    if job.model_kind == "qdm":
        if job.geometry == "square":
            return SquareQDMModel(**kwargs)
        if job.geometry == "triangular":
            return TriangularQDMModel(**kwargs)
        if job.geometry == "honeycomb":
            return HoneycombQDMModel(**kwargs)
        raise ValueError(f"Unsupported QDM geometry: {job.geometry!r}")

    if job.model_kind == "qlm":
        if job.geometry == "square":
            # User choice: square QLM uses zero charges, not staggered QDM charges.
            return SquareQLMModel(
                **kwargs,
                charges=0,
                charge_normalization=job.charge_normalization or "spin_half",
            )

        if job.geometry == "triangular":
            # Start with zero charges unless job explicitly overrides later.
            return TriangularQLMModel(
                **kwargs,
                charges=0,
                charge_normalization=job.charge_normalization or "spin_half",
            )

        if job.geometry == "honeycomb":
            # User choice: honeycomb zero charges are not meaningful, so use
            # staggered +/- charge_magnitude.
            return HoneycombQLMModel.from_staggered_background(
                **kwargs,
                charge_magnitude=job.charge_magnitude or 1,
                charge_convention=job.charge_convention or "even_positive",
            )

        raise ValueError(f"Unsupported QLM geometry: {job.geometry!r}")

    raise ValueError(f"Unsupported model kind: {job.model_kind!r}")


def basis_configs_from_build_result(build_result) -> np.ndarray:
    """Return explicit basis configurations, handling array or bitmask basis."""
    basis = build_result.basis

    if hasattr(basis, "states"):
        return np.asarray(basis.states)

    if hasattr(basis, "to_array_basis"):
        return np.asarray(basis.to_array_basis().states)

    raise TypeError(f"Cannot extract array basis states from {type(basis)!r}.")


def sparse_nnz(matrix: Any | None) -> int:
    if matrix is None:
        return 0
    return int(getattr(matrix, "nnz", 0))


def complex_array(values: list[complex]) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def write_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(values, dtype=dtype), dtype=dtype)


def write_attrs(group: h5py.Group | h5py.File, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue

        value = dataclass_to_jsonable(value)
        if isinstance(value, (dict, list, tuple)):
            group.attrs[key] = json.dumps(value, sort_keys=True)
        else:
            group.attrs[key] = value


def write_cage_hdf5(
    path: Path,
    *,
    job: CageSweepJob,
    settings: CageSweepSettings,
    build_result,
    cage_result,
    classification_reports: list[Any],
    elapsed_seconds: float,
) -> None:
    """Write cage-search result to HDF5 with partial-read friendly datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()

    records = list(cage_result.records)
    n_records = len(records)
    n_states = int(build_result.basis.n_states)

    max_support_size = max(
        [record.cage_state.support_size for record in records],
        default=0,
    )

    supports = np.full((n_records, max_support_size), -1, dtype=np.int64)
    local_states = np.zeros((n_records, max_support_size), dtype=np.complex128)
    support_sizes = np.zeros(n_records, dtype=np.int64)
    energies = np.zeros(n_records, dtype=np.complex128)
    boundary_residuals = np.zeros(n_records, dtype=np.float64)
    eigen_residuals = np.zeros(n_records, dtype=np.float64)
    full_residuals = np.full(n_records, np.nan, dtype=np.float64)
    signatures = np.zeros((n_records, 2), dtype=np.int64)

    labels: list[str] = []
    support_fractions = np.zeros(n_records, dtype=np.float64)
    n_nontrivial_zeros = np.zeros(n_records, dtype=np.int64)
    n_distinct_local_patterns = np.zeros(n_records, dtype=np.int64)
    n_complement_targets = np.zeros(n_records, dtype=np.int64)
    n_unexplained_complement_targets = np.zeros(n_records, dtype=np.int64)
    closed_fractions = np.zeros(n_records, dtype=np.float64)
    mean_q_weights = np.zeros(n_records, dtype=np.float64)
    max_q_weights = np.zeros(n_records, dtype=np.float64)
    mean_complement_norms = np.zeros(n_records, dtype=np.float64)
    max_complement_norms = np.zeros(n_records, dtype=np.float64)
    mean_reduced_norms = np.zeros(n_records, dtype=np.float64)
    max_reduced_norms = np.zeros(n_records, dtype=np.float64)
    n_q_empty_zeros = np.zeros(n_records, dtype=np.int64)
    n_closed_by_known_zero_zeros = np.zeros(n_records, dtype=np.int64)
    n_projector_like_zeros = np.zeros(n_records, dtype=np.int64)
    n_unexplained_leakage_zeros = np.zeros(n_records, dtype=np.int64)
    n_regional_mechanism_zeros = np.zeros(n_records, dtype=np.int64)
    n_extended_mechanism_zeros = np.zeros(n_records, dtype=np.int64)
    n_failure_mechanism_zeros = np.zeros(n_records, dtype=np.int64)

    for record_index, record in enumerate(records):
        cage_state = record.cage_state
        size = cage_state.support_size

        supports[record_index, :size] = cage_state.support
        local_states[record_index, :size] = cage_state.local_state
        support_sizes[record_index] = size
        energies[record_index] = cage_state.energy
        boundary_residuals[record_index] = cage_state.boundary_residual
        eigen_residuals[record_index] = cage_state.eigen_residual

        if cage_state.full_residual is not None:
            full_residuals[record_index] = cage_state.full_residual

        signatures[record_index, :] = np.asarray(record.signature, dtype=np.int64)

        if classification_reports:
            report = classification_reports[record_index]
            labels.append(str(report.label))
            support_fractions[record_index] = report.support_fraction
            n_nontrivial_zeros[record_index] = report.n_nontrivial_zeros
            n_distinct_local_patterns[record_index] = report.n_distinct_local_patterns
            n_complement_targets[record_index] = report.n_complement_targets
            n_unexplained_complement_targets[record_index] = report.n_unexplained_complement_targets
            closed_fractions[record_index] = report.fraction_zeros_with_closed_complement_targets
            mean_q_weights[record_index] = report.mean_q_sector_weight
            max_q_weights[record_index] = report.max_q_sector_weight
            mean_complement_norms[record_index] = report.mean_complement_action_norm
            max_complement_norms[record_index] = report.max_complement_action_norm
            mean_reduced_norms[record_index] = report.mean_reduced_action_norm
            max_reduced_norms[record_index] = report.max_reduced_action_norm
            n_q_empty_zeros[record_index] = report.n_q_empty_zeros
            n_closed_by_known_zero_zeros[record_index] = report.n_closed_by_known_zero_zeros
            n_projector_like_zeros[record_index] = report.n_projector_like_zeros
            n_unexplained_leakage_zeros[record_index] = report.n_unexplained_leakage_zeros
            n_regional_mechanism_zeros[record_index] = report.n_regional_mechanism_zeros
            n_extended_mechanism_zeros[record_index] = report.n_extended_mechanism_zeros
            n_failure_mechanism_zeros[record_index] = report.n_failure_mechanism_zeros

    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["format"] = "qlinks_cage_sweep"
        h5.attrs["schema_version"] = "0.2"
        h5.attrs["created_at"] = utc_now()

        write_attrs(
            h5.require_group("metadata/job"),
            dataclasses.asdict(job),
        )
        write_attrs(
            h5.require_group("metadata/settings"),
            dataclasses.asdict(settings),
        )

        write_attrs(
            h5.require_group("metadata/run"),
            {
                "elapsed_seconds": elapsed_seconds,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
        )

        write_attrs(
            h5.require_group("metadata/model"),
            {
                "model_class": type(build_result.model).__name__,
                "lattice_class": type(build_result.lattice).__name__,
                "layout_n_variables": getattr(build_result.layout, "n_variables", None),
                "n_constraints": len(build_result.constraints),
                "n_sectors": len(build_result.sectors),
                "n_states": n_states,
                "H_nnz": sparse_nnz(build_result.hamiltonian),
                "K_nnz": sparse_nnz(build_result.kinetic),
                "V_nnz": sparse_nnz(build_result.potential),
            },
        )

        basis_group = h5.require_group("basis")
        basis_configs = basis_configs_from_build_result(build_result)
        basis_group.create_dataset(
            "states",
            data=basis_configs,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        cages = h5.require_group("cages")
        cages.attrs["n_records"] = n_records
        cages.attrs["max_support_size"] = max_support_size
        cages.create_dataset("energies", data=energies, chunks=True)
        cages.create_dataset("signatures", data=signatures, chunks=True)
        cages.create_dataset("support_sizes", data=support_sizes, chunks=True)

        if n_records == 0:
            cages.create_dataset("supports", data=supports)
            cages.create_dataset("local_states", data=local_states)
        else:
            cages.create_dataset(
                "supports",
                data=supports,
                compression="gzip",
                compression_opts=4,
                chunks=(1, max(1, max_support_size)),
            )
            cages.create_dataset(
                "local_states",
                data=local_states,
                compression="gzip",
                compression_opts=4,
                chunks=(1, max(1, max_support_size)),
            )
            
        cages.create_dataset(
            "boundary_residuals",
            data=boundary_residuals,
            chunks=True,
        )
        cages.create_dataset("eigen_residuals", data=eigen_residuals, chunks=True)
        cages.create_dataset("full_residuals", data=full_residuals, chunks=True)

        if classification_reports:
            cls_group = h5.require_group("classification")
            write_string_dataset(cls_group, "labels", labels)
            cls_group.create_dataset(
                "support_fractions",
                data=support_fractions,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_nontrivial_zeros",
                data=n_nontrivial_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_distinct_local_patterns",
                data=n_distinct_local_patterns,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_complement_targets",
                data=n_complement_targets,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_unexplained_complement_targets",
                data=n_unexplained_complement_targets,
                chunks=True,
            )
            cls_group.create_dataset(
                "closed_complement_fractions",
                data=closed_fractions,
                chunks=True,
            )
            cls_group.create_dataset(
                "mean_q_sector_weights",
                data=mean_q_weights,
                chunks=True,
            )
            cls_group.create_dataset(
                "max_q_sector_weights",
                data=max_q_weights,
                chunks=True,
            )
            cls_group.create_dataset(
                "mean_complement_action_norms",
                data=mean_complement_norms,
                chunks=True,
            )
            cls_group.create_dataset(
                "max_complement_action_norms",
                data=max_complement_norms,
                chunks=True,
            )
            cls_group.create_dataset(
                "mean_reduced_action_norms",
                data=mean_reduced_norms,
                chunks=True,
            )
            cls_group.create_dataset(
                "max_reduced_action_norms",
                data=max_reduced_norms,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_q_empty_zeros",
                data=n_q_empty_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_closed_by_known_zero_zeros",
                data=n_closed_by_known_zero_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_projector_like_zeros",
                data=n_projector_like_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_unexplained_leakage_zeros",
                data=n_unexplained_leakage_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_regional_mechanism_zeros",
                data=n_regional_mechanism_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_extended_mechanism_zeros",
                data=n_extended_mechanism_zeros,
                chunks=True,
            )
            cls_group.create_dataset(
                "n_failure_mechanism_zeros",
                data=n_failure_mechanism_zeros,
                chunks=True,
            )
            cls_group.attrs["regional_mechanism_definition"] = (
                "q_empty + closed_by_known_zeros"
            )
            cls_group.attrs["extended_mechanism_definition"] = "projector_like"
            cls_group.attrs["failure_mechanism_definition"] = "unexplained_leakage"
            cls_group.attrs["label_rule"] = (
                "invalid if unexplained_leakage exists; "
                "extended if projector_like exists; "
                "otherwise regional"
            )

        mechanisms_group = cls_group.require_group("mechanism_zero_indices")

        for record_index, report in enumerate(classification_reports):
            record_group = mechanisms_group.require_group(str(record_index))
            record_group.create_dataset(
                "q_empty",
                data=report.q_empty_zero_indices,
            )
            record_group.create_dataset(
                "closed_by_known_zeros",
                data=report.closed_by_known_zero_indices,
            )
            record_group.create_dataset(
                "projector_like",
                data=report.projector_like_zero_indices,
            )
            record_group.create_dataset(
                "unexplained_leakage",
                data=report.unexplained_leakage_zero_indices,
            )
            record_group.create_dataset(
                "regional",
                data=report.regional_mechanism_zero_indices,
            )
            record_group.create_dataset(
                "extended",
                data=report.extended_mechanism_zero_indices,
            )
            record_group.create_dataset(
                "failure",
                data=report.failure_mechanism_zero_indices,
            )

    tmp_path.replace(path)


def run_one_job(task: CageSweepTask) -> dict[str, Any]:
    job = task.job
    settings = task.settings
    directory = job_dir(settings.output_root, job)
    directory.mkdir(parents=True, exist_ok=True)

    write_json(directory / "job.json", dataclasses.asdict(job))

    if should_skip_completed(task):
        return {
            "job_id": job.job_id,
            "status": "completed",
            "skipped_existing": True,
        }

    update_status(
        settings.output_root,
        job,
        "running",
        extra={"started_at": utc_now()},
    )

    start = time.perf_counter()

    try:
        model = make_model(job)

        # Preflight basis size before matrix construction.
        basis = model.build_basis(
            solver=job.basis_solver,
            sort=job.sort_basis,
        )
        n_states = int(basis.n_states)

        if n_states > settings.max_states:
            update_status(
                settings.output_root,
                job,
                "skipped_too_large",
                extra={
                    "n_states": n_states,
                    "max_states": settings.max_states,
                    "finished_at": utc_now(),
                },
            )
            return {
                "job_id": job.job_id,
                "status": "skipped_too_large",
                "n_states": n_states,
            }

        if n_states <= settings.min_states:
            update_status(
                settings.output_root,
                job,
                "skipped_too_small",
                extra={
                    "n_states": n_states,
                    "min_states": settings.min_states,
                    "finished_at": utc_now(),
                },
            )
            return {
                "job_id": job.job_id,
                "status": "skipped_too_small",
                "n_states": n_states,
            }

        build_result = model.build(
            basis=basis,
            basis_solver=job.basis_solver,
            builder=job.builder,
            sort_basis=job.sort_basis,
            on_missing="raise",
        )

        search_config = CageSearchConfig(
            search_type=job.search_type,
            tolerance=settings.search_tolerance,
            validate_full_residual=True,
            degenerate_basis_strategy="ipr",
            ipr_n_restarts=settings.ipr_n_restarts,
            ipr_max_iter=settings.ipr_max_iter,
            ipr_candidate_count=settings.ipr_candidate_count,
            ipr_random_seed=job.ipr_random_seed,
        )

        cage_result = CageSearcher.from_model_build_result(
            build_result,
            config=search_config,
        ).run()

        classification_reports: list[Any] = []
        if settings.classify:
            basis_configs = basis_configs_from_build_result(build_result)
            classification_config = CageClassificationConfig(
                amplitude_tolerance=settings.classification_amplitude_tolerance,
                cancellation_tolerance=settings.classification_cancellation_tolerance,
                action_tolerance=settings.classification_action_tolerance,
            )
            for record in cage_result.records:
                classification_reports.append(
                    classify_cage_state(
                        record.cage_state,
                        kinetic_matrix=build_result.kinetic,
                        basis_configs=basis_configs,
                        hilbert_size=n_states,
                        config=classification_config,
                    )
                )

        elapsed = time.perf_counter() - start

        write_cage_hdf5(
            hdf5_path(settings.output_root, job),
            job=job,
            settings=settings,
            build_result=build_result,
            cage_result=cage_result,
            classification_reports=classification_reports,
            elapsed_seconds=elapsed,
        )

        counts_by_signature = {
            str(signature): int(count)
            for signature, count in cage_result.counts_by_signature.items()
        }

        labels: dict[str, int] = {}
        for report in classification_reports:
            labels[report.label] = labels.get(report.label, 0) + 1

        mechanism_totals = {
            "q_empty": 0,
            "closed_by_known_zeros": 0,
            "projector_like": 0,
            "unexplained_leakage": 0,
            "regional": 0,
            "extended": 0,
            "failure": 0,
        }

        for report in classification_reports:
            mechanism_totals["q_empty"] += int(report.n_q_empty_zeros)
            mechanism_totals["closed_by_known_zeros"] += int(
                report.n_closed_by_known_zero_zeros
            )
            mechanism_totals["projector_like"] += int(report.n_projector_like_zeros)
            mechanism_totals["unexplained_leakage"] += int(
                report.n_unexplained_leakage_zeros
            )
            mechanism_totals["regional"] += int(report.n_regional_mechanism_zeros)
            mechanism_totals["extended"] += int(report.n_extended_mechanism_zeros)
            mechanism_totals["failure"] += int(report.n_failure_mechanism_zeros)

        n_invalid_reports = sum(
            int(report.label == "invalid_or_inconsistent")
            for report in classification_reports
        )
        n_reports_with_unexplained_leakage = sum(
            int(report.n_unexplained_leakage_zeros > 0)
            for report in classification_reports
        )

        summary = {
            "job_id": job.job_id,
            "status": "completed",
            "n_states": n_states,
            "H_nnz": sparse_nnz(build_result.hamiltonian),
            "K_nnz": sparse_nnz(build_result.kinetic),
            "V_nnz": sparse_nnz(build_result.potential),
            "n_records": len(cage_result.records),
            "counts_by_signature": counts_by_signature,
            "classification_counts": labels,
            "classification_mechanism_totals": mechanism_totals,
            "n_invalid_classification_reports": n_invalid_reports,
            "n_reports_with_unexplained_leakage": n_reports_with_unexplained_leakage,
            "elapsed_seconds": elapsed,
            "hdf5_path": str(hdf5_path(settings.output_root, job)),
        }

        write_json(directory / "summary.json", summary)
        if classification_reports:
            report_text = "\n\n".join(
                report.to_text(verbose=False) for report in classification_reports[:20]
            )
            (directory / "classification_preview.txt").write_text(
                report_text,
                encoding="utf-8",
            )

        update_status(
            settings.output_root,
            job,
            "completed",
            extra={
                **summary,
                "finished_at": utc_now(),
            },
        )

        return summary

    except Exception as error:
        error_text = traceback.format_exc()
        (directory / "error.txt").write_text(error_text, encoding="utf-8")

        update_status(
            settings.output_root,
            job,
            "failed",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "finished_at": utc_now(),
            },
        )

        return {
            "job_id": job.job_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def sector_product(labels: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    if not labels:
        return [{}]

    keys = sorted(labels)
    sectors: list[dict[str, Any]] = [{}]

    for key in keys:
        new_sectors: list[dict[str, Any]] = []
        for old in sectors:
            for value in labels[key]:
                item = dict(old)
                item[key] = value
                new_sectors.append(item)
        sectors = new_sectors

    return sectors


def make_sector_probe_model(
    *,
    model_kind: ModelKind,
    geometry: Geometry,
    lx: int,
    ly: int,
    boundary_condition: BoundaryCondition,
    kinetic: float,
    potential: float,
):
    dummy = CageSweepJob(
        job_id="probe",
        model_kind=model_kind,
        geometry=geometry,
        lx=lx,
        ly=ly,
        boundary_condition=boundary_condition,
        kinetic=kinetic,
        potential=potential,
        sector={},
        charges="zero" if model_kind == "qlm" and geometry != "honeycomb" else None,
        charge_magnitude=1 if model_kind == "qlm" and geometry == "honeycomb" else None,
        charge_convention=(
            "even_positive" if model_kind == "qlm" and geometry == "honeycomb" else None
        ),
        charge_normalization=(
            "spin_half" if model_kind == "qlm" and geometry == "honeycomb" else "integer_flux"
        ),
    )
    return make_model(dummy)


def make_jobs(
    *,
    model_kinds: list[ModelKind],
    geometries: list[Geometry],
    sizes_by_geometry: dict[Geometry, list[tuple[int, int]]],
    boundary_condition: BoundaryCondition = "periodic",
    kinetic: float = -1.0,
    potential: float = 1.0,
    builder: HamiltonianBuilderName = "sparse",
    basis_solver: BasisSolverName = "dfs",
    seed_base: int = 12345,
) -> list[CageSweepJob]:
    jobs: list[CageSweepJob] = []
    seed_counter = 0

    for model_kind in model_kinds:
        for geometry in geometries:
            for lx, ly in sizes_by_geometry[geometry]:
                probe_model = make_sector_probe_model(
                    model_kind=model_kind,
                    geometry=geometry,
                    lx=lx,
                    ly=ly,
                    boundary_condition=boundary_condition,
                    kinetic=kinetic,
                    potential=potential,
                )
                sectors = sector_product(probe_model.allowed_sector_labels())

                for sector in sectors:
                    sector_label = "_".join(
                        f"{key}{safe_label_value(value)}" for key, value in sorted(sector.items())
                    )
                    if not sector_label:
                        sector_label = "nosector"

                    charge_label = ""
                    charges: int | str | None = None
                    charge_magnitude: int | None = None
                    charge_convention: str | None = None
                    charge_normalization: str | None = None

                    if model_kind == "qlm":
                        if geometry == "honeycomb":
                            charge_magnitude = 1
                            charge_convention = "even_positive"
                            charge_normalization = "integer_flux"
                            charge_label = "_charges_stag_pm1"
                        else:
                            charges = "zero"
                            charge_normalization = "spin_half"
                            charge_label = "_charges_zero"

                    job_id = (
                        f"{model_kind}_{geometry}_{lx}x{ly}_"
                        f"{boundary_condition}_{sector_label}{charge_label}"
                    )

                    search_type = "qdm" if model_kind == "qdm" else "qlm"

                    jobs.append(
                        CageSweepJob(
                            job_id=job_id,
                            model_kind=model_kind,
                            geometry=geometry,
                            lx=lx,
                            ly=ly,
                            boundary_condition=boundary_condition,
                            sector=sector,
                            kinetic=kinetic,
                            potential=potential,
                            builder=builder,
                            basis_solver=basis_solver,
                            charges=charges,
                            charge_magnitude=charge_magnitude,
                            charge_convention=charge_convention,
                            charge_normalization=charge_normalization,
                            search_type=search_type,
                            ipr_random_seed=seed_base + seed_counter,
                        )
                    )
                    seed_counter += 1

    return jobs


def default_sizes() -> dict[Geometry, list[tuple[int, int]]]:
    return {
        "square": [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4)],
        "triangular": [(2, 2), (3, 2), (3, 3), (4, 3)],
        "honeycomb": [(2, 2), (3, 2), (3, 3), (4, 3)],
    }


def parse_size_list(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in value.split(","):
        lx_text, ly_text = token.lower().split("x")
        pairs.append((int(lx_text), int(ly_text)))
    return pairs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Ray/serial cage sweep for QDM/QLM lattices.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--backend",
        choices=["serial", "ray"],
        default="serial",
    )
    parser.add_argument("--num-cpus-per-task", type=float, default=1.0)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--models",
        default="qdm,qlm",
        help="Comma-separated model kinds: qdm,qlm.",
    )
    parser.add_argument(
        "--geometries",
        default="square,triangular,honeycomb",
        help="Comma-separated geometries.",
    )

    parser.add_argument("--square-sizes", default=None)
    parser.add_argument("--triangular-sizes", default=None)
    parser.add_argument("--honeycomb-sizes", default=None)

    parser.add_argument("--min-states", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=2**18)

    parser.add_argument("--kinetic", type=float, default=-1.0)
    parser.add_argument("--potential", type=float, default=1.0)
    parser.add_argument(
        "--builder",
        choices=["sparse", "optimized", "bitmask"],
        default="sparse",
    )
    parser.add_argument(
        "--basis-solver",
        choices=["dfs", "brute_force", "cpsat"],
        default="dfs",
    )

    parser.add_argument("--ipr-n-restarts", type=int, default=128)
    parser.add_argument("--ipr-max-iter", type=int, default=3000)
    parser.add_argument("--ipr-candidate-count", type=int, default=64)

    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    args = make_parser().parse_args()

    sizes = default_sizes()
    if args.square_sizes:
        sizes["square"] = parse_size_list(args.square_sizes)
    if args.triangular_sizes:
        sizes["triangular"] = parse_size_list(args.triangular_sizes)
    if args.honeycomb_sizes:
        sizes["honeycomb"] = parse_size_list(args.honeycomb_sizes)

    model_kinds = [item.strip() for item in args.models.split(",") if item.strip()]
    geometries = [item.strip() for item in args.geometries.split(",") if item.strip()]

    jobs = make_jobs(
        model_kinds=model_kinds,  # type: ignore[arg-type]
        geometries=geometries,  # type: ignore[arg-type]
        sizes_by_geometry=sizes,
        kinetic=args.kinetic,
        potential=args.potential,
        builder=args.builder,
        basis_solver=args.basis_solver,
    )

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": utc_now(),
        "n_jobs": len(jobs),
        "jobs": [dataclasses.asdict(job) for job in jobs],
    }
    write_json(output_root / "manifest.json", manifest)

    for job in jobs:
        update_status(
            output_root,
            job,
            "queued",
            extra={"queued_at": utc_now()},
        )

    if args.dry_run:
        print(f"Wrote manifest with {len(jobs)} jobs to {output_root}")
        return

    settings = CageSweepSettings(
        output_root=str(output_root),
        min_states=args.min_states,
        max_states=args.max_states,
        overwrite=args.overwrite,
        ipr_n_restarts=args.ipr_n_restarts,
        ipr_max_iter=args.ipr_max_iter,
        ipr_candidate_count=args.ipr_candidate_count,
        classify=not args.no_classify,
    )

    tasks = [CageSweepTask(job=job, settings=settings) for job in jobs]

    ray_init_kwargs = {}
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address

    results = map_tasks(
        run_one_job,
        tasks,
        config=DistributedConfig(
            backend=args.backend,
            preserve_order=False,
            show_progress=True,
            progress_description="Cage sweep jobs",
            failure_mode="return",
            ray_init_kwargs=ray_init_kwargs,
            ray_shutdown=False,
            num_cpus_per_task=args.num_cpus_per_task,
        ),
    )

    normalized_results: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, TaskFailure):
            normalized_results.append(dataclasses.asdict(result))
        else:
            normalized_results.append(result)

    write_json(
        output_root / "final_summary.json",
        {
            "finished_at": utc_now(),
            "n_jobs": len(jobs),
            "results": normalized_results,
        },
    )


if __name__ == "__main__":
    main()
