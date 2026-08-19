from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging.analysis.spectral import (
    basis_permutation_from_variable_permutation,
    cyclic_symmetry_sector_basis,
    diagnose_eigenpair,
    project_state_to_sector,
    select_microcanonical_window_by_width,
    spectral_observable_moments,
)
from qlinks.caging.analysis.thermodynamic import (
    LocalWitnessTemplate,
    directed_transition_witness_template,
    hermitianize_local_witness_template,
)
from qlinks.models import (
    spin_one_xy_hxy_h3_imaginary_j2_model,
    spin_one_xy_scar_tower_states,
)

from helpers import (
    charge_conserving_two_site_hermitian_basis,
    orthonormalize_columns,
    projector_deleted_basis,
    projector_deleted_block_covariance,
    projector_deleted_observable_moments,
)


TOL = 1.0e-10
DARK_TOL = 1.0e-9
TOTAL_SZ = -2
J_DRAFT = 1.0
J3_OVER_J = 0.10
J1_MATRIX = 2.0 * J_DRAFT
REPRESENTATIVE_KAPPA_OVER_J = 0.10
DEFAULT_FAMILY_KAPPA_OVER_J = 0.20


@dataclass(frozen=True, slots=True)
class Sec6ProvisioningConfig:
    output_dir: Path
    baseline_data_dir: Path | None = None
    sparse_convergence_data_dir: Path | None = None
    checkpoint_source_dir: Path | None = None
    checkpoint_dir: Path | None = None
    dense_sizes: tuple[int, ...] = (8, 10, 12)
    large_size: int = 14
    representative_kappa_over_j: float = REPRESENTATIVE_KAPPA_OVER_J
    family_kappa_over_j: float = DEFAULT_FAMILY_KAPPA_OVER_J
    representative_eigenpairs: int = 8192
    family_eigenpairs: int = 8192
    shift: float = 1.0e-7
    arpack_tolerance: float = 1.0e-9
    fixed_half_widths: tuple[float, ...] = (0.75, 1.0)
    include_quarter_window: bool = True
    include_sqrt_window_for_dense: bool = True
    concentration_half_width: float = 1.0
    energy_block_tolerance: float = 1.0e-10
    energy_block_tolerance_audit: tuple[float, ...] = (3.0e-10, 1.0e-9, 3.0e-9)
    convergence_tolerance: float = 1.0e-4
    reuse_checkpoints: bool = True
    write_checkpoints: bool = True
    run_large_representative: bool = True
    run_family_large_size: bool = False
    residual_chunk_size: int = 64

    @property
    def resolved_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir or (self.output_dir / "checkpoints")


def _make_spin1_witnesses() -> dict[str, object]:
    y_template = LocalWitnessTemplate(
        pattern_key=(),
        local_patterns=((0,),),
        local_operator=np.asarray([[-1.0]], dtype=np.complex128),
        metadata={"name": "Y_r", "support_sites": 1, "channel_type": "diagonal"},
    )
    a_template = directed_transition_witness_template(
        target_pattern=(0, 0),
        source_patterns=((1, -1), (-1, 1)),
        amplitudes=(J1_MATRIX, J1_MATRIX),
        metadata={"name": "Ared_r_r+1", "support_sites": 2},
    )
    z_template = hermitianize_local_witness_template(
        a_template,
        metadata={"name": "Zred_r_r+1", "support_sites": 2},
    )
    return {
        "Y": y_template.instantiate((0,)),
        "A": a_template.instantiate((0, 1)),
        "Z": z_template.instantiate((0, 1)),
    }


RAW_WITNESSES = _make_spin1_witnesses()


def _deformed_model(*, length: int, kappa_over_j: float):
    return spin_one_xy_hxy_h3_imaginary_j2_model(
        length=int(length),
        j=J_DRAFT,
        j3=J3_OVER_J * J_DRAFT,
        kappa=float(kappa_over_j) * J_DRAFT,
        total_sz=TOTAL_SZ,
        h_z=0.0,
        d_z=0.0,
    )


def _tower_state(configs: np.ndarray, *, length: int) -> np.ndarray:
    states, labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=int(length),
        normalize=True,
    )
    if states.shape[1] != 1:
        raise RuntimeError(f"expected one tower state in fixed-M basis; found {labels}")
    return states[:, 0]


def _tower_translation_sector(configs: np.ndarray, *, length: int):
    n_raised = (TOTAL_SZ + int(length)) // 2
    momentum_index = 0 if n_raised % 2 == 0 else int(length) // 2
    translation = basis_permutation_from_variable_permutation(
        configs,
        np.roll(np.arange(int(length)), 1),
    )
    sector = cyclic_symmetry_sector_basis(
        translation,
        order=int(length),
        momentum_index=int(momentum_index),
        labels={"total_sz": TOTAL_SZ},
    )
    return sector, int(momentum_index)


def _project_sparse(operator, sector) -> sp.csr_array:
    basis = sector.basis if hasattr(sector, "basis") else sector
    return sp.csr_array(basis.conj().T @ (operator @ basis))


def _projected_witness_square(witness, configs: np.ndarray, sector) -> sp.csr_array:
    local = witness.embed(configs)
    return _project_sparse(local.conj().T @ local, sector)


def _normalized_witness_q_ops(configs: np.ndarray, sector) -> dict[str, sp.csr_array]:
    return {
        "A": _projected_witness_square(RAW_WITNESSES["A"], configs, sector)
        / RAW_WITNESSES["A"].template.q_operator_norm,
        "Z": _projected_witness_square(RAW_WITNESSES["Z"], configs, sector)
        / RAW_WITNESSES["Z"].template.q_operator_norm,
        "Y": _projected_witness_square(RAW_WITNESSES["Y"], configs, sector),
    }


def _cleaned_trace_average(operator, exceptional_vectors: np.ndarray) -> dict[str, float | int]:
    dimension = int(operator.shape[0])
    raw_trace = (
        complex(operator.diagonal().sum())
        if sp.issparse(operator)
        else complex(np.trace(operator))
    )
    exceptional = np.asarray(exceptional_vectors, dtype=np.complex128)
    if exceptional.ndim == 1:
        exceptional = exceptional[:, None]
    rank = int(exceptional.shape[1]) if exceptional.size else 0
    removed_trace = 0.0j
    if rank:
        action = operator @ exceptional
        removed_trace = complex(np.einsum("ij,ij->", exceptional.conj(), action))
    if rank >= dimension:
        raise ValueError("exceptional projector exhausts resolved sector")
    return {
        "raw": float(raw_trace.real / dimension),
        "clean": float((raw_trace.real - removed_trace.real) / (dimension - rank)),
        "dimension": dimension,
        "removed_rank": rank,
        "removed_trace": float(removed_trace.real),
    }


def _translated_joint_dark_operator(*, configs: np.ndarray, sector, length: int) -> sp.csr_array:
    unit_witnesses = {
        name: witness.template.normalized("operator_norm")
        for name, witness in RAW_WITNESSES.items()
    }
    q_full = None
    for site in range(int(length)):
        translated = (
            unit_witnesses["A"].instantiate((site, (site + 1) % int(length))),
            unit_witnesses["Z"].instantiate((site, (site + 1) % int(length))),
            unit_witnesses["Y"].instantiate((site,)),
        )
        for witness in translated:
            local = witness.embed(configs)
            contribution = local.conj().T @ local
            q_full = contribution if q_full is None else q_full + contribution
    if q_full is None:
        raise RuntimeError("failed to construct translated joint-dark operator")
    return _project_sparse(q_full, sector)


def _joint_dark_kernel_from_spectrum(
    *,
    energies: np.ndarray,
    vectors: np.ndarray,
    q_all,
    tower: np.ndarray,
    energy_tolerance: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    energies = np.asarray(energies, dtype=float)
    groups: list[list[int]] = []
    if energies.size:
        current = [0]
        for index in range(1, energies.size):
            if abs(energies[index] - energies[current[-1]]) <= energy_tolerance:
                current.append(index)
            else:
                groups.append(current)
                current = [index]
        groups.append(current)

    columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for block_id, group in enumerate(groups):
        block = vectors[:, group]
        compressed = block.conj().T @ (q_all @ block)
        compressed = 0.5 * (compressed + compressed.conj().T)
        values, rotations = la.eigh(compressed, check_finite=False)
        scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
        keep = np.flatnonzero(values <= DARK_TOL * scale)
        dark = block @ rotations[:, keep] if keep.size else np.zeros((block.shape[0], 0), complex)
        target_weight = float(np.linalg.norm(dark.conj().T @ tower) ** 2) if keep.size else 0.0
        columns.extend(dark[:, column] for column in range(dark.shape[1]))
        rows.append(
            {
                "energy_block_id": int(block_id),
                "energy": float(np.mean(energies[group])),
                "block_dimension": int(len(group)),
                "joint_dark_rank": int(keep.size),
                "minimum_joint_dark_eigenvalue": float(np.min(values)) if values.size else np.nan,
                "maximum_retained_dark_eigenvalue": (
                    float(values[keep].max()) if keep.size else np.nan
                ),
                "target_tower_weight": target_weight,
            }
        )
    dark_basis = (
        orthonormalize_columns(np.column_stack(columns), tolerance=1.0e-9)
        if columns
        else np.zeros((vectors.shape[0], 0), dtype=np.complex128)
    )
    return dark_basis, rows


def _checkpoint_name(*, length: int, kappa_over_j: float, eigenpairs: int) -> str:
    kappa = f"{float(kappa_over_j):+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"spin1_L{int(length)}_kappa_{kappa}_eig{int(eigenpairs)}"


def _checkpoint_candidates(config: Sec6ProvisioningConfig, stem: str) -> Iterable[Path]:
    if config.checkpoint_source_dir is not None:
        yield Path(config.checkpoint_source_dir) / stem
    yield config.resolved_checkpoint_dir / stem


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(tmp, path)


def _save_checkpoint(
    directory: Path,
    *,
    energies: np.ndarray,
    vectors: np.ndarray,
    metadata: dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(directory / "energies.npy", np.asarray(energies, dtype=np.float64))
    _atomic_save_npy(directory / "vectors.npy", np.asarray(vectors, dtype=np.complex128))
    tmp = directory / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, directory / "metadata.json")


def _load_checkpoint(
    directory: Path,
    *,
    expected: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    metadata_path = directory / "metadata.json"
    energies_path = directory / "energies.npy"
    vectors_path = directory / "vectors.npy"
    if not (metadata_path.is_file() and energies_path.is_file() and vectors_path.is_file()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            return None
    energies = np.load(energies_path, mmap_mode="r")
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.shape[1] != energies.shape[0]:
        raise RuntimeError(f"invalid spectral checkpoint at {directory}: shape mismatch")
    return energies, vectors, metadata


def _partial_spectrum(
    h_sector,
    *,
    length: int,
    kappa_over_j: float,
    eigenpairs: int,
    config: Sec6ProvisioningConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dimension = int(h_sector.shape[0])
    requested = min(int(eigenpairs), dimension - 2)
    if requested <= 0:
        raise ValueError("resolved sector is too small for shift-invert")
    stem = _checkpoint_name(length=length, kappa_over_j=kappa_over_j, eigenpairs=requested)
    expected = {
        "schema_version": 1,
        "L": int(length),
        "M": TOTAL_SZ,
        "J3_over_J": float(J3_OVER_J),
        "kappa_over_J": float(kappa_over_j),
        "requested_eigenpairs": int(requested),
        "sector_dimension": dimension,
        "shift": float(config.shift),
        "arpack_tolerance": float(config.arpack_tolerance),
    }
    if config.reuse_checkpoints:
        for candidate in _checkpoint_candidates(config, stem):
            loaded = _load_checkpoint(candidate, expected=expected)
            if loaded is not None:
                energies, vectors, metadata = loaded
                metadata = dict(metadata)
                metadata["checkpoint_reused"] = True
                metadata["checkpoint_path"] = str(candidate)
                return energies, vectors, metadata

    started = time.perf_counter()
    energies, vectors = spla.eigsh(
        h_sector,
        k=requested,
        sigma=float(config.shift),
        which="LM",
        tol=float(config.arpack_tolerance),
    )
    order = np.argsort(energies)
    energies = np.asarray(energies[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    metadata = {
        **expected,
        "returned_eigenpairs": int(energies.size),
        "covered_spectral_half_width": float(min(abs(energies.min()), abs(energies.max()))),
        "solve_seconds": float(time.perf_counter() - started),
        "checkpoint_reused": False,
    }
    # Persist the expensive spectral payload before any covariance, fitting, pandas,
    # or plotting work. This is intentionally uncompressed: the L=14 vector block
    # is large and should be restartable without another week-scale eigensolve.
    if config.write_checkpoints:
        target = config.resolved_checkpoint_dir / stem
        _save_checkpoint(target, energies=energies, vectors=vectors, metadata=metadata)
        metadata["checkpoint_path"] = str(target)
    return energies, vectors, metadata


def _window_specs(
    length: int,
    config: Sec6ProvisioningConfig,
    *,
    dense: bool,
) -> list[tuple[str, float, float, float]]:
    specs: list[tuple[str, float, float, float]] = []
    for half_width in config.fixed_half_widths:
        specs.append((f"fixed_{half_width:g}", float(half_width), 0.0, float(half_width)))
    if config.include_quarter_window:
        specs.append(("L_quarter", float(length) ** 0.25, 0.25, 1.0))
    if dense and config.include_sqrt_window_for_dense:
        specs.append(("L_sqrt", float(length) ** 0.5, 0.5, 1.0))
    dedup: dict[str, tuple[str, float, float, float]] = {}
    for spec in specs:
        dedup[spec[0]] = spec
    return list(dedup.values())


def _window_indices(energies: np.ndarray, half_width: float, tolerance: float) -> np.ndarray:
    window = select_microcanonical_window_by_width(
        energies,
        target_energy=0.0,
        half_width=float(half_width),
        degeneracy_tolerance=float(tolerance),
    )
    return np.asarray(window.indices, dtype=np.int64)


def _window_residuals(
    h_sector,
    energies: np.ndarray,
    vectors: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[float, float]:
    selected = np.asarray(indices, dtype=np.int64)
    residuals: list[np.ndarray] = []
    for start in range(0, selected.size, max(1, int(chunk_size))):
        chunk = selected[start : start + max(1, int(chunk_size))]
        block = np.asarray(vectors[:, chunk])
        action = h_sector @ block
        diff = action - block * np.asarray(energies[chunk])[None, :]
        residuals.append(np.linalg.norm(diff, axis=0))
    if not residuals:
        return np.nan, np.nan
    values = np.concatenate(residuals)
    return float(np.max(values)), float(np.median(values))


def _pair_algebra_operators(configs: np.ndarray, sector):
    """Build the resolved-sector algebra and fixed-M trace coefficients.

    The coarse fixed-M trace only needs the diagonal local matrix element for
    every product configuration. Avoid retaining 19 full fixed-M operators at
    L=14; only the projected (M,k) operators are materialized.
    """
    patterns, names, matrices = charge_conserving_two_site_hermitian_basis()
    pattern_to_index = {tuple(pattern): index for index, pattern in enumerate(patterns)}
    local_indices = np.asarray(
        [pattern_to_index[tuple(map(int, row[:2]))] for row in np.asarray(configs)],
        dtype=np.int64,
    )
    fixed_m_coefficients = []
    sector_ops = []
    for name, matrix in zip(names, matrices, strict=True):
        matrix = np.asarray(matrix, dtype=np.complex128)
        fixed_m_coefficients.append(complex(np.mean(np.diag(matrix)[local_indices])))
        template = LocalWitnessTemplate(
            pattern_key=(),
            local_patterns=patterns,
            local_operator=matrix,
            metadata={"name": name},
        )
        full = template.instantiate((0, 1)).embed(configs)
        sector_ops.append(_project_sparse(full, sector))
    return (
        patterns,
        names,
        matrices,
        np.asarray(fixed_m_coefficients, dtype=np.complex128),
        tuple(sector_ops),
    )


def _rdm_from_coefficients(
    coefficients: Sequence[complex],
    basis_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    rho = np.zeros_like(np.asarray(basis_matrices[0]), dtype=np.complex128)
    for coefficient, matrix in zip(coefficients, basis_matrices, strict=True):
        rho += complex(coefficient) * np.asarray(matrix)
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def _microcanonical_coefficients(
    sector_ops: Sequence[object], vectors: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    block = np.asarray(vectors[:, selected])
    values = []
    for operator in sector_ops:
        action = operator @ block
        diagonal = np.einsum("ij,ij->j", block.conj(), action)
        values.append(complex(np.mean(diagonal)))
    return np.asarray(values, dtype=np.complex128)


def _trace_coefficients(operators: Sequence[object]) -> np.ndarray:
    dimension = int(operators[0].shape[0])
    values = []
    for operator in operators:
        trace = (
            complex(operator.diagonal().sum())
            if sp.issparse(operator)
            else complex(np.trace(operator))
        )
        values.append(trace / dimension)
    return np.asarray(values, dtype=np.complex128)


def _local_witness_matrices(
    patterns: Sequence[tuple[int, int]],
    basis_matrices: Sequence[np.ndarray],
):
    local_configs = np.asarray(patterns, dtype=np.int64)
    output: dict[str, np.ndarray] = {}
    for key, witness in RAW_WITNESSES.items():
        local = witness.embed(local_configs)
        q = np.asarray(
            (local.conj().T @ local).toarray()
            if sp.issparse(local)
            else local.conj().T @ local
        )
        if key in {"A", "Z"}:
            q = q / witness.template.q_operator_norm
        output[key] = np.asarray(q, dtype=np.complex128)
    coefficient_vectors = {
        key: np.asarray(
            [np.trace(matrix.conj().T @ q) for matrix in basis_matrices],
            dtype=np.complex128,
        )
        for key, q in output.items()
    }
    return output, coefficient_vectors


def _centered_direction(coefficients: np.ndarray) -> np.ndarray:
    vector = np.asarray(coefficients, dtype=np.complex128).copy()
    if vector.size:
        vector[0] = 0.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _bridge_diagnostics(
    *,
    length: int,
    kappa_over_j: float,
    window_role: str,
    half_width: float,
    raw_state_count: int,
    mc_coefficients: np.ndarray,
    resolved_coefficients: np.ndarray,
    fixed_m_coefficients: np.ndarray,
    basis_names: Sequence[str],
    basis_matrices: Sequence[np.ndarray],
    witness_matrices: dict[str, np.ndarray],
    witness_coefficient_vectors: dict[str, np.ndarray],
    metadata: dict[str, object],
):
    rho_mc = _rdm_from_coefficients(mc_coefficients, basis_matrices)
    rho_resolved = _rdm_from_coefficients(resolved_coefficients, basis_matrices)
    rho_fixed = _rdm_from_coefficients(fixed_m_coefficients, basis_matrices)

    distance_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    bridge_specs = (
        ("mc_to_beta0_resolved", rho_mc - rho_resolved, mc_coefficients - resolved_coefficients),
        (
            "beta0_resolved_to_fixedM",
            rho_resolved - rho_fixed,
            resolved_coefficients - fixed_m_coefficients,
        ),
    )
    for bridge, delta_rho, delta_coefficients in bridge_specs:
        delta_rho = 0.5 * (delta_rho + delta_rho.conj().T)
        eigenvalues = np.linalg.eigvalsh(delta_rho)
        trace_distance = float(0.5 * np.sum(np.abs(eigenvalues)))
        hs_norm = float(np.linalg.norm(delta_coefficients))
        worst_coefficients = (
            np.asarray(delta_coefficients) / hs_norm
            if hs_norm > 0.0
            else np.zeros_like(delta_coefficients)
        )
        eigvals, eigvecs = np.linalg.eigh(delta_rho)
        helstrom = eigvecs @ np.diag(np.sign(eigvals)) @ eigvecs.conj().T
        helstrom_hs_norm = float(np.linalg.norm(helstrom))
        if helstrom_hs_norm > 0.0:
            helstrom = helstrom / helstrom_hs_norm
        helstrom_coefficients = np.asarray(
            [np.trace(matrix.conj().T @ helstrom) for matrix in basis_matrices],
            dtype=np.complex128,
        )
        leading_index = (
            int(np.argmax(np.abs(worst_coefficients))) if worst_coefficients.size else 0
        )
        witness_differences = {
            key: float(np.trace(delta_rho @ q).real)
            for key, q in witness_matrices.items()
        }
        overlaps = {
            key: float(
                abs(
                    np.vdot(
                        _centered_direction(coeffs),
                        _centered_direction(worst_coefficients),
                    )
                )
            )
            for key, coeffs in witness_coefficient_vectors.items()
        }
        distance_rows.append(
            {
                "L": int(length),
                "M": TOTAL_SZ,
                "kappa_over_J": float(kappa_over_j),
                "window_role": window_role,
                "window_half_width": float(half_width),
                "raw_window_state_count": int(raw_state_count),
                "bridge": bridge,
                "trace_distance": trace_distance,
                "delta_rho_hs_norm": hs_norm,
                "leading_residual_basis_operator": str(basis_names[leading_index]),
                "leading_residual_coefficient": float(abs(worst_coefficients[leading_index])),
                "delta_tau_A": witness_differences["A"],
                "delta_tau_Z": witness_differences["Z"],
                "delta_tau_Y": witness_differences["Y"],
                "abs_delta_tau_A": abs(witness_differences["A"]),
                "abs_delta_tau_Z": abs(witness_differences["Z"]),
                "abs_delta_tau_Y": abs(witness_differences["Y"]),
                "overlap_with_A_direction": overlaps["A"],
                "overlap_with_Z_direction": overlaps["Z"],
                "overlap_with_Y_direction": overlaps["Y"],
                **metadata,
            }
        )
        for index, eigenvalue in enumerate(eigenvalues):
            spectrum_rows.append(
                {
                    "L": int(length),
                    "kappa_over_J": float(kappa_over_j),
                    "window_role": window_role,
                    "bridge": bridge,
                    "eigen_index": int(index),
                    "delta_rho_eigenvalue": float(eigenvalue),
                    "absolute_eigenvalue": float(abs(eigenvalue)),
                    "trace_distance": trace_distance,
                }
            )
        for name, delta_coefficient, worst_coefficient, helstrom_coefficient in zip(
            basis_names,
            delta_coefficients,
            worst_coefficients,
            helstrom_coefficients,
            strict=True,
        ):
            coefficient_rows.append(
                {
                    "L": int(length),
                    "kappa_over_J": float(kappa_over_j),
                    "window_role": window_role,
                    "bridge": bridge,
                    "basis_operator": str(name),
                    "delta_rho_coefficient_real": float(complex(delta_coefficient).real),
                    "delta_rho_coefficient_imag": float(complex(delta_coefficient).imag),
                    "worst_hs_operator_coefficient_real": float(complex(worst_coefficient).real),
                    "worst_hs_operator_coefficient_imag": float(complex(worst_coefficient).imag),
                    "helstrom_hs_operator_coefficient_real": float(
                        complex(helstrom_coefficient).real
                    ),
                    "helstrom_hs_operator_coefficient_imag": float(
                        complex(helstrom_coefficient).imag
                    ),
                }
            )
    return distance_rows, spectrum_rows, coefficient_rows


def _point_context(*, length: int, kappa_over_j: float):
    build = _deformed_model(length=length, kappa_over_j=kappa_over_j).build(
        builder="optimized", basis_solver="dfs", sort_basis=True
    )
    configs = basis_configs_from_build_result(build)
    tower = _tower_state(configs, length=length)
    sector, momentum_index = _tower_translation_sector(configs, length=length)
    tower_sector = project_state_to_sector(tower, sector)
    tower_sector = np.asarray(tower_sector, dtype=np.complex128)
    tower_sector /= np.linalg.norm(tower_sector)
    h_sector = _project_sparse(build.hamiltonian, sector)
    q_ops = _normalized_witness_q_ops(configs, sector)
    pair = _pair_algebra_operators(configs, sector)
    return {
        "build": build,
        "configs": configs,
        "tower": tower_sector,
        "sector": sector,
        "momentum_index": momentum_index,
        "h_sector": h_sector,
        "q_ops": q_ops,
        "pair": pair,
    }


def _copy_baseline_files(source: Path | None, output: Path) -> None:
    if source is None or not Path(source).is_dir():
        return
    names = (
        "spin1_xy_kappa0p1_sequence.csv",
        "spin1_xy_kappa0p1_eth_scatter_Lmax.csv",
        "spin1_xy_kappa0p1_eth_scatter_all_sizes.csv",
        "spin1_xy_kappa0p1_beta0_overlap.csv",
        "spin1_xy_kappa_matching_grid.csv",
        "spin1_xy_kappa_concentration_grid.csv",
        "spin1_xy_kappa_worst_eigenoperator.csv",
        "exact_fixed_M_activities.csv",
        "spin1_xy_complex_t2_obstruction_grid.csv",
    )
    for name in names:
        src = Path(source) / name
        dst = output / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def _derive_convergence_flags(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    derived = frame.copy()
    if derived.empty:
        return derived
    if "converged_vs_previous" not in derived:
        derived["converged_vs_previous"] = False
    if "requested_eigenpairs" not in derived:
        return derived
    grouping = [column for column in ("L", "window_role", "window_half_width") if column in derived]
    if not grouping:
        grouping = ["L"] if "L" in derived else []
    metrics = [
        column
        for column in (
            "tau_A_mc_clean",
            "tau_Z_mc_clean",
            "tau_Y_mc_clean",
            "delta_max_clean_clean",
        )
        if column in derived
    ]
    if not metrics:
        return derived
    groups = (
        [((), derived)]
        if not grouping
        else derived.groupby(grouping, dropna=False, sort=False)
    )
    for _, group in groups:
        ordered = group.sort_values("requested_eigenpairs")
        previous = None
        for index, row in ordered.iterrows():
            if previous is None:
                previous = row
                continue
            changes = [abs(float(row[column]) - float(previous[column])) for column in metrics]
            derived.loc[index, "converged_vs_previous"] = bool(
                max(changes, default=np.inf) <= tolerance
            )
            for column, change in zip(metrics, changes, strict=True):
                derived.loc[index, f"budget_change_{column}"] = float(change)
            previous = row
    return derived


def repair_sparse_convergence_table(config: Sec6ProvisioningConfig) -> pd.DataFrame:
    source_dir = config.sparse_convergence_data_dir
    if source_dir is None:
        return pd.DataFrame()
    source = Path(source_dir) / "spin1_xy_kappa0p1_L14_sparse_convergence.csv"
    if not source.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    frame = _derive_convergence_flags(frame, config.convergence_tolerance)
    frame.to_csv(config.output_dir / source.name, index=False)
    return frame


def _sparse_budget_certification(frame: pd.DataFrame) -> tuple[bool, str]:
    """Return whether the latest cross-budget row passes in every compared safe window."""
    if frame.empty or "converged_vs_previous" not in frame or "requested_eigenpairs" not in frame:
        return False, "missing_sparse_convergence_audit"
    grouping = [column for column in ("L", "window_role", "window_half_width") if column in frame]
    if not grouping:
        grouping = ["L"] if "L" in frame else []
    if grouping:
        latest = (
            frame.sort_values("requested_eigenpairs")
            .groupby(grouping, dropna=False, sort=False)
            .tail(1)
        )
    else:
        latest = frame.sort_values("requested_eigenpairs").tail(1)
    if latest.empty:
        return False, "no_cross_budget_comparison"
    passed = bool(latest["converged_vs_previous"].fillna(False).astype(bool).all())
    return passed, "representative_kappa_0p1_cross_budget_audit"


def _concentration_at_point(
    *,
    length: int,
    kappa_over_j: float,
    energies: np.ndarray,
    vectors: np.ndarray,
    context: dict[str, object],
    config: Sec6ProvisioningConfig,
    sparse_metadata: dict[str, object],
    sparse_convergence_passed: bool,
    budget_certification_source: str,
):
    half_width = float(config.concentration_half_width)
    coverage = float(min(abs(float(np.min(energies))), abs(float(np.max(energies)))))
    if half_width > coverage + config.energy_block_tolerance:
        raise RuntimeError(
            "requested concentration window "
            f"DeltaE={half_width:g} exceeds sparse coverage {coverage:g}"
        )
    indices = _window_indices(energies, half_width, config.energy_block_tolerance)
    max_residual, median_residual = _window_residuals(
        context["h_sector"], energies, vectors, indices, chunk_size=config.residual_chunk_size
    )
    q_all = _translated_joint_dark_operator(
        configs=context["configs"], sector=context["sector"], length=length
    )
    exceptional, dark_rows = _joint_dark_kernel_from_spectrum(
        energies=energies,
        vectors=vectors,
        q_all=q_all,
        tower=context["tower"],
        energy_tolerance=config.energy_block_tolerance,
    )
    empty = np.zeros((context["sector"].sector_dimension, 0), dtype=np.complex128)
    sector_ops = context["pair"][4]
    raw = projector_deleted_block_covariance(
        energies,
        vectors,
        empty,
        sector_ops,
        indices,
        energy_tolerance=config.energy_block_tolerance,
        vector_tolerance=1.0e-9,
    )
    clean = projector_deleted_block_covariance(
        energies,
        vectors,
        exceptional,
        sector_ops,
        indices,
        energy_tolerance=config.energy_block_tolerance,
        vector_tolerance=1.0e-9,
    )
    rows = []
    for variant, covariance in (("raw", raw), ("clean", clean)):
        rows.append(
            {
                "L": int(length),
                "M": TOTAL_SZ,
                "J3_over_J": float(J3_OVER_J),
                "kappa_over_J": float(kappa_over_j),
                "variant": variant,
                "window_role": "sparse_safe_fixed_width",
                "window_half_width": half_width,
                "window_state_count": int(raw["window_rank"]),
                "retained_state_count": int(clean["retained_rank"]),
                "removed_projector_rank": int(clean["exceptional_rank"]),
                "removed_fraction": float(clean["removed_fraction"]),
                "log_window_state_count_over_L": float(
                    math.log(max(1, raw["window_rank"])) / length
                ),
                "largest_covariance_eigenvalue": float(covariance["largest_eigenvalue"]),
                "largest_covariance_width": float(covariance["largest_width"]),
                "w_L": float(covariance["largest_width"]),
                "w_14": float(covariance["largest_width"]) if int(length) == 14 else np.nan,
                "median_nonidentity_width": float(covariance["median_nonidentity_width"]),
                "energy_block_count": int(covariance["energy_block_count"]),
                "covered_spectral_half_width": coverage,
                "window_max_eigenpair_residual": max_residual,
                "window_median_eigenpair_residual": median_residual,
                "joint_dark_rank": int(exceptional.shape[1]),
                "tower_residual": float(
                    diagnose_eigenpair(
                        context["h_sector"], context["tower"]
                    ).residual_norm
                ),
                "requested_eigenpairs": int(sparse_metadata["requested_eigenpairs"]),
                "checkpoint_reused": bool(sparse_metadata.get("checkpoint_reused", False)),
                "checkpoint_path": sparse_metadata.get("checkpoint_path", ""),
                "sparse_convergence_passed": bool(sparse_convergence_passed),
                "budget_certification_source": budget_certification_source,
            }
        )
    worst_rows = []
    for variant, covariance in (("raw", raw), ("clean", clean)):
        for name, coefficient in zip(
            context["pair"][1], covariance["worst_coefficients"], strict=True
        ):
            worst_rows.append(
                {
                    "L": int(length),
                    "kappa_over_J": float(kappa_over_j),
                    "variant": variant,
                    "basis_operator": name,
                    "coefficient": float(np.real(coefficient)),
                    "window_half_width": half_width,
                }
            )
    tolerance_rows = []
    for tolerance in config.energy_block_tolerance_audit:
        for variant, exceptional_basis in (("raw", empty), ("clean", exceptional)):
            covariance = projector_deleted_block_covariance(
                energies,
                vectors,
                exceptional_basis,
                sector_ops,
                indices,
                energy_tolerance=float(tolerance),
                vector_tolerance=1.0e-9,
            )
            tolerance_rows.append(
                {
                    "L": int(length),
                    "kappa_over_J": float(kappa_over_j),
                    "variant": variant,
                    "energy_block_tolerance": float(tolerance),
                    "energy_block_count": int(covariance["energy_block_count"]),
                    "largest_covariance_eigenvalue": float(covariance["largest_eigenvalue"]),
                    "largest_covariance_width": float(covariance["largest_width"]),
                }
            )
    return rows, worst_rows, tolerance_rows, dark_rows, exceptional


def _matching_at_point(
    *,
    length: int,
    kappa_over_j: float,
    half_width: float,
    energies: np.ndarray,
    vectors: np.ndarray,
    context: dict[str, object],
    exceptional: np.ndarray,
    config: Sec6ProvisioningConfig,
    sparse_metadata: dict[str, object],
    sparse_convergence_passed: bool,
    budget_certification_source: str,
) -> dict[str, object]:
    coverage = float(min(abs(float(np.min(energies))), abs(float(np.max(energies)))))
    if half_width > coverage + config.energy_block_tolerance:
        raise RuntimeError("matching window is not contained in sparse spectrum")
    indices = _window_indices(energies, half_width, config.energy_block_tolerance)
    split = projector_deleted_basis(np.asarray(vectors[:, indices]), exceptional, tolerance=1.0e-9)
    resolved_beta0 = {
        name: _cleaned_trace_average(operator, exceptional)
        for name, operator in context["q_ops"].items()
    }
    raw = {
        name: spectral_observable_moments(
            operator,
            vectors,
            squared_operator=operator,
            indices=indices,
        ).mean
        for name, operator in context["q_ops"].items()
    }
    clean = {
        name: projector_deleted_observable_moments(
            np.asarray(vectors[:, indices]),
            exceptional,
            operator,
            squared_operator=operator,
            tolerance=1.0e-9,
        )["mean"]
        for name, operator in context["q_ops"].items()
    }
    raw_deltas = {name: abs(float(raw[name]) - float(resolved_beta0[name]["raw"])) for name in raw}
    clean_deltas = {
        name: abs(float(clean[name]) - float(resolved_beta0[name]["clean"]))
        for name in clean
    }
    max_residual, median_residual = _window_residuals(
        context["h_sector"], energies, vectors, indices, chunk_size=config.residual_chunk_size
    )
    return {
        "L": int(length),
        "M": TOTAL_SZ,
        "J3_over_J": float(J3_OVER_J),
        "kappa_over_J": float(kappa_over_j),
        "window_role": "sparse_safe_fixed_width",
        "grid_role": "sparse_safe_fixed_width",
        "window_half_width": float(half_width),
        "window_coverage_complete": True,
        "window_state_count": int(indices.size),
        "retained_state_count": int(split["retained_rank"]),
        "removed_projector_rank": int(split["exceptional_rank"]),
        "removed_fraction": float(split["removed_fraction"]),
        "covered_spectral_half_width": coverage,
        "log_window_state_count_over_L": float(math.log(max(1, indices.size)) / length),
        "window_max_eigenpair_residual": max_residual,
        "window_median_eigenpair_residual": median_residual,
        **{f"tau_{key}_mc_raw": float(raw[key]) for key in raw},
        **{f"tau_{key}_mc_clean": float(clean[key]) for key in clean},
        **{
            f"tau_{key}_resolved_beta0_raw": float(resolved_beta0[key]["raw"])
            for key in resolved_beta0
        },
        **{
            f"tau_{key}_resolved_beta0_clean": float(resolved_beta0[key]["clean"])
            for key in resolved_beta0
        },
        **{f"delta_{key}_raw_raw": float(raw_deltas[key]) for key in raw_deltas},
        **{f"delta_{key}_clean_clean": float(clean_deltas[key]) for key in clean_deltas},
        "delta_max_raw_raw": float(max(raw_deltas.values())),
        "delta_max_clean_clean": float(max(clean_deltas.values())),
        "sparse_convergence_passed": bool(sparse_convergence_passed),
        "budget_certification_source": budget_certification_source,
        "requested_eigenpairs": int(sparse_metadata["requested_eigenpairs"]),
        "checkpoint_reused": bool(sparse_metadata.get("checkpoint_reused", False)),
        "checkpoint_path": sparse_metadata.get("checkpoint_path", ""),
    }


def _run_two_bridge_for_point(
    *,
    length: int,
    kappa_over_j: float,
    energies: np.ndarray,
    vectors: np.ndarray,
    context: dict[str, object],
    config: Sec6ProvisioningConfig,
    dense: bool,
    sparse_metadata: dict[str, object] | None = None,
):
    patterns, names, matrices, fixed_m_coefficients, sector_ops = context["pair"]
    resolved_coefficients = _trace_coefficients(sector_ops)
    witness_matrices, witness_coefficient_vectors = _local_witness_matrices(patterns, matrices)
    distance_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    microcanonical_rows: list[dict[str, object]] = []
    coverage = float(min(abs(float(np.min(energies))), abs(float(np.max(energies)))))
    for role, half_width, exponent, prefactor in _window_specs(length, config, dense=dense):
        if not dense and half_width > coverage + config.energy_block_tolerance:
            continue
        indices = _window_indices(energies, half_width, config.energy_block_tolerance)
        mc_coefficients = _microcanonical_coefficients(sector_ops, vectors, indices)
        residual_max = residual_median = np.nan
        if not dense:
            residual_max, residual_median = _window_residuals(
                context["h_sector"],
                energies,
                vectors,
                indices,
                chunk_size=config.residual_chunk_size,
            )
        metadata = {
            "resolved_sector_dimension": int(context["sector"].sector_dimension),
            "fixed_M_sector_dimension": int(context["configs"].shape[0]),
            "window_exponent": float(exponent),
            "window_prefactor": float(prefactor),
            "spectrum_method": "full_dense_eigh" if dense else "sparse_shift_invert",
            "covered_spectral_half_width": coverage,
            "window_max_eigenpair_residual": residual_max,
            "window_median_eigenpair_residual": residual_median,
        }
        if sparse_metadata is not None:
            metadata.update(
                {
                    "requested_eigenpairs": int(sparse_metadata["requested_eigenpairs"]),
                    "checkpoint_reused": bool(sparse_metadata.get("checkpoint_reused", False)),
                }
            )
        d_rows, s_rows, c_rows = _bridge_diagnostics(
            length=length,
            kappa_over_j=kappa_over_j,
            window_role=role,
            half_width=half_width,
            raw_state_count=int(indices.size),
            mc_coefficients=mc_coefficients,
            resolved_coefficients=resolved_coefficients,
            fixed_m_coefficients=fixed_m_coefficients,
            basis_names=names,
            basis_matrices=matrices,
            witness_matrices=witness_matrices,
            witness_coefficient_vectors=witness_coefficient_vectors,
            metadata=metadata,
        )
        distance_rows.extend(d_rows)
        spectrum_rows.extend(s_rows)
        coefficient_rows.extend(c_rows)
        rho_mc = _rdm_from_coefficients(mc_coefficients, matrices)
        microcanonical_rows.append(
            {
                "L": int(length),
                "M": TOTAL_SZ,
                "kappa_over_J": float(kappa_over_j),
                "window_role": role,
                "window_half_width": float(half_width),
                "window_exponent": float(exponent),
                "window_prefactor": float(prefactor),
                "window_state_count": int(indices.size),
                "log_window_state_count_over_L": float(math.log(max(1, indices.size)) / length),
                "tau_A_mc_raw": float(np.trace(rho_mc @ witness_matrices["A"]).real),
                "tau_Z_mc_raw": float(np.trace(rho_mc @ witness_matrices["Z"]).real),
                "tau_Y_mc_raw": float(np.trace(rho_mc @ witness_matrices["Y"]).real),
                **metadata,
            }
        )
    return distance_rows, spectrum_rows, coefficient_rows, microcanonical_rows


def _fit_linear_models(
    frame: pd.DataFrame,
    *,
    quantity: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty or quantity not in frame:
        return pd.DataFrame()
    for key, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        group = group.sort_values("L")
        lengths = group["L"].to_numpy(dtype=float)
        values = group[quantity].to_numpy(dtype=float)
        if lengths.size < 2:
            continue
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(group_columns, key_values, strict=True))
        for model in ("c/L", "delta_inf+c/L"):
            design = (
                (1.0 / lengths)[:, None]
                if model == "c/L"
                else np.column_stack((np.ones_like(lengths), 1.0 / lengths))
            )
            params, *_ = np.linalg.lstsq(design, values, rcond=None)
            prediction = design @ params
            row = {
                **labels,
                "quantity": quantity,
                "model": model,
                "included_sizes": ",".join(str(int(x)) for x in lengths),
                "n_sizes": int(lengths.size),
                "rmse": float(np.sqrt(np.mean((values - prediction) ** 2))),
                "delta_inf": 0.0 if model == "c/L" else float(params[0]),
                "c": float(params[-1]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _update_concentration_sequence(config: Sec6ProvisioningConfig, l14_rows: pd.DataFrame) -> None:
    frames = []
    baseline = (
        None
        if config.baseline_data_dir is None
        else Path(config.baseline_data_dir) / "spin1_xy_kappa0p1_concentration.csv"
    )
    if baseline is not None and baseline.is_file():
        frames.append(pd.read_csv(baseline))
    if not l14_rows.empty:
        raw = l14_rows[l14_rows["variant"] == "raw"].copy()
        frames.append(raw)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [
        column for column in ("L", "window_half_width") if column in combined
    ]
    dedup_columns = [column for column in ("L", "kappa_over_J") if column in combined]
    combined = combined.sort_values(sort_columns).drop_duplicates(
        dedup_columns, keep="last"
    )
    combined.to_csv(config.output_dir / "spin1_xy_kappa0p1_concentration.csv", index=False)
    if "largest_covariance_width" in combined:
        fit_frame = combined[
            (combined["L"] >= 8)
            & np.isfinite(combined["largest_covariance_width"])
        ].copy()
        if len(fit_frame) >= 2:
            lengths = fit_frame["L"].to_numpy(dtype=float)
            widths = fit_frame["largest_covariance_width"].to_numpy(dtype=float)
            log_design = np.column_stack((np.ones_like(lengths), np.log(lengths)))
            params, *_ = np.linalg.lstsq(log_design, np.log(widths), rcond=None)
            pd.DataFrame(
                [
                    {
                        "model": "a*L^p",
                        "included_sizes": ",".join(str(int(x)) for x in lengths),
                        "a": float(np.exp(params[0])),
                        "p": float(params[1]),
                        "rmse_log": float(
                            np.sqrt(
                                np.mean((np.log(widths) - log_design @ params) ** 2)
                            )
                        ),
                    }
                ]
            ).to_csv(config.output_dir / "spin1_xy_kappa0p1_concentration_fit.csv", index=False)


def _build_panel_b_sequence(
    config: Sec6ProvisioningConfig,
    matching_row: dict[str, object] | None,
) -> None:
    source = None
    if config.baseline_data_dir is not None:
        for name in (
            "spin1_xy_kappa0p1_panel_b_sequence.csv",
            "spin1_xy_kappa0p1_beta0_overlap.csv",
        ):
            candidate = Path(config.baseline_data_dir) / name
            if candidate.is_file():
                source = candidate
                break
    frames = [pd.read_csv(source)] if source is not None else []
    if matching_row is not None:
        row = dict(matching_row)
        row["window_role"] = "sparse_safe_fixed_width"
        frames.append(pd.DataFrame([row]))
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False).sort_values("L")
        combined.to_csv(config.output_dir / "spin1_xy_kappa0p1_panel_b_sequence.csv", index=False)


def run_sec6_provisioning(config: Sec6ProvisioningConfig) -> dict[str, pd.DataFrame]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _copy_baseline_files(config.baseline_data_dir, output)
    repaired_convergence = repair_sparse_convergence_table(config)
    representative_budget_passed, representative_budget_source = _sparse_budget_certification(
        repaired_convergence
    )

    bridge_distance_rows: list[dict[str, object]] = []
    residual_spectrum_rows: list[dict[str, object]] = []
    residual_coefficient_rows: list[dict[str, object]] = []
    microcanonical_rows: list[dict[str, object]] = []

    # Dense sizes are cheap enough to recompute and make the two local bridges
    # independent of whatever subset of operator moments an older evidence run saved.
    for length in config.dense_sizes:
        context = _point_context(
            length=int(length),
            kappa_over_j=config.representative_kappa_over_j,
        )
        energies, vectors = la.eigh(context["h_sector"].toarray(), check_finite=False)
        d_rows, s_rows, c_rows, m_rows = _run_two_bridge_for_point(
            length=int(length),
            kappa_over_j=config.representative_kappa_over_j,
            energies=energies,
            vectors=vectors,
            context=context,
            config=config,
            dense=True,
        )
        bridge_distance_rows.extend(d_rows)
        residual_spectrum_rows.extend(s_rows)
        residual_coefficient_rows.extend(c_rows)
        microcanonical_rows.extend(m_rows)

    representative_concentration_rows: list[dict[str, object]] = []
    representative_worst_rows: list[dict[str, object]] = []
    representative_tolerance_rows: list[dict[str, object]] = []
    representative_matching = None
    large_context = None
    large_energies = large_vectors = large_sparse_metadata = None

    if config.run_large_representative:
        large_context = _point_context(
            length=config.large_size,
            kappa_over_j=config.representative_kappa_over_j,
        )
        large_energies, large_vectors, large_sparse_metadata = _partial_spectrum(
            large_context["h_sector"],
            length=config.large_size,
            kappa_over_j=config.representative_kappa_over_j,
            eigenpairs=config.representative_eigenpairs,
            config=config,
        )
        (
            concentration_rows,
            worst_rows,
            tolerance_rows,
            dark_rows,
            exceptional,
        ) = _concentration_at_point(
            length=config.large_size,
            kappa_over_j=config.representative_kappa_over_j,
            energies=large_energies,
            vectors=large_vectors,
            context=large_context,
            config=config,
            sparse_metadata=large_sparse_metadata,
            sparse_convergence_passed=representative_budget_passed,
            budget_certification_source=representative_budget_source,
        )
        representative_concentration_rows.extend(concentration_rows)
        representative_worst_rows.extend(worst_rows)
        representative_tolerance_rows.extend(tolerance_rows)
        representative_matching = _matching_at_point(
            length=config.large_size,
            kappa_over_j=config.representative_kappa_over_j,
            half_width=config.concentration_half_width,
            energies=large_energies,
            vectors=large_vectors,
            context=large_context,
            exceptional=exceptional,
            config=config,
            sparse_metadata=large_sparse_metadata,
            sparse_convergence_passed=representative_budget_passed,
            budget_certification_source=representative_budget_source,
        )
        d_rows, s_rows, c_rows, m_rows = _run_two_bridge_for_point(
            length=config.large_size,
            kappa_over_j=config.representative_kappa_over_j,
            energies=large_energies,
            vectors=large_vectors,
            context=large_context,
            config=config,
            dense=False,
            sparse_metadata=large_sparse_metadata,
        )
        bridge_distance_rows.extend(d_rows)
        residual_spectrum_rows.extend(s_rows)
        residual_coefficient_rows.extend(c_rows)
        microcanonical_rows.extend(m_rows)
        pd.DataFrame(dark_rows).to_csv(
            output / "spin1_xy_kappa0p1_L14_joint_dark_blocks.csv", index=False
        )

    concentration_df = pd.DataFrame(representative_concentration_rows)
    worst_df = pd.DataFrame(representative_worst_rows)
    tolerance_df = pd.DataFrame(representative_tolerance_rows)
    if not concentration_df.empty:
        concentration_df.to_csv(
            output / "spin1_xy_kappa0p1_concentration_L14.csv", index=False
        )
    if not worst_df.empty:
        worst_df.to_csv(
            output / "spin1_xy_kappa0p1_worst_eigenoperator_L14.csv", index=False
        )
    if not tolerance_df.empty:
        tolerance_df.to_csv(
            output / "spin1_xy_kappa0p1_concentration_L14_tolerance_audit.csv",
            index=False,
        )
    _update_concentration_sequence(config, concentration_df)
    _build_panel_b_sequence(config, representative_matching)

    family_matching_rows: list[dict[str, object]] = []
    family_concentration_rows: list[dict[str, object]] = []
    if config.run_family_large_size:
        family_context = _point_context(
            length=config.large_size, kappa_over_j=config.family_kappa_over_j
        )
        family_energies, family_vectors, family_metadata = _partial_spectrum(
            family_context["h_sector"],
            length=config.large_size,
            kappa_over_j=config.family_kappa_over_j,
            eigenpairs=config.family_eigenpairs,
            config=config,
        )
        (
            family_c_rows,
            _family_worst,
            _family_tol,
            family_dark_rows,
            family_exceptional,
        ) = _concentration_at_point(
            length=config.large_size,
            kappa_over_j=config.family_kappa_over_j,
            energies=family_energies,
            vectors=family_vectors,
            context=family_context,
            config=config,
            sparse_metadata=family_metadata,
            sparse_convergence_passed=representative_budget_passed,
            budget_certification_source=(
                "representative_kappa_0p1_budget_transfer"
                if representative_budget_passed
                else representative_budget_source
            ),
        )
        family_concentration_rows.extend(family_c_rows)
        family_matching_rows.append(
            _matching_at_point(
                length=config.large_size,
                kappa_over_j=config.family_kappa_over_j,
                half_width=config.concentration_half_width,
                energies=family_energies,
                vectors=family_vectors,
                context=family_context,
                exceptional=family_exceptional,
                config=config,
                sparse_metadata=family_metadata,
                sparse_convergence_passed=representative_budget_passed,
                budget_certification_source=(
                    "representative_kappa_0p1_budget_transfer"
                    if representative_budget_passed
                    else representative_budget_source
                ),
            )
        )
        pd.DataFrame(family_dark_rows).to_csv(
            output / "spin1_xy_large_size_family_joint_dark_blocks.csv", index=False
        )

    family_matching_df = pd.DataFrame(family_matching_rows)
    large_size_matching_records = []
    if representative_matching is not None:
        large_size_matching_records.append(representative_matching)
    large_size_matching_records.extend(family_matching_rows)
    large_size_matching_df = pd.DataFrame(large_size_matching_records)
    family_concentration_variants_df = pd.DataFrame(family_concentration_rows)
    family_concentration_records = []
    if not family_concentration_variants_df.empty:
        for (length, kappa), frame in family_concentration_variants_df.groupby(
            ["L", "kappa_over_J"], sort=True
        ):
            raw = frame[frame["variant"] == "raw"]
            clean = frame[frame["variant"] == "clean"]
            if raw.empty:
                continue
            record = raw.iloc[-1].to_dict()
            if not clean.empty:
                clean_row = clean.iloc[-1]
                record["largest_covariance_eigenvalue_clean"] = float(
                    clean_row["largest_covariance_eigenvalue"]
                )
                record["largest_covariance_width_clean"] = float(
                    clean_row["largest_covariance_width"]
                )
                record["median_nonidentity_width_clean"] = float(
                    clean_row["median_nonidentity_width"]
                )
            record["variant"] = "raw_primary_with_clean_companion"
            family_concentration_records.append(record)
    family_concentration_df = pd.DataFrame(family_concentration_records)
    if not large_size_matching_df.empty:
        large_size_matching_df.to_csv(
            output / "spin1_xy_large_size_family_check.csv", index=False
        )
        large_size_matching_df.to_csv(
            output / "spin1_xy_kappa_matching_large_size_safe_window.csv", index=False
        )
    if not family_concentration_df.empty:
        family_concentration_df.to_csv(
            output / "spin1_xy_large_size_family_concentration.csv", index=False
        )

    envelope_matching_records = large_size_matching_records
    envelope_concentration_frames = []
    if not concentration_df.empty:
        envelope_concentration_frames.append(
            concentration_df[concentration_df["variant"] == "raw"].copy()
        )
    if not family_concentration_df.empty:
        envelope_concentration_frames.append(family_concentration_df.copy())
    if envelope_matching_records:
        matching_envelope_frame = pd.DataFrame(envelope_matching_records)
        concentration_envelope_frame = (
            pd.concat(envelope_concentration_frames, ignore_index=True, sort=False)
            if envelope_concentration_frames
            else pd.DataFrame()
        )
        envelope_row = {
            "L": int(config.large_size),
            "window_role": "sparse_safe_fixed_width",
            "sampled_kappa_values": ",".join(
                f"{value:g}" for value in sorted(matching_envelope_frame["kappa_over_J"].unique())
            ),
            "maximum_matching_distance_raw_raw": float(
                matching_envelope_frame["delta_max_raw_raw"].max()
            ),
            "maximum_matching_distance_clean_clean": float(
                matching_envelope_frame["delta_max_clean_clean"].max()
            ),
            "maximum_concentration_width_raw": (
                float(concentration_envelope_frame["largest_covariance_width"].max())
                if not concentration_envelope_frame.empty
                else np.nan
            ),
            "maximum_concentration_width_clean": (
                float(concentration_envelope_frame["largest_covariance_width_clean"].max())
                if (
                    not concentration_envelope_frame.empty
                    and "largest_covariance_width_clean" in concentration_envelope_frame
                )
                else np.nan
            ),
        }
        pd.DataFrame([envelope_row]).to_csv(
            output / "spin1_xy_kappa_uniform_envelope_large_size_safe_window.csv", index=False
        )

    bridge_df = pd.DataFrame(bridge_distance_rows)
    spectrum_df = pd.DataFrame(residual_spectrum_rows)
    coefficients_df = pd.DataFrame(residual_coefficient_rows)
    microcanonical_df = pd.DataFrame(microcanonical_rows)
    bridge_df.to_csv(output / "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv", index=False)
    spectrum_df.to_csv(output / "spin1_xy_kappa0p1_residual_operator_spectrum.csv", index=False)
    coefficients_df.to_csv(
        output / "spin1_xy_kappa0p1_residual_operator_coefficients.csv", index=False
    )
    microcanonical_df.to_csv(
        output / "spin1_xy_kappa0p1_microcanonical_windows_sec6.csv", index=False
    )

    fit_frames = []
    for quantity in ("tau_A_mc_raw", "tau_Z_mc_raw", "tau_Y_mc_raw"):
        fit = _fit_linear_models(
            microcanonical_df, quantity=quantity, group_columns=("window_role",)
        )
        if not fit.empty:
            fit_frames.append(fit)
    micro_fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    micro_fit_df.to_csv(output / "spin1_xy_kappa0p1_microcanonical_extrapolation.csv", index=False)

    bridge_fit_frames = []
    for bridge, frame in bridge_df.groupby("bridge", sort=True) if not bridge_df.empty else []:
        fit = _fit_linear_models(frame, quantity="trace_distance", group_columns=("window_role",))
        if not fit.empty:
            fit["bridge"] = bridge
            bridge_fit_frames.append(fit)
    bridge_fit_df = (
        pd.concat(bridge_fit_frames, ignore_index=True)
        if bridge_fit_frames
        else pd.DataFrame()
    )
    bridge_fit_df.to_csv(output / "spin1_xy_kappa0p1_two_bridge_rdm_distance_fit.csv", index=False)

    summary = {
        "schema_version": 1,
        "representative_kappa_over_J": config.representative_kappa_over_j,
        "family_kappa_over_J": config.family_kappa_over_j,
        "dense_sizes": list(config.dense_sizes),
        "large_size": config.large_size,
        "representative_eigenpairs": config.representative_eigenpairs,
        "family_eigenpairs": config.family_eigenpairs,
        "fixed_half_widths": list(config.fixed_half_widths),
        "include_quarter_window": config.include_quarter_window,
        "run_large_representative": config.run_large_representative,
        "run_family_large_size": config.run_family_large_size,
        "checkpoint_dir": str(config.resolved_checkpoint_dir),
        "repaired_sparse_convergence_rows": int(len(repaired_convergence)),
        "representative_sparse_budget_certified": bool(representative_budget_passed),
        "representative_budget_certification_source": representative_budget_source,
        "two_bridge_rows": int(len(bridge_df)),
        "representative_concentration_rows": int(len(concentration_df)),
        "family_matching_rows": int(len(family_matching_df)),
    }
    (output / "spin1_xy_sec6_provisioning_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "sparse_convergence": repaired_convergence,
        "concentration_L14": concentration_df,
        "bridge_distances": bridge_df,
        "residual_spectrum": spectrum_df,
        "residual_coefficients": coefficients_df,
        "microcanonical_windows": microcanonical_df,
        "microcanonical_fits": micro_fit_df,
        "bridge_fits": bridge_fit_df,
        "family_matching": family_matching_df,
        "large_size_matching": large_size_matching_df,
        "family_concentration": family_concentration_df,
    }
