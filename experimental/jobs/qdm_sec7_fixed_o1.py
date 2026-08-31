"""Shared Sec. VII fixed-window and target-block helpers.

This module extracts only the numerical primitives needed by the post-PRIMME
handoff. It deliberately does not own the old broad O(sqrt(V)) window lane.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from qdm_checkerboard_large_strip import (
    binary_basis_configs_uint8,
    materialize_periodic_product_state_from_basis,
    packed_binary_basis_index,
    process_peak_rss_gib,
    project_sparse_operator_to_sector,
    translation_permutation_from_binary_basis,
)
from qdm_checkerboard_symmetry import checkerboard_fully_resolved_sector
from scipy.optimize import brentq

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging import (
    SquareQDMPeriodicProductUnitCell,
    SquareQDMWitnessPlacement,
    certify_square_qdm_periodic_product_sequence,
    evaluate_square_qdm_environment_witnesses_on_strips,
)
from qlinks.caging.analysis import EnvironmentReductionConfig, diagnose_cage_environment_reduction
from qlinks.caging.analysis.spectral import project_state_to_sector
from qlinks.caging.analysis.thermodynamic import (
    LocalWitnessTemplate,
    directed_transition_witness_template,
)
from qlinks.caging.local_search import (
    LocalQDMCageSearchConfig,
    RobustQDMLocalCageSearchConfig,
    robust_qdm_local_cage_search,
)
from qlinks.models import SquareQDMModel
from qlinks.models.couplings import peierls_plaquette_coupling

TOL = 1.0e-10
RANK_TOL = 1.0e-9
DARK_TOL = 1.0e-9
ENERGY_BLOCK_TOL = 1.0e-9
REPRESENTATIVE_PHASE = 0.05
PILOT_HALF_WIDTHS = (0.10, 0.20, 0.25, 0.50)
TARGET_LX = 12
TARGET_ENERGY = 12.0
EXPECTED_SECTOR_DIMENSIONS = {4: 15, 8: 1125, 12: 114483}


@dataclass(frozen=True, slots=True)
class ReferenceGeometry:
    product_unit_cell: Any
    a_placement: SquareQDMWitnessPlacement
    z_placement: SquareQDMWitnessPlacement


@dataclass(slots=True)
class Sec7Context:
    repeats: int
    lx: int
    phase: float
    instance: Any
    model: Any
    build: Any
    basis: Any
    configs: np.ndarray
    packed_index: Any
    resolved_sector: Any
    sector: Any
    tower: np.ndarray
    cage_full: np.ndarray
    h_sector: sp.csr_array
    projected_q: dict[str, sp.csr_array]
    q_all: sp.csr_array
    tower_energy: float
    tower_residual: float
    cage_projection_norm: float
    cage_q: dict[str, float]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def checkerboard_sign(model: Any, plaquette_id: int) -> int:
    x, y = model.lattice.plaquette_anchor_cell(int(plaquette_id))
    return 1 if (int(x) + int(y)) % 2 == 0 else -1


def recover_reference_geometry() -> ReferenceGeometry:
    """Recover the certified four-column motif and its A/Z placements."""

    base_model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    local_config = LocalQDMCageSearchConfig(
        halo_layers=0,
        boundary_mode="relaxed",
        prune_inactive_local_basis_states=True,
        tolerance=TOL,
        degenerate_basis_strategy="ipr",
        ipr_random_seed=1234,
    )
    robust_config = RobustQDMLocalCageSearchConfig(
        local_config=local_config,
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0, 1),
        max_regions_per_strategy=None,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=None,
        max_product_support_size=2048,
        max_paddings_per_stage=100,
        max_paddings_per_packing=10,
        include_sectors=True,
        padding_stages=("static",),
        tolerance=1.0e-9,
        store_full_states=False,
    )
    certified, search_context = robust_qdm_local_cage_search(
        base_model,
        config=robust_config,
        return_context=True,
    )
    repeatable: list[tuple[int, Any, Any]] = []
    for report_index, report in enumerate(certified.reports):
        try:
            cell = SquareQDMPeriodicProductUnitCell.from_padding(
                base_model,
                search_context.blocks,
                report.padding,
                repeat_axis="x",
            )
            sequence = certify_square_qdm_periodic_product_sequence(cell)
        except ValueError:
            continue
        if sequence.is_certified:
            repeatable.append((report_index, cell, sequence))
    if not repeatable:
        raise RuntimeError("No repeatable compact checkerboard cage found")
    report_index, product_unit_cell, sequence = repeatable[0]

    stripe_record = certified.records[report_index]
    classification = diagnose_cage_environment_reduction(
        stripe_record.cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
        config=EnvironmentReductionConfig(sector_policy="infer_support_component"),
    )
    strip_report = evaluate_square_qdm_environment_witnesses_on_strips(
        classification,
        model=base_model,
        lengths=(4, 8, 12),
        winding_sector=(0, 0),
        normalization="operator_norm",
        winding_projection="fourier",
    )
    z_reference = strip_report.records[0].witness
    z_placement = strip_report.records[0].placement
    local_operator = np.asarray(z_reference.template.local_operator, dtype=np.complex128)
    adjacency = np.abs(local_operator) > TOL
    target = int(np.argmax(np.sum(adjacency, axis=1)))
    sources = np.flatnonzero(adjacency[target])
    a_template = directed_transition_witness_template(
        target_pattern=z_reference.template.local_patterns[target],
        source_patterns=[z_reference.template.local_patterns[index] for index in sources],
        amplitudes=[local_operator[target, index] for index in sources],
        metadata={"name": "A_R"},
        normalization="operator_norm",
    )
    a_reference = a_template.instantiate(z_reference.variable_indices)
    a_placement = SquareQDMWitnessPlacement.from_local_witness(base_model, a_reference)
    return ReferenceGeometry(
        product_unit_cell=product_unit_cell,
        a_placement=a_placement,
        z_placement=z_placement,
    )


def checkerboard_instance(reference: ReferenceGeometry, repeats: int, phase: float):
    raw = reference.product_unit_cell.instantiate(int(repeats))
    model0 = replace(raw.model, winding_x=0, winding_y=0)
    couplings = {
        int(plaquette_id): peierls_plaquette_coupling(
            1.0,
            float(phase) * checkerboard_sign(model0, plaquette_id),
        )
        for plaquette_id in model0.plaquette_ids()
    }
    model = replace(model0, coup_kin=couplings, coup_pot=1.0)
    return replace(raw, model=model)


def translated_joint_dark(
    configs: np.ndarray,
    model: Any,
    sector: Any,
    *,
    a_placement: SquareQDMWitnessPlacement,
    z_placement: SquareQDMWitnessPlacement,
) -> sp.csr_array:
    total = None
    for x in range(int(model.lx)):
        for placement in (a_placement, z_placement):
            operator = placement.instantiate_on_model(model, origin_x=x).embed(configs)
            q_operator = operator.conj().T @ operator
            total = q_operator if total is None else total + q_operator
    if total is None:
        raise RuntimeError("translated joint-dark operator has no terms")
    return project_sparse_operator_to_sector(total, sector)


def build_context(
    *,
    reference: ReferenceGeometry,
    repeats: int,
    phase: float = REPRESENTATIVE_PHASE,
    symmetry_chunk_size: int = 16384,
) -> Sec7Context:
    """Build the fully resolved checkerboard sector without an eigensolve."""

    instance = checkerboard_instance(reference, repeats, phase)
    model = instance.model
    build = model.build(
        basis_solver="dfs",
        builder="bitmask",
        backend="scipy",
        sort_basis=True,
    )
    basis = build.basis
    packed_index = packed_binary_basis_index(basis)
    resolved_sector, symmetry_permutations = checkerboard_fully_resolved_sector(
        model,
        basis,
        packed_index=packed_index,
        repeats=int(repeats),
        chunk_size=int(symmetry_chunk_size),
    )
    sector = resolved_sector.sector
    del symmetry_permutations

    if int(repeats) >= 3:
        configs = binary_basis_configs_uint8(basis, chunk_size=int(symmetry_chunk_size))
    else:
        configs = basis_configs_from_build_result(build)

    cage_full = materialize_periodic_product_state_from_basis(instance, basis)
    tower = project_state_to_sector(cage_full, sector)
    projection_norm = float(np.linalg.norm(tower))
    if projection_norm <= TOL:
        raise RuntimeError("compact cage has zero weight in the selected checkerboard irrep")
    tower = np.asarray(tower / projection_norm, dtype=np.complex128)

    h_sector = project_sparse_operator_to_sector(build.hamiltonian, sector)
    localized = {
        name: placement.instantiate_on_model(model).embed(configs)
        for name, placement in (
            ("A", reference.a_placement),
            ("Z", reference.z_placement),
        )
    }
    projected_q = {
        name: project_sparse_operator_to_sector(operator.conj().T @ operator, sector)
        for name, operator in localized.items()
    }
    projected_full = np.asarray(sector.basis @ tower).reshape(-1)
    cage_q = {
        name: float(
            np.vdot(
                projected_full,
                operator.conj().T @ (operator @ projected_full),
            ).real
        )
        for name, operator in localized.items()
    }
    q_all = translated_joint_dark(
        configs,
        model,
        sector,
        a_placement=reference.a_placement,
        z_placement=reference.z_placement,
    )
    tower_energy = float(np.vdot(tower, h_sector @ tower).real)
    tower_residual = float(np.linalg.norm(h_sector @ tower - tower_energy * tower))

    expected = EXPECTED_SECTOR_DIMENSIONS.get(int(model.lx))
    if expected is not None and int(sector.sector_dimension) != expected:
        raise RuntimeError(
            f"selected sector dimension changed at Lx={model.lx}: "
            f"expected {expected}, got {sector.sector_dimension}"
        )
    if abs(tower_energy - float(model.lx)) > 1.0e-8:
        raise RuntimeError(
            f"checkerboard cage energy changed: expected {model.lx}, got {tower_energy}"
        )
    if tower_residual > 1.0e-8:
        raise RuntimeError(f"checkerboard cage residual is too large: {tower_residual:.3e}")
    if max(cage_q.values(), default=0.0) > 1.0e-8:
        raise RuntimeError(f"A/Z darkness failed: {cage_q}")

    return Sec7Context(
        repeats=int(repeats),
        lx=int(model.lx),
        phase=float(phase),
        instance=instance,
        model=model,
        build=build,
        basis=basis,
        configs=np.asarray(configs),
        packed_index=packed_index,
        resolved_sector=resolved_sector,
        sector=sector,
        tower=tower,
        cage_full=np.asarray(cage_full, dtype=np.complex128),
        h_sector=sp.csr_array(h_sector),
        projected_q=projected_q,
        q_all=sp.csr_array(q_all),
        tower_energy=tower_energy,
        tower_residual=tower_residual,
        cage_projection_norm=projection_norm,
        cage_q=cage_q,
    )


def select_target_block_indices(
    energies: np.ndarray,
    *,
    target_energy: float,
    solver_tolerance: float,
    energy_block_tolerance: float = ENERGY_BLOCK_TOL,
) -> np.ndarray:
    values = np.asarray(energies, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("target-block selection requires at least one eigenvalue")
    distances = np.abs(values - float(target_energy))
    closest = float(np.min(distances))
    threshold = closest + max(
        100.0 * float(energy_block_tolerance),
        10.0 * float(solver_tolerance),
    )
    return np.flatnonzero(distances <= threshold)


def orthonormalize(columns: np.ndarray, *, tolerance: float = 1.0e-9) -> np.ndarray:
    matrix = np.asarray(columns, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("columns must be a matrix")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    keep = singular > float(tolerance) * max(1.0, float(singular[0]))
    return np.asarray(u[:, keep], dtype=np.complex128)


def target_dark_kernel(
    context: Sec7Context,
    energies: np.ndarray,
    vectors: np.ndarray,
    *,
    solver_tolerance: float,
) -> dict[str, Any]:
    indices = select_target_block_indices(
        energies,
        target_energy=context.tower_energy,
        solver_tolerance=solver_tolerance,
    )
    basis = np.asarray(vectors[:, indices], dtype=np.complex128)
    if basis.shape[1] == 0:
        raise RuntimeError("no eigenvectors selected for the target-energy block")
    gram = basis.conj().T @ basis
    orthogonality_residual = float(np.linalg.norm(gram - np.eye(basis.shape[1]), ord=2))
    compressed = basis.conj().T @ (context.q_all @ basis)
    compressed = 0.5 * (compressed + compressed.conj().T)
    q_values, rotation = la.eigh(compressed, check_finite=False)
    q_scale = max(1.0, float(np.max(np.abs(q_values), initial=0.0)))
    dark_indices = np.flatnonzero(q_values <= DARK_TOL * q_scale)
    dark = (
        np.asarray(basis @ rotation[:, dark_indices], dtype=np.complex128)
        if dark_indices.size
        else np.zeros((basis.shape[0], 0), dtype=np.complex128)
    )
    projector_weight = float(np.linalg.norm(basis.conj().T @ context.tower) ** 2)
    dark_weight = float(np.linalg.norm(dark.conj().T @ context.tower) ** 2)
    target_residuals = np.linalg.norm(
        context.h_sector @ basis - basis * np.asarray(energies[indices])[None, :],
        axis=0,
    )
    return {
        "indices": indices,
        "basis": basis,
        "q_values": np.asarray(q_values, dtype=float),
        "dark": orthonormalize(dark),
        "block_dimension": int(indices.size),
        "joint_dark_rank": int(dark_indices.size),
        "cage_projector_weight": projector_weight,
        "cage_dark_weight": dark_weight,
        "target_maximum_residual": float(np.max(target_residuals, initial=0.0)),
        "target_median_residual": (
            float(np.median(target_residuals)) if target_residuals.size else 0.0
        ),
        "target_orthogonality_residual": orthogonality_residual,
        "target_energy_min": float(np.min(np.asarray(energies)[indices])),
        "target_energy_max": float(np.max(np.asarray(energies)[indices])),
    }


def compact_type1_orbit(context: Sec7Context) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tx1 = translation_permutation_from_binary_basis(
        context.model,
        context.basis,
        packed_index=context.packed_index,
        dx=1,
        chunk_size=16384,
    )
    ty1 = translation_permutation_from_binary_basis(
        context.model,
        context.basis,
        packed_index=context.packed_index,
        dy=1,
        chunk_size=16384,
    )
    columns: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for dx in (0, 1):
        for dy in (0, 1):
            translated = np.asarray(context.cage_full, dtype=np.complex128)
            if dx:
                moved = np.zeros_like(translated)
                moved[tx1] = translated
                translated = moved
            if dy:
                moved = np.zeros_like(translated)
                moved[ty1] = translated
                translated = moved
            coordinates = project_state_to_sector(translated, context.sector)
            projection_norm = float(np.linalg.norm(coordinates))
            if projection_norm > TOL:
                coordinates = np.asarray(coordinates / projection_norm, dtype=np.complex128)
                q_dark = float(np.vdot(coordinates, context.q_all @ coordinates).real)
                energy_residual = float(
                    np.linalg.norm(
                        context.h_sector @ coordinates - context.tower_energy * coordinates
                    )
                )
            else:
                q_dark = math.nan
                energy_residual = math.nan
            include = bool(projection_norm > TOL and q_dark <= 1.0e-8 and energy_residual <= 1.0e-8)
            if include:
                columns.append(coordinates)
            rows.append(
                {
                    "Lx": context.lx,
                    "phase": context.phase,
                    "dx": dx,
                    "dy": dy,
                    "projection_norm": projection_norm,
                    "projected_Qall": q_dark,
                    "eigen_residual": energy_residual,
                    "included_in_compact_dark_span": include,
                    "inventory_method": "translated_compact_cage_coset_orbit",
                }
            )
    candidate = (
        orthonormalize(np.column_stack(columns))
        if columns
        else np.zeros((context.sector.sector_dimension, 0), dtype=np.complex128)
    )
    return candidate, rows


def compare_subspaces(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = orthonormalize(candidate)
    reference = orthonormalize(reference)
    if candidate.shape[0] != reference.shape[0]:
        raise ValueError("subspaces have incompatible ambient dimensions")
    if candidate.shape[1] and reference.shape[1]:
        singular = np.linalg.svd(candidate.conj().T @ reference, compute_uv=False)
        singular = np.clip(np.real(singular), 0.0, 1.0)
        principal_angles = np.arccos(singular)
    else:
        principal_angles = np.zeros(0, dtype=float)
    unexplained = (
        reference - candidate @ (candidate.conj().T @ reference)
        if candidate.shape[1]
        else reference.copy()
    )
    outside = (
        candidate - reference @ (reference.conj().T @ candidate)
        if reference.shape[1]
        else candidate.copy()
    )
    return {
        "type1_projected_rank": int(candidate.shape[1]),
        "joint_dark_rank": int(reference.shape[1]),
        "unexplained_joint_dark_norm": float(np.linalg.norm(unexplained)),
        "candidate_outside_joint_dark_norm": float(np.linalg.norm(outside)),
        "principal_angles_rad": [float(value) for value in principal_angles],
        "maximum_principal_angle_rad": (
            float(np.max(principal_angles)) if principal_angles.size else math.nan
        ),
    }


def canonical_weights(energies: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    energies = np.asarray(energies, dtype=float)

    def weights(beta: float) -> np.ndarray:
        values = -float(beta) * energies
        values -= np.max(values)
        result = np.exp(values)
        return result / result.sum()

    mismatch_zero = float(np.mean(energies) - target)
    if abs(mismatch_zero) < 1.0e-12:
        return 0.0, weights(0.0)
    direction = 1.0 if mismatch_zero > 0 else -1.0
    bound = direction
    while abs(bound) < 128 and (
        float(np.dot(weights(bound), energies) - target) * mismatch_zero > 0
    ):
        bound *= 2.0
    if abs(bound) >= 128:
        raise RuntimeError("Could not bracket the finite-beta energy match")
    beta = float(
        brentq(
            lambda value: float(np.dot(weights(value), energies) - target),
            *sorted((0.0, bound)),
        )
    )
    return beta, weights(beta)


def stripe_algebra(
    context: Sec7Context,
    *,
    z_placement: SquareQDMWitnessPlacement,
) -> tuple[
    tuple[sp.csr_array, ...],
    tuple[str, ...],
    dict[str, Any],
    tuple[str, ...],
    np.ndarray,
]:
    """Complete constrained two-column stripe algebra modulo projected null action."""

    witness = z_placement.instantiate_on_model(context.model)
    variables = tuple(map(int, witness.variable_indices))
    patterns = np.unique(context.configs[:, variables], axis=0)
    position = {value: index for index, value in enumerate(variables)}
    site_lookup = {tuple(site.cell): int(site.id) for site in context.model.lattice.sites}
    boundary: list[tuple[int, ...]] = []
    for cell in z_placement.affected_sites:
        site_id = site_lookup[tuple(cell)]
        incident = [
            int(context.model.layout.link_variable_index(link))
            for link in context.model.lattice.incident_links(site_id)
        ]
        if not all(value in position for value in incident):
            boundary.append(tuple(incident))
    signatures = [
        tuple(
            sum(int(pattern[position[value]]) for value in incident if value in position)
            for incident in boundary
        )
        for pattern in patterns
    ]
    groups: dict[Any, list[int]] = {}
    for index, signature in enumerate(signatures):
        groups.setdefault(signature, []).append(index)

    matrices: list[np.ndarray] = []
    ambient_names: list[str] = []
    pattern_count = len(patterns)
    for block_id, indices in enumerate(groups.values()):
        for i in indices:
            matrix = np.zeros((pattern_count, pattern_count), dtype=np.complex128)
            matrix[i, i] = 1.0
            matrices.append(matrix)
            ambient_names.append(f"b{block_id}_d{i}")
        for offset, i in enumerate(indices):
            for j in indices[offset + 1 :]:
                symmetric = np.zeros((pattern_count, pattern_count), dtype=np.complex128)
                symmetric[i, j] = symmetric[j, i] = 1.0 / math.sqrt(2.0)
                matrices.append(symmetric)
                ambient_names.append(f"b{block_id}_s{i}_{j}")
                antisymmetric = np.zeros((pattern_count, pattern_count), dtype=np.complex128)
                antisymmetric[i, j] = -1.0j / math.sqrt(2.0)
                antisymmetric[j, i] = 1.0j / math.sqrt(2.0)
                matrices.append(antisymmetric)
                ambient_names.append(f"b{block_id}_a{i}_{j}")

    local_patterns = tuple(tuple(map(int, pattern)) for pattern in patterns)
    projected_raw: list[sp.csr_array] = []
    for name, matrix in zip(ambient_names, matrices, strict=True):
        template = LocalWitnessTemplate(
            pattern_key=(),
            local_patterns=local_patterns,
            local_operator=matrix,
            metadata={"name": name, "boundary_flux_blocks": len(groups)},
        )
        full = template.instantiate(variables).embed(context.configs)
        projected_raw.append(project_sparse_operator_to_sector(full, context.sector))

    gram = np.empty((len(projected_raw), len(projected_raw)), dtype=float)
    for a, left in enumerate(projected_raw):
        for b in range(a, len(projected_raw)):
            value = float(np.real(left.conj().multiply(projected_raw[b]).sum()))
            gram[a, b] = value
            gram[b, a] = value
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    quotient_mask = eigenvalues > 1.0e-10 * scale
    quotient_coefficients = np.asarray(eigenvectors[:, quotient_mask], dtype=float)
    quotient_eigenvalues = np.asarray(eigenvalues[quotient_mask], dtype=float)

    projected: list[sp.csr_array] = []
    quotient_names: list[str] = []
    for quotient_index, coefficients in enumerate(quotient_coefficients.T):
        operator = sp.csr_array(projected_raw[0].shape, dtype=np.complex128)
        for coefficient, raw in zip(coefficients, projected_raw, strict=True):
            if abs(coefficient) > 1.0e-14:
                operator = operator + complex(coefficient) * raw
        projected.append(sp.csr_array(operator))
        quotient_names.append(f"q{quotient_index}")

    metadata = {
        "local_pattern_count": len(patterns),
        "boundary_flux_block_count": len(groups),
        "boundary_flux_block_dimensions": repr(sorted(map(len, groups.values()))),
        "formal_operator_dimension": sum(len(values) ** 2 for values in groups.values()),
        "ambient_nonidentity_dimension": sum(len(values) ** 2 for values in groups.values()) - 1,
        "projected_operator_dimension": len(projected),
        "projected_null_dimension": len(projected_raw) - len(projected),
        "projected_map_min_nonzero_gram_eigenvalue": float(
            np.min(quotient_eigenvalues, initial=np.nan)
        ),
        "projected_map_max_gram_eigenvalue": float(np.max(quotient_eigenvalues, initial=np.nan)),
        "operator_normalization": "local_HS_quotient_kernel_of_P_O_P",
    }
    return (
        tuple(projected),
        tuple(quotient_names),
        metadata,
        tuple(ambient_names),
        quotient_coefficients,
    )


def validate_authoritative_small_ed(base_data_dir: Path) -> dict[str, Any]:
    """Require the previously established Lx=4,8 full-symmetry evidence."""

    base = Path(base_data_dir)
    required = {
        "common_sector": base / "qdm_checkerboard_common_symmetry_sector.csv",
        "thermal": base / "qdm_checkerboard_thermal_overlap.csv",
        "concentration": base / "qdm_checkerboard_concentration_grid.csv",
        "canonical": base / "qdm_checkerboard_finite_beta_transfer_target.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing authoritative Lx=4,8 prerequisite files: " + ", ".join(missing))
    common = pd.read_csv(required["common_sector"])
    checks: dict[str, Any] = {"files": {name: str(path) for name, path in required.items()}}
    for lx, expected in ((4, 15), (8, 1125)):
        selected = common[
            (common["Lx"].astype(int) == lx)
            & np.isclose(common["phase"].astype(float), REPRESENTATIVE_PHASE)
        ]
        if selected.empty:
            raise RuntimeError(f"authoritative common-sector row is missing for Lx={lx}")
        dimension = int(selected["sector_dimension"].dropna().iloc[0])
        if dimension != expected:
            raise RuntimeError(
                f"authoritative Lx={lx} sector dimension changed: {dimension} != {expected}"
            )
        checks[f"Lx{lx}_sector_dimension"] = dimension
    return checks


def estimate_l12_window_budget(
    primme_data_dir: Path,
    *,
    half_width: float,
) -> float | None:
    path = Path(primme_data_dir) / "qdm_checkerboard_L12_spectral_convergence.csv"
    if not path.is_file():
        path = Path(primme_data_dir) / "qdm_checkerboard_L12_sparse_convergence.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "solver_status" in frame:
        completed = frame[frame["solver_status"].astype(str) == "completed"].copy()
    else:
        completed = frame.copy()
    if completed.empty:
        return None
    completed["returned_eigenpairs"] = pd.to_numeric(
        completed["returned_eigenpairs"], errors="coerce"
    )
    completed = completed.dropna(
        subset=["returned_eigenpairs", "partial_min_energy", "partial_max_energy"]
    )
    if completed.empty:
        return None
    row = completed.sort_values("returned_eigenpairs").iloc[-1]
    span = float(row["partial_max_energy"]) - float(row["partial_min_energy"])
    if span <= 0:
        return None
    density = float(row["returned_eigenpairs"]) / span
    return float(math.ceil(density * (2.0 * float(half_width))))


def recommend_fixed_width(
    frame: pd.DataFrame,
    *,
    estimated_budgets: dict[float, float | None],
    max_practical_budget: int = 8192,
) -> dict[str, Any]:
    """Create a conservative, inspectable recommendation from the Lx=4,8 pilot."""

    rows: list[dict[str, Any]] = []
    widths = sorted(float(value) for value in frame["window_half_width"].unique())
    l8 = frame[frame["Lx"].astype(int) == 8].set_index("window_half_width")
    l4 = frame[frame["Lx"].astype(int) == 4].set_index("window_half_width")
    for index, width in enumerate(widths):
        if width not in l8.index or width not in l4.index:
            continue
        row8 = l8.loc[width]
        row4 = l4.loc[width]
        neighbors = []
        if index:
            neighbors.append(widths[index - 1])
        if index + 1 < len(widths):
            neighbors.append(widths[index + 1])
        neighbor_changes: list[float] = []
        for neighbor in neighbors:
            if neighbor not in l8.index:
                continue
            other = l8.loc[neighbor]
            neighbor_changes.append(
                max(
                    abs(float(row8["tau_A_mc_raw"]) - float(other["tau_A_mc_raw"])),
                    abs(float(row8["tau_Z_mc_raw"]) - float(other["tau_Z_mc_raw"])),
                    abs(float(row8["w_raw"]) - float(other["w_raw"])),
                )
            )
        stability = max(neighbor_changes, default=math.inf)
        growth = float(row8["raw_window_state_count"]) / max(
            1.0, float(row4["raw_window_state_count"])
        )
        budget = estimated_budgets.get(width)
        criteria = {
            "state_count_growth_at_least_2": growth >= 2.0,
            "neighbor_change_at_most_0p05": stability <= 0.05,
            "dark_fraction_at_most_0p20": float(row8["removed_fraction"]) <= 0.20,
            "estimated_L12_budget_at_most_8192": (
                budget is not None and budget <= max_practical_budget
            ),
        }
        rows.append(
            {
                "window_half_width": width,
                "state_count_growth_L8_over_L4": growth,
                "max_neighbor_change_L8": stability,
                "estimated_L12_eigenpair_budget": budget,
                "criteria": criteria,
                "passes_all_heuristics": all(criteria.values()),
            }
        )

    preference = (0.20, 0.25, 0.10, 0.50)
    passing = {
        float(row["window_half_width"]) for row in rows if bool(row["passes_all_heuristics"])
    }
    recommended = next((value for value in preference if value in passing), None)
    return {
        "schema_version": 1,
        "pilot_half_widths": widths,
        "recommended_half_width": recommended,
        "status": "recommended" if recommended is not None else "scientific_review_required",
        "preference_order": list(preference),
        "heuristic_candidates": rows,
        "neighbor_controls": (
            [value for value in widths if recommended is not None and value != recommended]
            if recommended is not None
            else widths
        ),
        "claim_boundary": (
            "This recommendation selects a production protocol from finite-size systematics; "
            "it is not itself evidence for a thermodynamic limit."
        ),
    }


def process_memory_gib() -> float:
    return float(process_peak_rss_gib())
