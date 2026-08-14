from __future__ import annotations

import numpy as np


def _assemble_hamiltonian(
    boundary: np.ndarray,
    *,
    internal: np.ndarray | None = None,
    external: np.ndarray | None = None,
) -> np.ndarray:
    if internal is None:
        internal = np.zeros((2, 2), dtype=np.complex128)
    if external is None:
        external = np.diag([2.0, 3.0]).astype(np.complex128)
    return np.block(
        [
            [internal, boundary.conj().T],
            [boundary, external],
        ]
    )


def _toy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_boundary = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    base = _assemble_hamiltonian(base_boundary)

    strong_boundary = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    strong = _assemble_hamiltonian(
        strong_boundary,
        internal=np.eye(2, dtype=np.complex128),
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    structural_boundary = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    structural = _assemble_hamiltonian(
        structural_boundary,
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    incompatible_boundary = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    incompatible = _assemble_hamiltonian(
        incompatible_boundary,
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    cage_state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    return base, strong, structural, incompatible, cage_state


def _nonintegrable_tangent_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cage_state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    orthogonal_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    basis_change = np.column_stack([cage_state, orthogonal_state])

    base_internal_local = np.diag([0.0, 1.0]).astype(np.complex128)
    perturbation_internal_local = np.array(
        [[0.0, -1.0], [-1.0, 0.0]],
        dtype=np.complex128,
    )
    base_boundary_local = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    perturbation_boundary_local = np.array(
        [[-1.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )

    base_internal = basis_change @ base_internal_local @ basis_change.conj().T
    perturbation_internal = basis_change @ perturbation_internal_local @ basis_change.conj().T
    base_boundary = base_boundary_local @ basis_change.conj().T
    perturbation_boundary = perturbation_boundary_local @ basis_change.conj().T

    base = _assemble_hamiltonian(base_boundary, internal=base_internal)
    perturbation = _assemble_hamiltonian(
        perturbation_boundary,
        internal=perturbation_internal,
        external=np.zeros((2, 2), dtype=np.complex128),
    )
    return base, perturbation, cage_state


def _physical_square_qdm_periodic_cage_unit_cell():
    from qlinks.caging import SquareQDMPeriodicProductUnitCell
    from qlinks.caging.local_search import (
        LocalQDMCageSearchConfig,
        RobustQDMLocalCageSearchConfig,
        robust_qdm_local_cage_search,
    )
    from qlinks.models import SquareQDMModel

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    config = RobustQDMLocalCageSearchConfig(
        local_config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_random_seed=1234,
        ),
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
    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )
    return SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[4].padding,
        repeat_axis="x",
    )
