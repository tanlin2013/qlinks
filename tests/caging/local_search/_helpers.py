from __future__ import annotations

import numpy as np

from qlinks.caging import CageState, CandidateSubgraph
from qlinks.caging.local_search import (
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    RobustQDMLocalCageSearchConfig,
)
from qlinks.models import SquareQDMModel
from qlinks.operators.plaquette import alternating_binary_patterns


def _square_qdm_w00_model(lx: int, ly: int) -> SquareQDMModel:
    return SquareQDMModel(
        lx=lx,
        ly=ly,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )


def _square_qdm_stripe_pair_robust_config(
    *,
    max_paddings_per_stage: int = 100,
    max_paddings_per_packing: int = 10,
    max_product_support_size: int = 2048,
    stripe_directions: tuple[int, ...] = (0, 1),
) -> RobustQDMLocalCageSearchConfig:
    return RobustQDMLocalCageSearchConfig(
        local_config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
        ),
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=stripe_directions,
        max_regions_per_strategy=None,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=None,
        max_product_support_size=max_product_support_size,
        max_paddings_per_stage=max_paddings_per_stage,
        max_paddings_per_packing=max_paddings_per_packing,
        include_sectors=True,
        padding_stages=("static",),
        tolerance=1.0e-9,
        store_full_states=False,
    )


def _first_static_qdm_config(model: SquareQDMModel) -> np.ndarray:
    build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    for config in build.basis.states:
        if all(_qdm_flip_is_absent(model, config, int(pid)) for pid in model.plaquette_ids()):
            return np.asarray(config, dtype=np.int64)
    raise AssertionError("Expected at least one static QDM configuration.")


def _qdm_flip_is_absent(model: SquareQDMModel, config: np.ndarray, plaquette_id: int) -> bool:
    links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
    values = np.asarray(config, dtype=np.int64)[links]
    pattern0, pattern1 = alternating_binary_patterns(int(links.size))
    return not (np.array_equal(values, pattern0) or np.array_equal(values, pattern1))


def _static_local_record_from_global_config(
    global_config: np.ndarray,
    link_ids: list[int],
) -> LocalQDMCageRecord:
    local_link_ids = np.asarray(link_ids, dtype=np.int64)
    return LocalQDMCageRecord(
        cage_state=CageState(
            energy=0.0 + 0.0j,
            local_state=np.ones(1, dtype=np.complex128),
            support=np.asarray([0], dtype=np.int64),
            boundary_residual=0.0,
            eigen_residual=0.0,
            full_residual=0.0,
        ),
        signature=(0, 0),
        candidate=CandidateSubgraph(vertices=np.asarray([0], dtype=np.int64)),
        support_configs=np.asarray(global_config[local_link_ids], dtype=np.int64).reshape(1, -1),
        local_link_ids=local_link_ids,
        active_plaquette_ids=np.empty(0, dtype=np.int64),
        scoring_plaquette_ids=np.empty(0, dtype=np.int64),
        unresolved_boundary_plaquette_ids=np.empty(0, dtype=np.int64),
    )
