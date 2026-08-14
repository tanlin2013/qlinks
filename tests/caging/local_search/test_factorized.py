from __future__ import annotations

import numpy as np

from qlinks.caging import (
    CageState,
    CandidateSubgraph,
)
from qlinks.caging.local_search import (
    FactorizedLocalQDMPadding,
    LocalQDMCageRecord,
    LocalQDMMultiPaddingConfig,
    certify_qdm_factorized_product_state,
    certify_qdm_multi_block_padding,
    find_factorized_qdm_block_paddings,
    find_multi_qdm_block_paddings,
    make_qdm_cage_block,
)
from qlinks.models import SquareQDMModel
from qlinks.operators.plaquette import alternating_binary_patterns


def _qdm_flip_is_absent(model: SquareQDMModel, config: np.ndarray, plaquette_id: int) -> bool:
    links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
    values = np.asarray(config, dtype=np.int64)[links]
    pattern0, pattern1 = alternating_binary_patterns(int(links.size))
    return not (np.array_equal(values, pattern0) or np.array_equal(values, pattern1))


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


def test_factorized_padding_search_avoids_global_support_materialization() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    static_config = _first_static_qdm_config(model)
    blocks = [
        make_qdm_cage_block(
            model,
            _static_local_record_from_global_config(static_config, [4]),
            block_id=0,
        ),
        make_qdm_cage_block(
            model,
            _static_local_record_from_global_config(static_config, [16]),
            block_id=1,
        ),
    ]
    config = LocalQDMMultiPaddingConfig(
        min_blocks=2,
        max_blocks=2,
        max_paddings=1,
        max_paddings_per_packing=1,
        include_sectors=False,
        require_static_exterior=True,
        require_kinetic_separation=True,
        tolerance=1.0e-9,
    )

    paddings = find_factorized_qdm_block_paddings(model, blocks, config=config)

    assert len(paddings) == 1
    assert isinstance(paddings[0], FactorizedLocalQDMPadding)
    assert not hasattr(paddings[0], "global_support_configs")
    report = certify_qdm_factorized_product_state(
        model,
        blocks,
        paddings[0],
        config=config,
    )
    assert report.is_certified
    assert report.avoids_support_materialization
    assert report.signature == (0, 0)
    assert report.support_size == 1
    assert report.hamiltonian_residual < config.tolerance


def test_factorized_certificate_matches_explicit_certificate() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    static_config = _first_static_qdm_config(model)
    blocks = [
        make_qdm_cage_block(
            model,
            _static_local_record_from_global_config(static_config, [4]),
            block_id=0,
        ),
        make_qdm_cage_block(
            model,
            _static_local_record_from_global_config(static_config, [16]),
            block_id=1,
        ),
    ]
    config = LocalQDMMultiPaddingConfig(
        min_blocks=2,
        max_blocks=2,
        max_paddings=1,
        max_paddings_per_packing=1,
        include_sectors=False,
        require_static_exterior=True,
        require_kinetic_separation=True,
        tolerance=1.0e-9,
    )
    explicit_padding = find_multi_qdm_block_paddings(model, blocks, config=config)[0]
    explicit = certify_qdm_multi_block_padding(
        model,
        blocks,
        explicit_padding,
        config=config,
    )
    factorized = certify_qdm_factorized_product_state(
        model,
        blocks,
        explicit_padding,
        config=config,
    )

    assert explicit is not None
    assert factorized.is_certified
    assert factorized.signature == explicit.signature
    assert np.isclose(factorized.energy, explicit.energy)
    assert np.isclose(factorized.kinetic_eigenvalue, explicit.kinetic_eigenvalue)
    assert np.isclose(factorized.self_loop_value, explicit.self_loop_value)
    assert np.isclose(factorized.hamiltonian_residual, explicit.full_residual)


def test_factorized_certificate_handles_coherent_full_square_qdm_cage() -> None:
    from qlinks.caging.local_search import (
        LocalQDMCageSearchConfig,
        LocalQDMCageSearcher,
    )

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
    result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_candidate_count=128,
            ipr_random_seed=0,
        ),
    ).run()
    block = make_qdm_cage_block(
        model,
        result.records_by_signature((0, 4))[0],
        block_id=0,
    )
    padding = FactorizedLocalQDMPadding(
        block_ids=(0,),
        exterior_link_ids=np.empty(0, dtype=np.int64),
        exterior_config=np.empty(0, dtype=np.int64),
    )

    report = certify_qdm_factorized_product_state(
        model,
        [block],
        padding,
        config=LocalQDMMultiPaddingConfig(
            min_blocks=1,
            max_blocks=1,
            include_sectors=True,
            require_static_exterior=False,
            require_kinetic_separation=True,
            tolerance=1.0e-9,
        ),
    )

    assert report.is_certified
    assert report.signature == (0, 4)
    assert report.support_size == 4
    assert report.n_kinetic_product_terms > 0
    assert report.hamiltonian_residual < 1.0e-9
