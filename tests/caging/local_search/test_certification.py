from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qlinks.caging import CageRecord, CageSearchResult, CageState, CandidateSubgraph
from qlinks.caging.analysis import diagnose_cage_environment_reduction
from qlinks.caging.local_search import (
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMCageSearcher,
    LocalQDMMultiPaddingConfig,
    LocalQDMPaddingConfig,
    MultiLocalQDMPadding,
    QDMMultiPaddingDiagnostics,
    StripeRegionProposal,
    certified_qdm_result_from_multi_block_reports,
    certify_qdm_local_result,
    certify_qdm_multi_block_padding,
    certify_qdm_multi_block_result,
    collect_qdm_cage_blocks_from_region_proposals,
    diagnose_qdm_multi_block_paddings,
    find_multi_qdm_block_paddings,
    iter_multi_qdm_block_paddings,
    make_qdm_cage_block,
    qdm_multi_padding_config_schedule,
    robust_certify_qdm_multi_block_result,
)
from qlinks.models import SquareQDMModel
from tests.caging.local_search._helpers import (
    _first_static_qdm_config,
    _static_local_record_from_global_config,
)


def test_qdm_multi_block_diagnostics_and_schedule_report_successes() -> None:
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
        tolerance=1.0e-9,
    )

    scheduled = qdm_multi_padding_config_schedule(config, stages=("loose", "static", "strict"))
    assert [name for name, _stage_config in scheduled] == ["loose", "static", "strict"]
    assert scheduled[0][1].require_static_exterior is False
    assert scheduled[1][1].require_static_exterior is True
    assert scheduled[1][1].require_kinetic_separation is False
    assert scheduled[2][1].require_kinetic_separation is True

    diagnostics = diagnose_qdm_multi_block_paddings(model, blocks, config=config)
    assert isinstance(diagnostics, QDMMultiPaddingDiagnostics)
    assert diagnostics.n_paddings == 1
    assert diagnostics.n_certified == 1
    assert diagnostics.n_failed == 0
    assert diagnostics.counts_by_failure_reason == {}

    robust = robust_certify_qdm_multi_block_result(
        model,
        blocks,
        config=config,
        stages=("loose", "static"),
    )
    assert robust.counts_by_signature == {(0, 0): 1}


def test_multi_block_padding_iterator_preserves_raw_padding_cap() -> None:
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
    proposal = StripeRegionProposal(
        model,
        directions=(0,),
        width=1,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_random_seed=0,
        ),
    )
    blocks = collect_qdm_cage_blocks_from_region_proposals(
        [proposal],
        model=model,
        signatures=((0, 2),),
        max_records_per_region=2,
    )
    config = LocalQDMMultiPaddingConfig(
        min_blocks=2,
        max_blocks=2,
        max_paddings=2,
        max_padding_attempts=8,
        max_paddings_per_packing=4,
        include_sectors=True,
        require_static_exterior=True,
        require_kinetic_separation=False,
        tolerance=1.0e-9,
    )

    raw_limited = find_multi_qdm_block_paddings(model, blocks, config=config)
    raw_streamed = list(iter_multi_qdm_block_paddings(model, blocks, config=config))
    diagnostics = diagnose_qdm_multi_block_paddings(model, blocks, config=config)

    assert len(raw_limited) == 2
    assert len(raw_streamed) <= 8
    assert diagnostics.n_padding_attempts <= 8
    assert diagnostics.n_certified <= 2
    assert diagnostics.leakage_failure_counts_by_class

    uncapped_attempt_config = replace(config, max_padding_attempts=None)
    uncapped_diagnostics = diagnose_qdm_multi_block_paddings(
        model,
        blocks,
        config=uncapped_attempt_config,
    )
    assert uncapped_diagnostics.n_certified == 2
    assert uncapped_diagnostics.n_padding_attempts >= diagnostics.n_certified
    assert uncapped_diagnostics.first_certified_attempt_index is not None


def test_local_qdm_full_square_4x4_certifies_to_cage_search_result_protocol() -> None:
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

    local_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()

    certified = certify_qdm_local_result(
        model,
        local_result,
        config=LocalQDMPaddingConfig(tolerance=1.0e-9),
    )

    assert isinstance(certified.as_cage_search_result(), CageSearchResult)
    assert all(isinstance(record, CageRecord) for record in certified.records)
    assert certified.counts_by_signature == local_result.counts_by_signature
    assert certified.counts_by_signature == {(0, 4): 9, (0, 6): 1}
    assert certified.hilbert_size <= local_result.local_hilbert_size
    assert certified.basis.states.shape == (certified.hilbert_size, model.lattice.num_links)
    assert certified.kinetic_matrix.shape == (certified.hilbert_size, certified.hilbert_size)
    assert all(report.full_residual < 1.0e-9 for report in certified.reports)


def test_local_qdm_certified_result_can_feed_classification_on_limited_basis() -> None:
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

    local_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()
    certified = certify_qdm_local_result(
        model,
        local_result,
        config=LocalQDMPaddingConfig(tolerance=1.0e-9),
    )

    record = certified.first((0, 4))
    report = diagnose_cage_environment_reduction(
        record.cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
    )

    assert report.support_size == record.cage_state.support_size
    assert report.n_nontrivial_zeros >= 0


def test_qdm_multi_block_padding_finds_and_certifies_static_lego_blocks() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    static_config = _first_static_qdm_config(model)

    # Two occupied links far enough apart that no plaquette touches both blocks.
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
        tolerance=1.0e-9,
    )
    paddings = find_multi_qdm_block_paddings(model, blocks, config=config)

    assert len(paddings) == 1
    assert paddings[0].block_ids == (0, 1)
    assert paddings[0].global_support_configs.shape == (1, model.lattice.num_links)
    assert paddings[0].global_amplitudes.shape == (1,)

    report = certify_qdm_multi_block_padding(model, blocks, paddings[0], config=config)
    assert report is not None
    assert report.block_ids == (0, 1)
    assert report.signature == (0, 0)
    assert report.full_residual < 1.0e-9
    assert report.support_size == 1


def test_qdm_multi_block_padding_certifies_explicit_static_exterior() -> None:
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

    owned_links = {int(link_id) for block in blocks for link_id in block.link_ids}
    exterior_link_ids = np.asarray(
        [link_id for link_id in range(model.lattice.num_links) if link_id not in owned_links],
        dtype=np.int64,
    )
    padding = MultiLocalQDMPadding(
        block_ids=(0, 1),
        exterior_link_ids=exterior_link_ids,
        exterior_config=static_config[exterior_link_ids],
        global_support_configs=static_config.reshape(1, -1),
        global_amplitudes=np.ones(1, dtype=np.complex128),
        block_support_indices=np.zeros((1, 2), dtype=np.int64),
    )

    report = certify_qdm_multi_block_padding(
        model,
        blocks,
        padding,
        config=LocalQDMMultiPaddingConfig(
            min_blocks=2,
            max_blocks=2,
            include_sectors=False,
            require_static_exterior=True,
            tolerance=1.0e-9,
        ),
    )

    assert report is not None
    assert report.signature == (0, 0)
    assert report.leakage_residual == 0.0
    assert report.support_kinetic_residual == 0.0


def test_qdm_multi_block_certified_result_reuses_limited_result_protocol() -> None:
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
        tolerance=1.0e-9,
    )

    certified = certify_qdm_multi_block_result(model, blocks, config=config)

    assert len(certified) == 1
    assert certified.counts_by_signature == {(0, 0): 1}
    assert certified.hilbert_size == certified.basis.n_states
    assert certified.kinetic_matrix.shape == (certified.hilbert_size, certified.hilbert_size)
    assert certified.padding_config is config
    assert len(certified.reports) == 1

    record = certified.first((0, 0))
    assert record.cage_state.support_size == 1
    assert record.cage_state.full_residual is not None
    assert record.cage_state.full_residual < 1.0e-9
    assert record.full_state is not None

    environment_report = diagnose_cage_environment_reduction(
        record.cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
    )
    assert environment_report.support_size == record.cage_state.support_size

    from_reports = certified_qdm_result_from_multi_block_reports(
        model,
        certified.reports,
        config=config,
    )
    assert from_reports.counts_by_signature == certified.counts_by_signature


def test_make_qdm_cage_block_rejects_support_dependent_site_counts() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    link_ids = np.asarray([4], dtype=np.int64)
    record = LocalQDMCageRecord(
        cage_state=CageState(
            energy=0.0 + 0.0j,
            local_state=np.ones(2, dtype=np.complex128) / np.sqrt(2.0),
            support=np.asarray([0, 1], dtype=np.int64),
            boundary_residual=0.0,
            eigen_residual=0.0,
            full_residual=0.0,
        ),
        signature=(0, 0),
        candidate=CandidateSubgraph(vertices=np.asarray([0, 1], dtype=np.int64)),
        support_configs=np.asarray([[0], [1]], dtype=np.int64),
        local_link_ids=link_ids,
        active_plaquette_ids=np.empty(0, dtype=np.int64),
        scoring_plaquette_ids=np.empty(0, dtype=np.int64),
        unresolved_boundary_plaquette_ids=np.empty(0, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="site occupation contribution changes"):
        make_qdm_cage_block(model, record, block_id=0)
