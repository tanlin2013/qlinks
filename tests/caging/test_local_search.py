from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qlinks.caging import (
    AdaptiveRegionProposal,
    AdaptiveRegionProposalRecord,
    CageRecord,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
    CageState,
    CandidateSubgraph,
    ConnectedRegionProposal,
    ConnectedRegionProposalRecord,
    LocalCageSearchConfig,
    LocalCageSearcher,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMCageSearcher,
    LocalQDMMultiPaddingConfig,
    LocalQDMPaddingConfig,
    LocalRegionProposalSearchResult,
    MultiLocalQDMPadding,
    QDMLocalCageAdapter,
    QDMMultiPaddingDiagnostics,
    RobustQDMLocalCageSearchConfig,
    RobustQDMLocalCageSearchContext,
    SnakeStripeRegionProposal,
    SnakeStripeRegionProposalRecord,
    StripeMotifRegionProposal,
    StripeMotifRegionProposalRecord,
    StripeRegionProposal,
    certified_qdm_result_from_multi_block_reports,
    certify_qdm_multi_block_padding,
    certify_qdm_multi_block_result,
    classify_cage_state,
    collect_qdm_cage_blocks_from_region_proposals,
    collect_qdm_cage_blocks_with_scan_from_region_proposals,
    diagnose_qdm_multi_block_paddings,
    enumerate_qdm_local_basis,
    find_multi_qdm_block_paddings,
    iter_multi_qdm_block_paddings,
    make_qdm_cage_block,
    qdm_multi_padding_config_schedule,
    robust_certify_qdm_multi_block_result,
    robust_qdm_local_cage_search,
    run_local_region_proposal,
)
from qlinks.models import HoneycombQDMModel, SquareQDMModel, TriangularQDMModel
from qlinks.operators.plaquette import alternating_binary_patterns


def test_generic_local_cage_searcher_replaces_qdm_wrapper_on_full_square() -> None:
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

    generic_result = LocalCageSearcher.full_model_region(
        model,
        config=LocalCageSearchConfig(
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_candidate_count=128,
            ipr_random_seed=0,
        ),
    ).run()
    wrapper_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_candidate_count=128,
            ipr_random_seed=0,
        ),
    ).run()

    assert generic_result.counts_by_signature == wrapper_result.counts_by_signature
    assert generic_result.counts_by_signature == {(0, 4): 9, (0, 6): 1}


def test_generic_local_cage_searcher_accepts_explicit_qdm_adapter() -> None:
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

    searcher = LocalCageSearcher.from_plaquettes(
        model,
        plaquette_ids=[0],
        config=LocalCageSearchConfig(
            halo_layers=1,
            boundary_mode="relaxed",
            tolerance=1.0e-10,
        ),
        adapter=QDMLocalCageAdapter(model),
    )
    result = searcher.run()

    assert result.local_hilbert_size > 0
    assert result.region.link_ids.size < model.lattice.num_links


def test_stripe_motif_region_proposal_yields_small_regions_from_square_stripes() -> None:
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
    proposal = StripeMotifRegionProposal(
        model,
        sources=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        motif_sizes=(2,),
        subset_mode="all",
        max_records=4,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
        ),
    )

    records = list(proposal.iter_records())

    assert records
    assert all(isinstance(record, StripeMotifRegionProposalRecord) for record in records)
    assert all(record.motif_size == 2 for record in records)
    assert all(record.source == "stripe" for record in records)
    assert all(record.region.active_plaquette_ids.size == 2 for record in records)
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)


def test_robust_qdm_local_cage_search_accepts_stripe_motif_strategy() -> None:
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
    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe_motif",),
        stripe_motif_sources=("stripe",),
        stripe_motif_sizes=(2,),
        stripe_motif_subset_mode="all",
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=4,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=1,
        max_blocks=2,
        max_paddings_per_stage=0,
        padding_stages=("static",),
        store_full_states=False,
    )

    _certified, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert context.n_regions <= 4
    assert context.stage_names == ("static",)
    assert context.padding_config == robust_config.as_multi_padding_config()


def test_local_qdm_full_square_4x4_matches_exact_type1_counts() -> None:
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

    exact_build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    exact_result = CageSearcher.from_model_build_result(
        exact_build,
        config=CageSearchConfig(search_type="type1", tolerance=1.0e-10),
    ).run()

    assert local_result.local_hilbert_size == exact_build.basis.n_states == 132
    assert local_result.counts_by_signature == exact_result.counts_by_signature
    assert local_result.counts_by_signature == {(0, 4): 9, (0, 6): 1}


def test_local_qdm_full_honeycomb_4x4_matches_exact_type1_counts() -> None:
    model = HoneycombQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    local_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()

    exact_build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    exact_result = CageSearcher.from_model_build_result(
        exact_build,
        config=CageSearchConfig(search_type="type1", tolerance=1.0e-10),
    ).run()

    assert local_result.local_hilbert_size == exact_build.basis.n_states == 6
    assert local_result.counts_by_signature == exact_result.counts_by_signature == {}


def test_local_qdm_full_triangular_small_matches_exact_type1_counts() -> None:
    model = TriangularQDMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    local_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()

    exact_build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    exact_result = CageSearcher.from_model_build_result(
        exact_build,
        config=CageSearchConfig(search_type="type1", tolerance=1.0e-10),
    ).run()

    assert local_result.local_hilbert_size == exact_build.basis.n_states == 4
    assert local_result.counts_by_signature == exact_result.counts_by_signature == {(0, 2): 2}


def test_local_qdm_basis_enumeration_respects_shared_dfs_limits() -> None:
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
        config=LocalQDMCageSearchConfig(
            tolerance=1.0e-10,
            max_local_states=3,
            sort_basis=False,
        ),
    ).run()

    assert local_result.local_hilbert_size == 3


def test_stripe_region_proposal_generates_square_winding_stripes() -> None:
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
            tolerance=1.0e-10,
        ),
    )
    records = list(proposal.iter_records())

    assert len(records) == model.ly
    assert {record.direction for record in records} == {0}
    assert all(record.width == 1 for record in records)
    assert all(record.plaquette_kind == "square" for record in records)
    assert all(record.plaquette_ids.size == model.lx for record in records)
    assert all(
        record.region.active_plaquette_ids.tolist() == record.plaquette_ids.tolist()
        for record in records
    )
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)

    stripe_y_values = []
    for record in records:
        cells = [model.lattice.plaquette_anchor_cell(int(pid)) for pid in record.plaquette_ids]
        stripe_y_values.append(tuple(sorted({int(cell[1]) for cell in cells})))
        assert len({int(cell[0]) for cell in cells}) == model.lx

    assert sorted(stripe_y_values) == [(0,), (1,), (2,), (3,)]


def test_run_local_region_proposal_retains_stripe_metadata() -> None:
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
            tolerance=1.0e-10,
            prune_inactive_local_basis_states=True,
        ),
    )

    scan = run_local_region_proposal(proposal, max_regions=2)

    assert isinstance(scan, LocalRegionProposalSearchResult)
    assert len(scan) == 2
    assert [record.region_index for record in scan] == [0, 1]
    assert all(record.proposal_index == 0 for record in scan)
    assert all(record.proposal_record is not None for record in scan)
    assert all(hasattr(record.proposal_record, "plaquette_kind") for record in scan)
    assert all(record.result.local_hilbert_size > 0 for record in scan)
    assert len(scan.local_results) == 2


def test_collect_qdm_cage_blocks_from_region_proposals_respects_limits() -> None:
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
            tolerance=1.0e-10,
            prune_inactive_local_basis_states=True,
        ),
    )

    blocks = collect_qdm_cage_blocks_from_region_proposals(
        [proposal],
        max_regions=1,
        max_records_per_region=0,
    )

    assert blocks == []


def test_stripe_region_proposal_yields_ready_local_searchers() -> None:
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
            tolerance=1.0e-10,
            prune_inactive_local_basis_states=True,
        ),
    )
    searcher = next(proposal.iter_searchers())
    result = searcher.run()

    assert result.region.active_plaquette_ids.size == model.lx
    assert result.local_hilbert_size > 0
    assert result.region.link_ids.size < model.lattice.num_links


def test_stripe_region_proposal_groups_triangular_rhombus_kinds() -> None:
    model = TriangularQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    proposal = StripeRegionProposal(
        model,
        directions=(0,),
        width=1,
        plaquette_kinds=("rhombus_ab",),
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())

    assert len(records) == model.ly
    assert all(record.plaquette_kind == "rhombus_ab" for record in records)
    assert all(record.plaquette_ids.size == model.lx for record in records)


def test_snake_stripe_region_proposal_finds_square_noncontractible_cycles() -> None:
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

    proposal = SnakeStripeRegionProposal(
        model,
        max_plaquettes=4,
        min_plaquettes=4,
        max_records=16,
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())

    assert records
    assert all(isinstance(record, SnakeStripeRegionProposalRecord) for record in records)
    assert all(record.length == 4 for record in records)
    assert all(record.plaquette_ids.size == 4 for record in records)
    assert all(any(value != 0 for value in record.winding) for record in records)
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)
    absolute_windings = {tuple(abs(value) for value in record.winding) for record in records}
    assert {(1, 0), (0, 1)}.intersection(absolute_windings)


def test_snake_stripe_region_proposal_finds_honeycomb_snakes() -> None:
    model = HoneycombQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    proposal = SnakeStripeRegionProposal(
        model,
        max_plaquettes=4,
        min_plaquettes=4,
        max_records=16,
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())

    assert records
    assert all(record.plaquette_kinds == ("hexagon",) for record in records)
    assert all(any(value != 0 for value in record.winding) for record in records)
    assert all(
        record.region.active_plaquette_ids.size == record.plaquette_ids.size for record in records
    )


def test_snake_stripe_region_proposal_finds_triangular_rhombus_snakes() -> None:
    model = TriangularQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    proposal = SnakeStripeRegionProposal(
        model,
        max_plaquettes=3,
        min_plaquettes=3,
        max_records=16,
        plaquette_kinds=("rhombus_ab",),
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())

    assert records
    assert all(record.plaquette_kinds == ("rhombus_ab",) for record in records)
    assert all(any(value != 0 for value in record.winding) for record in records)
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)


def test_snake_stripe_region_proposal_filters_known_honeycomb_induced_snake() -> None:
    model = HoneycombQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=-2,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    proposal = SnakeStripeRegionProposal(
        model,
        max_plaquettes=8,
        min_plaquettes=8,
        max_records=64,
        require_induced_cycle=True,
        kind_pattern="constant_or_alternating",
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())
    target = (1, 2, 4, 7, 9, 10, 12, 15)

    assert target in {tuple(record.plaquette_ids.tolist()) for record in records}
    assert all(record.plaquette_kinds == ("hexagon",) for record in records)


def test_snake_stripe_region_proposal_filters_known_triangular_alternating_snake() -> None:
    model = TriangularQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    proposal = SnakeStripeRegionProposal(
        model,
        max_plaquettes=8,
        min_plaquettes=8,
        max_records=32,
        allow_kind_changes=True,
        kind_pattern="alternating",
        require_induced_cycle=True,
        config=LocalQDMCageSearchConfig(halo_layers=0, boundary_mode="relaxed"),
    )
    records = list(proposal.iter_records())
    target = (47, 48, 53, 54, 68, 69, 74, 75)

    assert target in {tuple(record.plaquette_ids.tolist()) for record in records}
    assert all(len(record.plaquette_kinds) == 2 for record in records)


def test_robust_config_can_use_snake_stripe_strategy() -> None:
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
        region_strategies=("snake_stripe",),
        max_region_plaquettes=4,
        min_region_plaquettes=4,
        max_regions_per_strategy=4,
        max_records_per_region=0,
        padding_stages=("static",),
    )

    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )

    assert certified.counts_by_signature == {}
    assert context.n_regions == 4
    assert context.n_blocks == 0


def test_adaptive_region_proposal_grows_from_seed_without_shape_assumption() -> None:
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

    proposal = AdaptiveRegionProposal(
        model,
        max_plaquettes=3,
        seed_plaquette_ids=[0],
        beam_width=2,
        branch_factor=3,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            tolerance=1.0e-10,
        ),
    )
    records = list(proposal.iter_records())

    assert records
    assert all(isinstance(record, AdaptiveRegionProposalRecord) for record in records)
    assert all(1 <= record.plaquette_ids.size <= 3 for record in records)
    assert all(0 in set(int(pid) for pid in record.seed_plaquette_ids) for record in records)
    assert any(record.plaquette_ids.size == 2 for record in records)
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)


def test_adaptive_region_proposal_runs_with_proposal_runner() -> None:
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

    proposal = AdaptiveRegionProposal(
        model,
        max_plaquettes=2,
        seed_plaquette_ids=[0],
        beam_width=2,
        branch_factor=2,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
        ),
    )

    scan = run_local_region_proposal(proposal, max_regions=2)

    assert len(scan) == 2
    assert all(record.proposal_record is not None for record in scan)
    assert all(hasattr(record.proposal_record, "score") for record in scan)
    assert all(record.result.local_hilbert_size > 0 for record in scan)


def test_connected_region_proposal_enumerates_connected_sets_under_budget() -> None:
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

    proposal = ConnectedRegionProposal(
        model,
        max_plaquettes=2,
        seed_plaquette_ids=[0],
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            tolerance=1.0e-10,
        ),
    )
    records = list(proposal.iter_records())

    assert records
    assert all(isinstance(record, ConnectedRegionProposalRecord) for record in records)
    assert all(1 <= record.plaquette_ids.size <= 2 for record in records)
    assert any(record.plaquette_ids.size == 2 for record in records)
    assert all(record.seed_plaquette_id == 0 for record in records)
    assert all(record.region.link_ids.size < model.lattice.num_links for record in records)


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


def test_qdm_block_collection_stops_after_max_blocks() -> None:
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
    local_config = LocalQDMCageSearchConfig(
        halo_layers=0,
        boundary_mode="relaxed",
        prune_inactive_local_basis_states=True,
        tolerance=1.0e-10,
    )
    proposal = StripeRegionProposal(
        model,
        directions=(0,),
        width=1,
        config=local_config,
    )

    scan, blocks = collect_qdm_cage_blocks_with_scan_from_region_proposals(
        [proposal],
        model=model,
        signatures=((0, 2),),
        max_records_per_region=2,
        max_blocks=2,
    )

    assert len(blocks) == 2
    assert len(scan) == 1
    assert scan.records[0].counts_by_signature[(0, 2)] == 2


def test_robust_qdm_context_uses_streaming_block_collection() -> None:
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
    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=None,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=2,
        max_paddings_per_stage=0,
        padding_stages=("static",),
        store_full_states=False,
    )

    _certified, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert context.n_blocks == 2
    assert context.n_regions == 1


def test_robust_qdm_local_cage_search_returns_certified_result_under_small_budget() -> None:
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

    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=2,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=2,
        max_paddings_per_stage=2,
        max_paddings_per_packing=1,
        max_product_support_size=2048,
        include_sectors=True,
        padding_stages=("static",),
        tolerance=1.0e-9,
        store_full_states=False,
    )

    certified = robust_qdm_local_cage_search(model, config=robust_config)

    assert certified.hilbert_size == certified.basis.n_states
    assert isinstance(certified.counts_by_signature, dict)

    certified_with_context, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert certified_with_context.counts_by_signature == certified.counts_by_signature
    assert isinstance(context, RobustQDMLocalCageSearchContext)
    assert context.n_regions > 0
    assert context.n_blocks >= 0
    assert context.stage_names == ("static",)
    assert set(context.n_paddings_by_stage) == {"static"}
    assert set(context.n_certified_by_stage) == {"static"}
    assert set(context.failure_counts_by_stage) == {"static"}
    assert context.padding_config == robust_config.as_multi_padding_config()


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


def test_robust_square_qdm_4x4_w00_recovers_known_stripe_pair_cages() -> None:
    """Recover the eight known two-stripe local-padding cages in 4x4 W00 QDM.

    The full exact search has nine ``(0, 4)`` cages.  Empirically, eight of
    them decompose into two separated local ``(0, 2)`` stripe cages with one
    inactive spacer stripe between them.  This test keeps the local-first path
    honest without requiring full-basis ED or relying on the ninth non-stripe
    cage.
    """
    model = _square_qdm_w00_model(4, 4)
    certified, context = robust_qdm_local_cage_search(
        model,
        config=_square_qdm_stripe_pair_robust_config(),
        return_context=True,
    )

    assert certified.counts_by_signature == {(0, 4): 8}
    assert all(record.cage_state.support_size == 4 for record in certified.records)
    assert all(report.one_hop_shell_size == 12 for report in certified.reports)
    assert max(report.full_residual for report in certified.reports) < 1.0e-8
    assert max(report.leakage_residual for report in certified.reports) < 1.0e-8

    assert context.n_regions == 8
    assert context.n_blocks == 16
    assert context.n_paddings_by_stage == {"static": 16}
    assert context.n_certified_by_stage == {"static": 8}
    assert context.failure_counts_by_stage == {"static": {"leakage_residual": 8}}


@pytest.mark.manual
@pytest.mark.skip(
    reason=(
        "Manual diagnostic scaffold for larger square-QDM stripe stacking; "
        "enable locally when tuning robust local/padding budgets."
    )
)
def test_robust_square_qdm_4x8_w00_stripe_stacking_diagnostic_manual() -> None:
    """Diagnostic scaffold for the known 4xLy stripe-stacking family.

    The robust local-first solver is not expected to recover every exact cage,
    but the stripe-pair mechanism should produce non-empty certified results
    under a sufficiently permissive budget.  Keep this manual until the exact
    4x8 target counts and runtime budget are finalized.
    """
    model = _square_qdm_w00_model(4, 8)
    config = _square_qdm_stripe_pair_robust_config(
        max_paddings_per_stage=64,
        max_paddings_per_packing=16,
        max_product_support_size=1024,
        stripe_directions=(0,),
    )

    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )

    assert context.n_regions > 0
    assert context.n_blocks > 0
    assert context.n_paddings_by_stage["static"] > 0
    assert len(certified) > 0
    assert all(report.full_residual < 1.0e-8 for report in certified.reports)


def test_local_qdm_active_plaquette_hook_prunes_kinetically_inactive_states() -> None:
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
    adapter = QDMLocalCageAdapter(model)
    config = LocalQDMCageSearchConfig(
        halo_layers=0,
        boundary_mode="relaxed",
        tolerance=1.0e-10,
    )
    region = adapter.build_region_from_plaquettes(
        plaquette_ids=[0],
        config=config,
    )

    unpruned = enumerate_qdm_local_basis(
        model,
        region,
        include_sectors_when_full=False,
        prune_inactive_states=False,
        sort=True,
    )
    pruned = enumerate_qdm_local_basis(
        model,
        region,
        include_sectors_when_full=False,
        prune_inactive_states=True,
        sort=True,
    )

    assert 0 < pruned.n_states < unpruned.n_states

    local_index_by_link = {int(link_id): i for i, link_id in enumerate(region.link_ids)}
    plaquette_variables = np.asarray(
        [local_index_by_link[int(link_id)] for link_id in model.lattice.plaquette_links(0)],
        dtype=np.int64,
    )
    pattern0, pattern1 = alternating_binary_patterns(int(plaquette_variables.size))
    for state in pruned.states:
        local_values = state[plaquette_variables]
        assert np.array_equal(local_values, pattern0) or np.array_equal(local_values, pattern1)


@pytest.mark.manual
@pytest.mark.skip(reason="Triangular 4x4 full local/exact comparison is a slow manual regression.")
def test_local_qdm_full_triangular_4x4_matches_exact_type1_counts_manual() -> None:
    model = TriangularQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )

    local_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()

    exact_build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    exact_result = CageSearcher.from_model_build_result(
        exact_build,
        config=CageSearchConfig(search_type="type1", tolerance=1.0e-10),
    ).run()

    assert local_result.local_hilbert_size == exact_build.basis.n_states
    assert local_result.counts_by_signature == exact_result.counts_by_signature


def test_local_qdm_plaquette_region_uses_small_relaxed_basis() -> None:
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

    local_result = LocalQDMCageSearcher.from_plaquettes(
        model,
        plaquette_ids=[0],
        config=LocalQDMCageSearchConfig(
            halo_layers=1,
            boundary_mode="relaxed",
            tolerance=1.0e-10,
        ),
    ).run()

    assert local_result.local_hilbert_size > 0
    assert local_result.region.link_ids.size < model.lattice.num_links
    assert local_result.local_hilbert_size < 2 ** int(local_result.region.link_ids.size)
    assert local_result.region.active_plaquette_ids.size >= 1
    # The local region is intentionally open to the exterior.  The result may
    # or may not contain a cage, but it should carry the unresolved kinetic
    # boundary information needed by a later padding/global-certification layer.
    assert isinstance(local_result.region.unresolved_boundary_plaquette_ids, np.ndarray)


def test_local_qdm_square_4x4_ipr_degenerate_policy_finds_compact_supports() -> None:
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

    localized_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_candidate_count=256,
            ipr_random_seed=0,
        ),
    ).run()

    plain_result = LocalQDMCageSearcher.full_model_region(
        model,
        config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
    ).run()

    localized_support_sizes = sorted(
        record.cage_state.support_size for record in localized_result.records
    )
    plain_support_sizes = sorted(record.cage_state.support_size for record in plain_result.records)

    assert localized_result.counts_by_signature == plain_result.counts_by_signature
    assert localized_support_sizes[0] < plain_support_sizes[0]
    assert localized_support_sizes[:8] == [4] * 8
    assert any(
        record.cage_state.metadata.get("degenerate_basis_strategy") == "ipr"
        for record in localized_result.records
    )


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

    certified = local_result.certify_paddings(
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

    certified = (
        LocalQDMCageSearcher.full_model_region(
            model,
            config=LocalQDMCageSearchConfig(tolerance=1.0e-10),
        )
        .run()
        .certify_paddings(config=LocalQDMPaddingConfig(tolerance=1.0e-9))
    )

    record = certified.first((0, 4))
    report = classify_cage_state(
        record.cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
    )

    assert report.support_size == record.cage_state.support_size
    assert report.n_nontrivial_zeros >= 0


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

    classification = classify_cage_state(
        record.cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
    )
    assert classification.support_size == record.cage_state.support_size

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
