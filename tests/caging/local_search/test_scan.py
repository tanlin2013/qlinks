from __future__ import annotations

from qlinks.caging.local_search import (
    LocalQDMCageSearchConfig,
    LocalRegionProposalSearchResult,
    StripeRegionProposal,
    collect_qdm_cage_blocks_from_region_proposals,
    collect_qdm_cage_blocks_with_scan_from_region_proposals,
    run_local_region_proposal,
)
from qlinks.models import SquareQDMModel


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
