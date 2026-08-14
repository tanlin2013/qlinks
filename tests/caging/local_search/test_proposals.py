from __future__ import annotations

from qlinks.caging.local_search import (
    AdaptiveRegionProposal,
    AdaptiveRegionProposalRecord,
    ConnectedRegionProposal,
    ConnectedRegionProposalRecord,
    LocalQDMCageSearchConfig,
    SnakeStripeRegionProposal,
    SnakeStripeRegionProposalRecord,
    StripeMotifComponentRegionProposal,
    StripeMotifComponentRegionProposalRecord,
    StripeMotifRegionProposal,
    StripeMotifRegionProposalRecord,
    StripeRegionProposal,
    run_local_region_proposal,
)
from qlinks.models import HoneycombQDMModel, SquareQDMModel, TriangularQDMModel


def test_stripe_motif_component_proposal_merges_seeded_square_stripes() -> None:
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
    proposal = StripeMotifComponentRegionProposal(
        model,
        sources=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        motif_sizes=(2,),
        motif_subset_mode="windows",
        min_seed_motifs=1,
        max_seed_motifs_per_stripe=1,
        component_subset_mode="full",
        max_components_per_stripe=1,
        max_records=2,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
        ),
    )

    records = list(proposal.iter_records())

    assert records
    assert all(isinstance(record, StripeMotifComponentRegionProposalRecord) for record in records)
    assert all(record.source == "stripe" for record in records)
    assert all(record.component_size == model.lx for record in records)
    assert all(record.n_seed_motifs >= 1 for record in records)
    assert all(record.region.active_plaquette_ids.size == model.lx for record in records)


def test_stripe_motif_component_proposal_can_emit_triangular_snake_windows() -> None:
    model = TriangularQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )
    proposal = StripeMotifComponentRegionProposal(
        model,
        sources=("snake_stripe",),
        motif_sizes=(2,),
        motif_subset_mode="windows",
        min_seed_motifs=1,
        max_seed_motifs_per_stripe=1,
        component_sizes=(4,),
        component_subset_mode="windows",
        max_components_per_stripe=1,
        max_records=4,
        max_links=30,
        snake_max_plaquettes=8,
        snake_min_plaquettes=8,
        snake_allow_kind_changes=True,
        snake_kind_pattern="alternating",
        snake_require_induced_cycle=True,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
        ),
    )

    records = list(proposal.iter_records())

    assert records
    assert all(record.source == "snake_stripe" for record in records)
    assert all(record.component_size == 4 for record in records)
    assert all(record.n_seed_motifs >= 1 for record in records)
    assert all(record.region.link_ids.size <= 30 for record in records)


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
