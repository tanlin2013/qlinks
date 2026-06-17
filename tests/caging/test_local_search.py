from __future__ import annotations

import numpy as np
import pytest

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
    LocalCageSearchConfig,
    LocalCageSearcher,
    LocalQDMCageSearchConfig,
    LocalQDMCageSearcher,
    LocalQDMPaddingConfig,
    QDMLocalCageAdapter,
    StripeRegionProposal,
    classify_cage_state,
    enumerate_qdm_local_basis,
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
