from __future__ import annotations

import numpy as np
import pytest

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
    LocalQDMCageSearchConfig,
    LocalQDMCageSearcher,
    LocalQDMPaddingConfig,
    classify_cage_state,
)
from qlinks.models import HoneycombQDMModel, SquareQDMModel, TriangularQDMModel


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
