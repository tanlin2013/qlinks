from __future__ import annotations

import numpy as np

from qlinks.caging import (
    QDMSingletTNProblem,
    analyze_square_qdm_singlet_product_tilings,
    analyze_square_qdm_singlet_stripe_product,
    build_square_qdm_singlet_boundary_tile,
    enumerate_square_qdm_singlet_exact_covers,
    square_qdm_two_plaquette_singlet_blocks,
)
from qlinks.models import SquareQDMModel


def _square_model(lx: int, ly: int) -> SquareQDMModel:
    return SquareQDMModel(
        lx=lx,
        ly=ly,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )


def test_find_all_two_plaquette_singlet_translations_on_6x6() -> None:
    model = _square_model(6, 6)

    blocks = square_qdm_two_plaquette_singlet_blocks(model)

    assert len(blocks) == 72
    assert sum(block.direction == "x" for block in blocks) == 36
    assert sum(block.direction == "y" for block in blocks) == 36
    assert all(block.support_size == 2 for block in blocks)
    assert all(len(block.covered_site_ids) == 6 for block in blocks)
    for block in blocks:
        amplitudes = block.block.amplitudes
        assert np.isclose(abs(amplitudes[0]), 1.0 / np.sqrt(2.0))
        assert np.isclose(amplitudes[1] / amplitudes[0], -1.0)


def test_square_singlet_stripe_product_has_full_rank_leakage() -> None:
    model = _square_model(6, 4)

    stripe = analyze_square_qdm_singlet_stripe_product(
        model,
        direction="x",
        transverse_coordinate=0,
        max_paddings=1,
    )

    assert stripe.failure_reason is None
    assert stripe.n_blocks == 2
    assert len(stripe.subspace_reports) == 1
    report = stripe.subspace_reports[0]
    assert report.support_size == 4
    assert report.leakage_rank == 4
    assert report.leakage_nullity == 0
    assert report.is_ruled_out_within_product_support
    assert not report.has_exact_state
    assert not report.product_state_is_exact

    problem = QDMSingletTNProblem.from_report(report)
    assert problem.physical_dimensions == (2, 2)
    assert problem.requires_enlarged_local_basis
    assert problem.loss(report.padding.global_amplitudes) > 0.0


def test_6x6_exact_covers_and_sampled_no_go_reports() -> None:
    model = _square_model(6, 6)
    blocks = square_qdm_two_plaquette_singlet_blocks(model)

    tilings = enumerate_square_qdm_singlet_exact_covers(
        model,
        singlet_blocks=blocks,
    )
    sampled = analyze_square_qdm_singlet_product_tilings(
        model,
        max_tilings=2,
    )

    assert len(tilings) == 120
    assert {(tiling.n_horizontal, tiling.n_vertical) for tiling in tilings} == {
        (6, 0),
        (0, 6),
    }
    assert len(sampled.records) == 2
    assert sampled.all_full_rank_no_go
    assert all(record.report.support_size == 64 for record in sampled.records)
    assert all(record.report.leakage_rank == 64 for record in sampled.records)
    assert all(record.report.leakage_nullity == 0 for record in sampled.records)


def test_boundary_resolved_halo_enlarges_the_local_physical_basis() -> None:
    model = _square_model(6, 6)
    singlet = square_qdm_two_plaquette_singlet_blocks(
        model,
        directions=("x",),
    )[0]

    tile = build_square_qdm_singlet_boundary_tile(
        model,
        singlet,
        halo_layers=1,
    )

    assert tile.enlarged_dimension == 786
    assert tile.core_compatible_dimension == 72
    assert tile.core_sector_indices[0].size == 36
    assert tile.core_sector_indices[1].size == 36
    assert tile.virtual_signature_count == 230
    assert tile.basis.local_hamiltonian.shape == (786, 786)
