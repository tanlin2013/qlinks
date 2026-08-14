from __future__ import annotations

from functools import lru_cache

import numpy as np

from qlinks.caging import (
    CageClassificationConfig,
    SquareQDMPeriodicProductUnitCell,
    certify_local_witness_on_square_qdm_periodic_sequence,
    certify_square_qdm_periodic_product_sequence,
    classify_cage_state,
    evaluate_square_qdm_classification_witnesses_on_strips,
    local_witnesses_from_classification_report,
    scan_square_qdm_beta_zero_energy_density,
)
from qlinks.caging.local_search import (
    LocalQDMCageSearchConfig,
    RobustQDMLocalCageSearchConfig,
    robust_qdm_local_cage_search,
)
from qlinks.models import SquareQDMModel


@lru_cache(maxsize=1)
def _stripe_cage_fixture():
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
    return model, certified, context


REPEATABLE_X_REPORT_INDEX = 4


def _repeatable_x_classification_report():
    _model, certified, _context = _stripe_cage_fixture()
    return classify_cage_state(
        certified.records[REPEATABLE_X_REPORT_INDEX].cage_state,
        kinetic_matrix=certified.kinetic_matrix,
        basis_configs=certified.basis.states,
        hilbert_size=certified.hilbert_size,
        config=CageClassificationConfig(sector_policy="infer_support_component"),
    )


def test_periodic_stripe_product_certifies_infinite_sequence() -> None:
    model, certified, context = _stripe_cage_fixture()
    unit_cell = SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[REPEATABLE_X_REPORT_INDEX].padding,
        repeat_axis="x",
    )

    sequence = certify_square_qdm_periodic_product_sequence(unit_cell)

    assert sequence.is_certified
    assert sequence.minimum_proven_repeats == 1
    assert sequence.is_symmetry_resolved
    assert sequence.unit_cell_winding_sector == (0, 0)
    assert sequence.winding_sector_for_repeats(7) == (0, 0)
    assert sequence.support_size_per_unit_cell == 4
    assert sequence.formal_support_size(5) == 4**5
    assert np.isclose(sequence.unit_cell_energy, 4.0)
    assert np.isclose(sequence.energy_for_repeats(7), 28.0)
    assert np.isclose(sequence.energy_density, 0.25)
    assert all(check.is_certified for check in sequence.finite_checks)
    assert all(check.n_flippable_inert_patterns == 0 for check in sequence.finite_checks)
    assert all(check.max_site_constraint_residual == 0 for check in sequence.finite_checks)


def test_actual_cage_witness_annihilates_entire_periodic_sequence() -> None:
    model, certified, context = _stripe_cage_fixture()
    unit_cell = SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[REPEATABLE_X_REPORT_INDEX].padding,
        repeat_axis="x",
    )
    sequence = certify_square_qdm_periodic_product_sequence(unit_cell)
    witness = local_witnesses_from_classification_report(_repeatable_x_classification_report())[0]

    certificate = certify_local_witness_on_square_qdm_periodic_sequence(
        sequence,
        witness,
        normalization="operator_norm",
    )

    assert certificate.is_infinite_sequence_witness
    assert certificate.annihilation_residual < 1.0e-12
    assert certificate.q_expectation < 1.0e-24
    assert np.isclose(certificate.witness.q_operator_norm, 1.0)


def test_actual_cage_witness_has_positive_same_sector_thermal_weight() -> None:
    model, _certified, _context = _stripe_cage_fixture()
    report = evaluate_square_qdm_classification_witnesses_on_strips(
        _repeatable_x_classification_report(),
        model=model,
        lengths=(4, 8, 12),
        winding_sector=(0, 0),
        normalization="operator_norm",
        winding_projection="fourier",
    )

    assert report.records
    assert all(np.isclose(record.witness.q_operator_norm, 1.0) for record in report.records)
    assert all(record.scaling_report.expectations[-1] > 0.0 for record in report.records)
    assert report.minimum_tail_expectation > 0.0


def test_pure_kinetic_sequence_matches_beta_zero_energy_exactly() -> None:
    model, certified, context = _stripe_cage_fixture()
    unit_cell = SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[REPEATABLE_X_REPORT_INDEX].padding,
        repeat_axis="x",
    ).with_couplings(coup_pot=0.0)
    sequence = certify_square_qdm_periodic_product_sequence(unit_cell)
    scaling = scan_square_qdm_beta_zero_energy_density(
        ((4, 4), (8, 4), (12, 4), (16, 4)),
        potential_coupling=0.0,
        winding_sector=(0, 0),
        winding_projection="fourier",
    )
    match = sequence.match_energy_density(
        scaling.evaluations[-1].energy_density,
        tolerance=1.0e-12,
    )
    witness_certificate = certify_local_witness_on_square_qdm_periodic_sequence(
        sequence,
        local_witnesses_from_classification_report(_repeatable_x_classification_report())[0],
        normalization="operator_norm",
    )

    assert sequence.is_certified
    assert np.isclose(sequence.energy_density, 0.0)
    assert all(np.isclose(value, 0.0) for value in scaling.energy_densities)
    assert match.is_matched
    assert witness_certificate.is_infinite_sequence_witness


def test_frozen_product_tile_certifies_true_two_dimensional_sequence() -> None:
    from qlinks.caging import (
        SquareQDMBiperiodicProductTile,
        certify_square_qdm_biperiodic_product_sequence,
    )
    from qlinks.caging.local_search import FactorizedLocalQDMPadding

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    frozen_config = np.asarray(
        [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
        ],
        dtype=np.int64,
    )
    padding = FactorizedLocalQDMPadding(
        block_ids=(),
        exterior_link_ids=np.arange(model.lattice.num_links, dtype=np.int64),
        exterior_config=frozen_config,
    )
    tile = SquareQDMBiperiodicProductTile(
        model=model,
        blocks=(),
        padding=padding,
    )

    certificate = certify_square_qdm_biperiodic_product_sequence(tile)

    assert certificate.is_certified
    assert certificate.is_true_2d_sequence
    assert certificate.minimum_proven_repeats == (1, 1)
    assert certificate.formal_support_size(7, 9) == 1
    assert np.isclose(certificate.energy_density, 0.0)
    assert certificate.winding_sector_for_repeats(2, 3) == (6, -4)
    assert len(certificate.finite_checks) == 9
    assert all(check.is_certified for check in certificate.finite_checks)


def test_repeatable_stripe_is_diagnosed_as_transverse_seam_failure() -> None:
    from qlinks.caging import (
        SquareQDMBiperiodicProductTile,
        diagnose_square_qdm_biperiodic_repeatability,
    )

    model, certified, context = _stripe_cage_fixture()
    tile = SquareQDMBiperiodicProductTile.from_padding(
        model,
        context.blocks,
        certified.reports[REPEATABLE_X_REPORT_INDEX].padding,
    )

    diagnosis = diagnose_square_qdm_biperiodic_repeatability(
        tile,
        check_smaller_repeats=False,
    )

    assert not diagnosis.is_certified
    assert len(diagnosis.failed_checks) == 1
    failed = diagnosis.failed_checks[0]
    assert failed.seam_diagnostics.max_site_constraint_residuals["internal"] == 0
    assert failed.seam_diagnostics.flippable_inert_patterns["internal"] == 0
    assert failed.seam_diagnostics.max_site_constraint_residuals["y_seam"] > 0
    assert failed.seam_diagnostics.flippable_inert_patterns["y_seam"] > 0


def test_direct_biperiodic_search_reports_failure_mechanisms() -> None:
    from qlinks.caging import (
        SquareQDMBiperiodicTileSearchConfig,
        search_square_qdm_biperiodic_product_tiles,
    )

    model, _certified, context = _stripe_cage_fixture()
    result = search_square_qdm_biperiodic_product_tiles(
        model,
        context.blocks,
        config=SquareQDMBiperiodicTileSearchConfig(
            min_blocks=2,
            max_blocks=2,
            max_padding_attempts=8,
            max_paddings_per_packing=1,
            max_results=8,
            verification_repeats=3,
            check_smaller_repeats=False,
            require_kinetic_separation=False,
        ),
    )

    assert result.n_padding_candidates_examined > 0
    assert result.records
    assert not result.certified_records
    assert result.failure_counts
    assert all(record.failure_reason is not None for record in result.failed_records)


def test_checkerboard_periodic_product_has_exact_all_repeat_certificate() -> None:
    """Protect the exact 4N x 4 checkerboard periodic-product cancellation theorem."""
    from qlinks.caging.checkerboard_exact import certify_checkerboard_periodic_product_exact

    model, certified, context = _stripe_cage_fixture()
    unit_cell = SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[REPEATABLE_X_REPORT_INDEX].padding,
        repeat_axis="x",
    )

    certificate = certify_checkerboard_periodic_product_exact(unit_cell)

    assert certificate.exact_for_all_positive_repeats
    assert certificate.active_plaquette_columns == (0, 2)
    assert certificate.inactive_plaquette_columns == (1, 3)
    assert certificate.boundary_inactive_exact
    assert certificate.checkerboard_phase_pairs_exact
    assert certificate.kinetic_symbolic_residual_terms == 0
    assert certificate.flippable_plaquettes_per_support_state == 4
    assert certificate.unit_cell_energy_per_lambda == 4
