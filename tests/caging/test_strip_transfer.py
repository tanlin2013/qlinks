from __future__ import annotations

import numpy as np

from qlinks.caging import (
    ReducedIZPatternSupport,
    SquareQDMStripTransferMatrix,
    SquareQDMWitnessPlacement,
    evaluate_local_witness_on_diagonal_ensemble,
    local_witness_template_from_pattern_support,
)
from qlinks.models import SquareQDMModel


def _plaquette_flip_witness(
    model: SquareQDMModel,
    *,
    x: int = 0,
    y: int = 0,
):
    plaquette_id = model.lattice.plaquette_id_from_cell(x, y)
    variable_indices = tuple(
        model.layout.link_variable_index(int(link_id))
        for link_id in model.lattice.plaquette_links(plaquette_id)
    )
    pattern_support = ReducedIZPatternSupport(
        pattern_key=(
            (
                (1, 0, 1, 0),
                (0, 1, 0, 1),
                (1.0, 0.0),
            ),
        ),
        variable_indices=variable_indices,
        source_zero_indices=(),
        mechanism_labels=(),
    )
    template = local_witness_template_from_pattern_support(pattern_support)
    return template.instantiate(variable_indices)


def test_square_qdm_column_transfer_counts_small_open_cylinders() -> None:
    transfer = SquareQDMStripTransferMatrix(circumference=4)
    matrix = transfer.transfer_matrix

    assert matrix.shape == (16, 16)
    assert np.isclose(matrix[0, 0], 2.0)
    assert np.isclose((matrix @ matrix)[0, 0], 9.0)


def test_periodic_strip_witness_matches_explicit_constrained_basis() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    strip = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
    )
    basis = model.build_basis(solver="dfs")
    exact = evaluate_local_witness_on_diagonal_ensemble(
        witness,
        basis_configs=basis.states,
    )

    assert placement.window_width == 2
    assert np.isclose(strip.partition_count, basis.n_states)
    assert np.isclose(strip.weighted_count, 32.0)
    assert np.isclose(strip.expectation, exact.expectation)


def test_periodic_reference_seam_is_unwrapped_to_a_bounded_placement() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model, x=3, y=0)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    strip = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
    )

    assert placement.reference_origin_x == 3
    assert placement.window_width == 2
    assert max(coordinate.x for coordinate in placement.link_coordinates) == 1
    assert np.isclose(strip.expectation, 32.0 / 272.0)


def test_single_link_lowering_has_zero_projected_qdm_weight() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    variable_index = model.layout.link_variable_index(0)
    pattern_support = ReducedIZPatternSupport(
        pattern_key=(((1,), (0,), (1.0, 0.0)),),
        variable_indices=(variable_index,),
        source_zero_indices=(),
        mechanism_labels=(),
    )
    witness = local_witness_template_from_pattern_support(pattern_support).instantiate(
        (variable_index,)
    )
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    strip = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
    )
    basis = model.build_basis(solver="dfs")
    exact = evaluate_local_witness_on_diagonal_ensemble(
        witness,
        basis_configs=basis.states,
    )

    assert np.isclose(strip.expectation, 0.0)
    assert np.isclose(exact.expectation, 0.0)


def test_open_strip_scan_centers_the_witness_and_reports_tail_spread() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    report = transfer.scan_witness(
        placement,
        lengths=(4, 6, 8, 10),
        boundary_x="open",
    )
    tail = report.tail_estimate(tail_points=3)

    assert report.lengths == (4, 6, 8, 10)
    assert tuple(evaluation.insertion_x for evaluation in report.evaluations) == (1, 2, 3, 4)
    assert tail["lengths"] == (6, 8, 10)
    assert float(tail["spread"]) > 0.0
    assert all(expectation > 0.0 for expectation in report.expectations)


def test_periodic_strip_scan_reuses_one_spectral_contraction() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    report = transfer.scan_witness(
        placement,
        lengths=(4, 6),
        boundary_x="periodic",
    )

    assert report.lengths == (4, 6)
    assert np.isclose(report.expectations[0], 32.0 / 272.0)
    assert all(evaluation.insertion_x is None for evaluation in report.evaluations)
