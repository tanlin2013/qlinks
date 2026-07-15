from __future__ import annotations

import numpy as np

from qlinks.caging import (
    LocalWitnessEmbeddingRecord,
    LocalWitnessFamily,
    ReducedIZPatternSupport,
    SquareQDMStripTransferMatrix,
    SquareQDMWitnessPlacement,
    evaluate_local_witness_on_diagonal_ensemble,
    evaluate_square_qdm_witness_family_on_strips,
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


def test_periodic_winding_sector_counts_match_known_4x4_decomposition() -> None:
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    counts = transfer.periodic_winding_sector_counts(length=4)
    by_label = {sector.label: count for sector, count in counts.items()}

    assert np.isclose(sum(by_label.values()), 272.0)
    assert np.isclose(by_label[(0, 0)], 132.0)
    assert np.isclose(by_label[(0, 2)], 32.0)
    assert np.isclose(by_label[(2, 0)], 32.0)
    assert np.isclose(by_label[(0, 4)], 1.0)
    assert np.isclose(by_label[(4, 0)], 1.0)


def test_winding_resolved_witness_matches_explicit_w00_basis() -> None:
    reference_model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(reference_model)
    placement = SquareQDMWitnessPlacement.from_local_witness(reference_model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    strip = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
        winding_sector=(0, 0),
    )
    sector_model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )
    basis = sector_model.build_basis(solver="dfs")
    exact = evaluate_local_witness_on_diagonal_ensemble(
        witness,
        basis_configs=basis.states,
    )

    assert strip.winding_sector is not None
    assert strip.winding_sector.label == (0, 0)
    assert np.isclose(strip.partition_count, 132.0)
    assert np.isclose(strip.weighted_count, 20.0)
    assert np.isclose(strip.expectation, exact.expectation)


def test_winding_resolved_periodic_scan_tracks_one_sector() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    report = transfer.scan_witness(
        placement,
        lengths=(4, 6, 8),
        boundary_x="periodic",
        winding_sector=(0, 0),
    )

    assert report.winding_sector is not None
    assert report.winding_sector.label == (0, 0)
    assert report.lengths == (4, 6, 8)
    assert all(
        evaluation.winding_sector == report.winding_sector for evaluation in report.evaluations
    )
    assert all(expectation > 0.0 for expectation in report.expectations)


def test_winding_resolution_rejects_open_x_boundaries() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    with np.testing.assert_raises_regex(ValueError, "requires periodic x"):
        transfer.evaluate_witness(
            placement,
            length=8,
            boundary_x="open",
            winding_sector=(0, 0),
        )


def test_common_witness_family_can_be_evaluated_directly_on_strips() -> None:
    model_4x4 = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    model_6x4 = SquareQDMModel(lx=6, ly=4, boundary_condition="periodic")
    witness_4x4 = _plaquette_flip_witness(model_4x4)
    witness_6x4_embedding = _plaquette_flip_witness(model_6x4)
    witness_6x4 = witness_4x4.template.instantiate(witness_6x4_embedding.variable_indices)
    family = LocalWitnessFamily(
        template=witness_4x4.template,
        embeddings=(
            LocalWitnessEmbeddingRecord(
                system_label="4x4",
                witnesses=(witness_4x4,),
            ),
            LocalWitnessEmbeddingRecord(
                system_label="6x4",
                witnesses=(witness_6x4,),
            ),
        ),
    )

    report = evaluate_square_qdm_witness_family_on_strips(
        family,
        models={"4x4": model_4x4, "6x4": model_6x4},
        lengths={"4x4": (4,), "6x4": (6,)},
        boundary_x="periodic",
        winding_sector=(0, 0),
        winding_projection="fourier",
    )

    assert report.system_labels == ("4x4", "6x4")
    assert np.isclose(
        report.record_for("4x4").scaling_report.expectations[0],
        20.0 / 132.0,
    )
    assert report.record_for("6x4").scaling_report.winding_sector is not None
    assert (
        report.record_for("4x4").scaling_report.evaluations[0].metadata["exact_fourier_projection"]
        is True
    )


def test_fourier_winding_projection_matches_dynamic_programming() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    dynamic = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
        winding_sector=(0, 0),
        winding_projection="dynamic_programming",
    )
    fourier = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
        winding_sector=(0, 0),
        winding_projection="fourier",
    )

    assert np.isclose(fourier.partition_count, dynamic.partition_count)
    assert np.isclose(fourier.weighted_count, dynamic.weighted_count)
    assert np.isclose(fourier.expectation, dynamic.expectation)
    assert fourier.metadata["exact_fourier_projection"] is True


def test_fourier_projection_allows_insertion_across_canonical_seam() -> None:
    model = SquareQDMModel(lx=4, ly=4, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    result = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
        insertion_x=3,
        winding_sector=(0, 0),
        winding_projection="fourier",
    )

    assert np.isclose(result.partition_count, 132.0)
    assert np.isclose(result.weighted_count, 20.0)
    assert result.metadata["insertion_crosses_canonical_seam"] is True


def test_fourier_sector_counts_match_dynamic_programming() -> None:
    transfer = SquareQDMStripTransferMatrix(circumference=4)

    dynamic = transfer.periodic_winding_sector_counts(
        length=4,
        winding_projection="dynamic_programming",
    )
    fourier = transfer.periodic_winding_sector_counts(
        length=4,
        winding_projection="fourier",
    )

    significant_dynamic = {sector: count for sector, count in dynamic.items() if count > 1.0e-8}
    assert set(significant_dynamic) == set(fourier)
    for sector, count in significant_dynamic.items():
        assert np.isclose(fourier[sector], count)


def test_auto_winding_projection_lifts_boundary_state_ceiling() -> None:
    model = SquareQDMModel(lx=4, ly=10, boundary_condition="periodic")
    witness = _plaquette_flip_witness(model)
    placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
    transfer = SquareQDMStripTransferMatrix(circumference=10)

    result = transfer.evaluate_witness(
        placement,
        length=4,
        boundary_x="periodic",
        winding_sector=(0, 0),
    )

    assert transfer.n_boundary_states == 1024
    assert result.expectation > 0.0
    assert result.metadata["contraction"] == ("electric_winding_fourier_projected_dense_transfer")
