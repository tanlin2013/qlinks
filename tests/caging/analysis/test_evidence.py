from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qlinks.caging.analysis.evidence import (
    Quasi1DSequencePoint,
    audit_quasi_1d_sequence,
    beta_zero_matching_subspace,
    cage_finite_size_scorecard,
    operator_coefficient_compatibility,
    project_coefficients_to_beta_zero_match,
    scan_windowed_operator_annihilators,
)


def test_cage_finite_size_scorecard_separates_shell_and_support() -> None:
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    state = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    report = cage_finite_size_scorecard(
        hamiltonian,
        candidate_shell=(0, 1),
        state=state,
        kinetic=hamiltonian,
    )
    assert report.is_certified
    assert report.boundary_shape == (1, 2)
    assert report.boundary_rank == 1
    assert report.boundary_nullity == 1
    assert report.actual_support_size == 2
    assert report.boundary_residual < 1.0e-12


def test_windowed_annihilator_scan_finds_two_operator_cancellation() -> None:
    op_a = sp.csr_array(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    op_b = -op_a
    op_c = sp.csr_array(np.asarray([[1.0, 0.0], [0.0, -1.0]]))
    state = np.asarray([1.0, 0.0])
    report = scan_windowed_operator_annihilators(
        (op_a, op_b, op_c),
        state,
        ((0.0,), (1.0,), (4.0,)),
        (0.0, 1.0),
        periodic_box=(8.0,),
    )
    assert report.points[0].minimum_residual > 0.9
    assert report.points[1].minimum_residual < 1.0e-12
    assert report.points[1].nullity == 1
    assert report.points[1].coefficient_support_size == 2
    assert report.bounded_annihilation_radius == 1.0


def test_beta_zero_matching_projection() -> None:
    report = beta_zero_matching_subspace(
        scar_term_expectations=(1.0, 0.0, 1.0),
        thermal_term_expectations=(0.5, 0.5, 0.5),
    )
    assert report.constraint_rank == 1
    assert report.compatible_dimension == 2
    coefficients = project_coefficients_to_beta_zero_match((1.0, 2.0, 3.0), report)
    mismatch = np.asarray(report.mismatch_vector)
    assert abs(np.dot(mismatch, coefficients)) < 1.0e-12


def test_quasi_one_dimensional_audit_flags_fixed_width_and_codimension() -> None:
    points = tuple(
        Quasi1DSequencePoint(
            length=length,
            width=4,
            exact_residual=1.0e-12,
            witness_radius=1.0,
            thermal_second_moment=0.2,
            compatibility_rank=2 * repeat,
            local_parameter_count=16 * repeat,
        )
        for repeat, length in enumerate((4, 8, 12), start=1)
    )
    report = audit_quasi_1d_sequence(
        points,
        energy_density_mismatch=0.1,
        level_gap_ratio=0.52,
    )
    assert report.is_exact_sequence
    assert report.has_bounded_witness
    assert report.has_positive_thermal_activity
    assert any("quasi-one-dimensional" in issue for issue in report.issues)
    assert any("extensive local compatibility" in issue for issue in report.issues)
    assert any("finite-temperature" in issue for issue in report.issues)


def test_quasi_one_dimensional_audit_accepts_energy_matched_microcanonical() -> None:
    points = (
        Quasi1DSequencePoint(
            length=8,
            width=4,
            exact_residual=1.0e-12,
            witness_radius=1.0,
            thermal_second_moment=0.2,
        ),
    )
    report = audit_quasi_1d_sequence(
        points,
        energy_density_mismatch=0.0,
        thermal_comparison="energy_matched_microcanonical",
    )
    assert report.thermal_comparison == "energy_matched_microcanonical"
    assert "microcanonical window centered at the scar energy" in report.established
    assert not any("beta-zero" in statement for statement in report.established)


def test_operator_coefficient_compatibility_distinguishes_fixed_vectors() -> None:
    identity = sp.identity(2, format="csr")
    pauli_x = sp.csr_array(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    pauli_z = sp.csr_array(np.asarray([[1.0, 0.0], [0.0, -1.0]]))
    vectors = np.eye(2)
    report = operator_coefficient_compatibility(
        (identity, pauli_x, pauli_z),
        vectors,
        mode="fixed_vectors",
    )
    assert report.rank == 1
    assert report.compatible_dimension == 2
