import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging.chain_complex import (
    HamiltonianGraphChainComplex,
    MotifRadiusHomologyPoint,
    MotifRadiusSaturationReport,
    build_hamiltonian_graph_chain_complex,
    diagnose_hamiltonian_graph_homology,
    diagnose_periodic_laurent_kernel,
    diagnose_term_resolved_caging,
    periodic_laurent_operator,
)


def _example_complex() -> HamiltonianGraphChainComplex:
    # ker(D) = span(e0, e1, e2); im(T) = span(e0, e1).
    constraint = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.complex128)
    generators = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return HamiltonianGraphChainComplex(constraint, generators)


def test_hamiltonian_graph_complex_computes_h1_h2_and_hodge_representative() -> None:
    report = diagnose_hamiltonian_graph_homology(_example_complex(), tolerance=1.0e-12)

    assert report.is_chain_complex
    assert report.constraint_rank == 1
    assert report.generator_rank == 2
    assert report.cage_dimension == 3
    assert report.h1_dimension == 1
    assert report.nu_mb == 1
    assert report.h2_dimension == 1
    assert report.hodge_gap is not None
    np.testing.assert_allclose(report.h1_basis[:, 0], [0.0, 0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(report.pairing_matrix(), np.eye(1), atol=1.0e-12)
    np.testing.assert_allclose(
        _example_complex().generator_map @ report.h2_basis,
        0.0,
        atol=1.0e-12,
    )


def test_hodge_gap_is_invariant_under_generator_rescaling() -> None:
    base = _example_complex()
    rescaled = HamiltonianGraphChainComplex(
        base.constraint_map,
        base.generator_map @ np.diag([2.0, 0.5, -3.0]),
    )

    base_report = diagnose_hamiltonian_graph_homology(base, tolerance=1.0e-12)
    rescaled_report = diagnose_hamiltonian_graph_homology(rescaled, tolerance=1.0e-12)

    np.testing.assert_allclose(base_report.hodge_gap, rescaled_report.hodge_gap, atol=1.0e-12)
    np.testing.assert_allclose(
        base_report.h1_basis @ base_report.h1_basis.conj().T,
        rescaled_report.h1_basis @ rescaled_report.h1_basis.conj().T,
        atol=1.0e-12,
    )


def test_nonzero_boundary_of_generator_is_rejected() -> None:
    bad = HamiltonianGraphChainComplex(
        constraint_map=np.asarray([[1.0, 0.0]], dtype=np.complex128),
        generator_map=np.asarray([[1.0], [0.0]], dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="do not form a chain complex"):
        diagnose_hamiltonian_graph_homology(bad, tolerance=1.0e-12)


def test_build_complex_from_hamiltonian_support_shell() -> None:
    hamiltonian = scipy_sparse.diags([0.0, 0.0, 1.0], format="csr", dtype=np.complex128)
    complex_ = build_hamiltonian_graph_chain_complex(
        hamiltonian,
        support_indices=(0, 1),
        local_generators=np.asarray([1.0, 0.0]),
        energy=0.0,
    )
    report = diagnose_hamiltonian_graph_homology(complex_, tolerance=1.0e-12)

    assert report.cage_dimension == 2
    assert report.generator_rank == 1
    assert report.nu_mb == 1


def test_term_resolved_quotient_detects_collective_cancellation() -> None:
    first = np.asarray([[1.0, 0.0]], dtype=np.complex128)
    second = np.asarray([[-1.0, 0.0]], dtype=np.complex128)

    report = diagnose_term_resolved_caging((first, second), tolerance=1.0e-12)

    assert report.physical_nullity == 2
    assert report.resolved_nullity == 1
    assert report.collective_quotient_dimension == 1
    assert report.has_collective_cancellation
    np.testing.assert_allclose(
        report.channel_activity(report.resolved_kernel_basis),
        0.0,
        atol=1.0e-12,
    )
    assert report.channel_activity(report.collective_quotient_basis)[0] > 1.0

    detuned = diagnose_term_resolved_caging(
        (first, second),
        coefficients=(1.0, 1.1),
        tolerance=1.0e-12,
    )
    assert detuned.physical_nullity == 1
    assert detuned.collective_quotient_dimension == 0


def test_motif_radius_scan_identifies_saturated_defect() -> None:
    points = (
        MotifRadiusHomologyPoint(1, 1, 2, 0, 0.0, 0.5),
        MotifRadiusHomologyPoint(2, 2, 1, 0, 0.0, 0.2),
        MotifRadiusHomologyPoint(3, 2, 1, 1, 0.0, 0.1),
    )
    report = MotifRadiusSaturationReport(points=points, plateau_length=2)

    assert report.classification == "saturated_defect_candidate"
    assert report.saturated_nu_mb == 1


def test_motif_radius_scan_requires_generator_rank_plateau() -> None:
    points = (
        MotifRadiusHomologyPoint(1, 1, 1, 0, 0.0, 0.5),
        MotifRadiusHomologyPoint(2, 2, 1, 0, 0.0, 0.4),
    )
    report = MotifRadiusSaturationReport(points=points, plateau_length=2)

    assert report.classification == "not_saturated"
    assert report.saturated_nu_mb is None


def test_order_two_laurent_factor_obeys_parity_twist_exchange() -> None:
    factor = {0: 1.0, 1: 1.0}

    even_periodic = diagnose_periodic_laurent_kernel(factor, 6, twist=0.0)
    odd_periodic = diagnose_periodic_laurent_kernel(factor, 5, twist=0.0)
    odd_antiperiodic = diagnose_periodic_laurent_kernel(factor, 5, twist=np.pi)
    even_antiperiodic = diagnose_periodic_laurent_kernel(factor, 6, twist=np.pi)

    assert even_periodic.nullity == 1
    assert odd_periodic.nullity == 0
    assert odd_antiperiodic.nullity == 1
    assert even_antiperiodic.nullity == 0

    expected_gap = 2.0 * np.sin(np.pi / 6.0)
    np.testing.assert_allclose(
        even_periodic.smallest_positive_singular_value,
        expected_gap,
        atol=1.0e-12,
    )


def test_matrix_valued_laurent_operator_has_expected_shape() -> None:
    operator = periodic_laurent_operator(
        {
            0: np.eye(2),
            -1: np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        },
        length=4,
    )

    assert operator.shape == (8, 8)
