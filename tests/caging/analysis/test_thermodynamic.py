from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlinks.caging.analysis import (
    LocalCancellationPatternSupport,
)
from qlinks.caging.analysis.thermodynamic import (
    ETHScalingReport,
    common_local_witness_families,
    diagnose_local_channel_spectrum,
    directed_transition_witness_template,
    evaluate_local_witness_microcanonical,
    evaluate_local_witness_on_diagonal_ensemble,
    evaluate_local_witness_on_states,
    hermitianize_local_witness_template,
    local_witness_template_from_pattern_support,
    make_eth_scaling_point,
    thermal_activity_margin_from_samples,
)


def _lowering_pattern_support(*, variable_index: int = 0) -> LocalCancellationPatternSupport:
    return LocalCancellationPatternSupport(
        pattern_key=(((1,), (0,), (1.0, 0.0)),),
        variable_indices=(variable_index,),
        source_zero_indices=(3,),
        mechanism_labels=("q_empty",),
    )


def test_local_witness_template_reconstructs_reduced_iz_row_operator() -> None:
    template = local_witness_template_from_pattern_support(_lowering_pattern_support())

    assert template.n_variables == 1
    assert template.local_patterns == ((0,), (1,))
    assert np.allclose(template.local_operator, np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(template.q_operator, np.diag([0.0, 1.0]))
    assert np.isclose(template.q_operator_norm, 1.0)


def test_local_witness_evaluation_detects_exact_annihilation_and_thermal_weight() -> None:
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    witness = local_witness_template_from_pattern_support(_lowering_pattern_support()).instantiate(
        (0,)
    )

    cage = evaluate_local_witness_on_states(
        witness,
        basis_configs=basis_configs,
        states=np.array([1.0, 0.0]),
    )
    thermal = evaluate_local_witness_on_diagonal_ensemble(
        witness,
        basis_configs=basis_configs,
    )

    assert np.isclose(cage.expectation, 0.0)
    assert np.isclose(cage.variance, 0.0)
    assert np.isclose(cage.annihilation_residual, 0.0)
    assert np.isclose(thermal.expectation, 0.5)
    assert np.isclose(thermal.second_moment, 0.5)
    assert np.isclose(thermal.variance, 0.25)
    assert np.isclose(thermal.normalized_expectation, 0.5)


def test_local_witness_uses_exact_constrained_basis_embedding() -> None:
    # The two local patterns never coexist with the same environment, so the
    # projected constrained-basis operator has no matrix element.
    basis_configs = np.array([[0, 0], [1, 1]], dtype=np.int64)
    witness = local_witness_template_from_pattern_support(_lowering_pattern_support()).instantiate(
        (0,)
    )

    thermal = evaluate_local_witness_on_diagonal_ensemble(
        witness,
        basis_configs=basis_configs,
    )

    assert np.isclose(thermal.expectation, 0.0)
    assert np.isclose(thermal.variance, 0.0)


def test_microcanonical_witness_evaluation_selects_requested_shell() -> None:
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    witness = local_witness_template_from_pattern_support(_lowering_pattern_support()).instantiate(
        (0,)
    )
    eigenvectors = np.eye(2, dtype=np.complex128)
    eigenvalues = np.array([-1.0, 1.0])

    shell = evaluate_local_witness_microcanonical(
        witness,
        basis_configs=basis_configs,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        energy_center=1.0,
        half_width=0.01,
        system_size=4,
    )

    assert shell.shell_indices == (1,)
    assert shell.n_shell_states == 1
    assert np.isclose(shell.evaluation.expectation, 1.0)
    assert np.isclose(shell.mean_energy_density, 0.25)


def test_eth_scaling_report_fits_constant_plus_inverse_size() -> None:
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    template = local_witness_template_from_pattern_support(_lowering_pattern_support())
    witness = template.instantiate((0,))

    points = []
    for system_size, probability in ((4, 0.75), (8, 0.625), (16, 0.5625)):
        points.append(
            make_eth_scaling_point(
                system_size=system_size,
                witness=witness,
                basis_configs=basis_configs,
                cage_state=np.array([1.0, 0.0]),
                diagonal_probabilities=np.array([1.0 - probability, probability]),
            )
        )

    report = ETHScalingReport(template=template, points=tuple(points))
    fit = report.fit_expectation_gap(order=1)

    assert report.system_sizes == (4, 8, 16)
    assert np.isclose(fit.thermodynamic_limit, 0.5)
    assert fit.root_mean_square_residual < 1.0e-12
    assert np.isclose(report.tail_liminf_lower_bound(tail_points=2), 0.5625)


@dataclass(frozen=True)
class _Transition:
    source_local: tuple[int, ...]
    target_local: tuple[int, ...]
    matrix_element: complex


@dataclass(frozen=True)
class _ZeroReport:
    zero_index: int
    local_mask: np.ndarray
    local_transitions: tuple[_Transition, ...]
    probe_mechanism_label: str = "q_empty"


@dataclass(frozen=True)
class _ClassificationReport:
    zero_reports: tuple[_ZeroReport, ...]


def _fake_report(*, variable_index: int, zero_index: int) -> _ClassificationReport:
    mask = np.zeros(3, dtype=bool)
    mask[variable_index] = True
    return _ClassificationReport(
        zero_reports=(
            _ZeroReport(
                zero_index=zero_index,
                local_mask=mask,
                local_transitions=(
                    _Transition(
                        source_local=(1,),
                        target_local=(0,),
                        matrix_element=1.0,
                    ),
                ),
            ),
        )
    )


def test_common_local_witness_families_match_translated_patterns() -> None:
    families = common_local_witness_families(
        {
            "L=4": _fake_report(variable_index=0, zero_index=2),
            "L=6": _fake_report(variable_index=2, zero_index=7),
        }  # type: ignore[arg-type]
    )

    assert len(families) == 1
    family = families[0]
    assert family.system_labels == ("L=4", "L=6")
    assert family.witnesses_for("L=4")[0].variable_indices == (0,)
    assert family.witnesses_for("L=6")[0].variable_indices == (2,)


def test_directed_transition_template_and_hermitianization() -> None:
    directed = directed_transition_witness_template(
        target_pattern=(0, 0),
        source_patterns=((1, -1), (-1, 1)),
        amplitudes=(2.0, 2.0),
    )
    hermitian = hermitianize_local_witness_template(directed)

    expected = np.zeros((3, 3), dtype=np.complex128)
    expected[0, 1:] = 2.0
    np.testing.assert_allclose(directed.local_operator, expected)
    np.testing.assert_allclose(
        hermitian.local_operator,
        expected + expected.conj().T,
    )
    np.testing.assert_allclose(
        directed.q_operator,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 4.0, 4.0],
                [0.0, 4.0, 4.0],
            ],
            dtype=np.complex128,
        ),
    )


def test_local_channel_spectrum_and_thermal_margin() -> None:
    directed = directed_transition_witness_template(
        target_pattern=(0, 0),
        source_patterns=((1, -1), (-1, 1)),
        amplitudes=(2.0, 2.0),
        normalization="operator_norm",
    )
    spectrum = diagnose_local_channel_spectrum(directed)
    assert spectrum.rank == 1
    assert spectrum.nullity == 2
    assert np.isclose(spectrum.dark_channel_gap, 1.0)

    margin = thermal_activity_margin_from_samples(
        [-0.1, 0.0, 0.1],
        [0.28, 0.30, 0.27],
        reference_parameter=0.0,
    )
    assert np.isclose(margin.reference_activity, 0.30)
    assert np.isclose(margin.susceptibility_bound, 0.30)
    assert np.isclose(margin.half_activity_radius, 0.5)
    assert np.isclose(margin.lower_bound(0.1), 0.27)
