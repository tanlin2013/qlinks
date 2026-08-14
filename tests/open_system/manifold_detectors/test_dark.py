import numpy as np
import scipy.sparse as sp

from qlinks.open_system import diagnose_manifold_dark_operator_basis
from qlinks.open_system.constructions.deprecated import build_degenerate_cage_lindblad_construction
from tests.open_system.manifold_detectors._helpers import (
    _equal_bit_manifold_rows,
    _single_site_x0_operator,
    _single_site_z_operators,
    _two_qubit_build_result,
)


def test_collective_dark_operator_can_exist_when_single_site_support_is_full():
    build_result = _two_qubit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_equal_bit_manifold_rows(),
        local_regions=((0,), (1,)),
    )

    support_summary = construction.local_subspace_support_report.to_summary_dict()
    assert support_summary["all_regions_have_full_local_support"] is True
    assert support_summary["n_regions_with_nullity"] == 0

    report = construction.diagnose_dark_operator_basis(
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )

    summary = report.to_summary_dict()
    assert summary["detector_nullity"] == 1
    assert summary["has_dark_detectors"] is True
    assert len(summary["candidates"]) == 1
    assert summary["candidates"][0]["action_residual"] < 1e-12

    coefficients = report.candidates[0].coefficients
    assert abs(abs(coefficients[0]) - abs(coefficients[1])) < 1e-12
    assert abs(coefficients[0] + coefficients[1]) < 1e-12


def test_manifold_dark_operator_basis_direct_api_and_rich_render():
    report = diagnose_manifold_dark_operator_basis(
        states=_equal_bit_manifold_rows(),
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )

    assert report.detector_nullity == 1
    assert report.candidates[0].terms[0].operator_name in {"Z0", "Z1"}

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Manifold dark-operator basis report" in rendered
    assert "dark-detector nullity" in rendered
    assert "Z0" in rendered or "Z1" in rendered


def test_dressed_dark_detector_finds_direct_inflow_from_left_multiplier():
    from qlinks.open_system import diagnose_dressed_manifold_dark_detectors

    z0, z1 = _single_site_z_operators()
    x0 = _single_site_x0_operator()
    report = diagnose_dressed_manifold_dark_detectors(
        states=_equal_bit_manifold_rows(),
        detector_operators=(z0, z1),
        detector_coefficients=np.asarray([1.0, -1.0]),
        detector_operator_names=("Z0", "Z1"),
        left_multipliers=(x0,),
        left_multiplier_names=("X0",),
    )

    summary = report.to_summary_dict()
    assert summary["n_detectors"] == 1
    assert summary["n_left_multipliers"] == 1
    assert summary["n_candidates_with_inflow"] == 1
    assert summary["has_attractive_candidates"] is True
    assert summary["best_inflow_norm"] > 0.0
    assert summary["candidates"][0]["relative_dark_residual"] < 1e-12
    assert summary["candidates"][0]["inflow_norm"] > 0.0


def test_construction_method_tests_dressed_dark_detector_report_and_rich_render():
    build_result = _two_qubit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_equal_bit_manifold_rows(),
        local_regions=((0,), (1,)),
    )
    dark_report = construction.diagnose_dark_operator_basis(
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )

    report = construction.diagnose_dressed_dark_detectors(
        detector_operators=_single_site_z_operators(),
        dark_operator_report=dark_report,
        left_multipliers=(_single_site_x0_operator(),),
        left_multiplier_names=("X0",),
    )

    assert report.n_candidates_with_inflow == 1
    assert report.best_inflow_norm > 0.0

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Dressed manifold dark-detector report" in rendered
    assert "candidates with inflow" in rendered
    assert "X0" in rendered


def test_dark_operator_coordinate_ipr_candidate_prefers_single_local_operator():
    target = np.asarray([1.0, 0.0], dtype=np.complex128)
    shared_action = sp.csr_array(
        (
            np.asarray([1.0], dtype=np.complex128),
            (np.asarray([1], dtype=np.int64), np.asarray([0], dtype=np.int64)),
        ),
        shape=(2, 2),
    )
    target_dark_local = sp.diags([0.0, 1.0], format="csr", dtype=np.complex128)

    report = diagnose_manifold_dark_operator_basis(
        states=target,
        operators=(shared_action, shared_action, target_dark_local),
        operator_names=("A", "A_copy", "P1"),
        candidate_strategy="coordinate_ipr",
        max_candidates=1,
    )

    assert report.candidate_strategy == "coordinate_ipr"
    assert report.detector_nullity == 2
    candidate = report.candidates[0]
    assert candidate.n_terms == 1
    assert candidate.terms[0].operator_name == "P1"
    assert candidate.coefficient_ipr == 1.0
    assert candidate.effective_operator_count == 1.0
    assert candidate.action_residual < 1e-12
