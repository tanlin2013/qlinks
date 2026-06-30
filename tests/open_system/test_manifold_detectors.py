from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from qlinks.open_system import diagnose_manifold_dark_operator_basis
from qlinks.open_system.constructions import build_degenerate_cage_lindblad_construction


class _ArrayBasis:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.int64)


def _two_qubit_build_result():
    basis = _ArrayBasis(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    hamiltonian = sp.csr_array((4, 4), dtype=np.complex128)
    return SimpleNamespace(basis=basis, hamiltonian=hamiltonian)


def _equal_bit_manifold_rows():
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def _single_site_z_operators():
    z0 = sp.diags([1.0, 1.0, -1.0, -1.0], format="csr", dtype=np.complex128)
    z1 = sp.diags([1.0, -1.0, 1.0, -1.0], format="csr", dtype=np.complex128)
    return z0, z1


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


def _single_site_x0_operator():
    rows = np.asarray([2, 3, 0, 1], dtype=np.int64)
    cols = np.asarray([0, 1, 2, 3], dtype=np.int64)
    data = np.ones(4, dtype=np.complex128)
    return sp.csr_array((data, (rows, cols)), shape=(4, 4))


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


def _single_qutrit_build_result():
    basis = _ArrayBasis([[0], [1], [2]])
    hamiltonian = sp.csr_array((3, 3), dtype=np.complex128)
    return SimpleNamespace(basis=basis, hamiltonian=hamiltonian)


def _single_qutrit_target_state():
    return np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)


def _single_qutrit_detector():
    return sp.diags([0.0, 1.0, 0.0], format="csr", dtype=np.complex128)


def test_recycled_dark_detector_finds_inflow_from_local_rdm_recycler():
    from qlinks.open_system import diagnose_recycled_manifold_dark_detectors

    report = diagnose_recycled_manifold_dark_detectors(
        states=_single_qutrit_target_state(),
        basis_configs=np.asarray([[0], [1], [2]], dtype=np.int64),
        detector_operators=(_single_qutrit_detector(),),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("D1",),
        local_regions=((0,),),
        recycler_source="rdm_support_matrix_units",
    )

    summary = report.to_summary_dict()
    assert summary["n_detectors"] == 1
    assert summary["n_regions"] == 1
    assert summary["max_region_size"] == 1
    assert summary["n_tested_candidates"] == 3
    assert summary["n_candidates_with_inflow"] >= 1
    assert summary["has_attractive_candidates"] is True
    assert summary["best_inflow_norm"] > 0.0
    assert summary["candidates"][0]["relative_dark_residual"] < 1e-12
    assert summary["candidates"][0]["inflow_norm"] > 0.0
    assert "support_0" in summary["candidates"][0]["recycler_name"]


def test_construction_method_tests_recycled_dark_detector_report_and_rich_render():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )

    report = construction.diagnose_recycled_dark_detectors(
        basis_configs=build_result.basis.states,
        detector_operators=(_single_qutrit_detector(),),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("D1",),
        recycler_source="matrix_units",
    )

    assert report.n_candidates_with_inflow >= 1
    assert report.best_inflow_norm > 0.0
    assert report.max_region_size == 1
    assert report.n_tested_candidates == 9

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Recycled manifold dark-detector report" in rendered
    assert "candidates with inflow" in rendered
    assert "D1" in rendered
