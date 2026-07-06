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


def _single_qutrit_detector_pair():
    d1 = sp.diags([0.0, 1.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0], format="csr", dtype=np.complex128)
    return d1, d2


def _single_qutrit_offdiagonal_detector():
    rows = np.asarray([1], dtype=np.int64)
    cols = np.asarray([2], dtype=np.int64)
    data = np.ones(1, dtype=np.complex128)
    return sp.csr_array((data, (rows, cols)), shape=(3, 3))


def test_select_recycled_matrix_unit_jumps_with_nondiagonal_detector():
    from qlinks.open_system import select_recycled_manifold_dark_detector_jumps

    build_result = _single_qutrit_build_result()
    detector = _single_qutrit_offdiagonal_detector()

    selection = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=(detector,),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("T12",),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=None,
        max_selected_jumps=1,
        selection_strategy="ranked_inflow",
        check_final_diagnostics=True,
    )

    assert selection.n_selected_jumps == 1
    assert selection.selected_candidates[0].recycler_name == "(0)<-(1)"
    assert selection.jumps[0].nnz == 1
    assert selection.jumps[0][0, 2] == 1.0
    assert selection.final_diagnostics is not None
    assert selection.final_diagnostics.max_target_jump_residual < 1e-12


def test_select_recycled_dark_detector_jumps_removes_complement_kernel():
    from qlinks.open_system import select_recycled_manifold_dark_detector_jumps

    build_result = _single_qutrit_build_result()
    selection = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=18,
        max_selected_jumps=4,
    )

    summary = selection.to_summary_dict()
    assert summary["n_selected_jumps"] == 2
    assert summary["final_bad_common_jump_kernel_dimension"] == 0
    assert summary["complement_common_kernel_removed"] is True
    assert summary["final_inflow_norm"] > 0.0
    assert len(selection.jumps) == 2
    assert selection.final_diagnostics is not None
    assert selection.final_diagnostics.bad_common_jump_kernel_dimension == 0


def test_select_recycled_dark_detector_jumps_expands_truncated_report():
    from qlinks.open_system import (
        diagnose_recycled_manifold_dark_detectors,
        select_recycled_manifold_dark_detector_jumps,
    )

    build_result = _single_qutrit_build_result()
    truncated_report = diagnose_recycled_manifold_dark_detectors(
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_report_candidates=1,
    )

    assert len(truncated_report.candidates) == 1
    assert truncated_report.n_tested_candidates > 1

    selection = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        candidate_report=truncated_report,
        max_candidate_pool=None,
        max_selected_jumps=4,
        expand_candidate_report=True,
    )

    assert selection.candidate_report_was_expanded is True
    assert selection.n_reported_candidates == selection.n_nonzero_candidates
    assert selection.n_reported_candidates < selection.n_tested_candidates
    assert selection.candidate_report_is_truncated is False
    assert selection.candidate_pool_is_truncated is False
    assert selection.complement_common_kernel_removed is True


def test_construction_selects_recycled_dark_detector_jumps_and_rich_render():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )

    selection = construction.select_recycled_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        recycler_source="rdm_support_matrix_units",
        max_candidate_pool=6,
        max_selected_jumps=4,
    )

    assert selection.n_selected_jumps == 2
    assert selection.complement_common_kernel_removed is True
    assert selection.final_bad_common_jump_kernel_dimension == 0

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(selection)
    rendered = console.export_text()
    assert "Recycled manifold jump-selection report" in rendered
    assert "complement kernel removed" in rendered
    assert "selected jumps" in rendered


def test_recycled_candidate_family_kernel_report_removes_complement_kernel():
    from qlinks.open_system import diagnose_recycled_manifold_candidate_family_kernel

    build_result = _single_qutrit_build_result()
    report = diagnose_recycled_manifold_candidate_family_kernel(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
    )

    assert report.n_candidate_jumps > 0
    assert report.n_nonzero_candidates == report.n_reported_candidates
    assert report.family_bad_common_jump_kernel_dimension == 0
    assert report.complement_common_kernel_removed is True
    assert report.family_inflow_norm > 0.0
    assert report.family_kernel_method == "streamed_kernel"
    assert report.to_summary_dict()["family_kernel_method"] == "streamed_kernel"

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Recycled candidate-family common-kernel report" in rendered
    assert "bad complement kernel" in rendered


def test_recycled_candidate_family_kernel_diagnostics_method_stores_jumps():
    from qlinks.open_system import diagnose_recycled_manifold_candidate_family_kernel

    build_result = _single_qutrit_build_result()
    report = diagnose_recycled_manifold_candidate_family_kernel(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        kernel_method="diagnostics",
    )

    assert report.n_candidate_jumps == len(report.candidate_jumps)
    assert report.n_candidate_jumps > 0
    assert report.family_kernel_method == "none"
    assert report.family_bad_common_jump_kernel_dimension == 0


def test_construction_recycled_candidate_family_kernel_report():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )

    report = construction.diagnose_recycled_candidate_family_kernel(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        recycler_source="rdm_support_matrix_units",
    )

    assert report.n_candidate_jumps > 0
    assert report.family_kernel_method == "streamed_kernel"
    assert report.family_bad_common_jump_kernel_dimension == 0
    assert report.to_summary_dict()["complement_common_kernel_removed"] is True


def test_expand_local_regions_to_pair_unions_overlap_and_all_modes():
    from qlinks.open_system import expand_local_regions_to_pair_unions

    regions = ((0, 1), (1, 2), (3, 4))

    overlap_pairs = expand_local_regions_to_pair_unions(regions)
    assert overlap_pairs == ((0, 1, 2),)

    all_pairs = expand_local_regions_to_pair_unions(regions, pair_mode="all")
    assert all_pairs == ((0, 1, 2), (0, 1, 3, 4), (1, 2, 3, 4))

    bounded_pairs = expand_local_regions_to_pair_unions(
        regions,
        pair_mode="all",
        max_region_size=3,
        include_single_regions=True,
    )
    assert bounded_pairs == ((0, 1), (1, 2), (3, 4), (0, 1, 2))


def test_expand_local_regions_to_cluster_unions_overlap_connected_and_all_modes():
    from qlinks.open_system import expand_local_regions_to_cluster_unions

    regions = ((0, 1), (1, 2), (2, 3), (4, 5))

    connected = expand_local_regions_to_cluster_unions(
        regions,
        cluster_size=3,
        cluster_mode="overlap_connected",
    )
    assert connected == ((0, 1, 2, 3),)

    all_clusters = expand_local_regions_to_cluster_unions(
        regions,
        cluster_size=3,
        cluster_mode="all",
    )
    assert all_clusters == (
        (0, 1, 2, 3),
        (0, 1, 2, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5),
    )

    with_smaller = expand_local_regions_to_cluster_unions(
        regions,
        cluster_size=3,
        cluster_mode="overlap_connected",
        include_single_regions=True,
        include_smaller_clusters=True,
        max_region_size=4,
    )
    assert (0, 1) in with_smaller
    assert (0, 1, 2) in with_smaller
    assert (1, 2, 3) in with_smaller


def test_degenerate_construction_local_region_pair_unions():
    build_result = _two_qubit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_equal_bit_manifold_rows(),
        local_regions=((0,), (1,)),
    )

    assert construction.local_region_pair_unions(pair_mode="all") == ((0, 1),)


def test_degenerate_construction_local_region_cluster_unions():
    build_result = _two_qubit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_equal_bit_manifold_rows(),
        local_regions=((0,), (1,), (0, 1)),
    )

    assert construction.local_region_cluster_unions(
        cluster_size=3,
        cluster_mode="overlap_connected",
    ) == ((0, 1),)


def test_recycled_jump_selection_kernel_projection_strategy():
    from qlinks.open_system import select_recycled_manifold_dark_detector_jumps

    build_result = _single_qutrit_build_result()
    report = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=None,
        selection_strategy="kernel_projection",
    )

    assert report.complement_common_kernel_removed is True
    assert report.final_bad_common_jump_kernel_dimension == 0
    assert report.n_selected_jumps <= 2


def test_recycled_residual_kernel_report_identifies_unseen_complement_sector():
    from qlinks.open_system import diagnose_recycled_manifold_residual_kernel

    build_result = _single_qutrit_build_result()
    d1, d2 = _single_qutrit_detector_pair()
    report = diagnose_recycled_manifold_residual_kernel(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=(d1,),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("D1",),
        local_regions=((0,),),
        recycler_source="matrix_units",
        operator_groups=(("projectors", (d1, d2), ("D1", "D2")),),
    )

    summary = report.to_summary_dict()
    assert summary["residual_dimension"] == 1
    assert summary["family_report"]["family_bad_common_jump_kernel_dimension"] == 1
    assert summary["hamiltonian_keeps_residual_sector"] is True
    assert summary["operator_action_reports"][0]["n_operators"] == 2
    assert summary["operator_action_reports"][0]["total_action_norm"] > 0.0
    assert summary["local_support_entries"][0]["residual_support_rank"] == 1


def test_construction_recycled_residual_kernel_report_and_rich_render():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )
    d1, d2 = _single_qutrit_detector_pair()

    report = construction.diagnose_recycled_residual_kernel(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1,),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("D1",),
        recycler_source="matrix_units",
        operator_groups=(("projectors", (d1, d2), ("D1", "D2")),),
    )

    assert report.residual_dimension == 1
    assert report.family_report.family_bad_common_jump_kernel_dimension == 1

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Recycled residual-kernel report" in rendered
    assert "residual bad-kernel dimension" in rendered


def test_targeted_residual_kernel_linear_search_finds_local_dark_jump():
    from qlinks.open_system import diagnose_targeted_residual_kernel_linear_search

    build_result = _single_qutrit_build_result()
    residual_basis = np.asarray([0.0, 1.0, 0.0], dtype=np.complex128)
    report = diagnose_targeted_residual_kernel_linear_search(
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        local_regions=((0,),),
        residual_basis=residual_basis,
        operator_source="matrix_units",
    )

    summary = report.to_summary_dict()
    assert summary["residual_dimension"] == 1
    assert summary["has_targeted_solution"] is True
    assert summary["best_residual_target_inflow_norm"] > 0.0
    assert summary["candidates"][0]["relative_dark_residual"] < 1e-12
    assert summary["candidates"][0]["residual_target_inflow_norm"] > 0.0
    assert len(report.candidate_jumps) >= 1


def test_construction_targeted_residual_kernel_linear_search_and_rich_render():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )

    report = construction.diagnose_targeted_residual_kernel_linear_search(
        basis_configs=build_result.basis.states,
        residual_basis=np.asarray([0.0, 1.0, 0.0], dtype=np.complex128),
        operator_source="matrix_units",
    )

    assert report.has_targeted_solution is True
    assert report.best_residual_target_inflow_norm > 0.0

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Targeted residual-kernel linear search" in rendered
    assert "hits residual" in rendered


def test_select_targeted_residual_kernel_jumps_removes_reported_residual():
    from qlinks.open_system import (
        diagnose_targeted_residual_kernel_linear_search,
        select_targeted_residual_kernel_jumps,
    )

    build_result = _single_qutrit_build_result()
    residual_basis = np.asarray([0.0, 1.0, 0.0], dtype=np.complex128)
    targeted = diagnose_targeted_residual_kernel_linear_search(
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        local_regions=((0,),),
        residual_basis=residual_basis,
        operator_source="matrix_units",
    )
    _d1, d2 = _single_qutrit_detector_pair()

    selection = select_targeted_residual_kernel_jumps(
        targeted_report=targeted,
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        base_jumps=(d2,),
        max_selected_jumps=2,
        liouvillian_spectrum_method="none",
    )

    summary = selection.to_summary_dict()
    assert summary["n_base_jumps"] == 1
    assert summary["n_selected_jumps"] == 1
    assert summary["final_residual_kernel_dimension"] == 0
    assert summary["residual_kernel_removed"] is True
    assert summary["combined_bad_common_jump_kernel_dimension"] == 0
    assert summary["combined_complement_common_kernel_removed"] is True
    assert len(selection.all_jumps) == 2


def test_construction_selects_targeted_residual_kernel_jumps_and_rich_render():
    build_result = _single_qutrit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_single_qutrit_target_state(),
        local_regions=((0,),),
    )
    targeted = construction.diagnose_targeted_residual_kernel_linear_search(
        basis_configs=build_result.basis.states,
        residual_basis=np.asarray([0.0, 1.0, 0.0], dtype=np.complex128),
        operator_source="matrix_units",
    )
    _d1, d2 = _single_qutrit_detector_pair()

    selection = construction.select_targeted_residual_kernel_jumps(
        targeted_report=targeted,
        hamiltonian=build_result.hamiltonian,
        base_jumps=(d2,),
        max_selected_jumps=2,
    )

    assert selection.residual_kernel_removed is True
    assert selection.combined_bad_common_jump_kernel_dimension == 0

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(selection)
    rendered = console.export_text()
    assert "Targeted residual-kernel jump-selection report" in rendered
    assert "final residual kernel" in rendered


def test_targeted_selector_can_minimize_combined_common_kernel_beyond_reported_residual():
    from qlinks.open_system import select_targeted_residual_kernel_jumps
    from qlinks.open_system.manifold_detectors import (
        TargetedResidualKernelLinearCandidate,
        TargetedResidualKernelLinearSearchReport,
    )

    target_state = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)
    residual_basis = np.asarray([[0.0], [1.0], [0.0]], dtype=np.complex128)
    jump_1 = sp.csr_array(
        ([1.0], ([0], [1])),
        shape=(3, 3),
        dtype=np.complex128,
    )
    jump_2 = sp.csr_array(
        ([1.0], ([0], [2])),
        shape=(3, 3),
        dtype=np.complex128,
    )

    def candidate(index: int, residual_inflow: float) -> TargetedResidualKernelLinearCandidate:
        return TargetedResidualKernelLinearCandidate(
            candidate_index=index,
            region_index=index,
            variable_indices=(index,),
            local_dim=3,
            operator_source="manual",
            dark_constraint_rank=1,
            dark_nullity=1,
            singular_value=residual_inflow,
            residual_target_inflow_norm=residual_inflow,
            dark_residual=0.0,
            relative_dark_residual=0.0,
            total_inflow_norm=1.0,
            target_block_norm=0.0,
            jump_frobenius_norm=1.0,
            jump_nnz=1,
            coefficients=np.asarray([1.0], dtype=np.complex128),
            terms=(),
        )

    targeted = TargetedResidualKernelLinearSearchReport(
        manifold_dimension=1,
        hilbert_dimension=3,
        residual_basis=residual_basis,
        region_variable_indices=((1,), (2,)),
        operator_source="manual",
        family_report=None,
        candidates=(candidate(0, 1.0), candidate(1, 0.0)),
        candidate_jumps=(jump_1, jump_2),
        tolerance=1e-10,
        dark_tolerance=1e-10,
        inflow_tolerance=1e-12,
    )

    reported_selection = select_targeted_residual_kernel_jumps(
        targeted_report=targeted,
        hamiltonian=sp.csr_array((3, 3), dtype=np.complex128),
        states=target_state,
        max_selected_jumps=4,
        selection_target="reported_residual_kernel",
    )
    assert reported_selection.n_selected_jumps == 1
    assert reported_selection.residual_kernel_removed is True
    assert reported_selection.combined_bad_common_jump_kernel_dimension == 1

    combined_selection = select_targeted_residual_kernel_jumps(
        targeted_report=targeted,
        hamiltonian=sp.csr_array((3, 3), dtype=np.complex128),
        states=target_state,
        max_selected_jumps=4,
        selection_target="combined_common_kernel",
    )
    targeted_summary = targeted.to_summary_dict()
    assert targeted_summary["reported_candidate_family_residual_kernel_dimension"] == (
        targeted_summary["reported_candidate_residual_kernel_dimension"]
    )
    assert targeted_summary["reported_candidates_remove_family_residual_kernel"] is (
        targeted_summary["reported_candidates_remove_residual_kernel"]
    )

    summary = combined_selection.to_summary_dict()
    assert summary["selection_target"] == "combined_common_kernel"
    assert summary["initial_selection_kernel_dimension"] == 2
    assert summary["n_selected_jumps"] == 2
    assert summary["final_residual_kernel_dimension"] == 0
    assert summary["final_selection_kernel_dimension"] == 0
    assert summary["selection_kernel_removed"] is True
    assert summary["combined_bad_common_jump_kernel_dimension"] == 0
    assert summary["combined_complement_common_kernel_removed"] is True


def test_dark_detector_and_recycled_matrix_readouts_are_available():
    from qlinks.open_system import (
        DarkDetectorMatrixReadout,
        LocalOperatorMatrixReadout,
        select_recycled_manifold_dark_detector_jumps,
    )

    build_result = _single_qutrit_build_result()
    selection = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=_single_qutrit_detector_pair(),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=18,
        max_selected_jumps=1,
    )

    detector_label = selection.candidate_report.detector_names[0]
    assert "D1" in detector_label

    readouts = selection.selected_recycler_readouts(
        basis_configs=build_result.basis.states,
    )
    assert len(readouts) == 1
    readout = readouts[0]
    assert isinstance(readout, LocalOperatorMatrixReadout)
    assert readout.variable_indices == (0,)
    assert readout.local_patterns == ((0,), (1,), (2,))
    assert readout.local_operator.shape == (3, 3)
    assert readout.nnz == 1
    assert readout.is_local_matrix_readout is True
    assert len(readout.nonzero_matrix_elements()) == 1
    assert readout.to_summary_dict()["nonzero_matrix_elements"] == (
        readout.nonzero_matrix_elements()
    )
    assert any(name in dict(readout.metadata)["detector_name"] for name in ("D1", "D2"))

    dark_report = diagnose_manifold_dark_operator_basis(
        states=_equal_bit_manifold_rows(),
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )
    dark_readouts = dark_report.detector_readouts()
    assert isinstance(dark_readouts[0], DarkDetectorMatrixReadout)
    assert dark_readouts[0].n_terms == 2
    assert dark_readouts[0].is_local_matrix_readout is False
    assert dark_readouts[0].operator_names == ("Z0", "Z1")


def test_targeted_matrix_unit_candidate_readout():
    from qlinks.open_system import diagnose_targeted_residual_kernel_linear_search

    build_result = _single_qutrit_build_result()
    residual_basis = np.asarray([0.0, 1.0, 0.0], dtype=np.complex128)
    report = diagnose_targeted_residual_kernel_linear_search(
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        local_regions=((0,),),
        residual_basis=residual_basis,
        operator_source="matrix_units",
        max_modes_per_region=1,
        residual_objective="action_norm",
    )

    assert report.candidates
    candidate = report.candidates[0]
    readout = report.candidate_readouts(
        basis_configs=build_result.basis.states,
    )[0]
    assert readout.source == "targeted_operator"
    assert readout.variable_indices == candidate.variable_indices
    assert readout.local_operator.shape == (3, 3)
    assert readout.nnz >= 1


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


def test_recycled_h_invariant_compression_removes_redundant_ranked_recyclers():
    from qlinks.open_system import (
        diagnose_common_kernel_h_invariant_sector,
        select_recycled_manifold_dark_detector_jumps,
    )

    build_result = _single_qutrit_build_result()
    d1, d2 = _single_qutrit_detector_pair()
    detector_coefficients = np.eye(3, dtype=np.complex128)

    uncompressed = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d1, d2),
        detector_coefficients=detector_coefficients,
        detector_operator_names=("D1", "D1_copy", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=None,
        max_selected_jumps=4,
        selection_strategy="ranked_inflow",
        check_final_diagnostics=False,
    )
    compressed = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d1, d2),
        detector_coefficients=detector_coefficients,
        detector_operator_names=("D1", "D1_copy", "D2"),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=None,
        max_selected_jumps=4,
        selection_strategy="ranked_inflow",
        compression_strategy="h_invariant",
        max_compression_passes=8,
        check_final_diagnostics=False,
    )

    assert uncompressed.n_selected_jumps >= 3
    assert compressed.n_selected_jumps < uncompressed.n_selected_jumps
    assert compressed.n_compressed_jumps_removed > 0
    assert compressed.compression_strategy == "h_invariant"

    hcert = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=build_result.hamiltonian,
        jumps=compressed.jumps,
        target_states=_single_qutrit_target_state(),
    )
    assert hcert.likely_attractive_by_h_invariant_kernel is True
