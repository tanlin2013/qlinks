import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.open_system.constructions.deprecated import build_degenerate_cage_lindblad_construction
from tests.open_system.manifold_detectors._helpers import (
    _equal_bit_manifold_rows,
    _single_qutrit_build_result,
    _single_qutrit_detector,
    _single_qutrit_detector_pair,
    _single_qutrit_offdiagonal_detector,
    _single_qutrit_target_state,
    _two_qubit_build_result,
)


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


def test_expand_local_regions_to_cluster_unions_accepts_single_unit_clusters():
    from qlinks.open_system import expand_local_regions_to_cluster_unions

    regions = ((0, 1), (1, 2), (3, 4))

    assert (
        expand_local_regions_to_cluster_unions(
            regions,
            cluster_size=1,
            cluster_mode="overlap_connected",
        )
        == regions
    )

    assert (
        expand_local_regions_to_cluster_unions(
            regions,
            cluster_size=1,
            cluster_mode="all",
        )
        == regions
    )


def test_expand_local_regions_to_cluster_unions_rejects_zero_cluster_size():
    from qlinks.open_system import expand_local_regions_to_cluster_unions

    with pytest.raises(ValueError, match="cluster_size must be at least one"):
        expand_local_regions_to_cluster_unions(((0, 1),), cluster_size=0)


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


def test_recycled_collective_recycler_bundles_region_detector_group():
    from qlinks.open_system import (
        RecycledManifoldCollectiveRecyclerGroup,
        select_recycled_manifold_dark_detector_jumps,
    )

    build_result = _single_qutrit_build_result()
    detector = sp.diags([0.0, 1.0, 1.0], format="csr", dtype=np.complex128)

    selection = select_recycled_manifold_dark_detector_jumps(
        hamiltonian=build_result.hamiltonian,
        states=_single_qutrit_target_state(),
        basis_configs=build_result.basis.states,
        detector_operators=(detector,),
        detector_coefficients=np.asarray([1.0]),
        detector_operator_names=("D12",),
        local_regions=((0,),),
        recycler_source="matrix_units",
        max_candidate_pool=None,
        max_selected_jumps=2,
        selection_strategy="ranked_inflow",
        check_final_diagnostics=True,
        collective_recycler_strategy="bundle_by_region_detector",
    )

    assert selection.uses_collective_recyclers is True
    assert selection.n_unbundled_jumps == 2
    assert selection.n_selected_jumps == 1
    assert selection.collective_jump_reduction == 1
    assert selection.n_collective_groups == 1
    assert selection.n_bundled_recyclers == 2
    assert isinstance(selection.collective_groups[0], RecycledManifoldCollectiveRecyclerGroup)
    assert selection.collective_groups[0].n_bundled_recyclers == 2
    assert selection.collective_groups[0].detector_index == 0
    assert selection.collective_groups[0].region_index == 0
    assert selection.final_diagnostics is not None
    assert selection.final_diagnostics.max_target_jump_residual < 1e-12

    readouts = selection.selected_recycler_readouts(
        basis_configs=build_result.basis.states,
    )
    assert len(readouts) == 1
    assert readouts[0].source == "collective_recycled_recycler"
    assert readouts[0].nnz == 2
    assert dict(readouts[0].metadata)["n_bundled_recyclers"] == 2

    summary = selection.to_summary_dict()
    assert summary["uses_collective_recyclers"] is True
    assert summary["collective_recycler_strategy"] == "bundle_by_region_detector"
    assert summary["n_unbundled_jumps"] == 2
    assert summary["n_selected_jumps"] == 1
    assert summary["unbundled_inflow_norm"] is not None
    assert summary["selected_inflow_norm"] is not None
    assert summary["final_inflow_norm"] == summary["selected_inflow_norm"]
    assert summary["collective_inflow_ratio"] is not None
    assert summary["collective_inflow_ratio"] > 0.0
    group_summary = summary["collective_groups"][0]
    assert group_summary["unbundled_inflow_norm"] is not None
    assert group_summary["bundled_inflow_norm"] is not None
