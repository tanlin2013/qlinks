import numpy as np
import scipy.sparse as sp

from qlinks.open_system.constructions.deprecated import build_degenerate_cage_lindblad_construction
from tests.open_system.manifold_detectors._helpers import (
    _single_qutrit_build_result,
    _single_qutrit_detector_pair,
    _single_qutrit_target_state,
)


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
    from qlinks.open_system.manifold_detector_types import TargetedResidualKernelLinearCandidate
    from qlinks.open_system.manifold_residual import TargetedResidualKernelLinearSearchReport

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
    assert (
        targeted_summary["reported_candidate_family_residual_kernel_dimension"]
        == (targeted_summary["reported_candidate_residual_kernel_dimension"])
    )
    assert (
        targeted_summary["reported_candidates_remove_family_residual_kernel"]
        is (targeted_summary["reported_candidates_remove_residual_kernel"])
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
