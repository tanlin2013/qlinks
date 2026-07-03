from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.search import CageRecord
from qlinks.caging.solver import CageState
from qlinks.models import LocalTermDescriptor
from qlinks.open_system import (
    build_local_recycling_jumps_from_subspace_regions,
    local_reduced_density_matrix_from_state_matrix,
)
from qlinks.open_system.constructions import build_degenerate_cage_lindblad_construction


class _ArrayBasis:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.int64)


class _FakeLocalTermModel:
    def local_term_descriptors(self, *, term_kind=None):
        if term_kind not in {None, "plaquette"}:
            return ()
        return (
            LocalTermDescriptor(
                term_id=0,
                term_kind="plaquette",
                operator_kind="kinetic",
                support_links=(0, 1),
                support_variables=(0, 1),
            ),
        )


def _two_bit_build_result():
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


def _two_state_manifold_rows():
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def test_local_reduced_density_matrix_from_state_matrix_uses_manifold_support():
    build_result = _two_bit_build_result()
    rdm = local_reduced_density_matrix_from_state_matrix(
        basis_configs=build_result.basis.states,
        states=_two_state_manifold_rows(),
        variable_indices=(0, 1),
    )

    assert rdm.support_rank == 2
    assert rdm.nullity == 2
    np.testing.assert_allclose(np.sort(rdm.eigenvalues), [0.0, 0.0, 0.5, 0.5])


def test_subspace_block_reset_recycling_annihilates_entire_manifold():
    build_result = _two_bit_build_result()
    states = _two_state_manifold_rows().T
    result = build_local_recycling_jumps_from_subspace_regions(
        basis_configs=build_result.basis.states,
        states=states,
        regions=((0, 1),),
        source="local_rdm_block_reset",
    )

    assert result.n_jumps == 1
    jump = result.jumps[0]
    np.testing.assert_allclose((jump @ states), np.zeros((4, 2)))
    assert result.selections[0].candidate.inflow_norm > 0.0


def test_degenerate_cage_lindblad_construction_from_states():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_two_state_manifold_rows(),
        local_regions=((0, 1),),
    )

    summary = construction.to_summary_dict()
    assert summary["manifold_dimension"] == 2
    assert summary["n_jumps"] == 1
    assert summary["local_regions"] == ((0, 1),)
    assert construction.inflow_norm > 0.0
    assert construction.max_jump_residual < 1e-12
    assert construction.liouvillian_residual is not None
    assert construction.liouvillian_residual < 1e-12
    np.testing.assert_allclose(
        construction.target_density_matrix,
        np.diag([0.5, 0.0, 0.0, 0.5]),
    )

    diagnostics = construction.diagnose_manifold(
        hamiltonian=build_result.hamiltonian,
        liouvillian_spectrum_method="dense",
    )
    assert diagnostics.manifold_dimension == 2
    assert diagnostics.max_target_jump_residual < 1e-12
    assert diagnostics.expected_internal_zero_mode_count == 4


def test_degenerate_cage_lindblad_construction_infers_regions_from_model():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        model=_FakeLocalTermModel(),
        states=_two_state_manifold_rows(),
        local_term_kind="plaquette",
    )

    assert construction.local_regions == ((0, 1),)
    assert construction.n_jumps == 1


def test_degenerate_cage_lindblad_construction_from_records_validates_signature():
    build_result = _two_bit_build_result()
    records = []
    for basis_index in (0, 3):
        records.append(
            CageRecord(
                cage_state=CageState(
                    energy=0.0,
                    local_state=np.asarray([1.0], dtype=np.complex128),
                    support=np.asarray([basis_index], dtype=np.int64),
                    boundary_residual=0.0,
                    eigen_residual=0.0,
                ),
                signature=(0, 4),
                candidate=CandidateSubgraph(np.asarray([basis_index], dtype=np.int64)),
                full_state=None,
            )
        )

    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        records=records,
        local_regions=((0, 1),),
    )

    assert construction.record_signature == (0, 4)
    assert construction.manifold_dimension == 2
    assert construction.n_jumps == 1


def test_degenerate_construction_reports_full_local_support_when_no_nullity():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=np.eye(4, dtype=np.complex128),
        local_regions=((0, 1),),
    )

    assert construction.n_jumps == 0
    report = construction.local_subspace_support_report
    summary = report.to_summary_dict()
    assert summary["n_regions"] == 1
    assert summary["n_regions_with_nullity"] == 0
    assert summary["all_regions_have_full_local_support"] is True
    assert summary["entries"][0]["status"] == "full_local_support"
    assert summary["entries"][0]["parent_detector_directions"] == 0


def test_degenerate_construction_rich_reports_render():
    from rich.console import Console

    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_two_state_manifold_rows(),
        local_regions=((0, 1),),
    )

    console = Console(record=True, width=120)
    console.print(construction)
    rendered = console.export_text()
    assert "Degenerate cage Lindblad construction" in rendered
    assert "Local manifold-support report" in rendered
    assert "selected" in rendered


def test_degenerate_jump_design_workflow_reuses_existing_stages():
    build_result = _two_bit_build_result()
    target_state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    d1 = sp.diags([0.0, 1.0, 0.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0, 1.0], format="csr", dtype=np.complex128)
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=target_state,
        local_regions=((0, 1),),
    )

    workflow = construction.design_dark_manifold_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d2),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        targeted_operator_source="matrix_units",
        max_recycled_selected_jumps=4,
        max_targeted_selected_jumps=4,
        liouvillian_spectrum_method="none",
    )

    summary = workflow.to_summary_dict()
    assert summary["dark_detector_nullity"] == 2
    assert summary["n_recycled_jumps"] >= 2
    assert summary["combined_bad_common_jump_kernel_dimension"] == 0
    assert summary["combined_complement_common_kernel_removed"] is True
    assert summary["likely_successful_common_kernel_design"] is True
    assert "targeted_reported_candidates_remove_family_residual" in summary
    assert "targeted_selected_candidates_remove_family_residual" in summary
    assert "targeted_selection_removes_combined_kernel" in summary
    assert len(workflow.jumps) == workflow.n_jumps
    problem = workflow.to_lindblad_problem(hamiltonian=build_result.hamiltonian)
    assert problem.jumps == workflow.jumps


def test_degenerate_jump_design_workflow_h_invariant_fast_stops_early():
    build_result = _two_bit_build_result()
    target_state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    d1 = sp.diags([0.0, 1.0, 0.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0, 1.0], format="csr", dtype=np.complex128)
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=target_state,
        local_regions=((0, 1),),
    )

    workflow = construction.design_dark_manifold_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d2),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        targeted_operator_source="matrix_units",
        max_recycled_selected_jumps=4,
        liouvillian_spectrum_method="none",
        design_mode="h_invariant_fast",
    )

    summary = workflow.to_summary_dict()
    assert summary["design_mode"] == "h_invariant_fast"
    assert summary["early_stop_reason"] == "recycled_h_invariant_success"
    assert summary["n_targeted_jumps"] == 0
    assert summary["targeted_candidates"] is None
    assert summary["family_candidate_jumps"] is None
    assert summary["likely_successful_h_invariant_design"] is True
    assert workflow.family_report is None
    assert workflow.residual_report is None
    assert workflow.targeted_report is None
    assert workflow.targeted_selection is None
    assert workflow.jumps == workflow.recycled_jumps


def test_degenerate_jump_design_workflow_rich_render():
    from rich.console import Console

    build_result = _two_bit_build_result()
    target_state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    d1 = sp.diags([0.0, 1.0, 0.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0, 1.0], format="csr", dtype=np.complex128)
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=target_state,
        local_regions=((0, 1),),
    )

    workflow = construction.design_dark_manifold_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d2),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        targeted_operator_source="matrix_units",
        max_recycled_selected_jumps=4,
        max_targeted_selected_jumps=4,
        liouvillian_spectrum_method="none",
    )

    console = Console(record=True, width=120)
    console.print(workflow)
    rendered = console.export_text()
    assert "Degenerate cage jump-design workflow" in rendered
    assert "Workflow stages" in rendered
    assert "combined bad" in rendered


def test_degenerate_jump_design_workflow_recycled_screening_stops_after_recycled_stage():
    build_result = _two_bit_build_result()
    target_state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    d1 = sp.diags([0.0, 1.0, 0.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0, 1.0], format="csr", dtype=np.complex128)
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=target_state,
        local_regions=((0, 1),),
    )

    workflow = construction.design_dark_manifold_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1, d2),
        detector_coefficients=np.eye(2, dtype=np.complex128),
        detector_operator_names=("D1", "D2"),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        max_recycled_selected_jumps=2,
        design_mode="recycled_screening",
        check_recycled_selection_diagnostics=False,
    )

    summary = workflow.to_summary_dict()
    assert summary["design_mode"] == "recycled_screening"
    assert summary["early_stop_reason"] == "recycled_screening"
    assert summary["n_recycled_jumps"] == 2
    assert summary["n_targeted_jumps"] == 0
    assert summary["targeted_candidates"] is None
    assert summary["combined_bad_common_jump_kernel_dimension"] is None
    assert workflow.family_report is None
    assert workflow.residual_report is None
    assert workflow.targeted_report is None
    assert workflow.targeted_selection is None
    assert workflow.final_diagnostics is None
    assert workflow.jumps == workflow.recycled_jumps


def test_degenerate_jump_design_h_invariant_completion_targets_remaining_bad_sector():
    build_result = SimpleNamespace(
        basis=_ArrayBasis([[0], [1], [2]]),
        hamiltonian=sp.csr_array((3, 3), dtype=np.complex128),
    )
    target_state = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)
    d1 = sp.diags([0.0, 1.0, 0.0], format="csr", dtype=np.complex128)
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=target_state,
        local_regions=((0,),),
    )

    workflow = construction.design_dark_manifold_jumps(
        hamiltonian=build_result.hamiltonian,
        basis_configs=build_result.basis.states,
        detector_operators=(d1,),
        detector_coefficients=np.asarray([1.0], dtype=np.complex128),
        detector_operator_names=("D1",),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        targeted_operator_source="matrix_units",
        max_recycled_selected_jumps=1,
        max_h_invariant_completion_selected_jumps=2,
        design_mode="h_invariant_completion",
        check_final_manifold_diagnostics=True,
    )

    summary = workflow.to_summary_dict()
    assert summary["design_mode"] == "h_invariant_completion"
    assert summary["early_stop_reason"] == "h_invariant_completion_success"
    assert summary["n_recycled_jumps"] == 1
    assert summary["n_targeted_jumps"] == 1
    assert summary["targeted_candidates"] >= 1
    assert workflow.targeted_report is not None
    assert workflow.targeted_report.residual_objective == "action_norm"
    assert workflow.targeted_report.best_residual_score_norm > 0.0
    assert workflow.h_invariant_report is not None
    assert workflow.h_invariant_report.bad_h_invariant_kernel_dimension == 0
    assert workflow.likely_successful_h_invariant_design is True
