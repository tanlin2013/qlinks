from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from qlinks.models import LocalTermDescriptor
from qlinks.open_system.constructions import (
    CageLindbladDetectorOperators,
    build_cage_lindblad_problem,
)
from qlinks.open_system.constructions.deprecated import (
    build_degenerate_cage_lindblad_construction,
)


class _ArrayBasis:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.int64)


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


def _detector_bundle():
    d1 = sp.diags([0.0, 1.0, 0.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0, 1.0], format="csr", dtype=np.complex128)
    return CageLindbladDetectorOperators(
        operators=(d1, d2),
        names=("D1", "D2"),
    )


def test_unified_problem_accepts_single_cage_state():
    build_result = _two_bit_build_result()
    problem = build_cage_lindblad_problem(
        build_result=build_result,
        target_state=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        local_regions=((0, 1),),
    )

    assert problem.manifold_dimension == 1
    assert problem.is_single_cage_target is True
    assert problem.local_regions == ((0, 1),)
    assert problem.basis_configs.shape == (4, 2)

    workflow = problem.design_jumps(
        detector_operators=_detector_bundle(),
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        targeted_operator_source="matrix_units",
        max_recycled_selected_jumps=4,
        max_targeted_selected_jumps=4,
        liouvillian_spectrum_method="none",
    )

    summary = workflow.to_summary_dict()
    assert summary["manifold_dimension"] == 1
    assert summary["n_recycled_jumps"] >= 2
    assert workflow.jumps


def test_unified_problem_accepts_degenerate_cage_manifold():
    build_result = _two_bit_build_result()
    target_states = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    problem = build_cage_lindblad_problem(
        build_result=build_result,
        target_states=target_states,
        local_regions=((0, 1),),
    )

    assert problem.manifold_dimension == 2
    assert problem.is_single_cage_target is False
    np.testing.assert_allclose(
        problem.target_density_matrix,
        np.diag([0.5, 0.0, 0.0, 0.5]),
    )


def test_unified_problem_raw_detectors_and_solver_packaging():
    build_result = _two_bit_build_result()
    problem = build_cage_lindblad_problem(
        build_result=build_result,
        target_state=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        local_regions=((0, 1),),
    )
    detectors = _detector_bundle()

    result = problem.design_jumps(
        detector_operators=detectors.operators,
        detector_operator_names=detectors.names,
        local_region_mode="construction",
        recycled_recycler_source="matrix_units",
        max_recycled_selected_jumps=2,
        design_mode="recycled_screening",
        check_recycled_selection_diagnostics=False,
    )

    assert result.lindblad_problem.hamiltonian is build_result.hamiltonian
    assert result.lindblad_problem.jumps == result.jumps
    assert result.to_lindblad_problem() is result.lindblad_problem
    assert result.workflow.jumps == result.jumps


def test_deprecated_namespace_keeps_legacy_builder_available():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        local_regions=((0, 1),),
    )

    assert construction.manifold_dimension == 1


def test_detector_operator_bundle_validates_lengths():
    d = sp.eye(2, dtype=np.complex128, format="csr")
    try:
        CageLindbladDetectorOperators(operators=(d,), names=("a", "b"))
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected mismatched detector names to fail")


def test_detector_operator_bundle_accepts_terms():
    d = sp.eye(2, dtype=np.complex128, format="csr")
    term = LocalTermDescriptor(
        term_id=0,
        term_kind="plaquette",
        operator_kind="potential",
        support_links=(0,),
        support_variables=(0,),
    )
    bundle = CageLindbladDetectorOperators(operators=(d,), names=("V0",), terms=(term,))

    assert bundle.to_summary_dict()["n_operators"] == 1
    assert bundle.terms[0].operator_kind == "potential"
