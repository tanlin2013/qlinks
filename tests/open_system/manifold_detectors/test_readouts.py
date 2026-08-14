import numpy as np

from qlinks.open_system import diagnose_manifold_dark_operator_basis
from tests.open_system.manifold_detectors._helpers import (
    _equal_bit_manifold_rows,
    _single_qutrit_build_result,
    _single_qutrit_detector_pair,
    _single_qutrit_target_state,
    _single_site_z_operators,
)


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
