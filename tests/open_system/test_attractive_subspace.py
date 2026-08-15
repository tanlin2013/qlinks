import numpy as np

from qlinks.open_system import (
    diagnose_attractive_subspace,
    diagnose_common_kernel_h_invariant_sector,
)


def _matrix_unit(dim: int, row: int, column: int) -> np.ndarray:
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix[row, column] = 1.0
    return matrix


def test_amplitude_damping_certifies_unique_target():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = _matrix_unit(2, 0, 1)
    target = np.asarray([1.0, 0.0], dtype=np.complex128)

    report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_basis=target,
    )

    assert report.no_inflow_dimension == 0
    assert report.invariant_obstruction_dimension == 0
    assert report.target_attractive_certified is True


def test_decoupled_spectator_is_exact_invariant_obstruction():
    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    jump = _matrix_unit(3, 0, 1)
    target = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)

    report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_basis=target,
    )

    assert report.no_inflow_dimension == 1
    assert report.invariant_obstruction_dimension == 1
    assert report.target_attractive_certified is False


def test_non_target_invariant_jump_sector_defeats_common_kernel_heuristic():
    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    jump = _matrix_unit(3, 1, 2) + _matrix_unit(3, 2, 1)
    target = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)

    old_report = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target,
    )
    report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_basis=target,
    )

    assert old_report.likely_attractive_by_h_invariant_kernel is True
    assert report.common_jump_kernel_dimension == 1
    assert report.no_inflow_dimension == 2
    assert report.invariant_obstruction_dimension == 2
    assert report.target_attractive_certified is False


def test_multidimensional_dark_target_can_be_attractive():
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    hamiltonian[0, 1] = hamiltonian[1, 0] = 0.25
    jumps = (_matrix_unit(4, 0, 2), _matrix_unit(4, 1, 3))
    target = np.eye(4, 2, dtype=np.complex128)

    report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_basis=target,
    )

    assert report.target_dimension == 2
    assert report.hamiltonian_target_invariance_residual < 1.0e-12
    assert report.max_jump_darkness_residual < 1.0e-12
    assert report.no_inflow_dimension == 0
    assert report.target_attractive_certified is True


def test_target_basis_rotation_and_projector_are_equivalent():
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    jumps = (_matrix_unit(4, 0, 2), _matrix_unit(4, 1, 3))
    target = np.eye(4, 2, dtype=np.complex128)
    theta = 0.37
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.complex128,
    )
    rotated = target @ rotation
    projector = target @ target.conj().T

    reference = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_basis=target,
    )
    rotated_report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_basis=rotated,
    )
    projector_report = diagnose_attractive_subspace(
        hamiltonian=hamiltonian,
        jumps=jumps,
        projector=projector,
    )

    expected = (
        reference.no_inflow_dimension,
        reference.invariant_obstruction_dimension,
        reference.common_jump_kernel_dimension,
    )
    assert (
        rotated_report.no_inflow_dimension,
        rotated_report.invariant_obstruction_dimension,
        rotated_report.common_jump_kernel_dimension,
    ) == expected
    assert (
        projector_report.no_inflow_dimension,
        projector_report.invariant_obstruction_dimension,
        projector_report.common_jump_kernel_dimension,
    ) == expected
