import numpy as np
import pytest

from qlinks.open_system import diagnose_dark_manifold


def _basis_vector(dim: int, index: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.complex128)
    vector[index] = 1.0
    return vector


def test_diagnose_dark_manifold_accepts_degenerate_target_zero_modes():
    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    manifold = np.column_stack([_basis_vector(3, 0), _basis_vector(3, 1)])
    jumps = [
        np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        ),
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        ),
    ]

    diagnostics = diagnose_dark_manifold(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_states=manifold,
        liouvillian_spectrum_method="dense",
    )

    assert diagnostics.manifold_dimension == 2
    assert diagnostics.max_target_jump_residual < 1e-12
    assert diagnostics.hamiltonian_closure_residual < 1e-12
    assert diagnostics.target_density_liouvillian_residual < 1e-12
    assert diagnostics.inflow_norm > 0.0
    assert diagnostics.expected_internal_zero_mode_count == 4
    assert diagnostics.liouvillian_zero_mode_count == 4
    assert diagnostics.extra_zero_mode_count == 0
    assert diagnostics.extra_nondecaying_mode_count == 0
    assert diagnostics.bad_common_jump_kernel_dimension == 0
    assert diagnostics.likely_attractive_dark_manifold is True


def test_diagnose_dark_manifold_allows_internal_imaginary_axis_modes():
    hamiltonian = np.diag([0.0, 2.0, 5.0]).astype(np.complex128)
    manifold = np.column_stack([_basis_vector(3, 0), _basis_vector(3, 1)])
    jumps = [
        np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        ),
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        ),
    ]

    diagnostics = diagnose_dark_manifold(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_states=manifold,
        liouvillian_spectrum_method="dense",
    )

    assert diagnostics.expected_internal_zero_mode_count == 2
    assert diagnostics.expected_internal_peripheral_mode_count == 2
    assert diagnostics.liouvillian_zero_mode_count == 2
    assert diagnostics.liouvillian_peripheral_mode_count == 2
    assert diagnostics.extra_nondecaying_mode_count == 0
    assert diagnostics.likely_attractive_dark_manifold is True

    expected = sorted(
        diagnostics.expected_internal_liouvillian_eigenvalues,
        key=lambda value: (value.real, value.imag),
    )
    expected_values = sorted(
        [0.0, 0.0, 2.0j, -2.0j],
        key=lambda value: (complex(value).real, complex(value).imag),
    )
    assert expected == pytest.approx(expected_values)


def test_diagnose_dark_manifold_flags_extra_external_dark_sector():
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    manifold = np.column_stack([_basis_vector(4, 0), _basis_vector(4, 1)])
    jumps = [
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    ]

    diagnostics = diagnose_dark_manifold(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_states=manifold,
        liouvillian_spectrum_method="dense",
    )

    assert diagnostics.expected_internal_zero_mode_count == 4
    assert diagnostics.liouvillian_zero_mode_count > 4
    assert diagnostics.extra_zero_mode_count is not None
    assert diagnostics.extra_zero_mode_count > 0
    assert diagnostics.extra_nondecaying_mode_count is not None
    assert diagnostics.extra_nondecaying_mode_count > 0
    assert diagnostics.bad_common_jump_kernel_dimension > 0
    assert diagnostics.likely_attractive_dark_manifold is False


def test_dark_manifold_diagnostics_rich_report_renders():
    from rich.console import Console

    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    manifold = np.column_stack([_basis_vector(3, 0), _basis_vector(3, 1)])
    jumps = [
        np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        )
    ]

    diagnostics = diagnose_dark_manifold(
        hamiltonian=hamiltonian,
        jumps=jumps,
        target_states=manifold,
        liouvillian_spectrum_method="dense",
    )

    console = Console(record=True, width=120)
    console.print(diagnostics)
    rendered = console.export_text()
    assert "Dark-manifold diagnostics" in rendered
    assert "Target manifold checks" in rendered
    assert "Liouvillian spectrum" in rendered


def test_dark_manifold_bad_kernel_uses_subspace_not_columnwise_projection():
    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    target = np.column_stack([_basis_vector(3, 0), _basis_vector(3, 1)])
    # Two dark jumps whose common kernel is a slightly rotated basis for the
    # same two-dimensional target subspace.  Column-wise projection would see a
    # tiny component outside the target in each arbitrary kernel vector and can
    # misclassify a spurious bad direction at tight tolerances.
    epsilon = 2.0e-10
    kernel_vector_0 = np.asarray([1.0, 0.0, epsilon], dtype=np.complex128)
    kernel_vector_0 = kernel_vector_0 / np.linalg.norm(kernel_vector_0)
    kernel_vector_1 = _basis_vector(3, 1)
    leaking_direction = np.cross(
        kernel_vector_0.real,
        kernel_vector_1.real,
    ).astype(np.complex128)
    leaking_direction = leaking_direction / np.linalg.norm(leaking_direction)
    jump = np.outer(_basis_vector(3, 0), leaking_direction.conj())

    diagnostics = diagnose_dark_manifold(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target,
        kernel_tolerance=1.0e-10,
        liouvillian_spectrum_method="none",
    )

    assert diagnostics.common_jump_kernel_dimension == 2
    assert diagnostics.bad_common_jump_kernel_dimension == 0


def test_common_kernel_h_invariant_sector_ignores_h_leaking_bad_vector():
    from qlinks.open_system import diagnose_common_kernel_h_invariant_sector

    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    hamiltonian[1, 2] = 1.0
    hamiltonian[2, 1] = 1.0
    target = _basis_vector(3, 0)
    jump = np.outer(_basis_vector(3, 0), _basis_vector(3, 1).conj())

    report = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target,
        kernel_tolerance=1.0e-10,
    )

    assert report.common_jump_kernel_dimension == 2
    assert report.bad_common_jump_kernel_dimension == 1
    assert report.h_leakage_norm_from_bad_kernel > 0.9
    assert report.bad_h_invariant_kernel_dimension == 0
    assert report.likely_attractive_by_h_invariant_kernel is True


def test_common_kernel_h_invariant_sector_flags_h_closed_bad_vector():
    from qlinks.open_system import diagnose_common_kernel_h_invariant_sector

    hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    target = _basis_vector(3, 0)
    jump = np.outer(_basis_vector(3, 0), _basis_vector(3, 1).conj())

    report = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target,
        kernel_tolerance=1.0e-10,
    )

    assert report.common_jump_kernel_dimension == 2
    assert report.bad_common_jump_kernel_dimension == 1
    assert report.h_leakage_norm_from_bad_kernel < 1.0e-12
    assert report.bad_h_invariant_kernel_dimension == 1
    assert report.likely_attractive_by_h_invariant_kernel is False
    assert report.to_summary_dict()["has_bad_h_invariant_kernel"] is True


def test_common_kernel_h_invariant_sector_survives_svd_nonconvergence(monkeypatch):
    from qlinks.open_system import diagnose_common_kernel_h_invariant_sector

    def raising_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", raising_svd)

    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    target = _basis_vector(4, 0)
    jump = np.outer(_basis_vector(4, 0), _basis_vector(4, 1).conj())

    report = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target,
        kernel_tolerance=1.0e-10,
    )

    assert report.common_jump_kernel_dimension == 3
    assert report.bad_common_jump_kernel_dimension == 2
    assert report.bad_h_invariant_kernel_dimension == 2
    assert report.likely_attractive_by_h_invariant_kernel is False


def test_bad_h_invariant_common_kernel_basis_returns_obstruction_vectors():
    from qlinks.open_system import bad_h_invariant_common_kernel_basis

    target_state = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)
    jump = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    hamiltonian = np.zeros((3, 3), dtype=np.complex128)

    basis = bad_h_invariant_common_kernel_basis(
        hamiltonian=hamiltonian,
        jumps=(jump,),
        target_states=target_state,
    )

    assert basis.shape == (3, 1)
    assert np.linalg.norm(jump @ basis) < 1e-12
    assert abs(np.vdot(target_state, basis[:, 0])) < 1e-12
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(1), atol=1e-12)
