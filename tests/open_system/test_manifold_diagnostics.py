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
