import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.open_sys.lindbladian import (
    build_liouvillian,
    dark_state_residual,
    fidelity_pure,
    liouvillian_residual_of_pure_state,
    lindblad_rhs_matrix,
    purity,
    rk4_step_liouville,
    rk4_step_matrix,
    trace_of_rho,
    unvec,
    vec,
    evolve_liouvillian_krylov,
    evolve_liouvillian_rk4,
    evolve_matrix_rk4,
)


@pytest.fixture
def two_level_system():
    d = 2
    H = sp.csc_matrix(np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128))
    gamma = 0.3
    J = sp.csc_matrix(np.sqrt(gamma) * np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))
    jumps = [J]

    rho_excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    rho_ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    psi_ground = np.array([1.0, 0.0], dtype=np.complex128)
    psi_excited = np.array([0.0, 1.0], dtype=np.complex128)

    return {
        "d": d,
        "H": H,
        "gamma": gamma,
        "J": J,
        "jumps": jumps,
        "rho_excited": rho_excited,
        "rho_ground": rho_ground,
        "psi_ground": psi_ground,
        "psi_excited": psi_excited,
    }


def test_vec_unvec_roundtrip():
    rho = np.array(
        [[1 + 1j, 2 - 1j], [3 + 0.5j, 4 - 2j]],
        dtype=np.complex128,
    )
    rho_vec = vec(rho)
    rho_back = unvec(rho_vec, 2)

    assert rho_vec.shape == (4,)
    assert np.allclose(rho_back, rho)


def test_vec_column_stacking_convention():
    rho = np.array(
        [[1, 2], [3, 4]],
        dtype=np.complex128,
    )
    expected = np.array([1, 3, 2, 4], dtype=np.complex128)
    assert np.allclose(vec(rho), expected)


def test_trace_of_rho():
    rho = np.array(
        [[0.7, 0.0], [0.0, 0.3]],
        dtype=np.complex128,
    )
    assert np.isclose(trace_of_rho(rho), 1.0)


def test_purity_pure_and_mixed_states():
    rho_pure = np.array(
        [[1.0, 0.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    rho_mixed = np.array(
        [[0.5, 0.0], [0.0, 0.5]],
        dtype=np.complex128,
    )

    assert np.isclose(purity(rho_pure), 1.0)
    assert np.isclose(purity(rho_mixed), 0.5)


def test_fidelity_pure(two_level_system):
    psi = two_level_system["psi_ground"]
    rho_same = two_level_system["rho_ground"]
    rho_orth = two_level_system["rho_excited"]

    assert np.isclose(fidelity_pure(psi, rho_same), 1.0)
    assert np.isclose(fidelity_pure(psi, rho_orth), 0.0)


def test_dark_state_residual_ground_state(two_level_system):
    psi_ground = two_level_system["psi_ground"]
    jumps = two_level_system["jumps"]
    res = dark_state_residual(psi_ground, jumps)

    assert len(res) == 1
    assert np.isclose(res[0], 0.0, atol=1e-13)


def test_dark_state_residual_excited_state_nonzero(two_level_system):
    psi_excited = two_level_system["psi_excited"]
    jumps = two_level_system["jumps"]
    gamma = two_level_system["gamma"]

    res = dark_state_residual(psi_excited, jumps)

    assert len(res) == 1
    assert np.isclose(res[0], np.sqrt(gamma), atol=1e-13)


def test_lindblad_rhs_matrix_ground_state_is_steady(two_level_system):
    rho_ground = two_level_system["rho_ground"]
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]

    drho = lindblad_rhs_matrix(rho_ground, H, jumps)
    assert np.allclose(drho, np.zeros_like(rho_ground), atol=1e-13)


def test_build_liouvillian_shape(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    Lio = build_liouvillian(H, jumps)

    d = H.shape[0]
    assert Lio.shape == (d * d, d * d)


def test_liouvillian_matches_matrix_rhs(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    Lio = build_liouvillian(H, jumps)

    rho = np.array(
        [[0.4, 0.1 + 0.2j], [0.1 - 0.2j, 0.6]],
        dtype=np.complex128,
    )

    rhs_matrix = lindblad_rhs_matrix(rho, H, jumps)
    rhs_vec = Lio @ vec(rho)
    rhs_from_liouville = unvec(rhs_vec, rho.shape[0])

    assert np.allclose(rhs_from_liouville, rhs_matrix, atol=1e-12)


def test_liouvillian_residual_of_pure_dark_state(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    psi_ground = two_level_system["psi_ground"]
    Lio = build_liouvillian(H, jumps)

    res = liouvillian_residual_of_pure_state(psi_ground, Lio)
    assert np.isclose(res, 0.0, atol=1e-12)


def test_rk4_step_liouville_preserves_trace_approximately(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]
    Lio = build_liouvillian(H, jumps)

    dt = 1e-3
    y1 = rk4_step_liouville(vec(rho0), dt, Lio)
    rho1 = unvec(y1, 2)

    assert np.isclose(np.trace(rho1), 1.0, atol=1e-10)


def test_rk4_step_matrix_preserves_trace_approximately(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]

    dt = 1e-3
    rho1 = rk4_step_matrix(rho0, dt, H, jumps)

    assert np.isclose(np.trace(rho1), 1.0, atol=1e-10)


def test_evolve_liouvillian_krylov_relaxes_excited_to_ground(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]
    rho_ground = two_level_system["rho_ground"]

    Lio = build_liouvillian(H, jumps)
    times = np.linspace(0.0, 40.0, 101)

    rhos = evolve_liouvillian_krylov(rho0, Lio, times)
    rho_final = rhos[-1]

    assert np.isclose(np.trace(rho_final), 1.0, atol=1e-10)
    assert np.allclose(rho_final, rho_ground, atol=5e-4)


def test_evolve_liouvillian_rk4_relaxes_excited_to_ground(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]
    rho_ground = two_level_system["rho_ground"]

    Lio = build_liouvillian(H, jumps)
    times = np.linspace(0.0, 40.0, 4001)

    rhos = evolve_liouvillian_rk4(
        rho0,
        Lio,
        times,
        renormalize_trace=False,
        enforce_hermiticity=False,
    )
    rho_final = rhos[-1]

    assert np.isclose(np.trace(rho_final), 1.0, atol=1e-8)
    assert np.allclose(rho_final, rho_ground, atol=2e-3)


def test_evolve_matrix_rk4_relaxes_excited_to_ground(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]
    rho_ground = two_level_system["rho_ground"]

    times = np.linspace(0.0, 40.0, 4001)

    rhos = evolve_matrix_rk4(
        rho0,
        H,
        jumps,
        times,
        renormalize_trace=False,
        enforce_hermiticity=False,
    )
    rho_final = rhos[-1]

    assert np.isclose(np.trace(rho_final), 1.0, atol=1e-8)
    assert np.allclose(rho_final, rho_ground, atol=2e-3)


def test_krylov_and_rk4_agree(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]
    rho0 = two_level_system["rho_excited"]

    Lio = build_liouvillian(H, jumps)
    times_k = np.linspace(0.0, 10.0, 101)
    times_rk = np.linspace(0.0, 10.0, 1001)

    rhos_k = evolve_liouvillian_krylov(rho0, Lio, times_k)
    rhos_rk = evolve_liouvillian_rk4(
        rho0,
        Lio,
        times_rk,
        renormalize_trace=False,
        enforce_hermiticity=False,
    )

    # Compare at final time only
    rho_k_final = rhos_k[-1]
    rho_rk_final = rhos_rk[-1]

    assert np.allclose(rho_k_final, rho_rk_final, atol=1e-3)


def test_optional_postprocessing_flags(two_level_system):
    H = two_level_system["H"]
    jumps = two_level_system["jumps"]

    rho0 = np.array(
        [[0.5, 0.2 + 0.1j], [0.2 - 0.1j, 0.5]],
        dtype=np.complex128,
    )

    Lio = build_liouvillian(H, jumps)
    times = np.linspace(0.0, 1.0, 11)

    rhos = evolve_liouvillian_rk4(
        rho0,
        Lio,
        times,
        renormalize_trace=True,
        enforce_hermiticity=True,
    )

    for rho in rhos:
        assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)
