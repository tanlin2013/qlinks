import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ============================================================
# Basic vec / unvec utilities
# ============================================================


def vec(rho: np.ndarray) -> np.ndarray:
    """
    Column-stack a density matrix into a vector.
    Convention: vec(A rho B) = (B^T ⊗ A) vec(rho)
    """
    return np.asarray(rho, dtype=np.complex128).reshape(-1, order="F")


def unvec(rho_vec: np.ndarray, dim: int) -> np.ndarray:
    """
    Inverse of vec(): recover dim x dim matrix from vector.
    """
    return np.asarray(rho_vec, dtype=np.complex128).reshape((dim, dim), order="F")


# ============================================================
# Sparse Liouvillian builder
# ============================================================


def build_liouvillian(
    H: sp.spmatrix,
    jumps: list[sp.spmatrix],
    *,
    fmt: str = "csc",
    dtype=np.complex128,
) -> sp.spmatrix:
    r"""
    Build the sparse Lindbladian superoperator L such that

        d/dt vec(rho) = L @ vec(rho)

    for the master equation

        dot(rho) = -i[H, rho]
                   + sum_mu (L_mu rho L_mu^\dagger
                            - 1/2 {L_mu^\dagger L_mu, rho})

    Parameters
    ----------
    H : sparse matrix, shape (d, d)
        Hermitian Hamiltonian.
    jumps : list of sparse matrices, each shape (d, d)
        Jump operators.
    fmt : str
        Sparse format of the returned matrix.
    dtype : dtype
        Complex dtype.

    Returns
    -------
    Lio : sparse matrix, shape (d^2, d^2)
        Sparse Liouvillian in vectorized form.
    """
    H = H.asformat(fmt).astype(dtype)
    d = H.shape[0]
    idty = sp.identity(d, format=fmt, dtype=dtype)

    # Coherent part: -i(I ⊗ H - H^T ⊗ I)
    Lio = -1j * (sp.kron(idty, H, format=fmt) - sp.kron(H.T, idty, format=fmt))

    for J in jumps:
        J = J.asformat(fmt).astype(dtype)
        JdagJ = (J.conj().T @ J).asformat(fmt)

        # Jump term: J rho J^\dagger -> (J^* ⊗ J) vec(rho)
        jump_term = sp.kron(J.conj(), J, format=fmt)

        # Anti-commutator terms
        left_loss = sp.kron(idty, JdagJ, format=fmt)
        right_loss = sp.kron(JdagJ.T, idty, format=fmt)

        Lio += jump_term - 0.5 * left_loss - 0.5 * right_loss

    return Lio.asformat(fmt)


# ============================================================
# Optional direct action of Liouvillian on rho (matrix form)
# Useful for RK4 without explicitly building full Liouvillian
# ============================================================


def lindblad_rhs_matrix(
    rho: np.ndarray,
    H,
    jumps,
) -> np.ndarray:
    """
    Compute dot(rho) directly in matrix form.

    H and jumps may be sparse or dense; rho is dense.
    """
    Hrho = H @ rho
    rhoH = rho @ H
    drho = -1j * (Hrho - rhoH)

    for J in jumps:
        Jdag = J.conj().T
        JdagJ = Jdag @ J
        drho += J @ rho @ Jdag
        drho += -0.5 * (JdagJ @ rho + rho @ JdagJ)

    return np.asarray(drho, dtype=np.complex128)


# ============================================================
# RK4 time evolution in Liouville space
# ============================================================


def rk4_step_liouville(y: np.ndarray, dt: float, Lio) -> np.ndarray:
    """
    One RK4 step for y' = Lio @ y
    """
    k1 = Lio @ y
    k2 = Lio @ (y + 0.5 * dt * k1)
    k3 = Lio @ (y + 0.5 * dt * k2)
    k4 = Lio @ (y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_liouvillian_rk4(
    rho0: np.ndarray,
    Lio,
    times: np.ndarray,
    *,
    renormalize_trace: bool = False,
    enforce_hermiticity: bool = False,
) -> list[np.ndarray]:
    """
    Time-evolve rho via RK4 in vectorized Liouville space.

    Parameters
    ----------
    rho0 : ndarray, shape (d, d)
        Initial density matrix.
    Lio : sparse matrix, shape (d^2, d^2)
        Liouvillian.
    times : 1D ndarray
        Time grid.
    renormalize_trace : bool
        Optionally enforce Tr(rho)=1 after each step.
    enforce_hermiticity : bool
        Optionally symmetrize rho after each step.

    Returns
    -------
    rhos : list of ndarray
        Density matrices at the requested times.
    """
    d = rho0.shape[0]
    y = vec(rho0)
    out = [rho0.astype(np.complex128).copy()]

    for n in range(len(times) - 1):
        dt = times[n + 1] - times[n]
        y = rk4_step_liouville(y, dt, Lio)

        rho = unvec(y, d)

        if enforce_hermiticity:
            rho = 0.5 * (rho + rho.conj().T)

        if renormalize_trace:
            tr = np.trace(rho)
            if abs(tr) > 0:
                rho = rho / tr

        y = vec(rho)
        out.append(rho.copy())

    return out


# ============================================================
# Krylov / expm_multiply time evolution in Liouville space
# ============================================================


def evolve_liouvillian_krylov(
    rho0: np.ndarray,
    Lio,
    times: np.ndarray,
) -> list[np.ndarray]:
    """
    Time-evolve rho via expm_multiply:
        vec(rho(t)) = exp(Lio * t) vec(rho0)

    This is often the best sparse "Krylov-like" approach in SciPy.

    Parameters
    ----------
    rho0 : ndarray, shape (d, d)
        Initial density matrix.
    Lio : sparse matrix, shape (d^2, d^2)
        Liouvillian.
    times : 1D ndarray
        Times at which rho(t) is returned.

    Returns
    -------
    rhos : list of ndarray
        Density matrices at the requested times.
    """
    d = rho0.shape[0]
    y0 = vec(rho0)

    # returns array of shape (len(times), d^2)
    Ys = spla.expm_multiply(Lio, y0, start=times[0], stop=times[-1], num=len(times), endpoint=True)

    rhos = [unvec(y, d) for y in Ys]
    return rhos


# ============================================================
# Optional RK4 directly on matrix rho, without flattening
# Sometimes cheaper than building full Lio
# ============================================================


def rk4_step_matrix(rho: np.ndarray, dt: float, H, jumps) -> np.ndarray:
    """
    One RK4 step for dot(rho) = Lindblad(rho), directly in matrix form.
    """
    k1 = lindblad_rhs_matrix(rho, H, jumps)
    k2 = lindblad_rhs_matrix(rho + 0.5 * dt * k1, H, jumps)
    k3 = lindblad_rhs_matrix(rho + 0.5 * dt * k2, H, jumps)
    k4 = lindblad_rhs_matrix(rho + dt * k3, H, jumps)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_matrix_rk4(
    rho0: np.ndarray,
    H,
    jumps,
    times: np.ndarray,
    *,
    renormalize_trace: bool = False,
    enforce_hermiticity: bool = False,
) -> list[np.ndarray]:
    """
    RK4 directly on the density matrix, without forming Liouvillian.
    """
    rho = rho0.astype(np.complex128).copy()
    out = [rho.copy()]

    for n in range(len(times) - 1):
        dt = times[n + 1] - times[n]
        rho = rk4_step_matrix(rho, dt, H, jumps)

        if enforce_hermiticity:
            rho = 0.5 * (rho + rho.conj().T)

        if renormalize_trace:
            tr = np.trace(rho)
            if abs(tr) > 0:
                rho = rho / tr

        out.append(rho.copy())

    return out


# ============================================================
# Diagnostics
# ============================================================


def trace_of_rho(rho: np.ndarray) -> complex:
    return np.trace(rho)


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def fidelity_pure(psi: np.ndarray, rho: np.ndarray) -> float:
    """
    F = <psi|rho|psi> for normalized pure state psi
    """
    return np.real(psi.conj().T @ rho @ psi).item()


def dark_state_residual(psi: np.ndarray, jumps) -> list[float]:
    """
    Return ||J psi|| for each jump operator.
    """
    return [float(np.linalg.norm(J @ psi)) for J in jumps]


def liouvillian_residual_of_pure_state(psi: np.ndarray, Lio) -> float:
    """
    Check ||Lio vec(|psi><psi|)||.
    """
    rho = np.outer(psi, psi.conj())
    return float(np.linalg.norm(Lio @ vec(rho)))
