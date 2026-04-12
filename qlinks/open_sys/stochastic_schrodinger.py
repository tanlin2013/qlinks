from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]
ArrayF = NDArray[np.float64]


@dataclass
class TrajectoryResult:
    times: ArrayF
    states: list[ArrayC]
    jump_records: list[tuple[float, int]]


@dataclass
class EnsembleResult:
    times: ArrayF
    rho_t: list[ArrayC]
    trajectories: list[TrajectoryResult] | None = None


def _as_complex_array(x: ArrayC) -> ArrayC:
    return np.asarray(x, dtype=np.complex128)


def normalize_state(psi: ArrayC, tol: float = 1e-14) -> ArrayC:
    norm = np.linalg.norm(psi)
    if norm < tol:
        raise ValueError("State norm is too small; cannot normalize.")
    return psi / norm


def projector(psi: ArrayC) -> ArrayC:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def expectation(psi: ArrayC, op: ArrayC) -> complex:
    return np.vdot(psi, op @ psi)


def effective_hamiltonian(H: ArrayC, jumps: list[ArrayC]) -> ArrayC:
    heff = H.astype(np.complex128).copy()
    for L in jumps:
        heff = heff - 0.5j * (L.conj().T @ L)
    return heff


def jump_probabilities(psi: ArrayC, jumps: list[ArrayC], dt: float) -> ArrayF:
    probs = np.array(
        [dt * np.real(np.vdot(psi, (L.conj().T @ L) @ psi)) for L in jumps],
        dtype=np.float64,
    )
    probs[probs < 0.0] = 0.0
    return probs


def choose_jump(probs: ArrayF, rng: np.random.Generator) -> int:
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("Total jump probability must be positive when choosing a jump.")
    return rng.choice(len(probs), p=probs / total)


def evolve_no_jump_first_order(psi: ArrayC, H_eff: ArrayC, dt: float) -> ArrayC:
    # First-order propagator:
    # |psi(t+dt)> ~ (I - i H_eff dt)|psi(t)>
    return psi - 1j * dt * (H_eff @ psi)


def run_quantum_jump_trajectory(
    H: ArrayC,
    jumps: list[ArrayC],
    psi0: ArrayC,
    times: ArrayF,
    rng: np.random.Generator | None = None,
) -> TrajectoryResult:
    """
    Run a single Monte Carlo wave-function trajectory.

    Parameters
    ----------
    H : (d, d) complex ndarray
        Hamiltonian.
    jumps : list of (d, d) complex ndarrays
        Lindblad jump operators.
    psi0 : (d,) complex ndarray
        Initial normalized wavefunction.
    times : (n_times,) float ndarray
        Time grid, assumed uniform.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    TrajectoryResult
    """
    if rng is None:
        rng = np.random.default_rng()

    H = _as_complex_array(H)
    jumps = [_as_complex_array(L) for L in jumps]
    psi = normalize_state(_as_complex_array(psi0).copy())
    times = np.asarray(times, dtype=np.float64)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a 1D array with at least two points.")

    dt_arr = np.diff(times)
    if not np.allclose(dt_arr, dt_arr[0]):
        raise ValueError("This implementation assumes a uniform time step.")
    dt = float(dt_arr[0])

    H_eff = effective_hamiltonian(H, jumps)

    states = [psi.copy()]
    jump_records: list[tuple[float, int]] = []

    for n in range(len(times) - 1):
        probs = jump_probabilities(psi, jumps, dt)
        p_jump = probs.sum()

        if p_jump > 1.0:
            raise RuntimeError(
                f"Time step too large: total jump probability = {p_jump:.6f} > 1. " "Reduce dt."
            )

        r = rng.random()

        if r < p_jump:
            # A jump occurs
            mu = choose_jump(probs, rng)
            psi = jumps[mu] @ psi
            psi = normalize_state(psi)
            jump_records.append((times[n + 1], mu))
        else:
            # No jump: evolve under H_eff
            psi = evolve_no_jump_first_order(psi, H_eff, dt)
            psi = normalize_state(psi)

        states.append(psi.copy())

    return TrajectoryResult(times=times, states=states, jump_records=jump_records)


def sample_lindblad_mcwf(
    H: ArrayC,
    jumps: list[ArrayC],
    psi0_sampler: Callable[[np.random.Generator], ArrayC],
    times: ArrayF,
    n_trajectories: int,
    seed: int | None = None,
    store_trajectories: bool = False,
) -> EnsembleResult:
    """
    Solve Lindblad dynamics by averaging quantum-jump trajectories.

    Parameters
    ----------
    H : (d, d) complex ndarray
        Hamiltonian.
    jumps : list of (d, d) complex ndarrays
        Jump operators.
    psi0_sampler : callable(rng) -> (d,) complex ndarray
        Function that samples the initial pure state for each trajectory.
        For a fixed pure initial state, just return the same vector every time.
    times : (n_times,) float ndarray
        Uniform time grid.
    n_trajectories : int
        Number of stochastic trajectories.
    seed : int, optional
        RNG seed.
    store_trajectories : bool
        Whether to keep all trajectories in memory.

    Returns
    -------
    EnsembleResult
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=np.float64)

    psi0_test = _as_complex_array(psi0_sampler(rng))
    d = psi0_test.shape[0]

    rho_t = [np.zeros((d, d), dtype=np.complex128) for _ in range(len(times))]
    stored: list[TrajectoryResult] | None = [] if store_trajectories else None

    # Reuse independent child seeds for reproducibility
    child_seeds = rng.integers(0, 2**63 - 1, size=n_trajectories, dtype=np.int64)

    for s in child_seeds:
        traj_rng = np.random.default_rng(int(s))
        psi0 = normalize_state(_as_complex_array(psi0_sampler(traj_rng)))

        traj = run_quantum_jump_trajectory(
            H=H,
            jumps=jumps,
            psi0=psi0,
            times=times,
            rng=traj_rng,
        )

        for k, psi in enumerate(traj.states):
            rho_t[k] += projector(psi)

        if store_trajectories and stored is not None:
            stored.append(traj)

    rho_t = [rho / n_trajectories for rho in rho_t]
    return EnsembleResult(times=times, rho_t=rho_t, trajectories=stored)


def observable_vs_time(rho_t: Iterable[ArrayC], opt: ArrayC) -> ArrayF:
    opt = _as_complex_array(opt)
    vals = [np.real_if_close(np.trace(rho @ opt)) for rho in rho_t]
    return np.asarray(vals, dtype=np.float64)
