from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

DensityMatrixKind = Literal[
    "pure",
    "mixed",
    "maximally_mixed",
]


def _rng_from_seed(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def normalize_state(
    state: npt.ArrayLike,
    *,
    atol: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Return a normalized complex state vector."""
    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))

    if norm <= atol:
        raise ValueError("Cannot normalize a zero-norm state.")

    return vector / norm


def random_pure_state(
    dim: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.complex128]:
    """Draw a Haar-like random complex state vector."""
    if dim <= 0:
        raise ValueError("dim must be positive.")

    generator = _rng_from_seed(rng)

    real = generator.normal(size=dim)
    imag = generator.normal(size=dim)
    state = real + 1j * imag

    return normalize_state(state)


def density_matrix_from_state(
    state: npt.ArrayLike,
    *,
    normalize: bool = True,
) -> npt.NDArray[np.complex128]:
    """Return the pure density matrix associated with a state vector.

    Args:
        state: One-dimensional state vector.
        normalize: Whether to normalize the vector before forming
            ``|psi><psi|``.

    Returns:
        Complex density matrix ``|psi><psi|``.
    """
    return pure_density_matrix(state, normalize=normalize)


def pure_density_matrix(
    state: npt.ArrayLike,
    *,
    normalize: bool = True,
) -> npt.NDArray[np.complex128]:
    """Return |psi><psi|."""
    vector = np.asarray(state, dtype=np.complex128)

    if vector.ndim != 1:
        raise ValueError("state must be a one-dimensional vector.")

    if normalize:
        vector = normalize_state(vector)

    return np.outer(vector, np.conjugate(vector)).astype(
        np.complex128,
        copy=False,
    )


def random_pure_density_matrix(
    dim: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.complex128]:
    """Draw a random pure density matrix |psi><psi|."""
    state = random_pure_state(dim, rng=rng)
    return pure_density_matrix(state, normalize=False)


def random_mixed_density_matrix(
    dim: int,
    *,
    rank: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Draw a random mixed density matrix using a Ginibre ensemble.

    The state is generated as ``rho = X X^\dagger / Tr(X X^\dagger)`` where
    ``X`` has shape ``(dim, rank)``.

    Args:
        dim: Hilbert-space dimension.
        rank: Optional effective rank.  If ``None``, use full rank ``dim``.
        rng: NumPy random generator or seed.

    Returns:
        Hermitian, positive-semidefinite density matrix with trace one.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")

    if rank is None:
        rank = dim

    if rank <= 0:
        raise ValueError("rank must be positive.")

    if rank > dim:
        raise ValueError("rank must be less than or equal to dim.")

    generator = _rng_from_seed(rng)

    real = generator.normal(size=(dim, rank))
    imag = generator.normal(size=(dim, rank))
    ginibre = real + 1j * imag

    rho = ginibre @ np.conjugate(ginibre).T
    trace = np.trace(rho)

    if abs(trace) == 0:
        raise RuntimeError("Generated a zero-trace random density matrix.")

    rho = rho / trace

    # Symmetrize to remove tiny numerical anti-Hermitian noise.
    rho = 0.5 * (rho + np.conjugate(rho).T)

    # Re-normalize after symmetrization.
    rho = rho / np.trace(rho)

    return rho.astype(np.complex128, copy=False)


def random_density_matrix(
    dim: int,
    *,
    kind: DensityMatrixKind = "mixed",
    rank: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Draw a random pure or mixed density matrix.

    Args:
        dim: Hilbert-space dimension.
        kind: ``"pure"`` for ``|psi><psi|`` or ``"mixed"`` for a Ginibre
            mixed state.
        rank: Optional effective rank for ``kind="mixed"``.
        rng: NumPy random generator or seed.

    Returns:
        Density matrix with trace one.

    Raises:
        ValueError: If ``kind`` is unsupported or ``rank`` is incompatible.
    """
    if kind == "pure":
        if rank is not None and rank != 1:
            raise ValueError("rank is only meaningful for kind='mixed'.")
        return random_pure_density_matrix(dim, rng=rng)

    if kind == "mixed":
        return random_mixed_density_matrix(dim, rank=rank, rng=rng)

    raise ValueError(f"Unsupported density-matrix kind: {kind!r}")


def initial_density_matrix(
    dim: int,
    *,
    kind: DensityMatrixKind = "mixed",
    rank: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.complex128]:
    """Create a convenient initial density matrix."""
    if dim <= 0:
        raise ValueError("dim must be positive.")

    if kind == "maximally_mixed":
        if rank is not None:
            raise ValueError("rank is not meaningful for kind='maximally_mixed'.")
        return np.eye(dim, dtype=np.complex128) / dim

    if kind == "pure":
        if rank is not None and rank != 1:
            raise ValueError("rank is not meaningful for kind='pure'.")
        return random_pure_density_matrix(dim, rng=rng)

    if kind == "mixed":
        return random_mixed_density_matrix(dim, rank=rank, rng=rng)

    raise ValueError(f"Unsupported initial density matrix kind: {kind!r}")
