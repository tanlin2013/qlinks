from __future__ import annotations

import numpy as np
import pytest

from qlinks.open_system import (
    initial_density_matrix,
    pure_density_matrix,
    random_density_matrix,
    random_mixed_density_matrix,
    random_pure_density_matrix,
    random_pure_state,
    verify_density_matrix,
)


def _assert_density_matrix(rho: np.ndarray, *, dim: int) -> None:
    assert rho.shape == (dim, dim)
    np.testing.assert_allclose(rho, np.conjugate(rho).T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)

    eigenvalues = np.linalg.eigvalsh(rho)
    assert np.min(eigenvalues) >= -1e-12


def test_random_pure_state_is_normalized() -> None:
    state = random_pure_state(5, rng=123)

    assert state.shape == (5,)
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)


def test_pure_density_matrix_from_state() -> None:
    state = np.array([1.0, 1.0j], dtype=np.complex128)

    rho = pure_density_matrix(state)

    _assert_density_matrix(rho, dim=2)
    np.testing.assert_allclose(np.trace(rho @ rho), 1.0, atol=1e-12)


def test_random_pure_density_matrix_is_rank_one_projector() -> None:
    rho = random_pure_density_matrix(6, rng=123)

    _assert_density_matrix(rho, dim=6)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho @ rho), 1.0, atol=1e-12)


def test_random_mixed_density_matrix_is_valid() -> None:
    rho = random_mixed_density_matrix(6, rng=123)

    _assert_density_matrix(rho, dim=6)

    purity = np.real(np.trace(rho @ rho))
    assert purity < 1.0
    assert purity > 1.0 / 6.0 - 1e-12


def test_random_mixed_density_matrix_with_rank() -> None:
    rho = random_mixed_density_matrix(6, rank=2, rng=123)

    _assert_density_matrix(rho, dim=6)

    eigenvalues = np.linalg.eigvalsh(rho)
    numerical_rank = int(np.count_nonzero(eigenvalues > 1e-10))

    assert numerical_rank <= 2


def test_random_density_matrix_dispatches_pure() -> None:
    rho = random_density_matrix(4, kind="pure", rng=123)

    _assert_density_matrix(rho, dim=4)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)


def test_random_density_matrix_dispatches_mixed() -> None:
    rho = random_density_matrix(4, kind="mixed", rng=123)

    _assert_density_matrix(rho, dim=4)


def test_random_density_matrix_rejects_rank_for_pure() -> None:
    with pytest.raises(ValueError, match="rank"):
        random_density_matrix(4, kind="pure", rank=2, rng=123)


def test_random_density_matrix_rejects_bad_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        random_density_matrix(4, kind="bad", rng=123)  # type: ignore[arg-type]


def test_random_mixed_density_matrix_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        random_mixed_density_matrix(4, rank=0, rng=123)

    with pytest.raises(ValueError, match="rank"):
        random_mixed_density_matrix(4, rank=5, rng=123)


def test_initial_density_matrix_mixed_is_valid():
    density_matrix = initial_density_matrix(5, kind="mixed", rng=0)
    verification = verify_density_matrix(density_matrix)
    assert verification.is_density_matrix


def test_initial_density_matrix_pure_is_projector():
    density_matrix = initial_density_matrix(5, kind="pure", rng=0)
    verification = verify_density_matrix(density_matrix)
    assert verification.is_density_matrix
    assert verification.purity == pytest.approx(1.0)


def test_initial_density_matrix_maximally_mixed():
    density_matrix = initial_density_matrix(4, kind="maximally_mixed")
    np.testing.assert_allclose(
        density_matrix,
        np.eye(4, dtype=np.complex128) / 4,
    )


def test_initial_density_matrix_rejects_rank_for_maximally_mixed():
    with pytest.raises(ValueError, match="rank"):
        initial_density_matrix(4, kind="maximally_mixed", rank=2)
