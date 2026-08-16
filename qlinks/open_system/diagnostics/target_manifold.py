from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


def _orthonormal_target_basis(
    target_states: npt.ArrayLike,
    *,
    dim: int | None = None,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return orthonormal target states as columns."""
    matrix = np.asarray(target_states, dtype=np.complex128)

    if matrix.ndim == 1:
        if dim is not None and matrix.size != dim:
            raise ValueError(f"target state has dimension {matrix.size}; expected {dim}.")
        matrix = matrix.reshape(matrix.size, 1)
    elif matrix.ndim == 2:
        if dim is not None:
            if matrix.shape[0] == dim:
                pass
            elif matrix.shape[1] == dim:
                matrix = matrix.T
            else:
                raise ValueError(
                    "target states must have shape (dim, n_states) or (n_states, dim)."
                )
        elif matrix.shape[0] < matrix.shape[1]:
            # Common notebook convention for a small manifold is one state per row.
            matrix = matrix.T
    else:
        raise ValueError("target states must be one- or two-dimensional.")

    if matrix.shape[1] == 0:
        raise ValueError("target_states must contain at least one state.")

    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(diagonal > tolerance))
    if rank == 0:
        raise ValueError("target_states have numerical rank zero.")

    return np.asarray(q[:, :rank], dtype=np.complex128)


def target_manifold_projector(
    target_states: npt.ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the projector onto a target manifold.

    ``target_states`` may be one target vector, a ``(dim, n_states)`` matrix, or
    an ``(n_states, dim)`` matrix.  The columns/rows are orthonormalized before
    building the projector, so linearly dependent target vectors are harmless.
    """
    target_basis = _orthonormal_target_basis(target_states, tolerance=tolerance)
    return target_basis @ target_basis.conj().T


def target_manifold_weight(
    density_matrix: npt.ArrayLike,
    *,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    projector: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> float:
    """Return ``Tr(P_target rho)`` for one density matrix.

    Pass either ``target_states``/``target_basis`` or an explicit ``projector``.
    When a target basis is supplied, the computation uses
    ``Tr(Q^dagger rho Q)`` and avoids materializing the full projector.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")

    n_target_specs = sum(spec is not None for spec in (target_states, target_basis, projector))
    if n_target_specs != 1:
        raise ValueError("Pass exactly one of target_states, target_basis, or projector.")

    if projector is not None:
        p_target = np.asarray(projector, dtype=np.complex128)
        if p_target.shape != rho.shape:
            raise ValueError("projector must have the same shape as density_matrix.")
        value = np.trace(p_target @ rho)
        return float(np.real_if_close(value).real)

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(raw_basis, dim=rho.shape[0], tolerance=tolerance)
    value = np.trace(q.conj().T @ rho @ q)
    return float(np.real_if_close(value).real)


def target_manifold_weight_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    projector: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return ``Tr(P_target rho(t))`` for evolution or MCWF output.

    Exactly one data source must be supplied: ``density_matrices``, a
    ``LindbladEvolutionResult`` via ``evolution_result``, an ``EnsembleResult``
    via ``ensemble_result``, or MCWF ``state_snapshots``.  For state snapshots,
    each snapshot is expected to be a ``(dim, n_trajectories)`` state matrix and
    the returned weight is the trajectory average of ``<psi|P_target|psi>``.
    """
    sources = (density_matrices, evolution_result, ensemble_result, state_snapshots)
    if sum(source is not None for source in sources) != 1:
        raise ValueError(
            "Pass exactly one of density_matrices, evolution_result, "
            "ensemble_result, or state_snapshots."
        )

    resolved_density_matrices = density_matrices
    resolved_state_snapshots = state_snapshots

    if evolution_result is not None:
        resolved_density_matrices = getattr(evolution_result, "density_matrices", None)
        if resolved_density_matrices is None:
            raise ValueError("evolution_result does not contain density_matrices.")

    if ensemble_result is not None:
        rho_t = tuple(getattr(ensemble_result, "rho_t", ()) or ())
        if rho_t:
            resolved_density_matrices = rho_t
        else:
            snapshots = getattr(ensemble_result, "state_snapshots", None)
            if snapshots is None:
                raise ValueError(
                    "ensemble_result must contain rho_t or state_snapshots for "
                    "target-manifold weights."
                )
            resolved_state_snapshots = tuple(snapshots)

    n_target_specs = sum(spec is not None for spec in (target_states, target_basis, projector))
    if n_target_specs != 1:
        raise ValueError("Pass exactly one of target_states, target_basis, or projector.")

    if resolved_density_matrices is not None:
        return np.asarray(
            [
                target_manifold_weight(
                    density_matrix,
                    target_states=target_states,
                    target_basis=target_basis,
                    projector=projector,
                    tolerance=tolerance,
                )
                for density_matrix in resolved_density_matrices
            ],
            dtype=np.float64,
        )

    assert resolved_state_snapshots is not None
    snapshot_tuple = tuple(resolved_state_snapshots)
    if len(snapshot_tuple) == 0:
        raise ValueError("state_snapshots must be non-empty.")

    first_snapshot = np.asarray(snapshot_tuple[0], dtype=np.complex128)
    if first_snapshot.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")

    if projector is not None:
        p_target = np.asarray(projector, dtype=np.complex128)
        if p_target.shape != (first_snapshot.shape[0], first_snapshot.shape[0]):
            raise ValueError("projector has incompatible shape for state_snapshots.")
        weights = []
        for snapshot in snapshot_tuple:
            states = np.asarray(snapshot, dtype=np.complex128)
            values = np.einsum("ij,ij->j", states.conj(), p_target @ states)
            weights.append(float(np.real(np.mean(values))))
        return np.asarray(weights, dtype=np.float64)

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(
        raw_basis,
        dim=first_snapshot.shape[0],
        tolerance=tolerance,
    )
    weights = []
    for snapshot in snapshot_tuple:
        states = np.asarray(snapshot, dtype=np.complex128)
        if states.ndim != 2 or states.shape[0] != q.shape[0]:
            raise ValueError("state snapshot has incompatible shape.")
        overlaps = q.conj().T @ states
        values = np.sum(np.abs(overlaps) ** 2, axis=0)
        weights.append(float(np.real(np.mean(values))))
    return np.asarray(weights, dtype=np.float64)


def _resolve_density_matrix_or_snapshot_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Resolve one density/state time-series source.

    The returned source kind is either ``"density_matrices"`` or
    ``"state_snapshots"``.  This helper intentionally accepts duck-typed solver
    outputs so diagnostics can be used with lightweight notebook objects.
    """
    sources = (density_matrices, evolution_result, ensemble_result, state_snapshots)
    if sum(source is not None for source in sources) != 1:
        raise ValueError(
            "Pass exactly one of density_matrices, evolution_result, "
            "ensemble_result, or state_snapshots."
        )

    if evolution_result is not None:
        resolved_density_matrices = getattr(evolution_result, "density_matrices", None)
        if resolved_density_matrices is None:
            raise ValueError("evolution_result does not contain density_matrices.")
        return "density_matrices", tuple(resolved_density_matrices)

    if ensemble_result is not None:
        rho_t = tuple(getattr(ensemble_result, "rho_t", ()) or ())
        if rho_t:
            return "density_matrices", rho_t
        snapshots = getattr(ensemble_result, "state_snapshots", None)
        if snapshots is None:
            raise ValueError(
                "ensemble_result must contain rho_t or state_snapshots for "
                "target-manifold diagnostics."
            )
        return "state_snapshots", tuple(snapshots)

    if density_matrices is not None:
        return "density_matrices", tuple(density_matrices)

    assert state_snapshots is not None
    return "state_snapshots", tuple(state_snapshots)


def target_manifold_density_matrix(
    density_matrix: npt.ArrayLike,
    *,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    normalize: bool = True,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the density matrix reduced to the target manifold basis.

    The returned matrix is ``Q^dagger rho Q`` where columns of ``Q`` are an
    orthonormal target basis.  When ``normalize=True`` this is conditioned on
    being in the manifold by dividing by ``Tr(Q^dagger rho Q)``.  If the weight
    is numerically zero, the unnormalized zero matrix is returned.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")

    if (target_states is None) == (target_basis is None):
        raise ValueError("Pass exactly one of target_states or target_basis.")

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(raw_basis, dim=rho.shape[0], tolerance=tolerance)
    reduced = np.asarray(q.conj().T @ rho @ q, dtype=np.complex128)

    if normalize:
        weight = float(np.real_if_close(np.trace(reduced)).real)
        if abs(weight) > tolerance:
            reduced = reduced / weight

    return reduced


def _target_manifold_density_matrix_from_snapshot(
    snapshot: npt.ArrayLike,
    *,
    target_basis: npt.ArrayLike,
    normalize: bool,
    tolerance: float,
) -> np.ndarray:
    states = np.asarray(snapshot, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")

    q = _orthonormal_target_basis(
        target_basis,
        dim=states.shape[0],
        tolerance=tolerance,
    )
    overlaps = q.conj().T @ states
    reduced = np.asarray(overlaps @ overlaps.conj().T, dtype=np.complex128)
    reduced /= float(states.shape[1])

    if normalize:
        weight = float(np.real_if_close(np.trace(reduced)).real)
        if abs(weight) > tolerance:
            reduced = reduced / weight

    return reduced


def target_manifold_density_matrix_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    normalize: bool = True,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return ``Q^dagger rho(t) Q`` for a target manifold basis.

    The result has shape ``(n_times, manifold_dimension, manifold_dimension)``.
    With ``normalize=True`` each time slice is conditioned by its target-manifold
    weight, i.e. it is divided by ``Tr(P_target rho(t))`` when the weight is
    nonzero.
    """
    if (target_states is None) == (target_basis is None):
        raise ValueError("Pass exactly one of target_states or target_basis.")

    raw_basis = target_basis if target_basis is not None else target_states
    source_kind, values = _resolve_density_matrix_or_snapshot_series(
        density_matrices=density_matrices,
        evolution_result=evolution_result,
        ensemble_result=ensemble_result,
        state_snapshots=state_snapshots,
    )
    if not values:
        raise ValueError(f"{source_kind} must be non-empty.")

    if source_kind == "density_matrices":
        return np.stack(
            [
                target_manifold_density_matrix(
                    density_matrix,
                    target_basis=raw_basis,
                    normalize=normalize,
                    tolerance=tolerance,
                )
                for density_matrix in values
            ]
        )

    first_snapshot = np.asarray(values[0], dtype=np.complex128)
    if first_snapshot.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")
    q = _orthonormal_target_basis(
        raw_basis,
        dim=first_snapshot.shape[0],
        tolerance=tolerance,
    )
    return np.stack(
        [
            _target_manifold_density_matrix_from_snapshot(
                snapshot,
                target_basis=q,
                normalize=normalize,
                tolerance=tolerance,
            )
            for snapshot in values
        ]
    )


def target_manifold_populations_series(**kwargs: Any) -> np.ndarray:
    """Return diagonal populations of the target-manifold density matrices."""
    reduced = target_manifold_density_matrix_series(**kwargs)
    return np.real(np.diagonal(reduced, axis1=1, axis2=2))


def target_manifold_coherence_series(
    *,
    norm: Literal["fro", "l1"] = "fro",
    **kwargs: Any,
) -> np.ndarray:
    """Return off-diagonal coherence of target-manifold density matrices.

    ``norm="fro"`` returns the Frobenius norm of off-diagonal entries, while
    ``norm="l1"`` returns their elementwise absolute sum.
    """
    reduced = target_manifold_density_matrix_series(**kwargs)
    offdiag = reduced.copy()
    indices = np.arange(offdiag.shape[1])
    offdiag[:, indices, indices] = 0.0

    if norm == "fro":
        return np.linalg.norm(offdiag.reshape(offdiag.shape[0], -1), axis=1)
    if norm == "l1":
        return np.sum(np.abs(offdiag), axis=(1, 2))
    raise ValueError("norm must be 'fro' or 'l1'.")


def target_manifold_purity_series(**kwargs: Any) -> np.ndarray:
    """Return ``Tr(rho_target(t)^2)`` for target-manifold density matrices."""
    reduced = target_manifold_density_matrix_series(**kwargs)
    values = np.einsum("tij,tji->t", reduced, reduced)
    return np.real_if_close(values).real.astype(np.float64, copy=False)


def target_manifold_entropy_series(
    *,
    base: float | None = None,
    tolerance: float = 1.0e-12,
    **kwargs: Any,
) -> np.ndarray:
    """Return von Neumann entropy of target-manifold density matrices.

    The input density matrices are usually conditioned by leaving
    ``normalize=True`` in ``kwargs``.  Eigenvalues below ``tolerance`` are
    ignored in the logarithm.
    """
    reduced = target_manifold_density_matrix_series(**kwargs)
    entropies: list[float] = []
    log_base = 1.0 if base is None else float(np.log(base))
    for rho in reduced:
        hermitian = 0.5 * (rho + rho.conj().T)
        eigvals = np.linalg.eigvalsh(hermitian)
        eigvals = np.real(eigvals[eigvals > tolerance])
        if eigvals.size == 0:
            entropies.append(0.0)
            continue
        entropy = -float(np.sum(eigvals * np.log(eigvals)))
        entropies.append(entropy / log_base)
    return np.asarray(entropies, dtype=np.float64)
