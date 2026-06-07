from __future__ import annotations

from typing import Any

import numpy as np

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    as_backend_dense_array,
    as_backend_sparse_matrix,
    get_open_system_backend,
)


def vectorize_density_matrix(density_matrix: Any) -> Any:
    return density_matrix.reshape(-1, order="F")


def unvectorize_density_matrix(vectorized_density_matrix: Any, dim: int) -> Any:
    return vectorized_density_matrix.reshape((dim, dim), order="F")


def build_liouvillian(
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    *,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    sparse_format: str = "csc",
    dtype=np.complex128,
):
    backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = as_backend_sparse_matrix(
        hamiltonian,
        backend=backend_obj,
        format=sparse_format,
        dtype=dtype,
    )
    dim = hamiltonian_sparse.shape[0]
    identity = backend_obj.sparse_identity(
        dim,
        format=sparse_format,
        dtype=dtype,
    )

    liouvillian = -1j * (
        backend_obj.sparse_kron(identity, hamiltonian_sparse, format=sparse_format)
        - backend_obj.sparse_kron(hamiltonian_sparse.T, identity, format=sparse_format)
    )

    for jump in jumps:
        jump_sparse = as_backend_sparse_matrix(
            jump,
            backend=backend_obj,
            format=sparse_format,
            dtype=dtype,
        )
        jump_dagger_jump = (jump_sparse.conj().T @ jump_sparse).asformat(sparse_format)

        jump_term = backend_obj.sparse_kron(
            jump_sparse.conj(),
            jump_sparse,
            format=sparse_format,
        )
        left_loss = backend_obj.sparse_kron(
            identity,
            jump_dagger_jump,
            format=sparse_format,
        )
        right_loss = backend_obj.sparse_kron(
            jump_dagger_jump.T,
            identity,
            format=sparse_format,
        )

        liouvillian = liouvillian + jump_term - 0.5 * left_loss - 0.5 * right_loss

    return liouvillian.asformat(sparse_format)


def lindblad_rhs_density_matrix(
    density_matrix: Any,
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
):
    backend_obj = get_open_system_backend(backend)
    array_module = backend_obj.array_module

    density_matrix = as_backend_dense_array(
        density_matrix,
        backend=backend_obj,
        dtype=np.complex128,
    )
    hamiltonian = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_operators = [
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    ]

    derivative = -1j * (hamiltonian @ density_matrix - density_matrix @ hamiltonian)

    for jump_operator in jump_operators:
        jump_dagger = jump_operator.conj().T
        jump_dagger_jump = jump_dagger @ jump_operator

        derivative = derivative + (jump_operator @ density_matrix @ jump_dagger)
        derivative = derivative - 0.5 * (
            jump_dagger_jump @ density_matrix + density_matrix @ jump_dagger_jump
        )

    return array_module.asarray(derivative, dtype=array_module.complex128)


def estimate_lindblad_scale(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> float:
    """Cheap stiffness scale for RK4 step-size sanity checks."""
    backend_obj = get_open_system_backend(backend)
    array_module = backend_obj.array_module

    hamiltonian_dense = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    scale = float(array_module.linalg.norm(hamiltonian_dense))

    for jump in jumps:
        jump_dense = as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        jump_norm = float(array_module.linalg.norm(jump_dense))
        scale += jump_norm * jump_norm

    return max(scale, 1.0)
