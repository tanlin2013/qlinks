from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    as_backend_dense_array,
    as_backend_sparse_matrix,
    get_open_system_backend,
)


@dataclass(frozen=True, slots=True)
class DenseLindbladOperators:
    """Dense Lindblad operators prepared for repeated RHS evaluations."""

    backend: OpenSystemBackend
    hamiltonian: Any
    jumps: tuple[Any, ...]
    jump_daggers: tuple[Any, ...]
    jump_dagger_jumps: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SparseLindbladOperators:
    """Sparse Lindblad operators prepared for Liouville-space construction."""

    backend: OpenSystemBackend
    sparse_format: str
    hamiltonian: Any
    identity: Any
    jumps: tuple[Any, ...]
    jump_dagger_jumps: tuple[Any, ...]

    @property
    def dim(self) -> int:
        return int(self.hamiltonian.shape[0])


def prepare_dense_lindblad_operators(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    dtype=np.complex128,
) -> DenseLindbladOperators:
    """Convert Lindblad operators once for repeated dense RHS calls."""
    backend_obj = get_open_system_backend(backend)

    hamiltonian_dense = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=dtype,
    )
    jump_operators = tuple(
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=dtype,
        )
        for jump in jumps
    )
    jump_daggers = tuple(jump.conj().T for jump in jump_operators)
    jump_dagger_jumps = tuple(
        jump_dagger @ jump for jump_dagger, jump in zip(jump_daggers, jump_operators)
    )

    return DenseLindbladOperators(
        backend=backend_obj,
        hamiltonian=hamiltonian_dense,
        jumps=jump_operators,
        jump_daggers=jump_daggers,
        jump_dagger_jumps=jump_dagger_jumps,
    )


def prepare_sparse_lindblad_operators(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    sparse_format: str = "csc",
    dtype=np.complex128,
) -> SparseLindbladOperators:
    """Convert sparse Lindblad operators once for Liouville-space construction."""
    backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = as_backend_sparse_matrix(
        hamiltonian,
        backend=backend_obj,
        format=sparse_format,
        dtype=dtype,
    )
    dim = int(hamiltonian_sparse.shape[0])
    identity = backend_obj.sparse_identity(
        dim,
        format=sparse_format,
        dtype=dtype,
    )

    jump_operators = tuple(
        as_backend_sparse_matrix(
            jump,
            backend=backend_obj,
            format=sparse_format,
            dtype=dtype,
        )
        for jump in jumps
    )
    jump_dagger_jumps = tuple(
        (jump.conj().T @ jump).asformat(sparse_format) for jump in jump_operators
    )

    return SparseLindbladOperators(
        backend=backend_obj,
        sparse_format=sparse_format,
        hamiltonian=hamiltonian_sparse,
        identity=identity,
        jumps=jump_operators,
        jump_dagger_jumps=jump_dagger_jumps,
    )


def vectorize_density_matrix(density_matrix: Any) -> Any:
    return density_matrix.reshape(-1, order="F")


def unvectorize_density_matrix(vectorized_density_matrix: Any, dim: int) -> Any:
    return vectorized_density_matrix.reshape((dim, dim), order="F")


def build_liouvillian_from_prepared(
    sparse_operators: SparseLindbladOperators,
):
    """Build the Liouvillian from preconverted sparse Lindblad operators."""
    backend_obj = sparse_operators.backend
    sparse_format = sparse_operators.sparse_format
    hamiltonian = sparse_operators.hamiltonian
    identity = sparse_operators.identity

    liouvillian = -1j * (
        backend_obj.sparse_kron(identity, hamiltonian, format=sparse_format)
        - backend_obj.sparse_kron(hamiltonian.T, identity, format=sparse_format)
    )

    for jump, jump_dagger_jump in zip(
        sparse_operators.jumps,
        sparse_operators.jump_dagger_jumps,
    ):
        jump_term = backend_obj.sparse_kron(
            jump.conj(),
            jump,
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


def build_liouvillian(
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    *,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    sparse_format: str = "csc",
    dtype=np.complex128,
):
    sparse_operators = prepare_sparse_lindblad_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
        sparse_format=sparse_format,
        dtype=dtype,
    )
    return build_liouvillian_from_prepared(sparse_operators)


def lindblad_rhs_density_matrix(
    density_matrix: Any,
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
):
    dense_operators = prepare_dense_lindblad_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
    )
    return lindblad_rhs_density_matrix_prepared(
        density_matrix,
        dense_operators=dense_operators,
    )


def lindblad_rhs_density_matrix_prepared(
    density_matrix: Any,
    *,
    dense_operators: DenseLindbladOperators,
):
    """Evaluate the dense Lindblad RHS using preconverted operators."""
    backend_obj = dense_operators.backend
    array_module = backend_obj.array_module

    density_matrix = as_backend_dense_array(
        density_matrix,
        backend=backend_obj,
        dtype=np.complex128,
    )
    hamiltonian = dense_operators.hamiltonian

    derivative = -1j * (hamiltonian @ density_matrix - density_matrix @ hamiltonian)

    for jump_operator, jump_dagger, jump_dagger_jump in zip(
        dense_operators.jumps,
        dense_operators.jump_daggers,
        dense_operators.jump_dagger_jumps,
    ):
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


def estimate_lindblad_scale_prepared(
    dense_operators: DenseLindbladOperators,
) -> float:
    """Cheap stiffness scale from preconverted dense operators."""
    array_module = dense_operators.backend.array_module
    scale = float(array_module.linalg.norm(dense_operators.hamiltonian))

    for jump in dense_operators.jumps:
        jump_norm = float(array_module.linalg.norm(jump))
        scale += jump_norm * jump_norm

    return max(scale, 1.0)
