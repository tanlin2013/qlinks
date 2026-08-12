"""Embedding of local pattern operators into constrained Hilbert spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.local_structure.reduced_density import (
    _local_pattern_basis_context_from_basis,
    _LocalPatternBasisContext,
)


@dataclass(frozen=True, slots=True)
class _LocalPatternEmbeddingContext:
    """Precomputed constrained-basis embedding data for one local region."""

    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    source_full_indices: npt.NDArray[np.int64]
    target_full_indices: npt.NDArray[np.int64]
    source_local_indices: npt.NDArray[np.int64]
    target_local_indices: npt.NDArray[np.int64]
    dim: int

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)


def _embedding_context_from_basis_context(
    context: _LocalPatternBasisContext,
) -> _LocalPatternEmbeddingContext:
    source_full_chunks: list[npt.NDArray[np.int64]] = []
    target_full_chunks: list[npt.NDArray[np.int64]] = []
    source_local_chunks: list[npt.NDArray[np.int64]] = []
    target_local_chunks: list[npt.NDArray[np.int64]] = []

    for full_indices, local_indices in context.environment_groups:
        group_size = int(full_indices.size)

        if group_size == 0:
            continue

        source_full_chunks.append(np.repeat(full_indices, group_size))
        target_full_chunks.append(np.tile(full_indices, group_size))
        source_local_chunks.append(np.repeat(local_indices, group_size))
        target_local_chunks.append(np.tile(local_indices, group_size))

    if len(source_full_chunks) == 0:
        source_full_indices = np.asarray((), dtype=np.int64)
        target_full_indices = np.asarray((), dtype=np.int64)
        source_local_indices = np.asarray((), dtype=np.int64)
        target_local_indices = np.asarray((), dtype=np.int64)
    else:
        source_full_indices = np.concatenate(source_full_chunks).astype(np.int64, copy=False)
        target_full_indices = np.concatenate(target_full_chunks).astype(np.int64, copy=False)
        source_local_indices = np.concatenate(source_local_chunks).astype(np.int64, copy=False)
        target_local_indices = np.concatenate(target_local_chunks).astype(np.int64, copy=False)

    return _LocalPatternEmbeddingContext(
        variable_indices=context.variable_indices,
        local_patterns=context.local_patterns,
        source_full_indices=source_full_indices,
        target_full_indices=target_full_indices,
        source_local_indices=source_local_indices,
        target_local_indices=target_local_indices,
        dim=context.dim,
    )


def _embedding_context_from_basis(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...] | list[int],
    local_patterns: tuple[tuple[int, ...], ...],
) -> _LocalPatternEmbeddingContext:
    """Precompute constrained-basis transitions induced by local pattern changes."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
        local_patterns=local_patterns,
    )
    return _embedding_context_from_basis_context(basis_context)


def _embed_local_pattern_operator_from_context(
    *,
    context: _LocalPatternEmbeddingContext,
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    local_dim = context.local_dim

    if local_operator.shape != (local_dim, local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != {(local_dim, local_dim)}."
        )

    if context.source_full_indices.size == 0:
        return sp.csr_array((context.dim, context.dim), dtype=np.complex128)

    data = np.asarray(
        local_operator[context.target_local_indices, context.source_local_indices],
        dtype=np.complex128,
    )
    nonzero_mask = data != 0.0

    return sp.csr_array(
        (
            data[nonzero_mask],
            (
                context.target_full_indices[nonzero_mask],
                context.source_full_indices[nonzero_mask],
            ),
        ),
        shape=(context.dim, context.dim),
        dtype=np.complex128,
    )


def embed_local_pattern_operator(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
    local_patterns: tuple[tuple[int, ...], ...],
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    """Embed a local operator into the constrained full basis."""
    context = _embedding_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
        local_patterns=local_patterns,
    )
    return _embed_local_pattern_operator_from_context(
        context=context,
        local_operator=local_operator,
    )
