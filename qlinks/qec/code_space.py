from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.encoded.binary_basis import BinaryEncodedBasis


@dataclass(frozen=True, slots=True)
class CodeSpace:
    """Orthonormal code-space basis embedded in a constrained Hilbert space.

    The columns of ``vectors`` are the orthonormal code states in the basis
    ordering fixed by ``basis``.  This class is intentionally dense and small-
    code-space oriented; it is meant for diagnostics on candidate cage manifolds
    before building heavier Liouvillian objects.
    """

    basis: Basis | BinaryEncodedBasis
    vectors: npt.NDArray[np.complex128]
    labels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors, dtype=np.complex128)

        if vectors.ndim != 2:
            raise ValueError("CodeSpace.vectors must be a two-dimensional array.")

        if vectors.shape[0] != self.basis.n_states:
            raise ValueError(
                f"Expected vectors with {self.basis.n_states} rows, got {vectors.shape[0]}."
            )

        if vectors.shape[1] == 0:
            raise ValueError("CodeSpace must contain at least one code vector.")

        labels = tuple(self.labels)
        if labels and len(labels) != vectors.shape[1]:
            raise ValueError("labels must be empty or have one entry per code vector.")

        gram = vectors.conj().T @ vectors
        if not np.allclose(gram, np.eye(vectors.shape[1]), atol=1e-10):
            raise ValueError("CodeSpace vectors must be orthonormal. Use from_vectors().")

        if not labels:
            labels = tuple(range(vectors.shape[1]))

        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_vectors(
        cls,
        basis: Basis | BinaryEncodedBasis,
        vectors: npt.ArrayLike,
        *,
        labels: Sequence[Any] = (),
        orthonormalize: bool = True,
        rank_tolerance: float = 1e-12,
        allow_rank_deficient: bool = False,
    ) -> CodeSpace:
        """Build a code space from column vectors.

        Args:
            basis: Computational basis that fixes the ambient Hilbert space.
            vectors: Array with shape ``(basis.n_states, code_dimension)``.
            labels: Optional labels for the input vectors.
            orthonormalize: Whether to orthonormalize the supplied columns.
            rank_tolerance: Singular-value cutoff used during orthonormalization.
            allow_rank_deficient: If false, linearly dependent input vectors are
                rejected.  If true, dependent columns are dropped by SVD.

        Returns:
            CodeSpace with orthonormal column vectors.
        """
        arr = np.asarray(vectors, dtype=np.complex128)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError("vectors must be one- or two-dimensional.")

        if arr.shape[0] != basis.n_states:
            raise ValueError(f"Expected {basis.n_states} rows, got {arr.shape[0]}.")

        input_labels = tuple(labels)
        if input_labels and len(input_labels) != arr.shape[1]:
            raise ValueError("labels must have one entry per input vector.")

        if not orthonormalize:
            return cls(
                basis=basis,
                vectors=arr,
                labels=input_labels,
            )

        _u, singular_values, _vh = np.linalg.svd(arr, full_matrices=False)
        if singular_values.size == 0:
            raise ValueError("CodeSpace must contain at least one nonzero vector.")

        cutoff = float(rank_tolerance) * max(1.0, float(singular_values[0]))
        rank = int(np.count_nonzero(singular_values > cutoff))

        if rank == 0:
            raise ValueError("CodeSpace input vectors are numerically zero.")

        if rank < arr.shape[1] and not allow_rank_deficient:
            raise ValueError(
                "CodeSpace input vectors are linearly dependent. "
                "Pass allow_rank_deficient=True to keep an orthonormal basis for their span."
            )

        if rank < arr.shape[1]:
            # Rank-deficient input has no one-to-one map from original labels
            # to output columns; use an SVD basis for the span.
            u, _singular_values, _vh = np.linalg.svd(arr, full_matrices=False)
            vectors_out = u[:, :rank]
            output_labels: tuple[Any, ...] = tuple(range(rank))
        else:
            # Full-rank input keeps the original column order as much as
            # possible, which is useful when records label intended logical
            # basis states.
            q, _r = np.linalg.qr(arr, mode="reduced")
            vectors_out = q[:, :rank]
            output_labels = input_labels

        return cls(
            basis=basis,
            vectors=vectors_out,
            labels=output_labels,
        )

    @classmethod
    def from_row_vectors(
        cls,
        basis: Basis | BinaryEncodedBasis,
        row_vectors: npt.ArrayLike,
        **kwargs: Any,
    ) -> CodeSpace:
        """Build from row-major state vectors such as ``CageSearchResult.full_state_matrix()``."""
        arr = np.asarray(row_vectors, dtype=np.complex128)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("row_vectors must be one- or two-dimensional.")

        return cls.from_vectors(
            basis,
            arr.T,
            **kwargs,
        )

    @classmethod
    def from_basis_indices(
        cls,
        basis: Basis | BinaryEncodedBasis,
        indices: Sequence[int],
        *,
        labels: Sequence[Any] = (),
    ) -> CodeSpace:
        """Build a computational-basis code from selected basis indices."""
        if len(indices) == 0:
            raise ValueError("At least one basis index is required.")

        vectors = np.zeros((basis.n_states, len(indices)), dtype=np.complex128)
        for column, index in enumerate(indices):
            index = int(index)
            if index < 0 or index >= basis.n_states:
                raise IndexError(f"basis index {index} outside [0, {basis.n_states}).")
            vectors[index, column] = 1.0

        return cls(
            basis=basis,
            vectors=vectors,
            labels=tuple(labels),
        )

    @classmethod
    def from_cage_records(
        cls,
        basis: Basis | BinaryEncodedBasis,
        records: Sequence[Any],
        *,
        labels: Sequence[Any] = (),
        rank_tolerance: float = 1e-12,
        allow_rank_deficient: bool = False,
    ) -> CodeSpace:
        """Build a code space from ``CageRecord``-like objects.

        Each record may expose ``full_state`` directly, or the compact pair
        ``support`` and ``local_state``.  This duck-typed constructor avoids a
        hard import from ``qlinks.caging`` and also works for local-search
        records with the same attributes.
        """
        row_vectors = np.zeros((len(records), basis.n_states), dtype=np.complex128)

        for row, record in enumerate(records):
            full_state = getattr(record, "full_state", None)
            if full_state is not None:
                vector = np.asarray(full_state, dtype=np.complex128)
                if vector.shape != (basis.n_states,):
                    raise ValueError(
                        f"Record {row} full_state has shape {vector.shape}, "
                        f"expected ({basis.n_states},)."
                    )
                row_vectors[row, :] = vector
                continue

            support = np.asarray(record.support, dtype=np.int64)
            local_state = np.asarray(record.local_state, dtype=np.complex128)
            if support.ndim != 1 or local_state.ndim != 1 or support.size != local_state.size:
                raise ValueError(
                    "Each compact cage record must provide matching support/local_state."
                )
            row_vectors[row, support] = local_state

        default_labels = tuple(labels)
        if not default_labels:
            default_labels = tuple(
                getattr(record, "signature", i) for i, record in enumerate(records)
            )

        return cls.from_row_vectors(
            basis,
            row_vectors,
            labels=default_labels,
            rank_tolerance=rank_tolerance,
            allow_rank_deficient=allow_rank_deficient,
        )

    @classmethod
    def from_cage_collection(
        cls,
        collection: Any,
        *,
        labels: Sequence[Any] = (),
        rank_tolerance: float = 1e-12,
        allow_rank_deficient: bool = False,
    ) -> CodeSpace:
        """Build a code space from a cross-sector cage collection.

        The collection is duck-typed and must expose ``ambient_basis`` and
        ``to_ambient_row_vectors()``.  :class:`qlinks.qec.CageSectorCollection`
        is the intended producer, but the loose dependency keeps this class
        independent of the collection module.
        """
        ambient_basis = collection.ambient_basis
        row_vectors = collection.to_ambient_row_vectors()
        default_labels = tuple(labels)
        if not default_labels:
            default_labels = tuple(getattr(collection, "labels", ()))

        return cls.from_row_vectors(
            ambient_basis,
            row_vectors,
            labels=default_labels,
            rank_tolerance=rank_tolerance,
            allow_rank_deficient=allow_rank_deficient,
        )

    @property
    def dimension(self) -> int:
        """Number of encoded/code states."""
        return int(self.vectors.shape[1])

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the constrained Hilbert space containing the code."""
        return int(self.vectors.shape[0])

    @property
    def projector(self) -> npt.NDArray[np.complex128]:
        """Dense ambient-space projector onto the code space."""
        return self.vectors @ self.vectors.conj().T

    def projected_matrix(self, image: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Return ``V^dagger image`` for an ambient vector or matrix image."""
        arr = np.asarray(image, dtype=np.complex128)
        return self.vectors.conj().T @ arr

    def project_operator_action(self, image: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Project an ambient image back into the code space."""
        return self.vectors @ self.projected_matrix(image)

    def leakage_image(self, image: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Return the component of an operator image orthogonal to the code."""
        arr = np.asarray(image, dtype=np.complex128)
        return arr - self.project_operator_action(arr)

    def leakage_norm(
        self,
        image: npt.ArrayLike,
        *,
        ord: int | float | str | None = "fro",
    ) -> float:
        """Norm of the component of ``image`` outside the code space."""
        return float(np.linalg.norm(self.leakage_image(image), ord=ord))

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact, serialization-friendly summary of the code space."""
        return {
            "dimension": self.dimension,
            "ambient_dimension": self.ambient_dimension,
            "labels": tuple(repr(label) for label in self.labels),
        }

    def to_text(self) -> str:
        """Return a compact human-readable code-space summary."""
        from qlinks.qec.reporting import format_key_value_lines

        summary = self.to_summary_dict()
        return format_key_value_lines(
            "Code space",
            (
                ("dimension", summary["dimension"]),
                ("ambient dimension", summary["ambient_dimension"]),
                ("labels", summary["labels"]),
            ),
        )

    def format_summary(self) -> str:
        """Alias for :meth:`to_text`, useful in notebooks."""
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        """Return a rich renderable summary of the code space."""
        from qlinks.qec.reporting import add_summary_rows, require_rich

        _group, panel_cls, table_cls, _text = require_rich("CodeSpace")
        table = table_cls.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        add_summary_rows(
            table,
            (
                ("dimension", self.dimension),
                ("ambient dimension", self.ambient_dimension),
                ("labels", tuple(repr(label) for label in self.labels)),
            ),
        )
        return panel_cls(table, title="Code space")
