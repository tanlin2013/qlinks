from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

INTERNAL_KINETIC_MATRIX_METADATA_KEY = "_cage_internal_kinetic_matrix"
BOUNDARY_OVERLAP_MATRIX_METADATA_KEY = "_cage_boundary_overlap_matrix"


@dataclass(frozen=True)
class CandidateSubgraph:
    """Candidate support for an interference-caged state."""

    vertices: NDArray[np.int_]
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        support_indices = np.asarray(self.vertices, dtype=np.int64)

        if support_indices.ndim != 1:
            raise ValueError("CandidateSubgraph.vertices must be a 1D array.")

        if support_indices.size == 0:
            raise ValueError("CandidateSubgraph.vertices must not be empty.")

        if len(np.unique(support_indices)) != len(support_indices):
            raise ValueError("CandidateSubgraph.vertices contains duplicates.")

        object.__setattr__(self, "vertices", np.sort(support_indices))

    @property
    def size(self) -> int:
        """Number of vertices in the candidate support."""
        return int(self.vertices.size)
