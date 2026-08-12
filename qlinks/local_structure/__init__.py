"""Neutral local-structure primitives shared by research layers.

The public names here describe local algebra and constrained-basis structure;
they carry no caging- or Lindblad-specific semantics.  Higher-level research
layers should depend on these primitives rather than on each other.
"""

from qlinks.local_structure.embedding import embed_local_pattern_operator
from qlinks.local_structure.matrix_units import (
    LocalMatrixUnitTerm,
    local_operator_matrix_unit_expansion,
    local_rank_one_matrix_unit_expansion,
)
from qlinks.local_structure.reduced_density import (
    LocalReducedDensityMatrix,
    local_reduced_density_matrix_from_state,
    local_reduced_density_matrix_from_state_matrix,
)

__all__ = [
    "LocalMatrixUnitTerm",
    "LocalReducedDensityMatrix",
    "embed_local_pattern_operator",
    "local_operator_matrix_unit_expansion",
    "local_rank_one_matrix_unit_expansion",
    "local_reduced_density_matrix_from_state",
    "local_reduced_density_matrix_from_state_matrix",
]
