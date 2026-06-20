from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from qlinks.open_system.local_recycling import (  # noqa: F401
    LocalMatrixUnitTerm,
    LocalReducedDensityMatrix,
    local_operator_matrix_unit_expansion,
    local_rank_one_matrix_unit_expansion,
    local_reduced_density_matrix_from_state,
)

if TYPE_CHECKING:
    from qlinks.caging.classification import (
        CageClassificationReport,
        ReducedIZMonitorDecomposition,
    )
else:
    ReducedIZMonitorDecomposition = str


@dataclass(frozen=True, slots=True)
class LocalReducedDensityMatrixReadout:
    """Notebook-friendly readout for one local reduced density matrix.

    The readout keeps the full :class:`LocalReducedDensityMatrix` object and a
    truncated local matrix-unit expansion of its density matrix.  The optional
    component metadata is populated when the readout comes from a reduced-IZ
    frustration-free decomposition of a classification report.
    """

    variable_indices: tuple[int, ...]
    reduced_density_matrix: LocalReducedDensityMatrix
    n_matrix_unit_terms: int
    matrix_unit_terms: tuple[LocalMatrixUnitTerm, ...]
    matrix_unit_terms_truncated: bool
    component_index: int | None = None
    component_id: int | None = None
    decomposition: ReducedIZMonitorDecomposition | None = None
    zero_indices: tuple[int, ...] = ()

    @property
    def local_patterns(self) -> tuple[tuple[int, ...], ...]:
        return self.reduced_density_matrix.local_patterns

    @property
    def density_matrix(self) -> npt.NDArray[np.complex128]:
        return self.reduced_density_matrix.density_matrix

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        return self.reduced_density_matrix.eigenvalues

    @property
    def support_basis(self) -> npt.NDArray[np.complex128]:
        return self.reduced_density_matrix.support_basis

    @property
    def null_basis(self) -> npt.NDArray[np.complex128]:
        return self.reduced_density_matrix.null_basis

    @property
    def local_dim(self) -> int:
        return self.reduced_density_matrix.local_dim

    @property
    def support_rank(self) -> int:
        return self.reduced_density_matrix.support_rank

    @property
    def nullity(self) -> int:
        return self.reduced_density_matrix.nullity

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        return {
            "component_index": self.component_index,
            "component_id": self.component_id,
            "decomposition": self.decomposition,
            "zero_indices": self.zero_indices,
            "variable_indices": self.variable_indices,
            "n_local_patterns": len(self.local_patterns),
            "local_dim": self.local_dim,
            "support_rank": self.support_rank,
            "nullity": self.nullity,
            "eigenvalues": tuple(float(value) for value in self.eigenvalues),
            "n_matrix_unit_terms": self.n_matrix_unit_terms,
            "matrix_unit_terms_truncated": self.matrix_unit_terms_truncated,
            "matrix_unit_terms": tuple(
                {
                    "coefficient": term.coefficient,
                    "target_pattern": term.target_pattern,
                    "source_pattern": term.source_pattern,
                }
                for term in self.matrix_unit_terms
            ),
        }


def local_reduced_density_matrix_readout_from_state(
    *,
    basis_configs: npt.NDArray[np.integer],
    state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    tolerance: float = 1e-10,
    matrix_unit_tolerance: float | None = None,
    max_matrix_unit_terms: int | None = 64,
    component_index: int | None = None,
    component_id: int | None = None,
    decomposition: ReducedIZMonitorDecomposition | None = None,
    zero_indices: tuple[int, ...] | list[int] = (),
) -> LocalReducedDensityMatrixReadout:
    """Compute a local RDM and expose its matrix-unit expansion.

    This is a thin caging-facing wrapper around the local-RDM utilities used by
    the open-system local-recycling layer.  No global basis outside
    ``basis_configs`` is constructed.
    """
    variable_key = tuple(int(index) for index in variable_indices)
    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=variable_key,
        tolerance=tolerance,
    )
    matrix_terms = local_operator_matrix_unit_expansion(
        local_patterns=rdm.local_patterns,
        local_operator=rdm.density_matrix,
        tolerance=tolerance if matrix_unit_tolerance is None else matrix_unit_tolerance,
    )

    if max_matrix_unit_terms is None:
        shown_terms = matrix_terms
        truncated = False
    else:
        max_terms = int(max_matrix_unit_terms)
        shown_terms = matrix_terms[:max_terms]
        truncated = len(matrix_terms) > max_terms

    return LocalReducedDensityMatrixReadout(
        variable_indices=variable_key,
        reduced_density_matrix=rdm,
        n_matrix_unit_terms=len(matrix_terms),
        matrix_unit_terms=tuple(shown_terms),
        matrix_unit_terms_truncated=bool(truncated),
        component_index=component_index,
        component_id=component_id,
        decomposition=decomposition,
        zero_indices=tuple(int(index) for index in zero_indices),
    )


def reduced_iz_local_rdm_readouts_from_report(
    report: CageClassificationReport,
    *,
    basis_configs: npt.NDArray[np.integer],
    state: npt.ArrayLike,
    decomposition: ReducedIZMonitorDecomposition = "exact_support",
    tolerance: float = 1e-10,
    matrix_unit_tolerance: float | None = None,
    max_matrix_unit_terms: int | None = 64,
    include_empty_supports: bool = False,
) -> tuple[LocalReducedDensityMatrixReadout, ...]:
    """Return local-RDM readouts for reduced-IZ monitor components.

    The components are the same frustration-free reduced-IZ groups cached by
    :class:`CageClassificationReport` and used by the Lindblad-construction
    layer.  For each component support, this function computes the target state
    reduced density matrix and expands that RDM in local matrix units so that it
    can be inspected in notebooks.
    """
    readouts: list[LocalReducedDensityMatrixReadout] = []
    for component_index, component in enumerate(
        report.reduced_iz_component_groups(decomposition=decomposition)
    ):
        variable_indices = tuple(int(index) for index in component.support_variables)
        if not variable_indices and not include_empty_supports:
            continue

        readouts.append(
            local_reduced_density_matrix_readout_from_state(
                basis_configs=basis_configs,
                state=state,
                variable_indices=variable_indices,
                tolerance=tolerance,
                matrix_unit_tolerance=matrix_unit_tolerance,
                max_matrix_unit_terms=max_matrix_unit_terms,
                component_index=int(component_index),
                component_id=int(component.component_id),
                decomposition=decomposition,
                zero_indices=component.zero_indices,
            )
        )

    return tuple(readouts)
