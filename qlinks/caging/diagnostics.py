from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

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


@dataclass(frozen=True, slots=True)
class SupportPermutationDiagnostic:
    """Diagnostics for one support permutation as a possible CLS symmetry.

    The permutation maps old local support positions to new local support
    positions.  It is tested as a unitary ``U`` acting inside the support.  A
    boundary-fixed local symmetry satisfies ``K U = K`` for the leakage matrix
    ``K = H_{boundary,support}``.  If the state is also an eigenvector of ``U``
    with a nontrivial phase, then the permutation gives a symmetry-sector
    explanation for the destructive-interference closure.
    """

    permutation: tuple[int, ...]
    moved_support_count: int
    cycle_lengths: tuple[int, ...]
    boundary_fixed_residual: float
    boundary_fixed_relative_residual: float
    internal_commutator_residual: float | None
    internal_commutator_relative_residual: float | None
    state_phase: complex
    state_eigen_residual: float
    state_eigen_relative_residual: float
    is_nontrivial_state_sector: bool
    explains_boundary_closure: bool

    @property
    def is_identity(self) -> bool:
        return self.moved_support_count == 0

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        return {
            "permutation": self.permutation,
            "moved_support_count": self.moved_support_count,
            "cycle_lengths": self.cycle_lengths,
            "boundary_fixed_residual": self.boundary_fixed_residual,
            "boundary_fixed_relative_residual": self.boundary_fixed_relative_residual,
            "internal_commutator_residual": self.internal_commutator_residual,
            "internal_commutator_relative_residual": self.internal_commutator_relative_residual,
            "state_phase": self.state_phase,
            "state_eigen_residual": self.state_eigen_residual,
            "state_eigen_relative_residual": self.state_eigen_relative_residual,
            "is_nontrivial_state_sector": self.is_nontrivial_state_sector,
            "explains_boundary_closure": self.explains_boundary_closure,
        }


@dataclass(frozen=True, slots=True)
class FockSpaceAutomorphismDiagnostic:
    """Boundary-fixed Fock-space automorphism diagnostics for a cage state.

    The diagnostic searches automorphisms of the support-plus-boundary graph
    while fixing every boundary vertex individually.  This is intentionally the
    conservative CLS test: a nontrivial support irrep is certified only when the
    exterior/boundary channels are symmetry-trivial.
    """

    support_indices: tuple[int, ...]
    boundary_indices: tuple[int, ...]
    hilbert_size: int
    support_size: int
    boundary_size: int
    selected_graph_size: int
    state_boundary_residual: float
    state_boundary_relative_residual: float
    n_automorphisms_tested: int
    n_nontrivial_automorphisms: int
    n_explaining_automorphisms: int
    automorphism_search_truncated: bool
    skipped_reason: str | None
    permutation_diagnostics: tuple[SupportPermutationDiagnostic, ...]

    @property
    def was_skipped(self) -> bool:
        return self.skipped_reason is not None

    @property
    def has_nontrivial_automorphism_explanation(self) -> bool:
        return self.n_explaining_automorphisms > 0

    @property
    def best_explaining_permutation(self) -> SupportPermutationDiagnostic | None:
        explaining = [
            diagnostic
            for diagnostic in self.permutation_diagnostics
            if diagnostic.explains_boundary_closure
        ]
        if not explaining:
            return None

        return min(
            explaining,
            key=lambda item: (
                item.state_eigen_relative_residual,
                item.boundary_fixed_relative_residual,
                -item.moved_support_count,
            ),
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        best = self.best_explaining_permutation
        return {
            "support_size": self.support_size,
            "boundary_size": self.boundary_size,
            "selected_graph_size": self.selected_graph_size,
            "state_boundary_residual": self.state_boundary_residual,
            "state_boundary_relative_residual": self.state_boundary_relative_residual,
            "n_automorphisms_tested": self.n_automorphisms_tested,
            "n_nontrivial_automorphisms": self.n_nontrivial_automorphisms,
            "n_explaining_automorphisms": self.n_explaining_automorphisms,
            "automorphism_search_truncated": self.automorphism_search_truncated,
            "skipped_reason": self.skipped_reason,
            "best_explaining_permutation": None if best is None else best.to_summary_dict(),
        }


@dataclass(frozen=True, slots=True)
class OperatorStateActionDiagnostic:
    """Action of one candidate operator on the target state."""

    name: str
    action_norm: float
    relative_action_norm: float
    eigenvalue: complex
    eigen_residual: float
    relative_eigen_residual: float
    annihilates_state: bool
    is_eigen_operator: bool

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        return {
            "name": self.name,
            "action_norm": self.action_norm,
            "relative_action_norm": self.relative_action_norm,
            "eigenvalue": self.eigenvalue,
            "eigen_residual": self.eigen_residual,
            "relative_eigen_residual": self.relative_eigen_residual,
            "annihilates_state": self.annihilates_state,
            "is_eigen_operator": self.is_eigen_operator,
        }


@dataclass(frozen=True, slots=True)
class LocalAnnihilatorDiagnostic:
    """Linear-annihilator diagnostics in a supplied operator basis.

    If the operator basis is ``{O_a}``, this diagnostic computes the nullspace of
    the action matrix with columns ``O_a |psi>``.  Every null vector gives a
    linear combination ``sum_a c_a O_a`` that annihilates the state.  The same
    action data also reports which individual operators have the state as an
    eigenvector.
    """

    operator_names: tuple[str, ...]
    state_norm: float
    action_matrix_shape: tuple[int, int]
    action_matrix_rank: int
    annihilator_nullity: int
    singular_values: npt.NDArray[np.float64]
    annihilator_coefficients: npt.NDArray[np.complex128]
    annihilator_residuals: npt.NDArray[np.float64]
    operator_action_diagnostics: tuple[OperatorStateActionDiagnostic, ...]
    tolerance: float

    @property
    def has_annihilator(self) -> bool:
        return self.annihilator_nullity > 0

    @property
    def n_individual_annihilators(self) -> int:
        return sum(
            int(diagnostic.annihilates_state) for diagnostic in self.operator_action_diagnostics
        )

    @property
    def n_individual_eigen_operators(self) -> int:
        return sum(
            int(diagnostic.is_eigen_operator) for diagnostic in self.operator_action_diagnostics
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        return {
            "operator_names": self.operator_names,
            "state_norm": self.state_norm,
            "action_matrix_shape": self.action_matrix_shape,
            "action_matrix_rank": self.action_matrix_rank,
            "annihilator_nullity": self.annihilator_nullity,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "annihilator_residuals": tuple(float(value) for value in self.annihilator_residuals),
            "n_individual_annihilators": self.n_individual_annihilators,
            "n_individual_eigen_operators": self.n_individual_eigen_operators,
            "operator_action_diagnostics": tuple(
                diagnostic.to_summary_dict() for diagnostic in self.operator_action_diagnostics
            ),
        }


@dataclass(frozen=True, slots=True)
class CommutantAlgebraDiagnostic:
    """Commutant diagnostics in a supplied operator basis.

    Given candidate operators ``O_a`` and Hamiltonian/local terms ``h_j``, this
    solves for linear combinations ``X=sum_a c_a O_a`` satisfying
    ``[X, h_j]=0`` for all supplied terms.  This is a small-system diagnostic and
    is meant for reverse engineering candidate commutant generators after exact
    cage states have been found.
    """

    operator_names: tuple[str, ...]
    term_names: tuple[str, ...]
    commutator_matrix_shape: tuple[int, int]
    commutant_rank: int
    commutant_nullity: int
    singular_values: npt.NDArray[np.float64]
    commutant_coefficients: npt.NDArray[np.complex128]
    commutant_residuals: npt.NDArray[np.float64]
    individual_commutator_norms: npt.NDArray[np.float64]
    tolerance: float

    @property
    def has_commutant_generator(self) -> bool:
        return self.commutant_nullity > 0

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact dictionary useful in notebooks and logs."""
        return {
            "operator_names": self.operator_names,
            "term_names": self.term_names,
            "commutator_matrix_shape": self.commutator_matrix_shape,
            "commutant_rank": self.commutant_rank,
            "commutant_nullity": self.commutant_nullity,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "commutant_residuals": tuple(float(value) for value in self.commutant_residuals),
            "individual_commutator_norms": tuple(
                float(value) for value in self.individual_commutator_norms
            ),
        }


def fock_space_automorphism_diagnostic(
    *,
    kinetic_matrix,
    support_indices: Sequence[int],
    state: npt.ArrayLike,
    tolerance: float = 1.0e-10,
    weight_tolerance: float = 0.0,
    max_graph_vertices: int | None = 96,
    max_automorphisms: int = 128,
    include_identity: bool = False,
) -> FockSpaceAutomorphismDiagnostic:
    """Diagnose boundary-fixed Fock-space automorphisms of a cage support.

    Args:
        kinetic_matrix: Sparse or dense matrix defining Fock-space hopping.
        support_indices: Global basis indices carrying the compact state.
        state: Either the local state on ``support_indices`` or a full Hilbert
            vector in the same basis as ``kinetic_matrix``.
        tolerance: Numerical tolerance used for symmetry/eigenstate tests.
        weight_tolerance: Matrix entries with smaller absolute value are ignored
            when building the graph and boundary map.
        max_graph_vertices: Skip automorphism enumeration above this selected
            support-plus-boundary graph size.  Use ``None`` to disable.
        max_automorphisms: Maximum number of graph automorphisms to test.
        include_identity: Whether to keep the identity permutation in the
            returned per-permutation diagnostics.

    Returns:
        Boundary-fixed automorphism diagnostic.  ``skipped_reason`` is populated
        when the selected graph is intentionally too large for this lightweight
        diagnostic.
    """
    matrix = sp.csr_array(kinetic_matrix, dtype=np.complex128)
    hilbert_size = int(matrix.shape[0])
    support = _as_index_tuple(support_indices, upper_bound=hilbert_size, name="support_indices")
    local_state = _state_on_support(state=state, support=support, hilbert_size=hilbert_size)

    boundary = _boundary_indices_from_support(
        matrix=matrix,
        support=support,
        tolerance=weight_tolerance,
    )
    support_array = np.asarray(support, dtype=np.int64)
    boundary_array = np.asarray(boundary, dtype=np.int64)
    leakage = matrix[boundary_array, :][:, support_array]
    leakage = _threshold_sparse_matrix(leakage, tolerance=weight_tolerance)
    state_boundary_action = leakage @ local_state
    state_norm = float(np.linalg.norm(local_state))
    state_boundary_residual = float(np.linalg.norm(state_boundary_action))
    state_boundary_relative_residual = state_boundary_residual / max(state_norm, tolerance)

    selected_graph_size = len(support) + len(boundary)
    skipped_reason = None
    if max_graph_vertices is not None and selected_graph_size > int(max_graph_vertices):
        skipped_reason = (
            f"selected support-plus-boundary graph has {selected_graph_size} vertices, "
            f"which exceeds max_graph_vertices={max_graph_vertices}"
        )
        return FockSpaceAutomorphismDiagnostic(
            support_indices=support,
            boundary_indices=boundary,
            hilbert_size=hilbert_size,
            support_size=len(support),
            boundary_size=len(boundary),
            selected_graph_size=selected_graph_size,
            state_boundary_residual=state_boundary_residual,
            state_boundary_relative_residual=state_boundary_relative_residual,
            n_automorphisms_tested=0,
            n_nontrivial_automorphisms=0,
            n_explaining_automorphisms=0,
            automorphism_search_truncated=False,
            skipped_reason=skipped_reason,
            permutation_diagnostics=(),
        )

    support_permutations, truncated = _boundary_fixed_support_automorphisms(
        matrix=matrix,
        support=support,
        boundary=boundary,
        weight_tolerance=weight_tolerance,
        max_automorphisms=max_automorphisms,
        include_identity=include_identity,
    )

    internal = matrix[support_array, :][:, support_array]
    internal = _threshold_sparse_matrix(internal, tolerance=weight_tolerance)
    leakage_norm = _sparse_frobenius_norm(leakage)
    internal_norm = _sparse_frobenius_norm(internal)

    diagnostics = tuple(
        _diagnose_support_permutation(
            permutation=permutation,
            leakage=leakage,
            leakage_norm=leakage_norm,
            internal=internal,
            internal_norm=internal_norm,
            state=local_state,
            tolerance=tolerance,
        )
        for permutation in support_permutations
    )

    return FockSpaceAutomorphismDiagnostic(
        support_indices=support,
        boundary_indices=boundary,
        hilbert_size=hilbert_size,
        support_size=len(support),
        boundary_size=len(boundary),
        selected_graph_size=selected_graph_size,
        state_boundary_residual=state_boundary_residual,
        state_boundary_relative_residual=state_boundary_relative_residual,
        n_automorphisms_tested=len(diagnostics),
        n_nontrivial_automorphisms=sum(
            int(not diagnostic.is_identity) for diagnostic in diagnostics
        ),
        n_explaining_automorphisms=sum(
            int(diagnostic.explains_boundary_closure) for diagnostic in diagnostics
        ),
        automorphism_search_truncated=truncated,
        skipped_reason=skipped_reason,
        permutation_diagnostics=diagnostics,
    )


def fock_space_automorphism_diagnostic_for_cage_state(
    *,
    kinetic_matrix,
    cage_state,
    tolerance: float = 1.0e-10,
    weight_tolerance: float = 0.0,
    max_graph_vertices: int | None = 96,
    max_automorphisms: int = 128,
    include_identity: bool = False,
) -> FockSpaceAutomorphismDiagnostic:
    """Run :func:`fock_space_automorphism_diagnostic` on a ``CageState``."""
    return fock_space_automorphism_diagnostic(
        kinetic_matrix=kinetic_matrix,
        support_indices=cage_state.support,
        state=cage_state.local_state,
        tolerance=tolerance,
        weight_tolerance=weight_tolerance,
        max_graph_vertices=max_graph_vertices,
        max_automorphisms=max_automorphisms,
        include_identity=include_identity,
    )


def local_annihilator_diagnostic_from_operators(
    *,
    operators: Sequence[object],
    state: npt.ArrayLike,
    operator_names: Sequence[str] | None = None,
    tolerance: float = 1.0e-10,
    max_annihilator_vectors: int | None = 16,
) -> LocalAnnihilatorDiagnostic:
    """Find linear combinations of supplied operators that annihilate a state."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    state_norm = float(np.linalg.norm(vector))
    if state_norm <= tolerance:
        raise ValueError("state must have nonzero norm.")

    names = _default_names(operator_names, prefix="O", count=len(operators))
    action_columns = []
    action_reports = []
    for name, operator in zip(names, operators, strict=True):
        action = _matrix_vector_product(operator, vector)
        action_columns.append(action)
        action_norm = float(np.linalg.norm(action))
        eigenvalue = complex(np.vdot(vector, action) / np.vdot(vector, vector))
        eigen_residual = float(np.linalg.norm(action - eigenvalue * vector))
        action_reports.append(
            OperatorStateActionDiagnostic(
                name=name,
                action_norm=action_norm,
                relative_action_norm=action_norm / state_norm,
                eigenvalue=eigenvalue,
                eigen_residual=eigen_residual,
                relative_eigen_residual=eigen_residual / state_norm,
                annihilates_state=action_norm / state_norm <= tolerance,
                is_eigen_operator=eigen_residual / state_norm <= tolerance,
            )
        )

    action_matrix = (
        np.column_stack(action_columns) if action_columns else np.empty((vector.size, 0))
    )
    rank, singular_values, null_vectors = _right_nullspace_from_dense_matrix(
        action_matrix,
        tolerance=tolerance,
    )
    if max_annihilator_vectors is not None:
        null_vectors = null_vectors[:, : int(max_annihilator_vectors)]

    residuals = np.asarray(
        [
            float(np.linalg.norm(action_matrix @ null_vectors[:, i]))
            for i in range(null_vectors.shape[1])
        ],
        dtype=np.float64,
    )

    return LocalAnnihilatorDiagnostic(
        operator_names=names,
        state_norm=state_norm,
        action_matrix_shape=(int(action_matrix.shape[0]), int(action_matrix.shape[1])),
        action_matrix_rank=rank,
        annihilator_nullity=int(null_vectors.shape[1]),
        singular_values=singular_values,
        annihilator_coefficients=np.asarray(null_vectors, dtype=np.complex128),
        annihilator_residuals=residuals,
        operator_action_diagnostics=tuple(action_reports),
        tolerance=tolerance,
    )


def commutant_algebra_diagnostic_from_operator_basis(
    *,
    operators: Sequence[object],
    hamiltonian_terms: Sequence[object],
    operator_names: Sequence[str] | None = None,
    term_names: Sequence[str] | None = None,
    tolerance: float = 1.0e-10,
    max_commutant_vectors: int | None = 16,
) -> CommutantAlgebraDiagnostic:
    """Find operator-basis combinations commuting with all supplied terms."""
    names = _default_names(operator_names, prefix="O", count=len(operators))
    terms = _default_names(term_names, prefix="h", count=len(hamiltonian_terms))
    operator_matrices = [_as_dense_matrix(operator) for operator in operators]
    term_matrices = [_as_dense_matrix(term) for term in hamiltonian_terms]

    if operator_matrices and term_matrices:
        shape = operator_matrices[0].shape
        for matrix in operator_matrices + term_matrices:
            if matrix.shape != shape:
                raise ValueError("operators and hamiltonian_terms must have matching shapes.")
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("operators and hamiltonian_terms must be square matrices.")

    commutator_columns = []
    individual_norms = []
    for operator in operator_matrices:
        pieces = []
        for term in term_matrices:
            commutator = operator @ term - term @ operator
            pieces.append(commutator.reshape(-1))
        stacked = np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.complex128)
        commutator_columns.append(stacked)
        individual_norms.append(float(np.linalg.norm(stacked)))

    commutator_matrix = (
        np.column_stack(commutator_columns)
        if commutator_columns
        else np.empty((0, 0), dtype=np.complex128)
    )
    rank, singular_values, null_vectors = _right_nullspace_from_dense_matrix(
        commutator_matrix,
        tolerance=tolerance,
    )
    if max_commutant_vectors is not None:
        null_vectors = null_vectors[:, : int(max_commutant_vectors)]

    residuals = np.asarray(
        [
            float(np.linalg.norm(commutator_matrix @ null_vectors[:, i]))
            for i in range(null_vectors.shape[1])
        ],
        dtype=np.float64,
    )

    return CommutantAlgebraDiagnostic(
        operator_names=names,
        term_names=terms,
        commutator_matrix_shape=(
            int(commutator_matrix.shape[0]),
            int(commutator_matrix.shape[1]),
        ),
        commutant_rank=rank,
        commutant_nullity=int(null_vectors.shape[1]),
        singular_values=singular_values,
        commutant_coefficients=np.asarray(null_vectors, dtype=np.complex128),
        commutant_residuals=residuals,
        individual_commutator_norms=np.asarray(individual_norms, dtype=np.float64),
        tolerance=tolerance,
    )


def _as_index_tuple(
    indices: Sequence[int],
    *,
    upper_bound: int,
    name: str,
) -> tuple[int, ...]:
    result = tuple(int(index) for index in indices)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate indices.")
    if any(index < 0 or index >= upper_bound for index in result):
        raise ValueError(f"{name} contains indices outside [0, {upper_bound}).")
    return result


def _state_on_support(
    *,
    state: npt.ArrayLike,
    support: tuple[int, ...],
    hilbert_size: int,
) -> npt.NDArray[np.complex128]:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size == len(support):
        return vector
    if vector.size == hilbert_size:
        return vector[np.asarray(support, dtype=np.int64)]
    raise ValueError("state must either be a local support vector or a full Hilbert-space vector.")


def _boundary_indices_from_support(
    *,
    matrix: sp.csr_array,
    support: tuple[int, ...],
    tolerance: float,
) -> tuple[int, ...]:
    support_array = np.asarray(support, dtype=np.int64)
    support_mask = np.zeros(matrix.shape[0], dtype=bool)
    support_mask[support_array] = True
    block = matrix[:, support_array].tocsr()
    block = _threshold_sparse_matrix(block, tolerance=tolerance)
    row_norms = _sparse_row_norms(block)
    boundary = np.flatnonzero((row_norms > tolerance) & ~support_mask).astype(np.int64)
    return tuple(int(index) for index in boundary)


def _boundary_fixed_support_automorphisms(
    *,
    matrix: sp.csr_array,
    support: tuple[int, ...],
    boundary: tuple[int, ...],
    weight_tolerance: float,
    max_automorphisms: int,
    include_identity: bool,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    try:
        import networkx as nx
        from networkx.algorithms import isomorphism as nx_iso
    except ImportError as error:
        raise ImportError("Fock-space automorphism diagnostics require networkx.") from error

    selected = tuple(support) + tuple(boundary)
    selected_array = np.asarray(selected, dtype=np.int64)
    n_support = len(support)
    selected_block = matrix[selected_array, :][:, selected_array].tocoo()
    graph = nx.DiGraph()
    for node in range(len(selected)):
        if node < n_support:
            graph.add_node(node, color=("support", 0))
        else:
            graph.add_node(node, color=("boundary", node - n_support))

    for row, col, value in zip(
        selected_block.row,
        selected_block.col,
        selected_block.data,
        strict=True,
    ):
        if int(row) == int(col) or abs(value) <= weight_tolerance:
            continue
        graph.add_edge(
            int(row),
            int(col),
            weight=_complex_label(complex(value), tolerance=max(weight_tolerance, 1.0e-12)),
        )

    matcher = nx_iso.DiGraphMatcher(
        graph,
        graph,
        node_match=nx_iso.categorical_node_match("color", None),
        edge_match=nx_iso.categorical_edge_match("weight", None),
    )

    identity = tuple(range(n_support))
    seen: set[tuple[int, ...]] = set()
    permutations: list[tuple[int, ...]] = []
    truncated = False
    for mapping in matcher.isomorphisms_iter():
        permutation = tuple(int(mapping[index]) for index in range(n_support))
        if permutation in seen:
            continue
        seen.add(permutation)
        if permutation == identity and not include_identity:
            continue
        permutations.append(permutation)
        if len(permutations) >= max_automorphisms:
            truncated = True
            break

    return tuple(permutations), truncated


def _diagnose_support_permutation(
    *,
    permutation: tuple[int, ...],
    leakage: sp.csr_array,
    leakage_norm: float,
    internal: sp.csr_array,
    internal_norm: float,
    state: npt.NDArray[np.complex128],
    tolerance: float,
) -> SupportPermutationDiagnostic:
    perm_matrix = _permutation_matrix(permutation)
    boundary_residual = _sparse_frobenius_norm(leakage @ perm_matrix - leakage)
    boundary_relative = boundary_residual / max(leakage_norm, tolerance)

    commutator = internal @ perm_matrix - perm_matrix @ internal
    internal_residual = _sparse_frobenius_norm(commutator)
    internal_relative = internal_residual / max(internal_norm, tolerance)

    permuted_state = _apply_permutation_to_vector(state, permutation)
    state_norm = float(np.linalg.norm(state))
    if state_norm <= tolerance:
        phase = 0.0 + 0.0j
        state_eigen_residual = float(np.linalg.norm(permuted_state))
    else:
        phase = complex(np.vdot(state, permuted_state) / np.vdot(state, state))
        state_eigen_residual = float(np.linalg.norm(permuted_state - phase * state))
    state_relative = state_eigen_residual / max(state_norm, tolerance)
    nontrivial = state_relative <= tolerance and abs(phase - 1.0) > tolerance

    return SupportPermutationDiagnostic(
        permutation=permutation,
        moved_support_count=sum(int(index != image) for index, image in enumerate(permutation)),
        cycle_lengths=_permutation_cycle_lengths(permutation),
        boundary_fixed_residual=boundary_residual,
        boundary_fixed_relative_residual=boundary_relative,
        internal_commutator_residual=internal_residual,
        internal_commutator_relative_residual=internal_relative,
        state_phase=phase,
        state_eigen_residual=state_eigen_residual,
        state_eigen_relative_residual=state_relative,
        is_nontrivial_state_sector=nontrivial,
        explains_boundary_closure=boundary_relative <= tolerance and nontrivial,
    )


def _permutation_matrix(permutation: tuple[int, ...]) -> sp.csr_array:
    n_items = len(permutation)
    rows = np.asarray(permutation, dtype=np.int64)
    cols = np.arange(n_items, dtype=np.int64)
    data = np.ones(n_items, dtype=np.complex128)
    return sp.csr_array((data, (rows, cols)), shape=(n_items, n_items))


def _apply_permutation_to_vector(
    vector: npt.NDArray[np.complex128],
    permutation: tuple[int, ...],
) -> npt.NDArray[np.complex128]:
    result = np.empty_like(vector)
    result[np.asarray(permutation, dtype=np.int64)] = vector
    return result


def _permutation_cycle_lengths(permutation: tuple[int, ...]) -> tuple[int, ...]:
    seen = [False] * len(permutation)
    lengths = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        node = start
        length = 0
        while not seen[node]:
            seen[node] = True
            length += 1
            node = permutation[node]
        if length > 1:
            lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _threshold_sparse_matrix(matrix, *, tolerance: float) -> sp.csr_array:
    result = sp.csr_array(matrix, dtype=np.complex128).copy()
    if tolerance > 0.0:
        result.data[np.abs(result.data) <= tolerance] = 0.0
        result.eliminate_zeros()
    return result


def _sparse_row_norms(matrix: sp.csr_array) -> npt.NDArray[np.float64]:
    squared = matrix.copy()
    squared.data = np.abs(squared.data) ** 2
    return np.sqrt(np.asarray(squared.sum(axis=1)).reshape(-1)).astype(np.float64)


def _sparse_frobenius_norm(matrix) -> float:
    sparse = sp.csr_array(matrix, dtype=np.complex128)
    return float(np.sqrt(np.sum(np.abs(sparse.data) ** 2)))


def _complex_label(value: complex, *, tolerance: float) -> tuple[int, int]:
    scale = max(tolerance, 1.0e-15)
    return (int(round(float(np.real(value)) / scale)), int(round(float(np.imag(value)) / scale)))


def _matrix_vector_product(
    operator,
    vector: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    if sp.issparse(operator):
        return np.asarray(operator @ vector, dtype=np.complex128).reshape(-1)
    matrix = np.asarray(operator, dtype=np.complex128)
    return np.asarray(matrix @ vector, dtype=np.complex128).reshape(-1)


def _as_dense_matrix(operator) -> npt.NDArray[np.complex128]:
    if sp.issparse(operator):
        return np.asarray(operator.toarray(), dtype=np.complex128)
    return np.asarray(operator, dtype=np.complex128)


def _right_nullspace_from_dense_matrix(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_columns = int(matrix.shape[1])
    if n_columns == 0:
        return 0, np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.complex128)

    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tolerance))
    null_vectors = vh[rank:].conj().T
    return (
        rank,
        np.asarray(singular_values, dtype=np.float64),
        np.asarray(
            null_vectors,
            dtype=np.complex128,
        ),
    )


def _default_names(
    names: Sequence[str] | None,
    *,
    prefix: str,
    count: int,
) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}{index}" for index in range(count))
    result = tuple(str(name) for name in names)
    if len(result) != count:
        raise ValueError("names must have the same length as the supplied objects.")
    return result
