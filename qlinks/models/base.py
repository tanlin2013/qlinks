from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.basis import (
    Basis,
    BruteForceBasisSolver,
    CPSATBasisSolver,
    DFSBasisSolver,
    full_basis_from_layout,
)
from qlinks.builders import (
    OptimizedSparseHamiltonianBuilder,
    SparseHamiltonianBuilder,
)
from qlinks.constraints import Constraint, SectorCondition
from qlinks.encoded import BinaryEncodedBasis, BitmaskSparseHamiltonianBuilder
from qlinks.models.local_terms import (
    LocalOperatorKind,
    LocalTermDescriptor,
    LocalTermKind,
)
from qlinks.operators import BasisOperator
from qlinks.variables import VariableLayout

BasisSolverName = Literal["brute_force", "dfs", "cpsat"]
HamiltonianBuilderName = Literal["sparse", "optimized", "bitmask"]
TermKind = Literal["kinetic", "potential", "other"]


def normalize_sector_label_for_display(label: Any) -> Any:
    """Normalize sector labels exposed by model-level APIs.

    Fractions with denominator 1 are converted to ints recursively.
    Other Fractions are kept exact.
    """
    if isinstance(label, Fraction):
        if label.denominator == 1:
            return int(label.numerator)
        return label

    if isinstance(label, tuple):
        return tuple(normalize_sector_label_for_display(value) for value in label)

    if isinstance(label, list):
        return [normalize_sector_label_for_display(value) for value in label]

    if isinstance(label, Mapping):
        return {key: normalize_sector_label_for_display(value) for key, value in label.items()}

    return label


def normalize_sector_labels_for_display(labels: Any) -> Any:
    """Normalize a collection of model-facing sector labels."""
    if isinstance(labels, tuple):
        return tuple(normalize_sector_label_for_display(label) for label in labels)

    if isinstance(labels, list):
        return [normalize_sector_label_for_display(label) for label in labels]

    return normalize_sector_label_for_display(labels)


@dataclass(frozen=True, slots=True)
class HamiltonianTermSpec:
    """Symbolic Hamiltonian term before sparse matrix construction.

    A model returns these specs from ``make_terms()``.  The shared builder then
    chooses the requested sparse backend and turns each operator tuple into a
    matrix.

    Attributes:
        name: Stable term name, such as ``"kinetic"`` or ``"potential"``.
        operators: Local operator objects that contribute to this term.
        kind: Coarse term category used by diagnostics and local-term assembly.

    Examples:
        >>> HamiltonianTermSpec(
        ...     name="kinetic",
        ...     kind="kinetic",
        ...     operators=(op0, op1),
        ... )
    """

    name: str
    operators: tuple[object, ...]
    kind: TermKind = "other"

    @classmethod
    def from_operators(
        cls,
        name: str,
        operators: Sequence[object],
        *,
        kind: TermKind = "other",
    ) -> HamiltonianTermSpec:
        return cls(
            name=name,
            operators=tuple(operators),
            kind=kind,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.operators) == 0


@dataclass(frozen=True, slots=True)
class BuiltHamiltonianTerm:
    """Sparse matrix and metadata for one built Hamiltonian term.

    Attributes:
        name: Term name copied from :class:`HamiltonianTermSpec`.
        kind: Coarse term category.
        operators: Operators used to build the term.
        matrix: Built sparse matrix, or ``None`` when the term has no operators.
    """

    name: str
    kind: TermKind
    operators: tuple[object, ...]
    matrix: Any | None


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """Basis, terms, and total Hamiltonian returned by ``model.build()``.

    Use this result when downstream code needs both the total Hamiltonian and
    named pieces such as the kinetic or potential terms.  Keeping them together
    avoids rebuilding the basis and matrices repeatedly.

    Attributes:
        model: Model instance that produced the result.
        lattice: Cached lattice used by the model.
        layout: Physical variable layout.
        constraints: Constraints used for basis construction.
        sectors: Sector filters used for basis construction.
        basis: Basis object used by the selected builder.  For ``builder=
            "bitmask"`` this can be a :class:`BinaryEncodedBasis`.
        terms: Mapping from term name to built term metadata.
        hamiltonian: Sparse total Hamiltonian, equal to the sum of all built
            nonempty term matrices.

    Examples:
        >>> result = model.build(builder="optimized")
        >>> hamiltonian = result.hamiltonian
        >>> kinetic = result.kinetic
    """

    model: object
    lattice: object
    layout: VariableLayout
    constraints: tuple[Constraint, ...]
    sectors: tuple[SectorCondition, ...]
    basis: Basis | BinaryEncodedBasis
    terms: dict[str, BuiltHamiltonianTerm]
    hamiltonian: Any

    @property
    def kinetic(self) -> Any | None:
        term = self.terms.get("kinetic")
        return None if term is None else term.matrix

    @property
    def potential(self) -> Any | None:
        term = self.terms.get("potential")
        return None if term is None else term.matrix

    @property
    def kinetic_operators(self) -> tuple[object, ...]:
        term = self.terms.get("kinetic")
        return () if term is None else term.operators

    @property
    def potential_operators(self) -> tuple[object, ...]:
        term = self.terms.get("potential")
        return () if term is None else term.operators

    @property
    def operators(self) -> tuple[object, ...]:
        operators: list[object] = []
        for term in self.terms.values():
            operators.extend(term.operators)
        return tuple(operators)

    def basis_operator(self, name: str) -> BasisOperator:
        term = self.terms[name]
        return BasisOperator(
            basis=self.basis,
            operators=term.operators,
        )


@dataclass(frozen=True, slots=True)
class SparseBuildOptions:
    """Shared sparse-build options.

    Attributes:
        backend: Sparse backend name or backend object.
        dtype: Matrix dtype passed to the builder.
        on_missing: Policy for operator actions that leave the constrained
            basis.  Use ``"raise"`` for debugging and ``"skip"`` for boundary
            terms that intentionally leak outside a subspace.
        drop_zero_atol: Absolute threshold below which coefficients are dropped.
    """

    backend: SparseBackendName | SparseBackend = "scipy"
    dtype: npt.DTypeLike = np.complex128
    on_missing: Literal["skip", "raise"] = "raise"
    drop_zero_atol: float = 0.0


def solve_basis(
    layout: VariableLayout,
    constraints: Sequence[Constraint] = (),
    sectors: Sequence[SectorCondition] = (),
    *,
    solver: BasisSolverName = "dfs",
    sort: bool = True,
    max_states: int | None = None,
) -> Basis:
    """Build an array basis using the requested solver.

    When no constraints or sectors are supplied, this function uses direct
    Cartesian-product enumeration instead of invoking DFS, brute force, or
    CP-SAT.

    Args:
        layout: Variable layout defining the local state space.
        constraints: Local/global constraints that every basis state must obey.
        sectors: Diagonal sector filters.
        solver: Solver name: ``"dfs"``, ``"brute_force"``, or ``"cpsat"``.
        sort: Whether to lexicographically sort the final basis.
        max_states: Optional early-stop limit for first-solution or existence
            checks.

    Returns:
        Basis containing the satisfying configurations.

    Raises:
        ValueError: If ``max_states`` is negative or ``solver`` is unknown.
    """
    if max_states is not None and max_states < 0:
        raise ValueError("max_states must be non-negative or None.")

    if len(constraints) == 0 and len(sectors) == 0:
        return full_basis_from_layout(
            layout,
            sort=sort,
            max_states=max_states,
        )

    if solver == "brute_force":
        return BruteForceBasisSolver(sort=sort).solve(
            layout,
            constraints=constraints,
            sectors=sectors,
            max_states=max_states,
        )

    if solver == "dfs":
        return DFSBasisSolver(sort=sort).solve(
            layout,
            constraints=constraints,
            sectors=sectors,
            max_states=max_states,
        )

    if solver == "cpsat":
        return CPSATBasisSolver(sort=sort).solve(
            layout,
            constraints=constraints,
            sectors=sectors,
            max_states=max_states,
        )

    raise ValueError("solver must be one of 'brute_force', 'dfs', or 'cpsat'.")


def validate_builder_name(builder: HamiltonianBuilderName) -> None:
    """Validate a Hamiltonian builder name.

    Args:
        builder: Candidate builder name.

    Raises:
        ValueError: If ``builder`` is not one of the supported names.
    """
    if builder not in ("sparse", "optimized", "bitmask"):
        raise ValueError("builder must be one of 'sparse', 'optimized', or 'bitmask'.")


def combine_hamiltonian_terms(matrices: Sequence[Any | None]) -> Any:
    """Sum all nonempty sparse matrices.

    Args:
        matrices: Term matrices.  ``None`` entries are ignored.

    Returns:
        Sum of all non-``None`` matrices, preserving the backend matrix type.

    Raises:
        ValueError: If every entry is ``None``.
    """

    nonempty = [matrix for matrix in matrices if matrix is not None]

    if len(nonempty) == 0:
        raise ValueError("Cannot combine zero Hamiltonian terms.")

    total = nonempty[0]
    for matrix in nonempty[1:]:
        total = total + matrix

    return total


class HamiltonianModelBase:
    """Base class for model-level basis and Hamiltonian construction.

    Subclasses implement the geometry-specific pieces: lattice construction,
    variable layout, constraints, optional sectors, and symbolic Hamiltonian
    terms.  The base class provides cached ``lattice``/``layout`` properties and
    shared ``build_basis()``, ``build()``, and ``build_hamiltonian()`` methods.

    Notes:
        Subclasses should usually use ``@dataclass(frozen=True)`` without
        ``slots=True``.  The cached properties require an instance ``__dict__``.
    """

    @cached_property
    def lattice(self) -> object:
        return self._make_lattice()

    @cached_property
    def layout(self) -> VariableLayout:
        return self._make_layout()

    @cached_property
    def model_builder(self) -> GenericModelBuilder:
        return GenericModelBuilder(self)

    def allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        """Return allowed user-facing labels for symmetry sectors.

        Models without user-selectable sectors return an empty dictionary.
        Geometry-specific subclasses override this to expose labels such as
        winding numbers.

        Returns:
            Mapping from sector name to allowed user-facing values.

        Examples:
            ``SquareQLMModel(...).allowed_sector_labels()`` may return
            ``{"winding_x": (...), "winding_y": (...)}``.
        """
        return {}

    def _allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        raise NotImplementedError

    def _nonempty_sector_labels(
        self, *, solver: BasisSolverName = "dfs"
    ) -> dict[str, tuple[object, ...]]:
        raise NotImplementedError

    def _make_lattice(self) -> object:
        raise NotImplementedError

    def _make_layout(self) -> VariableLayout:
        raise NotImplementedError

    def make_lattice(self) -> object:
        """Return the cached lattice.

        This is a backward-compatible alias for the ``lattice`` property.
        """
        return self.lattice

    def make_layout(self) -> VariableLayout:
        """Return the cached variable layout.

        This is a backward-compatible alias for the ``layout`` property.
        """
        return self.layout

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ) -> Sequence[Constraint]:
        return ()

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ) -> Sequence[SectorCondition]:
        return ()

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> Sequence[HamiltonianTermSpec]:
        raise NotImplementedError

    def prepare_builder_basis(
        self,
        *,
        physical_layout: VariableLayout,
        array_basis: Basis,
        input_basis: Basis | BinaryEncodedBasis | None,
        builder: HamiltonianBuilderName,
        sort_basis: bool,
    ) -> tuple[VariableLayout, Basis | BinaryEncodedBasis]:
        """Convert the physical basis to the representation required by a builder.

        The default sparse and optimized builders use the array
        :class:`Basis`.  The bitmask builder uses :class:`BinaryEncodedBasis`.
        QLM flux models override this method because their physical variables
        are ``{-1, +1}``, while the bitmask encoding is binary.

        Args:
            physical_layout: Original model layout.
            array_basis: Basis in physical variable values.
            input_basis: User-supplied basis, if any.
            builder: Requested builder name.
            sort_basis: Whether the caller requested sorted basis generation.

        Returns:
            Pair ``(operator_layout, build_basis)`` used for operator assembly.
        """

        validate_builder_name(builder)

        if builder == "bitmask":
            if input_basis is None:
                encoded_basis = BinaryEncodedBasis.from_basis(array_basis, sort=False)
            elif isinstance(input_basis, BinaryEncodedBasis):
                encoded_basis = input_basis
            elif isinstance(input_basis, Basis):
                encoded_basis = BinaryEncodedBasis.from_basis(input_basis, sort=False)
            else:
                raise TypeError("basis must be Basis, BinaryEncodedBasis, or None.")

            return encoded_basis.layout, encoded_basis

        return physical_layout, array_basis

    def has_basis_state(
        self,
        *,
        solver: BasisSolverName = "dfs",
    ) -> bool:
        basis = self.build_basis(
            solver=solver,
            sort=False,
            max_states=1,
        )
        return basis.n_states > 0

    def build_basis(
        self,
        *,
        solver: BasisSolverName = "dfs",
        sort: bool = True,
        max_states: int | None = None,
    ) -> Basis:
        return self.model_builder.build_basis(
            solver=solver,
            sort=sort,
            max_states=max_states,
        )

    def build(
        self,
        *,
        basis: Basis | BinaryEncodedBasis | None = None,
        basis_solver: BasisSolverName = "dfs",
        builder: HamiltonianBuilderName = "sparse",
        backend: SparseBackendName | SparseBackend = "scipy",
        dtype: npt.DTypeLike = np.complex128,
        sort_basis: bool = True,
        on_missing: Literal["skip", "raise"] = "raise",
        drop_zero_atol: float = 0.0,
    ) -> ModelBuildResult:
        """Build the constrained basis, named terms, and total Hamiltonian.

        Args:
            basis: Optional precomputed basis.  Supplying one avoids basis
                enumeration and fixes the row/column order.
            basis_solver: Solver used when ``basis`` is not supplied.
            builder: Matrix builder: ``"sparse"``, ``"optimized"``, or
                ``"bitmask"``.
            backend: Sparse backend name or backend object.
            dtype: Matrix dtype.
            sort_basis: Whether to sort an internally generated basis.
            on_missing: Policy for operator actions that leave the basis.
            drop_zero_atol: Absolute threshold for dropping small coefficients.

        Returns:
            Full model build result containing the basis, named term matrices,
            and total Hamiltonian.
        """

        return self.model_builder.build(
            basis=basis,
            basis_solver=basis_solver,
            builder=builder,
            backend=backend,
            dtype=dtype,
            sort_basis=sort_basis,
            on_missing=on_missing,
            drop_zero_atol=drop_zero_atol,
        )

    def build_hamiltonian(
        self,
        *,
        basis: Basis | BinaryEncodedBasis | None = None,
        basis_solver: BasisSolverName = "dfs",
        builder: HamiltonianBuilderName = "sparse",
        backend: SparseBackendName | SparseBackend = "scipy",
        dtype: npt.DTypeLike = np.complex128,
        sort_basis: bool = True,
        on_missing: Literal["skip", "raise"] = "raise",
        drop_zero_atol: float = 0.0,
    ) -> Any:
        """Build and return only the total Hamiltonian matrix.

        Args:
            basis: Optional precomputed basis.
            basis_solver: Solver used when ``basis`` is not supplied.
            builder: Matrix builder name.
            backend: Sparse backend name or backend object.
            dtype: Matrix dtype.
            sort_basis: Whether to sort an internally generated basis.
            on_missing: Policy for operator actions that leave the basis.
            drop_zero_atol: Absolute threshold for dropping small coefficients.

        Returns:
            Sparse total Hamiltonian matrix.

        Notes:
            Prefer :meth:`build` when you also need the basis, kinetic term, or
            potential term.
        """

        return self.model_builder.build_hamiltonian(
            basis=basis,
            basis_solver=basis_solver,
            builder=builder,
            backend=backend,
            dtype=dtype,
            sort_basis=sort_basis,
            on_missing=on_missing,
            drop_zero_atol=drop_zero_atol,
        )

    def local_term_descriptors(
        self,
        *,
        operator_kind: LocalOperatorKind | None = None,
        term_kind: LocalTermKind | None = None,
    ) -> tuple[LocalTermDescriptor, ...]:
        """Return matrix-free descriptors for local Hamiltonian pieces.

        Args:
            operator_kind: Optional filter such as kinetic or potential.
            term_kind: Optional physical/local-term category filter.

        Returns:
            Tuple of local term descriptors.  The base implementation returns
            an empty tuple because not every model exposes local terms.
        """
        return ()

    def make_local_term(
        self,
        descriptor: LocalTermDescriptor,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> HamiltonianTermSpec:
        """Return the symbolic operator spec for one local term.

        Args:
            descriptor: Descriptor previously returned by
                :meth:`local_term_descriptors`.
            layout: Layout used for operator construction.
            builder: Builder name, allowing subclasses to return optimized or
                bitmask-specific operator implementations.

        Returns:
            Symbolic Hamiltonian term containing the local operators.

        Raises:
            NotImplementedError: If the model does not support local terms.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support local term assembly yet."
        )

    def build_local_term(
        self,
        descriptor: LocalTermDescriptor,
        build_result: ModelBuildResult,
        *,
        builder: HamiltonianBuilderName = "sparse",
        backend: SparseBackendName | SparseBackend = "scipy",
        dtype: npt.DTypeLike = np.complex128,
        on_missing: Literal["skip", "raise"] = "raise",
        drop_zero_atol: float = 0.0,
    ) -> Any:
        """Build one local-term matrix in an existing model basis.

        Args:
            descriptor: Local term descriptor to build.
            build_result: Existing model build result whose basis fixes the
                matrix row/column order.
            builder: Matrix builder name.
            backend: Sparse backend name or backend object.
            dtype: Matrix dtype.
            on_missing: Policy for operator actions outside the basis.
            drop_zero_atol: Absolute threshold for dropping small coefficients.

        Returns:
            Sparse matrix for the requested local term.
        """
        validate_builder_name(builder)

        term = self.make_local_term(
            descriptor,
            build_result.layout,
            builder=builder,
        )

        return self.model_builder._build_term_matrix(
            basis=build_result.basis,
            term=term,
            builder=builder,
            backend=backend,
            dtype=dtype,
            on_missing=on_missing,
            drop_zero_atol=drop_zero_atol,
        )


@dataclass(frozen=True, slots=True)
class GenericModelBuilder:
    """Shared implementation behind :class:`HamiltonianModelBase.build`.

    The builder owns the repeated workflow of collecting constraints/sectors,
    solving or converting the basis, asking the model for symbolic terms,
    building each term matrix, and summing the total Hamiltonian.

    Attributes:
        model: Model instance whose hooks provide geometry, constraints, and
            symbolic terms.
    """

    model: HamiltonianModelBase

    def build_basis(
        self,
        *,
        solver: BasisSolverName = "dfs",
        sort: bool = True,
        max_states: int | None = None,
    ) -> Basis:
        layout = self.model.layout
        constraints = tuple(self.model.make_constraints(layout))
        sectors = tuple(self.model.make_sectors(layout))

        return solve_basis(
            layout,
            constraints=constraints,
            sectors=sectors,
            solver=solver,
            sort=sort,
            max_states=max_states,
        )

    def build(
        self,
        *,
        basis: Basis | BinaryEncodedBasis | None = None,
        basis_solver: BasisSolverName = "dfs",
        builder: HamiltonianBuilderName = "sparse",
        backend: SparseBackendName | SparseBackend = "scipy",
        dtype: npt.DTypeLike = np.complex128,
        sort_basis: bool = True,
        on_missing: Literal["skip", "raise"] = "raise",
        drop_zero_atol: float = 0.0,
    ) -> ModelBuildResult:
        validate_builder_name(builder)

        physical_layout = self.model.layout
        constraints = tuple(self.model.make_constraints(physical_layout))
        sectors = tuple(self.model.make_sectors(physical_layout))

        if basis is None:
            array_basis = solve_basis(
                physical_layout,
                constraints=constraints,
                sectors=sectors,
                solver=basis_solver,
                sort=sort_basis,
            )
        elif isinstance(basis, Basis):
            array_basis = basis
        elif isinstance(basis, BinaryEncodedBasis):
            array_basis = basis.to_array_basis()
        else:
            raise TypeError("basis must be Basis, BinaryEncodedBasis, or None.")

        operator_layout, build_basis_obj = self.model.prepare_builder_basis(
            physical_layout=physical_layout,
            array_basis=array_basis,
            input_basis=basis,
            builder=builder,
            sort_basis=sort_basis,
        )

        term_specs = tuple(
            self.model.make_terms(
                operator_layout,
                builder=builder,
            )
        )

        built_terms: dict[str, BuiltHamiltonianTerm] = {}

        for term_spec in term_specs:
            matrix = self._build_term_matrix(
                basis=build_basis_obj,
                term=term_spec,
                builder=builder,
                backend=backend,
                dtype=dtype,
                on_missing=on_missing,
                drop_zero_atol=drop_zero_atol,
            )

            built_terms[term_spec.name] = BuiltHamiltonianTerm(
                name=term_spec.name,
                kind=term_spec.kind,
                operators=term_spec.operators,
                matrix=matrix,
            )

        term_matrices = [term.matrix for term in built_terms.values()]

        try:
            hamiltonian = combine_hamiltonian_terms(term_matrices)
        except ValueError:
            sparse_backend = get_sparse_backend(backend)
            hamiltonian = sparse_backend.empty_csr(
                shape=(build_basis_obj.n_states, build_basis_obj.n_states),
                dtype=dtype,
            )

        return ModelBuildResult(
            model=self.model,
            lattice=self.model.lattice,
            layout=physical_layout,
            constraints=constraints,
            sectors=sectors,
            basis=build_basis_obj,
            terms=built_terms,
            hamiltonian=hamiltonian,
        )

    def build_hamiltonian(
        self,
        *,
        basis: Basis | BinaryEncodedBasis | None = None,
        basis_solver: BasisSolverName = "dfs",
        builder: HamiltonianBuilderName = "sparse",
        backend: SparseBackendName | SparseBackend = "scipy",
        dtype: npt.DTypeLike = np.complex128,
        sort_basis: bool = True,
        on_missing: Literal["skip", "raise"] = "raise",
        drop_zero_atol: float = 0.0,
    ) -> Any:
        return self.build(
            basis=basis,
            basis_solver=basis_solver,
            builder=builder,
            backend=backend,
            dtype=dtype,
            sort_basis=sort_basis,
            on_missing=on_missing,
            drop_zero_atol=drop_zero_atol,
        ).hamiltonian

    def _build_term_matrix(
        self,
        *,
        basis: Basis | BinaryEncodedBasis,
        term: HamiltonianTermSpec,
        builder: HamiltonianBuilderName,
        backend: SparseBackendName | SparseBackend,
        dtype: npt.DTypeLike,
        on_missing: Literal["skip", "raise"],
        drop_zero_atol: float,
    ) -> Any | None:
        if term.is_empty:
            return None

        if builder == "sparse":
            if not isinstance(basis, Basis):
                raise TypeError("builder='sparse' requires an array Basis.")

            return SparseHamiltonianBuilder(
                backend=backend,
                dtype=dtype,
                on_missing=on_missing,
                drop_zero_atol=drop_zero_atol,
            ).build(basis, term.operators)

        if builder == "optimized":
            if not isinstance(basis, Basis):
                raise TypeError("builder='optimized' requires an array Basis.")

            return OptimizedSparseHamiltonianBuilder(
                backend=backend,
                dtype=dtype,
                on_missing=on_missing,
                drop_zero_atol=drop_zero_atol,
            ).build(basis, term.operators)

        if builder == "bitmask":
            if not isinstance(basis, BinaryEncodedBasis):
                raise TypeError("builder='bitmask' requires a BinaryEncodedBasis.")

            return BitmaskSparseHamiltonianBuilder(
                backend=backend,
                dtype=dtype,
                on_missing=on_missing,
                drop_zero_atol=drop_zero_atol,
            ).build(basis, term.operators)

        raise ValueError(f"Unsupported builder: {builder}")
