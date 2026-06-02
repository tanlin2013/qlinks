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
    """
    A symbolic Hamiltonian term before matrix construction.

    Examples
    --------
    kinetic term:
        HamiltonianTermSpec(
            name="kinetic",
            kind="kinetic",
            operators=(op0, op1, ...),
        )

    potential term:
        HamiltonianTermSpec(
            name="potential",
            kind="potential",
            operators=(diag_op0, diag_op1, ...),
        )
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
    """
    A Hamiltonian term after sparse matrix construction.
    """

    name: str
    kind: TermKind
    operators: tuple[object, ...]
    matrix: Any | None


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """
    Full model build result.

    The intended usage is:

        result = model.build(...)
        H = result.hamiltonian
        K = result.kinetic
        V = result.potential
        basis = result.basis

    This avoids repeatedly calling expensive build methods.
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
    """
    Shared sparse-build options.

    This is mostly a convenience container for scripts and future APIs.
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
    """
    Common array-basis solver dispatch.

    If no constraints/sectors are present, use a direct Cartesian-product
    construction instead of DFS/CP-SAT/brute force.
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
    if builder not in ("sparse", "optimized", "bitmask"):
        raise ValueError("builder must be one of 'sparse', 'optimized', or 'bitmask'.")


def combine_hamiltonian_terms(matrices: Sequence[Any | None]) -> Any:
    """
    Sum all non-None sparse matrices.

    Raises
    ------
    ValueError
        If all terms are None.
    """

    nonempty = [matrix for matrix in matrices if matrix is not None]

    if len(nonempty) == 0:
        raise ValueError("Cannot combine zero Hamiltonian terms.")

    total = nonempty[0]
    for matrix in nonempty[1:]:
        total = total + matrix

    return total


class HamiltonianModelBase:
    """
    Base class for Hamiltonian models.

    Concrete models should implement:

        _make_lattice()
        _make_layout()
        make_constraints()
        make_sectors()
        make_terms()

    The base class provides:

        cached lattice
        cached layout
        cached model_builder
        build_basis()
        build()
        build_hamiltonian()

    Notes
    -----
    Model classes that inherit from this base should usually NOT use
    dataclass(slots=True), because functools.cached_property needs an instance
    __dict__. Use:

        @dataclass(frozen=True)
        class MyModel(HamiltonianModelBase):
            ...
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
        """
        Return allowed user-facing quantum labels for diagonal symmetry sectors.

        Models without user-selectable sectors return an empty dictionary.
        Geometry-specific subclasses should override this method.

        Examples
        --------
        SquareQLMModel(...).allowed_sector_labels()

        may return:

            {
                "winding_x": (...),
                "winding_y": (...),
            }
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
        """
        Backward-compatible alias.
        """
        return self.lattice

    def make_layout(self) -> VariableLayout:
        """
        Backward-compatible alias.
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
        """
        Convert the physical layout/basis into the representation needed by
        the requested builder.

        Default behavior
        ----------------
        sparse / optimized:
            use the array Basis.

        bitmask:
            convert binary array Basis into BinaryEncodedBasis.

        QLM flux models using {-1,+1} variables should override this method
        because they need the flux-to-binary encoding convention.
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
        """
        Build basis, Hamiltonian terms, and total Hamiltonian once.

        Preferred usage:

            result = model.build(...)
            H = result.hamiltonian
            K = result.kinetic
            V = result.potential
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
        """
        Convenience method returning only the total Hamiltonian.

        For accessing kinetic/potential separately, prefer calling build()
        once and using result.kinetic/result.potential.
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


@dataclass(frozen=True, slots=True)
class GenericModelBuilder:
    """
    Shared Hamiltonian model builder.

    It owns the repeated model-building workflow:

        1. get cached layout
        2. build constraints/sectors
        3. solve or convert basis
        4. ask model for Hamiltonian terms
        5. build each sparse term matrix
        6. sum terms into total Hamiltonian
        7. return ModelBuildResult
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
