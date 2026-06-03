from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

from qlinks.constraints import (
    DimerCoveringConstraint,
    HoneycombElectricWindingSector,
    SquareQDMElectricWindingSector,
    SquareWindingSector,
    TriangularZ2WindingSector,
    WindingTarget,
)
from qlinks.encoded import (
    BitmaskAlternatingPlaquetteFlipOperator,
    BitmaskQDMFlipOperator,
    bitmask_alternating_flippability_projectors,
    bitmask_qdm_flippability_projectors,
)
from qlinks.lattice import (
    BoundaryCondition,
    HoneycombLattice,
    LatticeGraph,
    SquareLattice,
    TriangularLattice,
)
from qlinks.models.base import (
    BasisSolverName,
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    normalize_sector_labels_for_display,
    validate_builder_name,
)
from qlinks.models.couplings import (
    DirectedPlaquetteCoupling,
    DirectedPlaquetteCouplingLike,
    PlaquetteCoupling,
    directed_plaquette_coupling_value,
    is_zero_coupling,
    plaquette_coupling_value,
)
from qlinks.models.local_terms import (
    LocalOperatorKind,
    LocalTermDescriptor,
    LocalTermKind,
)
from qlinks.operators import (
    PlaquettePatternOperator,
    UpdatePlaquettePatternOperator,
    alternating_binary_flippability_projectors,
    qdm_flippability_projectors,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class QDMBase(HamiltonianModelBase):
    """
    Shared implementation for link-binary quantum dimer models.

    Subclasses provide the lattice geometry by implementing `_make_lattice()`.
    They may also override `plaquette_ids()` or `make_sectors()` for
    geometry-specific topological sectors.

    Variables:
        n_l in {0, 1}

    Constraint:
        sum of occupied links touching each site = required_count

    Hamiltonian:
        H = kinetic * sum_p flip_p
          + potential * sum_p flippability_p
    """

    coup_kin: DirectedPlaquetteCouplingLike = -1.0
    coup_pot: PlaquetteCoupling = 0.0
    required_count: int = 1

    def allowed_sector_labels(self):
        return normalize_sector_labels_for_display(self._allowed_sector_labels())

    def nonempty_sector_labels(self, *args, **kwargs):
        return normalize_sector_labels_for_display(self._nonempty_sector_labels(*args, **kwargs))

    def _coup_kin_at(self, plaquette_id: int) -> DirectedPlaquetteCoupling:
        return directed_plaquette_coupling_value(
            self.coup_kin,
            int(plaquette_id),
            name="coup_kin",
        )

    def _coup_pot_at(self, plaquette_id: int) -> complex:
        return plaquette_coupling_value(
            self.coup_pot,
            int(plaquette_id),
            name="coup_pot",
        )

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_links(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        return DimerCoveringConstraint.all_sites(
            lattice=self.lattice,
            layout=layout,
            required_counts=self.required_count,
        )

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        """
        Default QDM sector list.

        Geometry-specific subclasses can override this.
        """
        return ()

    def plaquette_ids(self) -> list[int]:
        """
        Plaquettes used by the QDM resonance move.

        The lattice may define qdm_plaquette_ids() to select only the relevant
        resonance loops. For example, triangular QDM should use rhombi rather
        than elementary triangles.
        """

        if hasattr(self.lattice, "qdm_plaquette_ids"):
            return list(self.lattice.qdm_plaquette_ids())

        return [int(p) for p in self.lattice.plaquette_ids]

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        operators: list[object] = []
        for plaquette_id in self.plaquette_ids():
            operators.extend(
                self._make_single_kinetic_operator(
                    layout,
                    plaquette_id=int(plaquette_id),
                    builder=builder,
                )
            )

        return tuple(operators)

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        if is_zero_coupling(self.coup_pot, self.plaquette_ids()):
            return ()

        operators: list[object] = []
        for plaquette_id in self.plaquette_ids():
            operators.extend(
                self._make_single_potential_operators(
                    layout,
                    plaquette_id=int(plaquette_id),
                    builder=builder,
                )
            )

        return tuple(operators)

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        return self.make_kinetic_operators(layout, builder=builder) + self.make_potential_operators(
            layout, builder=builder
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[HamiltonianTermSpec, ...]:
        kinetic_operators = self.make_kinetic_operators(
            layout,
            builder=builder,
        )

        potential_operators = self.make_potential_operators(
            layout,
            builder=builder,
        )

        terms: list[HamiltonianTermSpec] = [
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=kinetic_operators,
                kind="kinetic",
            )
        ]

        if potential_operators:
            terms.append(
                HamiltonianTermSpec.from_operators(
                    name="potential",
                    operators=potential_operators,
                    kind="potential",
                )
            )

        return tuple(terms)

    def _make_single_kinetic_operator(
        self,
        layout: VariableLayout,
        *,
        plaquette_id: int,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)
        coupling = self._coup_kin_at(plaquette_id)

        if builder == "sparse":
            return (
                PlaquettePatternOperator.alternating_binary_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coupling.resolved_forward(),
                    reverse_coefficient=coupling.resolved_backward(),
                ),
            )

        if builder == "bitmask":
            return (
                BitmaskAlternatingPlaquetteFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coupling.resolved_forward(),
                    reverse_coefficient=coupling.resolved_backward(),
                ),
            )

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QDM operators are not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def _make_single_potential_operators(
        self,
        layout: VariableLayout,
        *,
        plaquette_id: int,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)
        coefficient = self._coup_pot_at(plaquette_id)

        if coefficient == 0:
            return ()

        if builder == "sparse":
            return tuple(
                alternating_binary_flippability_projectors(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coefficient,
                )
            )

        if builder == "bitmask":
            return tuple(
                bitmask_alternating_flippability_projectors(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coefficient,
                )
            )

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QDM potential is not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def local_term_descriptors(
        self,
        *,
        operator_kind: LocalOperatorKind | None = None,
        term_kind: LocalTermKind | None = None,
    ) -> tuple[LocalTermDescriptor, ...]:
        if term_kind not in (None, "plaquette"):
            return ()

        descriptors: list[LocalTermDescriptor] = []

        for plaquette_id in self.plaquette_ids():
            plaquette_id = int(plaquette_id)
            support_links = tuple(
                int(link_id) for link_id in self.lattice.plaquette_links(plaquette_id)
            )

            if operator_kind in (None, "kinetic", "hamiltonian"):
                descriptors.append(
                    LocalTermDescriptor(
                        term_id=plaquette_id,
                        term_kind="plaquette",
                        operator_kind="kinetic",
                        support_links=support_links,
                        support_plaquettes=(plaquette_id,),
                        label=f"K_{plaquette_id}",
                    )
                )

            if operator_kind in (None, "potential", "hamiltonian"):
                descriptors.append(
                    LocalTermDescriptor(
                        term_id=plaquette_id,
                        term_kind="plaquette",
                        operator_kind="potential",
                        support_links=support_links,
                        support_plaquettes=(plaquette_id,),
                        label=f"V_{plaquette_id}",
                    )
                )

        return tuple(descriptors)

    def make_local_term(
        self,
        descriptor: LocalTermDescriptor,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> HamiltonianTermSpec:
        if descriptor.term_kind != "plaquette":
            raise ValueError("QDM local terms currently only support plaquette terms.")

        plaquette_id = int(descriptor.term_id)

        if descriptor.operator_kind == "kinetic":
            operators = self._make_single_kinetic_operator(
                layout,
                plaquette_id=plaquette_id,
                builder=builder,
            )
            return HamiltonianTermSpec.from_operators(
                name=f"kinetic_{plaquette_id}",
                operators=operators,
                kind="kinetic",
            )

        if descriptor.operator_kind == "potential":
            operators = self._make_single_potential_operators(
                layout,
                plaquette_id=plaquette_id,
                builder=builder,
            )
            return HamiltonianTermSpec.from_operators(
                name=f"potential_{plaquette_id}",
                operators=operators,
                kind="potential",
            )

        raise ValueError("descriptor.operator_kind must be 'kinetic' or 'potential' for QDM.")


@dataclass(frozen=True)
class SquareQDMModel(QDMBase):
    """
    Square-lattice quantum dimer model.

    This subclass keeps square-specific functionality, especially winding
    sectors.

    winding_convention:
        "cut_count":
            raw count of occupied wrapping links.

        "electric":
            staggered electric-flux winding compatible with the square QDM
            to staggered-charge QLM mapping.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_x: WindingTarget | None = None
    winding_y: WindingTarget | None = None
    winding_convention: Literal["cut_count", "electric"] = "electric"

    def _make_lattice(self) -> SquareLattice:
        return SquareLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return [int(p) for p in self.lattice.plaquette_ids]

    def _allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        if self.winding_convention == "cut_count":
            winding_cls = SquareWindingSector
            kwargs = {"flux_normalization": "integer_flux"}
        elif self.winding_convention == "electric":
            winding_cls = SquareQDMElectricWindingSector
            kwargs = {}
        else:
            raise ValueError("winding_convention must be 'cut_count' or 'electric'.")

        return {
            "winding_x": winding_cls.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="x",
                **kwargs,
            ),
            "winding_y": winding_cls.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="y",
                **kwargs,
            ),
        }

    def _nonempty_sector_labels(
        self,
        *,
        solver: BasisSolverName = "dfs",
    ) -> dict[str, tuple[tuple[object, object], ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        allowed = self.allowed_sector_labels()
        nonempty: list[tuple[object, object]] = []

        for winding_x, winding_y in product(
            allowed["winding_x"],
            allowed["winding_y"],
        ):
            trial_model = replace(
                self,
                winding_x=winding_x,
                winding_y=winding_y,
            )
            if trial_model.has_basis_state(solver=solver):
                nonempty.append((winding_x, winding_y))

        return {"winding": tuple(nonempty)}

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_convention == "cut_count":
            winding_cls = SquareWindingSector
        elif self.winding_convention == "electric":
            winding_cls = SquareQDMElectricWindingSector
        else:
            raise ValueError("winding_convention must be 'cut_count' or 'electric'.")

        if self.winding_x is not None:
            sectors.append(
                winding_cls(
                    layout=layout,
                    lattice=self.lattice,
                    direction="x",
                    target=self.winding_x,
                )
            )

        if self.winding_y is not None:
            sectors.append(
                winding_cls(
                    layout=layout,
                    lattice=self.lattice,
                    direction="y",
                    target=self.winding_y,
                )
            )

        return tuple(sectors)

    def _make_single_kinetic_operator(
        self,
        layout: VariableLayout,
        *,
        plaquette_id: int,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        """
        Square QDM keeps the specialized length-4 qdm_flip implementation.

        The generic alternating plaquette flip would also work for square
        plaquettes, but the specialized operator preserves the old API and
        square-QDM convention explicitly.
        """
        validate_builder_name(builder)
        coupling = self._coup_kin_at(plaquette_id)

        if builder == "sparse":
            return (
                PlaquettePatternOperator.qdm_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coupling.resolved_forward(),
                    reverse_coefficient=coupling.resolved_backward(),
                ),
            )

        if builder == "optimized":
            return (
                UpdatePlaquettePatternOperator.qdm_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coupling.resolved_forward(),
                    reverse_coefficient=coupling.resolved_backward(),
                ),
            )

        if builder == "bitmask":
            return (
                BitmaskQDMFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coupling.resolved_forward(),
                    reverse_coefficient=coupling.resolved_backward(),
                ),
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def _make_single_potential_operators(
        self,
        layout: VariableLayout,
        *,
        plaquette_id: int,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        """
        Square QDM keeps the specialized 1010/0101 flippability projectors.
        """
        validate_builder_name(builder)
        coefficient = self._coup_pot_at(plaquette_id)

        if coefficient == 0:
            return ()

        if builder == "sparse":
            return tuple(
                qdm_flippability_projectors(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coefficient,
                )
            )

        if builder == "bitmask":
            return tuple(
                bitmask_qdm_flippability_projectors(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=plaquette_id,
                    coefficient=coefficient,
                )
            )

        if builder == "optimized":
            raise NotImplementedError(
                "Optimized QDM potential is not implemented as update actions yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")


@dataclass(frozen=True)
class TriangularQDMModel(QDMBase):
    """
    Triangular-lattice QDM.

    The QDM resonance plaquettes are rhombi/lozenges, not elementary triangles.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_a: int | None = None
    winding_b: int | None = None

    def _make_lattice(self) -> TriangularLattice:
        return TriangularLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
            include_triangles=True,
            include_rhombi=True,
        )

    def plaquette_ids(self) -> list[int]:
        return list(self.lattice.qdm_plaquette_ids())

    def _allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        return {
            "winding_a": TriangularZ2WindingSector.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="a",
                value_convention="binary",
            ),
            "winding_b": TriangularZ2WindingSector.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="b",
                value_convention="binary",
            ),
        }

    def _nonempty_sector_labels(
        self,
        *,
        solver: BasisSolverName = "dfs",
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        allowed = self.allowed_sector_labels()
        nonempty: list[tuple[int, int]] = []

        for winding_a, winding_b in product(
            allowed["winding_a"],
            allowed["winding_b"],
        ):
            trial_model = replace(
                self,
                winding_a=winding_a,
                winding_b=winding_b,
            )
            if trial_model.has_basis_state(solver=solver):
                nonempty.append((int(winding_a), int(winding_b)))

        return {"z2_winding": tuple(nonempty)}

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_a is not None:
            sectors.append(
                TriangularZ2WindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="a",
                    target=self.winding_a,
                    value_convention="binary",
                )
            )

        if self.winding_b is not None:
            sectors.append(
                TriangularZ2WindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="b",
                    target=self.winding_b,
                    value_convention="binary",
                )
            )

        return tuple(sectors)


@dataclass(frozen=True)
class HoneycombQDMModel(QDMBase):
    """
    Honeycomb-lattice QDM.

    The QDM resonance plaquettes are hexagons.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_x: WindingTarget | None = None
    winding_y: WindingTarget | None = None

    def _make_lattice(self) -> HoneycombLattice:
        return HoneycombLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return list(self.lattice.qdm_plaquette_ids())

    def _allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        return {
            "winding_x": HoneycombElectricWindingSector.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="x",
                value_convention="binary",
            ),
            "winding_y": HoneycombElectricWindingSector.allowed_targets(
                layout=self.layout,
                lattice=self.lattice,
                direction="y",
                value_convention="binary",
            ),
        }

    def _nonempty_sector_labels(
        self,
        *,
        solver: BasisSolverName = "dfs",
    ) -> dict[str, tuple[tuple[object, object], ...]]:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            return {}

        allowed = self.allowed_sector_labels()
        nonempty: list[tuple[object, object]] = []

        for winding_x, winding_y in product(
            allowed["winding_x"],
            allowed["winding_y"],
        ):
            trial_model = replace(
                self,
                winding_x=winding_x,
                winding_y=winding_y,
            )
            if trial_model.has_basis_state(solver=solver):
                nonempty.append((winding_x, winding_y))

        return {"winding": tuple(nonempty)}

    def make_sectors(self, layout: VariableLayout | None = None):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_x is not None:
            sectors.append(
                HoneycombElectricWindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="x",
                    target=self.winding_x,
                    value_convention="binary",
                )
            )

        if self.winding_y is not None:
            sectors.append(
                HoneycombElectricWindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="y",
                    target=self.winding_y,
                    value_convention="binary",
                )
            )

        return tuple(sectors)


@dataclass(frozen=True)
class QDMModel(QDMBase):
    """
    Generic lattice-backed QDM model.

    Use this when you already have a LatticeGraph instance.

    For named geometries, prefer:
        SquareQDMModel
        TriangularQDMModel
        HoneycombQDMModel
    """

    lattice_input: LatticeGraph | None = None

    def _make_lattice(self) -> LatticeGraph:
        if self.lattice_input is None:
            raise ValueError(
                "QDMModel requires lattice_input. "
                "Use SquareQDMModel, TriangularQDMModel, or HoneycombQDMModel "
                "for built-in geometries."
            )
        return self.lattice_input

    @classmethod
    def triangular(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        coup_kin: PlaquetteCoupling = -1.0,
        coup_pot: PlaquetteCoupling = 0.0,
        required_count: int = 1,
    ) -> TriangularQDMModel:
        return TriangularQDMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            coup_kin=coup_kin,
            coup_pot=coup_pot,
            required_count=required_count,
        )

    @classmethod
    def honeycomb(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        coup_kin: PlaquetteCoupling = -1.0,
        coup_pot: PlaquetteCoupling = 0.0,
        required_count: int = 1,
    ) -> HoneycombQDMModel:
        return HoneycombQDMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            coup_kin=coup_kin,
            coup_pot=coup_pot,
            required_count=required_count,
        )
