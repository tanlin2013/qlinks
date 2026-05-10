from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.constraints import (
    ChargeNormalization,
    FluxNormalization,
    GaussLawConstraint,
    HoneycombElectricWindingSector,
    SquareWindingSector,
    TriangularZ2WindingSector,
)
from qlinks.conventions import SublatticeSignConvention, square_qdm_staggered_charges
from qlinks.encoded import (
    BinaryEncodedBasis,
    BitmaskAlternatingPlaquetteFlipOperator,
    BitmaskQLMFluxFlipOperator,
    binary_encoded_basis_from_flux_basis,
    binary_layout_like_flux_layout,
    bitmask_alternating_flippability_projectors,
    bitmask_qlm_flippability_projectors,
)
from qlinks.lattice import (
    BoundaryCondition,
    HoneycombLattice,
    LatticeGraph,
    SquareLattice,
    TriangularLattice,
)
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import (
    PatternDiagonalOperator,
    PlaquettePatternOperator,
    PlaquettePatternTransition,
    UpdatePlaquettePatternOperator,
    UpdatePlaquettePatternTransition,
    alternating_flux_flippability_projectors,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class QLMBase(HamiltonianModelBase):
    """
    Shared implementation for spin-1/2 quantum link models.

    Variables:
        link electric flux E_l in {-1, +1}

    Constraint:
        Gauss law at each site.

    Hamiltonian:
        H = kinetic * sum_p ring_exchange_p
          + potential * sum_p flippability_p

    Bitmask convention:
        physical flux:
            -1, +1

        encoded binary:
            -1 -> 0
            +1 -> 1
    """

    kinetic: complex = -1.0
    potential: complex = 0.0
    charges: int | Sequence[int] | npt.NDArray[np.int64] = 0
    charge_normalization: ChargeNormalization = "spin_half"

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_links(
            self.lattice,
            LocalSpace.spin_half_flux(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        return GaussLawConstraint.all_sites(
            lattice=self.lattice,
            layout=layout,
            charges=self.charges,
            charge_normalization=self.charge_normalization,
        )

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        """
        Default QLM sectors.

        Geometry-specific subclasses can override this.
        """
        return ()

    def plaquette_ids(self) -> list[int]:
        """
        Plaquettes used by QLM ring exchange.

        The lattice may define qlm_plaquette_ids() to select only valid
        even-length ring-exchange loops.
        """

        if hasattr(self.lattice, "qlm_plaquette_ids"):
            return list(self.lattice.qlm_plaquette_ids())

        return [int(p) for p in self.lattice.plaquette_ids]

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
        Override the default bitmask conversion because physical QLM variables
        are {-1,+1}, while the bitmask backend needs binary {0,1}.
        """

        validate_builder_name(builder)

        if builder != "bitmask":
            return physical_layout, array_basis

        binary_layout = binary_layout_like_flux_layout(physical_layout)

        if input_basis is None:
            encoded_basis = binary_encoded_basis_from_flux_basis(
                array_basis,
                sort=False,
            )
        elif isinstance(input_basis, BinaryEncodedBasis):
            encoded_basis = input_basis
        elif isinstance(input_basis, Basis):
            encoded_basis = binary_encoded_basis_from_flux_basis(
                input_basis,
                sort=False,
            )
        else:
            raise TypeError("basis must be Basis, BinaryEncodedBasis, or None.")

        return binary_layout, encoded_basis

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QLM operators are not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        if layout is None:
            if builder == "bitmask":
                layout = binary_layout_like_flux_layout(self.layout)
            else:
                layout = self.layout

        if builder == "sparse":
            return tuple(
                PlaquettePatternOperator.alternating_flux_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        if builder == "bitmask":
            return tuple(
                BitmaskAlternatingPlaquetteFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if self.potential == 0:
            return ()

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QLM potential is not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        if layout is None:
            if builder == "bitmask":
                layout = binary_layout_like_flux_layout(self.layout)
            else:
                layout = self.layout

        operators: list[object] = []

        if builder == "sparse":
            for p in self.plaquette_ids():
                operators.extend(
                    alternating_flux_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        if builder == "bitmask":
            for p in self.plaquette_ids():
                operators.extend(
                    bitmask_alternating_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        raise ValueError(f"Unsupported builder: {builder}")

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


@dataclass(frozen=True)
class SquareQLMModel(QLMBase):
    """
    Square-lattice spin-1/2 QLM.

    Square-specific functionality:
        - square lattice construction
        - optional winding sectors
        - optimized update operators for the kinetic term
        - specialized bitmask QLM flux flip/projectors
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_x: int | None = None
    winding_y: int | None = None

    def _make_lattice(self) -> SquareLattice:
        return SquareLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return [int(p) for p in self.lattice.plaquette_ids]

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_x is not None:
            sectors.append(
                SquareWindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="x",
                    target=self.winding_x,
                    flux_normalization=self.charge_normalization,
                )
            )

        if self.winding_y is not None:
            sectors.append(
                SquareWindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="y",
                    target=self.winding_y,
                    flux_normalization=self.charge_normalization,
                )
            )

        return tuple(sectors)

    @staticmethod
    def _flux_transitions(coefficient: complex):
        return (
            PlaquettePatternTransition(
                initial=np.asarray([1, -1, 1, -1], dtype=np.int64),
                final=np.asarray([-1, 1, -1, 1], dtype=np.int64),
                coefficient=coefficient,
            ),
            PlaquettePatternTransition(
                initial=np.asarray([-1, 1, -1, 1], dtype=np.int64),
                final=np.asarray([1, -1, 1, -1], dtype=np.int64),
                coefficient=coefficient,
            ),
        )

    @staticmethod
    def _update_flux_transitions(coefficient: complex):
        return (
            UpdatePlaquettePatternTransition(
                initial=np.asarray([1, -1, 1, -1], dtype=np.int64),
                final=np.asarray([-1, 1, -1, 1], dtype=np.int64),
                coefficient=coefficient,
            ),
            UpdatePlaquettePatternTransition(
                initial=np.asarray([-1, 1, -1, 1], dtype=np.int64),
                final=np.asarray([1, -1, 1, -1], dtype=np.int64),
                coefficient=coefficient,
            ),
        )

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            if builder == "bitmask":
                layout = binary_layout_like_flux_layout(self.layout)
            else:
                layout = self.layout

        if builder == "sparse":
            return tuple(
                PlaquettePatternOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    transitions=self._flux_transitions(self.kinetic),
                    name="qlm_plaquette_ring_exchange",
                )
                for p in self.plaquette_ids()
            )

        if builder == "optimized":
            return tuple(
                UpdatePlaquettePatternOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    transitions=self._update_flux_transitions(self.kinetic),
                    name="update_qlm_plaquette_ring_exchange",
                )
                for p in self.plaquette_ids()
            )

        if builder == "bitmask":
            return tuple(
                BitmaskQLMFluxFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if self.potential == 0:
            return ()

        if layout is None:
            if builder == "bitmask":
                layout = binary_layout_like_flux_layout(self.layout)
            else:
                layout = self.layout

        operators: list[object] = []

        if builder == "sparse":
            for p in self.plaquette_ids():
                link_ids = self.lattice.plaquette_links(int(p))
                variable_indices = np.asarray(
                    [layout.link_variable_index(int(link_id)) for link_id in link_ids],
                    dtype=np.int64,
                )

                operators.append(
                    PatternDiagonalOperator(
                        layout=layout,
                        variable_indices=variable_indices,
                        pattern=np.asarray([1, -1, 1, -1], dtype=np.int64),
                        coefficient=self.potential,
                        name="qlm_flippability_pos",
                    )
                )

                operators.append(
                    PatternDiagonalOperator(
                        layout=layout,
                        variable_indices=variable_indices,
                        pattern=np.asarray([-1, 1, -1, 1], dtype=np.int64),
                        coefficient=self.potential,
                        name="qlm_flippability_neg",
                    )
                )

            return tuple(operators)

        if builder == "bitmask":
            for p in self.plaquette_ids():
                operators.extend(
                    bitmask_qlm_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )

            return tuple(operators)

        if builder == "optimized":
            raise NotImplementedError(
                "QLM potential is currently implemented only for builder='sparse' "
                "or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")

    @classmethod
    def from_qdm_staggered_background(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        kinetic: complex = -1.0,
        potential: complex = 0.0,
        charge_magnitude: int = 2,
        charge_convention: SublatticeSignConvention = "even_positive",
        winding_x: int | None = None,
        winding_y: int | None = None,
    ) -> SquareQLMModel:
        lattice = SquareLattice(
            lx,
            ly,
            boundary_condition=boundary_condition,
        )

        charges = square_qdm_staggered_charges(
            lattice,
            magnitude=charge_magnitude,
            convention=charge_convention,
        )

        return cls(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            kinetic=kinetic,
            potential=potential,
            charges=charges,
            winding_x=winding_x,
            winding_y=winding_y,
        )


@dataclass(frozen=True)
class TriangularQLMModel(QLMBase):
    """
    Triangular-lattice QLM.

    By default, QLM ring exchange uses rhombus/lozenge plaquettes rather than
    elementary triangular loops, because the alternating flux pattern requires
    even-length loops.
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
        return list(self.lattice.qlm_plaquette_ids())

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
                    value_convention="flux_pm",
                )
            )

        if self.winding_b is not None:
            sectors.append(
                TriangularZ2WindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="b",
                    target=self.winding_b,
                    value_convention="flux_pm",
                )
            )

        return tuple(sectors)


@dataclass(frozen=True)
class HoneycombQLMModel(QLMBase):
    """
    Honeycomb-lattice QLM.

    The ring-exchange plaquettes are hexagons.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_x: int | None = None
    winding_y: int | None = None
    flux_normalization: FluxNormalization = "spin_half"

    def _make_lattice(self) -> HoneycombLattice:
        return HoneycombLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return list(self.lattice.qlm_plaquette_ids())

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
                    value_convention="flux_pm",
                    flux_normalization=self.charge_normalization,
                )
            )

        if self.winding_y is not None:
            sectors.append(
                HoneycombElectricWindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="y",
                    target=self.winding_y,
                    value_convention="flux_pm",
                    flux_normalization=self.charge_normalization,
                )
            )

        return tuple(sectors)


@dataclass(frozen=True)
class QLMModel(QLMBase):
    """
    Generic lattice-backed QLM model.

    Use this when you already have a LatticeGraph instance.

    For named geometries, prefer:
        SquareQLMModel
        TriangularQLMModel
        HoneycombQLMModel
    """

    lattice_input: LatticeGraph | None = None

    def _make_lattice(self) -> LatticeGraph:
        if self.lattice_input is None:
            raise ValueError(
                "QLMModel requires lattice_input. "
                "Use SquareQLMModel, TriangularQLMModel, or HoneycombQLMModel "
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
        kinetic: complex = -1.0,
        potential: complex = 0.0,
        charges: int | Sequence[int] | npt.NDArray[np.int64] = 0,
    ) -> TriangularQLMModel:
        return TriangularQLMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            kinetic=kinetic,
            potential=potential,
            charges=charges,
        )

    @classmethod
    def honeycomb(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        kinetic: complex = -1.0,
        potential: complex = 0.0,
        charges: int | Sequence[int] | npt.NDArray[np.int64] = 0,
    ) -> HoneycombQLMModel:
        return HoneycombQLMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            kinetic=kinetic,
            potential=potential,
            charges=charges,
        )
