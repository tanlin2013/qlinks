from __future__ import annotations

from dataclasses import dataclass

from qlinks.constraints import TotalValueSector
from qlinks.lattice import BoundaryCondition, ChainLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.models.local_terms import (
    LocalOperatorKind,
    LocalTermDescriptor,
    LocalTermKind,
)
from qlinks.operators import (
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    SpinOneXYBondOperator,
    UpdateSpinOneXYBondOperator,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class SpinOneXYChainModel(HamiltonianModelBase):
    """
    Spin-1 XY chain in the S^z product basis.

    Local basis:

        m_i in {-1, 0, +1}

    Hamiltonian:

        H = J_xy * sum_<ij> (S^x_i S^x_j + S^y_i S^y_j)
          = J_xy/2 * sum_<ij> (S^+_i S^-_j + S^-_i S^+_j)

    No constraints are imposed at this stage.
    """

    length: int
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    j_xy: complex = 1.0
    h_z: complex = 0.0
    d_z: complex = 0.0
    total_sz: int | None = None

    def _make_lattice(self) -> ChainLattice:
        return ChainLattice(
            self.length,
            boundary_condition=self.boundary_condition,
        )

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.spin_one(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        return ()

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        if self.total_sz is None:
            return ()

        return (
            TotalValueSector(
                layout=layout,
                target=int(self.total_sz),
                name="total_sz_sector",
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
            layout = self.layout

        if builder == "sparse":
            return tuple(
                SpinOneXYBondOperator(
                    layout=layout,
                    lattice=self.lattice,
                    link_id=int(link_id),
                    coefficient=self.j_xy,
                )
                for link_id in self.lattice.link_ids
            )

        if builder == "optimized":
            return tuple(
                UpdateSpinOneXYBondOperator(
                    layout=layout,
                    lattice=self.lattice,
                    link_id=int(link_id),
                    coefficient=self.j_xy,
                )
                for link_id in self.lattice.link_ids
            )

        raise NotImplementedError(
            "SpinOneXYChainModel currently supports kinetic terms only for "
            "builder='sparse' or builder='optimized'."
        )

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        operators: list[object] = []

        if builder not in ("sparse", "optimized"):
            if self.h_z == 0 and self.d_z == 0:
                return ()

            raise NotImplementedError(
                "SpinOneXYChainModel currently supports potential terms only for "
                "builder='sparse' or builder='optimized'."
            )

        for site_id in self.lattice.site_ids:
            variable_index = int(layout.site_variable_index(int(site_id)))

            if self.h_z != 0:
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.h_z,
                        name="spin_one_zeeman_z",
                    )
                )

            if self.d_z != 0:
                operators.append(
                    LocalSquareValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.d_z,
                        name="spin_one_single_ion_anisotropy",
                    )
                )

        return tuple(operators)

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        return (
            *self.make_kinetic_operators(layout, builder=builder),
            *self.make_potential_operators(layout, builder=builder),
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

        terms = [
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=kinetic_operators,
                kind="kinetic",
            ),
        ]

        if len(potential_operators) > 0:
            terms.append(
                HamiltonianTermSpec.from_operators(
                    name="potential",
                    operators=potential_operators,
                    kind="potential",
                )
            )

        return tuple(terms)

    def local_term_descriptors(
        self,
        *,
        operator_kind: LocalOperatorKind | None = None,
        term_kind: LocalTermKind | None = None,
    ) -> tuple[LocalTermDescriptor, ...]:
        """Return site/bond local terms for generic open-system builders."""
        descriptors: list[LocalTermDescriptor] = []

        if term_kind in (None, "bond") and operator_kind in (None, "kinetic", "hamiltonian"):
            for link in self.lattice.links:
                support_sites = (int(link.source), int(link.target))
                support_variables = tuple(
                    int(self.layout.site_variable_index(site_id)) for site_id in support_sites
                )
                descriptors.append(
                    LocalTermDescriptor(
                        term_id=int(link.id),
                        term_kind="bond",
                        operator_kind="kinetic",
                        support_links=(int(link.id),),
                        support_sites=support_sites,
                        support_variables=support_variables,
                        label=f"XY_{link.source}_{link.target}",
                    )
                )

        if term_kind in (None, "site") and operator_kind in (None, "potential", "hamiltonian"):
            for site_id in self.lattice.site_ids:
                variable_index = int(self.layout.site_variable_index(int(site_id)))

                if self.h_z != 0:
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=int(site_id),
                            term_kind="site",
                            operator_kind="potential",
                            support_links=(),
                            support_sites=(int(site_id),),
                            support_variables=(variable_index,),
                            label=f"Sz_{site_id}",
                        )
                    )

                if self.d_z != 0:
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=int(site_id),
                            term_kind="site",
                            operator_kind="potential",
                            support_links=(),
                            support_sites=(int(site_id),),
                            support_variables=(variable_index,),
                            label=f"Sz2_{site_id}",
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
        validate_builder_name(builder)

        if descriptor.term_kind == "bond" and descriptor.operator_kind == "kinetic":
            operators = (
                (
                    SpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(descriptor.term_id),
                        coefficient=self.j_xy,
                    )
                    if builder == "sparse"
                    else UpdateSpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(descriptor.term_id),
                        coefficient=self.j_xy,
                    )
                ),
            )
            return HamiltonianTermSpec.from_operators(
                name=f"kinetic_{descriptor.term_id}",
                operators=operators,
                kind="kinetic",
            )

        if descriptor.term_kind == "site" and descriptor.operator_kind == "potential":
            site_id = int(descriptor.term_id)
            variable_index = int(layout.site_variable_index(site_id))
            operators: list[object] = []

            if self.h_z != 0 and (
                descriptor.label is None or str(descriptor.label).startswith("Sz_")
            ):
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.h_z,
                        name="spin_one_zeeman_z",
                    )
                )

            if self.d_z != 0 and (
                descriptor.label is None or str(descriptor.label).startswith("Sz2_")
            ):
                operators.append(
                    LocalSquareValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.d_z,
                        name="spin_one_single_ion_anisotropy",
                    )
                )

            return HamiltonianTermSpec.from_operators(
                name=f"potential_{descriptor.label or site_id}",
                operators=tuple(operators),
                kind="potential",
            )

        raise ValueError(
            "SpinOneXYChainModel local terms support bond kinetic terms and site potential terms."
        )
