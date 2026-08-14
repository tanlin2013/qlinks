from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.caging.analysis.thermodynamic import (
    EnergyDensityMatchReport,
    LocalWitness,
    WitnessNormalization,
)
from qlinks.caging.local_search.global_ops import _qdm_global_plaquette_actions
from qlinks.caging.local_search.padding import (
    factorized_qdm_padding_from_multi_padding,
    make_qdm_cage_block,
)
from qlinks.caging.local_search.types import (
    FactorizedLocalQDMPadding,
    LocalQDMCageBlock,
    LocalQDMCageRecord,
    MultiLocalQDMPadding,
)
from qlinks.constraints import SquareQDMElectricWindingSector
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models import SquareQDMModel

SquareQDMRepeatAxis = Literal["x", "y"]
_LinkCoordinate = tuple[int, int, str]
_PlaquetteCoordinate = tuple[int, int]
_ActionKey = tuple[tuple[int, ...], tuple[tuple[int, int], ...]]


def _link_coordinate(model: SquareQDMModel, link_id: int) -> _LinkCoordinate:
    link = model.lattice.links[int(link_id)]
    x, y = model.lattice.sites[int(link.source)].cell
    return int(x), int(y), str(link.kind)


def _plaquette_coordinate(model: SquareQDMModel, plaquette_id: int) -> _PlaquetteCoordinate:
    x, y = model.lattice.plaquette_anchor_cell(int(plaquette_id))
    return int(x), int(y)


def _link_lookup(model: SquareQDMModel) -> dict[_LinkCoordinate, int]:
    return {_link_coordinate(model, int(link.id)): int(link.id) for link in model.lattice.links}


def _translate_coordinate(
    coordinate: tuple[int, int],
    *,
    axis: SquareQDMRepeatAxis,
    offset: int,
) -> tuple[int, int]:
    x, y = coordinate
    if axis == "x":
        return x + offset, y
    return x, y + offset


def _translate_link_coordinate(
    coordinate: _LinkCoordinate,
    *,
    axis: SquareQDMRepeatAxis,
    offset: int,
) -> _LinkCoordinate:
    x, y = _translate_coordinate(coordinate[:2], axis=axis, offset=offset)
    return x, y, coordinate[2]


def _as_factorized_padding(
    padding: FactorizedLocalQDMPadding | MultiLocalQDMPadding,
) -> FactorizedLocalQDMPadding:
    if isinstance(padding, MultiLocalQDMPadding):
        return factorized_qdm_padding_from_multi_padding(padding)
    return padding


def _validate_complete_product_assignment(
    model: SquareQDMModel,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
) -> None:
    block_ids = tuple(int(block.block_id) for block in blocks)
    if block_ids != tuple(int(value) for value in padding.block_ids):
        raise ValueError("blocks must match padding.block_ids and order.")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("block ids must be unique.")

    owned: set[int] = set()
    for block in blocks:
        links = set(int(link_id) for link_id in block.link_ids)
        if owned.intersection(links):
            raise ValueError("periodic product blocks must be link-disjoint.")
        owned.update(links)

    exterior = set(int(link_id) for link_id in padding.exterior_link_ids)
    if owned.intersection(exterior):
        raise ValueError("block and exterior link assignments must be disjoint.")
    expected = set(range(int(model.lattice.num_links)))
    if owned.union(exterior) != expected:
        missing = sorted(expected.difference(owned.union(exterior)))
        extra = sorted(owned.union(exterior).difference(expected))
        raise ValueError(
            "block plus exterior assignments must cover every model link; "
            f"missing={missing[:8]}, extra={extra[:8]}."
        )


def _uniform_couplings_are_repeatable(model: SquareQDMModel) -> bool:
    return bool(np.isscalar(model.coup_kin) and np.isscalar(model.coup_pot))


@dataclass(frozen=True, slots=True)
class SquareQDMPeriodicProductInstance:
    """One finite member of a periodically repeated factorized cage family."""

    model: SquareQDMModel
    blocks: tuple[LocalQDMCageBlock, ...]
    padding: FactorizedLocalQDMPadding
    repeats: int
    repeat_axis: SquareQDMRepeatAxis

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            raise ValueError("repeats must be positive.")
        _validate_complete_product_assignment(self.model, self.blocks, self.padding)

    @property
    def formal_support_size(self) -> int:
        return int(np.prod([block.support_size for block in self.blocks], dtype=object))


@dataclass(frozen=True, slots=True)
class SquareQDMPeriodicProductUnitCell:
    """Coordinate-level unit cell for an arbitrary-repeat square-QDM cage.

    The unit cell may contain several coherent blocks and a fixed exterior.
    Repetition is exact at the link-pattern level; no global basis or Cartesian
    product support is formed.
    """

    model: SquareQDMModel
    blocks: tuple[LocalQDMCageBlock, ...]
    padding: FactorizedLocalQDMPadding
    repeat_axis: SquareQDMRepeatAxis = "y"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, SquareQDMModel):
            raise TypeError("model must be a SquareQDMModel.")
        if not isinstance(self.model.lattice, SquareLattice):
            raise TypeError("model must use SquareLattice geometry.")
        if self.model.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("periodic product sequences require periodic boundary conditions.")
        if self.repeat_axis not in ("x", "y"):
            raise ValueError("repeat_axis must be 'x' or 'y'.")
        if not self.blocks:
            raise ValueError("blocks must not be empty.")
        _validate_complete_product_assignment(self.model, self.blocks, self.padding)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_padding(
        cls,
        model: SquareQDMModel,
        block_pool: Sequence[LocalQDMCageBlock],
        padding: FactorizedLocalQDMPadding | MultiLocalQDMPadding,
        *,
        repeat_axis: SquareQDMRepeatAxis = "y",
        metadata: Mapping[str, object] | None = None,
    ) -> SquareQDMPeriodicProductUnitCell:
        factorized = _as_factorized_padding(padding)
        block_by_id = {int(block.block_id): block for block in block_pool}
        try:
            selected = tuple(block_by_id[int(block_id)] for block_id in factorized.block_ids)
        except KeyError as exc:
            raise ValueError(f"padding references an unknown block id: {exc.args[0]}.") from exc
        return cls(
            model=model,
            blocks=selected,
            padding=factorized,
            repeat_axis=repeat_axis,
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def repeat_period(self) -> int:
        return int(self.model.lx if self.repeat_axis == "x" else self.model.ly)

    @property
    def sites_per_unit_cell(self) -> int:
        return int(self.model.lattice.num_sites)

    @property
    def support_size_per_unit_cell(self) -> int:
        return int(np.prod([block.support_size for block in self.blocks], dtype=object))

    def with_couplings(
        self,
        *,
        coup_kin: object | None = None,
        coup_pot: object | None = None,
    ) -> SquareQDMPeriodicProductUnitCell:
        """Reuse the geometric cage cell with new translation-invariant couplings.

        This is useful when a cage was discovered in a model with a diagnostic
        potential term but the thermodynamic ETH comparison is made for the pure
        kinetic Hamiltonian.  The local sequence certificate is recomputed after
        the replacement; no spectral property is assumed to survive automatically.
        """
        updates: dict[str, object] = {}
        if coup_kin is not None:
            updates["coup_kin"] = coup_kin
        if coup_pot is not None:
            updates["coup_pot"] = coup_pot
        if not updates:
            return self
        return replace(self, model=replace(self.model, **updates))

    def instantiate(self, repeats: int) -> SquareQDMPeriodicProductInstance:
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        if not _uniform_couplings_are_repeatable(self.model):
            raise ValueError(
                "arbitrary repetition currently requires scalar, translation-invariant "
                "kinetic and potential couplings."
            )

        dimensions = {
            "lx": int(self.model.lx * repeats if self.repeat_axis == "x" else self.model.lx),
            "ly": int(self.model.ly * repeats if self.repeat_axis == "y" else self.model.ly),
            "winding_x": None,
            "winding_y": None,
        }
        target_model = replace(self.model, **dimensions)
        target_lookup = _link_lookup(target_model)
        period = self.repeat_period

        translated_blocks: list[LocalQDMCageBlock] = []
        for repeat_index in range(repeats):
            offset = repeat_index * period
            for source_block in self.blocks:
                link_ids = np.asarray(
                    [
                        target_lookup[
                            _translate_link_coordinate(
                                _link_coordinate(self.model, int(link_id)),
                                axis=self.repeat_axis,
                                offset=offset,
                            )
                        ]
                        for link_id in source_block.link_ids
                    ],
                    dtype=np.int64,
                )

                def translated_plaquettes(
                    ids: npt.ArrayLike,
                    *,
                    repeat_offset: int = offset,
                ) -> npt.NDArray[np.int64]:
                    result: list[int] = []
                    for plaquette_id in np.asarray(ids, dtype=np.int64):
                        x, y = _translate_coordinate(
                            _plaquette_coordinate(self.model, int(plaquette_id)),
                            axis=self.repeat_axis,
                            offset=repeat_offset,
                        )
                        result.append(int(target_model.lattice.plaquette_id_from_cell(x, y)))
                    return np.asarray(result, dtype=np.int64)

                translated_record = LocalQDMCageRecord(
                    cage_state=source_block.record.cage_state,
                    signature=source_block.record.signature,
                    candidate=source_block.record.candidate,
                    support_configs=source_block.support_configs.copy(),
                    local_link_ids=link_ids,
                    active_plaquette_ids=translated_plaquettes(
                        source_block.record.active_plaquette_ids
                    ),
                    scoring_plaquette_ids=translated_plaquettes(
                        source_block.record.scoring_plaquette_ids
                    ),
                    unresolved_boundary_plaquette_ids=translated_plaquettes(
                        source_block.record.unresolved_boundary_plaquette_ids
                    ),
                )
                translated_blocks.append(
                    make_qdm_cage_block(
                        target_model,
                        translated_record,
                        block_id=len(translated_blocks),
                        guard_plaquette_ids=translated_plaquettes(source_block.guard_plaquette_ids),
                    )
                )

        exterior_link_ids: list[int] = []
        exterior_config: list[int] = []
        for repeat_index in range(repeats):
            offset = repeat_index * period
            for link_id, value in zip(
                self.padding.exterior_link_ids,
                self.padding.exterior_config,
                strict=True,
            ):
                coordinate = _translate_link_coordinate(
                    _link_coordinate(self.model, int(link_id)),
                    axis=self.repeat_axis,
                    offset=offset,
                )
                exterior_link_ids.append(int(target_lookup[coordinate]))
                exterior_config.append(int(value))

        padding = FactorizedLocalQDMPadding(
            block_ids=tuple(block.block_id for block in translated_blocks),
            exterior_link_ids=np.asarray(exterior_link_ids, dtype=np.int64),
            exterior_config=np.asarray(exterior_config, dtype=np.int64),
        )
        return SquareQDMPeriodicProductInstance(
            model=target_model,
            blocks=tuple(translated_blocks),
            padding=padding,
            repeats=int(repeats),
            repeat_axis=self.repeat_axis,
        )


@dataclass(frozen=True, slots=True)
class QDMIndependentBlockCertificate:
    """Local eigenvalue certificate for one coherent product factor."""

    block_id: int
    n_plaquettes: int
    kinetic_eigenvalue: complex
    potential_eigenvalue: complex
    kinetic_residual: float
    potential_residual: float
    leakage_residual: float

    @property
    def energy(self) -> complex:
        return complex(self.kinetic_eigenvalue + self.potential_eigenvalue)

    def is_certified(self, *, tolerance: float) -> bool:
        return bool(
            self.kinetic_residual <= tolerance
            and self.potential_residual <= tolerance
            and self.leakage_residual <= tolerance
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "n_plaquettes": self.n_plaquettes,
            "kinetic_eigenvalue": self.kinetic_eigenvalue,
            "potential_eigenvalue": self.potential_eigenvalue,
            "energy": self.energy,
            "kinetic_residual": self.kinetic_residual,
            "potential_residual": self.potential_residual,
            "leakage_residual": self.leakage_residual,
        }


@dataclass(frozen=True, slots=True)
class QDMPeriodicInstanceCertificate:
    """Local-decomposition certificate for one finite repeated instance."""

    repeats: int
    block_certificates: tuple[QDMIndependentBlockCertificate, ...]
    n_exterior_only_plaquettes: int
    n_multi_block_plaquettes: int
    n_inert_pattern_checks: int
    n_flippable_inert_patterns: int
    max_site_constraint_residual: int
    energy: complex
    winding_sector: tuple[int, int] | None
    tolerance: float

    @property
    def is_certified(self) -> bool:
        return bool(
            self.max_site_constraint_residual == 0
            and self.n_flippable_inert_patterns == 0
            and self.winding_sector is not None
            and all(
                certificate.is_certified(tolerance=self.tolerance)
                for certificate in self.block_certificates
            )
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeats": self.repeats,
            "n_blocks": len(self.block_certificates),
            "n_exterior_only_plaquettes": self.n_exterior_only_plaquettes,
            "n_multi_block_plaquettes": self.n_multi_block_plaquettes,
            "n_inert_pattern_checks": self.n_inert_pattern_checks,
            "n_flippable_inert_patterns": self.n_flippable_inert_patterns,
            "max_site_constraint_residual": self.max_site_constraint_residual,
            "energy": self.energy,
            "winding_sector": self.winding_sector,
            "tolerance": self.tolerance,
            "is_certified": self.is_certified,
            "block_certificates": tuple(
                certificate.to_summary_dict() for certificate in self.block_certificates
            ),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMPeriodicSequenceCertificate:
    """Certificate for an exact one-axis infinite sequence of cage states."""

    unit_cell: SquareQDMPeriodicProductUnitCell
    finite_checks: tuple[QDMPeriodicInstanceCertificate, ...]
    verification_repeats: int
    minimum_proven_repeats: int
    unit_cell_energy: complex
    energy_density: float
    unit_cell_winding_sector: tuple[int, int] | None
    tolerance: float
    proof_statement: str

    @property
    def is_certified(self) -> bool:
        return bool(
            self.verification_repeats >= 3
            and _uniform_couplings_are_repeatable(self.unit_cell.model)
            and all(report.is_certified for report in self.finite_checks)
        )

    @property
    def support_size_per_unit_cell(self) -> int:
        return self.unit_cell.support_size_per_unit_cell

    def formal_support_size(self, repeats: int) -> int:
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        return int(self.support_size_per_unit_cell**repeats)

    def energy_for_repeats(self, repeats: int) -> complex:
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        return complex(repeats * self.unit_cell_energy)

    def winding_sector_for_repeats(self, repeats: int) -> tuple[int, int] | None:
        """Return the electric winding label of the repeated family member.

        Translation by an even unit-cell period preserves the staggered electric
        convention.  Repeating along y adds the x-cut flux of each copy while
        leaving the y-cut flux fixed; repeating along x gives the converse.
        """
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        if self.unit_cell_winding_sector is None:
            return None
        winding_x, winding_y = self.unit_cell_winding_sector
        if self.unit_cell.repeat_axis == "y":
            return int(repeats * winding_x), int(winding_y)
        return int(winding_x), int(repeats * winding_y)

    @property
    def is_symmetry_resolved(self) -> bool:
        return self.unit_cell_winding_sector is not None

    def match_energy_density(
        self,
        thermal_energy_density: float,
        *,
        tolerance: float = 1.0e-8,
        comparator: str = "beta_zero",
        metadata: Mapping[str, object] | None = None,
    ) -> EnergyDensityMatchReport:
        return EnergyDensityMatchReport(
            cage_energy_density=self.energy_density,
            thermal_energy_density=float(thermal_energy_density),
            tolerance=tolerance,
            comparator=comparator,
            metadata={} if metadata is None else dict(metadata),
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "is_certified": self.is_certified,
            "repeat_axis": self.unit_cell.repeat_axis,
            "verification_repeats": self.verification_repeats,
            "minimum_proven_repeats": self.minimum_proven_repeats,
            "unit_cell_energy": self.unit_cell_energy,
            "energy_density": self.energy_density,
            "unit_cell_winding_sector": self.unit_cell_winding_sector,
            "is_symmetry_resolved": self.is_symmetry_resolved,
            "support_size_per_unit_cell": self.support_size_per_unit_cell,
            "proof_statement": self.proof_statement,
            "finite_checks": tuple(report.to_summary_dict() for report in self.finite_checks),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMPeriodicWitnessCertificate:
    """Local annihilation certificate propagated to the repeated sequence."""

    witness: LocalWitness
    touched_block_ids: tuple[int, ...]
    annihilation_residual: float
    q_expectation: float
    tolerance: float
    sequence_is_certified: bool
    support_crosses_repeat_seam: bool

    @property
    def is_annihilated(self) -> bool:
        return self.annihilation_residual <= self.tolerance

    @property
    def is_infinite_sequence_witness(self) -> bool:
        return bool(
            self.sequence_is_certified
            and self.is_annihilated
            and not self.support_crosses_repeat_seam
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "touched_block_ids": self.touched_block_ids,
            "annihilation_residual": self.annihilation_residual,
            "q_expectation": self.q_expectation,
            "tolerance": self.tolerance,
            "is_annihilated": self.is_annihilated,
            "sequence_is_certified": self.sequence_is_certified,
            "support_crosses_repeat_seam": self.support_crosses_repeat_seam,
            "is_infinite_sequence_witness": self.is_infinite_sequence_witness,
            "q_operator_norm": self.witness.q_operator_norm,
        }


def _constant_product_winding_value(
    instance: SquareQDMPeriodicProductInstance,
    direction: Literal["x", "y"],
) -> int | None:
    """Evaluate one electric winding cut without materializing product support."""
    cut = SquareQDMElectricWindingSector.cut_data(
        layout=instance.model.layout,
        lattice=instance.model.lattice,
        direction=direction,
    )
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(instance.blocks)
        for link_id in block.link_ids
    }
    positions = tuple(
        {int(link_id): position for position, link_id in enumerate(block.link_ids)}
        for block in instance.blocks
    )

    total = 0
    contributions_by_block: dict[int, list[tuple[int, int]]] = {}
    for link_id, sign in zip(cut.link_ids, cut.signs, strict=True):
        link_id = int(link_id)
        sign = int(sign)
        owner = owner_by_link.get(link_id)
        if owner is None:
            total += sign * (2 * exterior[link_id] - 1)
        else:
            contributions_by_block.setdefault(owner, []).append((link_id, sign))

    for block_index, terms in contributions_by_block.items():
        block = instance.blocks[block_index]
        values = {
            sum(
                sign
                * (
                    2 * int(block.support_configs[support_index, positions[block_index][link_id]])
                    - 1
                )
                for link_id, sign in terms
            )
            for support_index in range(block.support_size)
        }
        if len(values) != 1:
            return None
        total += values.pop()
    return int(total)


def _constant_product_winding_sector(
    instance: SquareQDMPeriodicProductInstance,
) -> tuple[int, int] | None:
    winding_x = _constant_product_winding_value(instance, "x")
    winding_y = _constant_product_winding_value(instance, "y")
    if winding_x is None or winding_y is None:
        return None
    return int(winding_x), int(winding_y)


def _constant_product_constraint_residual(
    instance: SquareQDMPeriodicProductInstance,
) -> int:
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    residual = 0
    for site_id in range(int(instance.model.lattice.num_sites)):
        count = sum(int(block.site_counts[site_id]) for block in instance.blocks)
        count += sum(
            exterior.get(int(link_id), 0)
            for link_id in instance.model.lattice.incident_links(site_id)
        )
        residual = max(residual, abs(count - int(instance.model.required_count)))
    return int(residual)


def _block_local_action_certificate(
    instance: SquareQDMPeriodicProductInstance,
    block_index: int,
    actions: Sequence[object],
) -> QDMIndependentBlockCertificate:
    block = instance.blocks[block_index]
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    local_position = {int(link_id): position for position, link_id in enumerate(block.link_ids)}
    action_vector: dict[_ActionKey, complex] = {}
    potential_values: list[complex] = []

    for config, amplitude in zip(block.support_configs, block.amplitudes, strict=True):
        potential_value = 0.0 + 0.0j
        for action in actions:
            local_values = np.asarray(
                [
                    (
                        int(config[local_position[int(link_id)]])
                        if int(link_id) in local_position
                        else int(exterior[int(link_id)])
                    )
                    for link_id in action.links
                ],
                dtype=np.int64,
            )
            if np.array_equal(local_values, action.pattern0):
                target_pattern = action.pattern1
                coefficient = complex(action.forward)
            elif np.array_equal(local_values, action.pattern1):
                target_pattern = action.pattern0
                coefficient = complex(action.backward)
            else:
                continue

            potential_value += complex(action.potential)
            target_config = np.asarray(config, dtype=np.int64).copy()
            exterior_delta: list[tuple[int, int]] = []
            for link_id, value in zip(action.links, target_pattern, strict=True):
                link_id = int(link_id)
                value = int(value)
                if link_id in local_position:
                    target_config[local_position[link_id]] = value
                elif value != exterior[link_id]:
                    exterior_delta.append((link_id, value))
            key: _ActionKey = (
                tuple(int(value) for value in target_config),
                tuple(sorted(exterior_delta)),
            )
            action_vector[key] = action_vector.get(key, 0.0 + 0.0j) + (
                coefficient * complex(amplitude)
            )
        potential_values.append(potential_value)

    kinetic_eigenvalue = sum(
        np.conj(amplitude)
        * action_vector.get((tuple(int(value) for value in config), ()), 0.0 + 0.0j)
        for config, amplitude in zip(block.support_configs, block.amplitudes, strict=True)
    )
    residual_vector = dict(action_vector)
    for config, amplitude in zip(block.support_configs, block.amplitudes, strict=True):
        key = (tuple(int(value) for value in config), ())
        residual_vector[key] = residual_vector.get(key, 0.0 + 0.0j) - (
            kinetic_eigenvalue * complex(amplitude)
        )
    kinetic_residual = float(np.sqrt(sum(abs(value) ** 2 for value in residual_vector.values())))
    leakage_residual = float(
        np.sqrt(
            sum(
                abs(value) ** 2
                for (_target, exterior_delta), value in action_vector.items()
                if exterior_delta
            )
        )
    )

    potential_arr = np.asarray(potential_values, dtype=np.complex128)
    probabilities = np.abs(block.amplitudes) ** 2
    potential_eigenvalue = complex(np.sum(probabilities * potential_arr))
    potential_residual = float(
        np.linalg.norm((potential_arr - potential_eigenvalue) * block.amplitudes)
    )
    return QDMIndependentBlockCertificate(
        block_id=int(block.block_id),
        n_plaquettes=len(actions),
        kinetic_eigenvalue=complex(kinetic_eigenvalue),
        potential_eigenvalue=potential_eigenvalue,
        kinetic_residual=kinetic_residual,
        potential_residual=potential_residual,
        leakage_residual=leakage_residual,
    )


def _inert_plaquette_pattern_counts(
    instance: SquareQDMPeriodicProductInstance,
    inert_actions: Sequence[tuple[object, tuple[int, ...]]],
) -> tuple[int, int]:
    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(instance.blocks)
        for link_id in block.link_ids
    }
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    position_by_block = tuple(
        {int(link_id): position for position, link_id in enumerate(block.link_ids)}
        for block in instance.blocks
    )

    checked = 0
    flippable = 0
    for action, owners in inert_actions:
        support_ranges = [range(instance.blocks[index].support_size) for index in owners]
        combinations = itertools.product(*support_ranges) if support_ranges else [()]
        for support_indices in combinations:
            selected = dict(zip(owners, support_indices, strict=True))
            values: list[int] = []
            for link_id in action.links:
                link_id = int(link_id)
                owner = owner_by_link.get(link_id)
                if owner is None:
                    values.append(exterior[link_id])
                else:
                    values.append(
                        int(
                            instance.blocks[owner].support_configs[
                                selected[owner],
                                position_by_block[owner][link_id],
                            ]
                        )
                    )
            checked += 1
            values_arr = np.asarray(values, dtype=np.int64)
            if np.array_equal(values_arr, action.pattern0) or np.array_equal(
                values_arr, action.pattern1
            ):
                flippable += 1
    return checked, flippable


def certify_square_qdm_periodic_product_instance(
    instance: SquareQDMPeriodicProductInstance,
    *,
    tolerance: float = 1.0e-9,
) -> QDMPeriodicInstanceCertificate:
    """Certify a finite repeated product through strictly local identities."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(instance.blocks)
        for link_id in block.link_ids
    }
    actions_by_block: dict[int, list[object]] = {index: [] for index in range(len(instance.blocks))}
    inert_actions: list[tuple[object, tuple[int, ...]]] = []
    exterior_only = 0
    multi_block = 0
    for action in _qdm_global_plaquette_actions(instance.model):
        owners = tuple(
            sorted(
                {
                    owner_by_link[int(link_id)]
                    for link_id in action.links
                    if int(link_id) in owner_by_link
                }
            )
        )
        if len(owners) == 1:
            actions_by_block[owners[0]].append(action)
        else:
            inert_actions.append((action, owners))
            if owners:
                multi_block += 1
            else:
                exterior_only += 1

    block_certificates = tuple(
        _block_local_action_certificate(instance, index, actions_by_block[index])
        for index in range(len(instance.blocks))
    )
    n_checks, n_flippable = _inert_plaquette_pattern_counts(instance, inert_actions)
    energy = sum(
        (certificate.energy for certificate in block_certificates),
        start=0.0 + 0.0j,
    )
    return QDMPeriodicInstanceCertificate(
        repeats=instance.repeats,
        block_certificates=block_certificates,
        n_exterior_only_plaquettes=exterior_only,
        n_multi_block_plaquettes=multi_block,
        n_inert_pattern_checks=n_checks,
        n_flippable_inert_patterns=n_flippable,
        max_site_constraint_residual=_constant_product_constraint_residual(instance),
        energy=complex(energy),
        winding_sector=_constant_product_winding_sector(instance),
        tolerance=float(tolerance),
    )


def materialize_square_qdm_periodic_product_state(
    instance: SquareQDMPeriodicProductInstance,
    basis_configs: npt.ArrayLike,
    *,
    normalize: bool = True,
    tolerance: float = 1.0e-12,
) -> npt.NDArray[np.complex128]:
    """Embed a modest repeated product cage in an enumerated QDM basis.

    The coordinate-level sequence certificate avoids an exponentially large
    Cartesian product.  For ED-accessible repeat counts, this helper explicitly
    forms that product and maps every complete dimer covering to the supplied
    basis ordering.  It is intended for microcanonical overlap checks and
    figure generation, not for proving the arbitrary-repeat sequence.
    """
    configs = np.asarray(basis_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_states, n_links).")
    if configs.shape[1] != int(instance.model.lattice.num_links):
        raise ValueError("basis_configs has the wrong number of link variables.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    lookup = {tuple(int(value) for value in config): index for index, config in enumerate(configs)}
    if len(lookup) != configs.shape[0]:
        raise ValueError("basis_configs must not contain duplicate configurations.")

    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    state = np.zeros(configs.shape[0], dtype=np.complex128)
    support_ranges = tuple(range(block.support_size) for block in instance.blocks)
    for support_indices in itertools.product(*support_ranges):
        complete = np.zeros(configs.shape[1], dtype=np.int64)
        for link_id, value in exterior.items():
            complete[link_id] = value
        amplitude = 1.0 + 0.0j
        for block, support_index in zip(instance.blocks, support_indices, strict=True):
            complete[np.asarray(block.link_ids, dtype=np.int64)] = block.support_configs[
                support_index
            ]
            amplitude *= complex(block.amplitudes[support_index])
        key = tuple(int(value) for value in complete)
        try:
            basis_index = lookup[key]
        except KeyError as exc:
            raise ValueError(
                "a product-support configuration is absent from the supplied basis sector."
            ) from exc
        state[basis_index] += amplitude

    norm = float(np.linalg.norm(state))
    if norm <= tolerance:
        raise ValueError("materialized product state has zero norm.")
    if normalize:
        state /= norm
    return state


def certify_square_qdm_periodic_product_sequence(
    unit_cell: SquareQDMPeriodicProductUnitCell,
    *,
    verification_repeats: int = 3,
    check_smaller_repeats: bool = True,
    tolerance: float = 1.0e-9,
) -> SquareQDMPeriodicSequenceCertificate:
    """Prove an arbitrary-repeat one-axis cage sequence from local identities.

    Three repeats expose a left neighbor, a central unit, and a right neighbor.
    Because square-QDM plaquette terms have range one and the construction is an
    exact coordinate translation with uniform couplings, the verified local
    action classes then repeat for every larger system.  Smaller periodic rings
    are checked separately by default because they can identify neighboring
    copies through the short circumference.
    """
    if verification_repeats < 3:
        raise ValueError("verification_repeats must be at least three.")
    if not _uniform_couplings_are_repeatable(unit_cell.model):
        raise ValueError("sequence certification requires scalar translation-invariant couplings.")

    repeat_counts = (
        tuple(range(1, verification_repeats + 1))
        if check_smaller_repeats
        else (verification_repeats,)
    )
    finite_checks = tuple(
        certify_square_qdm_periodic_product_instance(
            unit_cell.instantiate(repeats),
            tolerance=tolerance,
        )
        for repeats in repeat_counts
    )
    verification = next(
        report for report in finite_checks if report.repeats == verification_repeats
    )
    unit_energy = complex(verification.energy / verification_repeats)
    if abs(unit_energy.imag) > max(tolerance, 1.0e-12):
        raise ValueError("certified sequence has a non-real energy density.")
    density = float(unit_energy.real / unit_cell.sites_per_unit_cell)
    unit_winding = finite_checks[0].winding_sector
    if unit_cell.repeat_period % 2 != 0:
        unit_winding = None
    if unit_winding is not None:
        winding_x, winding_y = unit_winding
        for report in finite_checks:
            predicted = (
                (report.repeats * winding_x, winding_y)
                if unit_cell.repeat_axis == "y"
                else (winding_x, report.repeats * winding_y)
            )
            if report.winding_sector != predicted:
                unit_winding = None
                break

    all_small_repeats = bool(
        check_smaller_repeats
        and tuple(report.repeats for report in finite_checks)
        == tuple(range(1, verification_repeats + 1))
        and all(report.is_certified for report in finite_checks)
    )
    minimum_proven = 1 if all_small_repeats else verification_repeats
    statement = (
        "All dimer constraints and electric winding labels are support-independent; "
        "every plaquette touching multiple coherent factors is inactive for every "
        "local support pattern; and the plaquettes touching one factor close on that "
        "factor with a fixed eigenvalue.  Exact periodic translation and range-one "
        "plaquette terms therefore certify every "
        f"repeat count n >= {verification_repeats}."
    )
    if all_small_repeats:
        statement += (
            f" The separately checked rings 1 <= n < {verification_repeats} are also "
            "exact, so the family is certified for every positive integer n."
        )
    return SquareQDMPeriodicSequenceCertificate(
        unit_cell=unit_cell,
        finite_checks=finite_checks,
        verification_repeats=int(verification_repeats),
        minimum_proven_repeats=int(minimum_proven),
        unit_cell_energy=unit_energy,
        energy_density=density,
        unit_cell_winding_sector=unit_winding,
        tolerance=float(tolerance),
        proof_statement=statement,
    )


def _witness_crosses_repeat_seam(
    unit_cell: SquareQDMPeriodicProductUnitCell,
    witness: LocalWitness,
) -> bool:
    coordinates = [
        _link_coordinate(unit_cell.model, int(link_id)) for link_id in witness.variable_indices
    ]
    values = [
        coordinate[0] if unit_cell.repeat_axis == "x" else coordinate[1]
        for coordinate in coordinates
    ]
    period = unit_cell.repeat_period
    return bool(0 in values and period - 1 in values)


def certify_local_witness_on_square_qdm_periodic_sequence(
    sequence: SquareQDMPeriodicSequenceCertificate,
    witness: LocalWitness,
    *,
    normalization: WitnessNormalization = "operator_norm",
    tolerance: float = 1.0e-10,
) -> SquareQDMPeriodicWitnessCertificate:
    """Verify ``L_R |Psi_n> = 0`` from one unit-cell-local calculation."""
    unit_cell = sequence.unit_cell
    if max(witness.variable_indices, default=-1) >= int(unit_cell.model.lattice.num_links):
        raise ValueError("witness variable indices do not belong to the unit-cell model.")
    normalized = LocalWitness(
        template=witness.template.normalized(normalization),
        variable_indices=witness.variable_indices,
    )
    crosses_seam = _witness_crosses_repeat_seam(unit_cell, normalized)

    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(unit_cell.blocks)
        for link_id in block.link_ids
    }
    touched = tuple(
        sorted(
            {
                owner_by_link[int(link_id)]
                for link_id in normalized.variable_indices
                if int(link_id) in owner_by_link
            }
        )
    )
    support_ranges = [range(unit_cell.blocks[index].support_size) for index in touched]
    combinations = itertools.product(*support_ranges) if support_ranges else [()]
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            unit_cell.padding.exterior_link_ids,
            unit_cell.padding.exterior_config,
            strict=True,
        )
    }
    local_pattern_to_index = {
        pattern: index for index, pattern in enumerate(normalized.local_patterns)
    }
    output: dict[tuple[int, ...], complex] = {}

    for support_indices in combinations:
        selected = dict(zip(touched, support_indices, strict=True))
        config = np.zeros(int(unit_cell.model.lattice.num_links), dtype=np.int64)
        for link_id, value in exterior.items():
            config[link_id] = value
        amplitude = 1.0 + 0.0j
        for block_index, block in enumerate(unit_cell.blocks):
            support_index = selected.get(block_index, 0)
            config[block.link_ids] = block.support_configs[support_index]
            if block_index in selected:
                amplitude *= complex(block.amplitudes[support_index])

        source_pattern = tuple(int(config[index]) for index in normalized.variable_indices)
        source_index = local_pattern_to_index.get(source_pattern)
        if source_index is None:
            continue
        for target_index, target_pattern in enumerate(normalized.local_patterns):
            coefficient = complex(normalized.local_operator[target_index, source_index])
            if coefficient == 0.0:
                continue
            target = config.copy()
            target[np.asarray(normalized.variable_indices, dtype=np.int64)] = np.asarray(
                target_pattern,
                dtype=np.int64,
            )
            if any(
                int(np.sum(target[unit_cell.model.lattice.incident_links(site_id)]))
                != int(unit_cell.model.required_count)
                for site_id in range(int(unit_cell.model.lattice.num_sites))
            ):
                continue
            key = tuple(int(value) for value in target)
            output[key] = output.get(key, 0.0 + 0.0j) + coefficient * amplitude

    residual = float(np.sqrt(sum(abs(value) ** 2 for value in output.values())))
    return SquareQDMPeriodicWitnessCertificate(
        witness=normalized,
        touched_block_ids=tuple(int(unit_cell.blocks[index].block_id) for index in touched),
        annihilation_residual=residual,
        q_expectation=float(residual**2),
        tolerance=float(tolerance),
        sequence_is_certified=sequence.is_certified,
        support_crosses_repeat_seam=crosses_seam,
    )
