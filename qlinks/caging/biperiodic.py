from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.caging.local_search_global import _qdm_global_plaquette_actions
from qlinks.caging.local_search_padding import (
    factorized_qdm_padding_from_multi_padding,
    iter_multi_qdm_block_paddings,
)
from qlinks.caging.local_search_types import (
    FactorizedLocalQDMPadding,
    LocalQDMCageBlock,
    LocalQDMMultiPaddingConfig,
    MultiLocalQDMPadding,
)
from qlinks.caging.periodic_sequence import (
    QDMIndependentBlockCertificate,
    _block_local_action_certificate,
    _constant_product_winding_sector,
    _inert_plaquette_pattern_counts,
    _link_coordinate,
    _link_lookup,
    _plaquette_coordinate,
    _uniform_couplings_are_repeatable,
    _validate_complete_product_assignment,
)
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models import SquareQDMModel

_PlaquetteClass = Literal["internal", "x_seam", "y_seam", "corner"]
_SiteClass = Literal["internal", "x_seam", "y_seam", "corner"]


def _as_factorized_padding(
    padding: FactorizedLocalQDMPadding | MultiLocalQDMPadding,
) -> FactorizedLocalQDMPadding:
    if isinstance(padding, MultiLocalQDMPadding):
        return factorized_qdm_padding_from_multi_padding(padding)
    return padding


def _translate_link_coordinate_2d(
    coordinate: tuple[int, int, str],
    *,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, str]:
    x, y, kind = coordinate
    return int(x + offset_x), int(y + offset_y), kind


def _translate_plaquette_ids_2d(
    source_model: SquareQDMModel,
    target_model: SquareQDMModel,
    plaquette_ids: npt.ArrayLike,
    *,
    offset_x: int,
    offset_y: int,
) -> npt.NDArray[np.int64]:
    translated: list[int] = []
    for plaquette_id in np.asarray(plaquette_ids, dtype=np.int64):
        x, y = _plaquette_coordinate(source_model, int(plaquette_id))
        translated.append(
            int(target_model.lattice.plaquette_id_from_cell(x + offset_x, y + offset_y))
        )
    return np.asarray(translated, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class QDMBiperiodicPlacedBlock:
    """One translated tile block, retaining support-dependent boundary charge."""

    block_id: int
    source_block_id: int
    link_ids: npt.NDArray[np.int64]
    active_plaquette_ids: npt.NDArray[np.int64]
    guard_plaquette_ids: npt.NDArray[np.int64]
    support_configs: npt.NDArray[np.int64]
    amplitudes: npt.NDArray[np.complex128]
    site_counts_by_support: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        link_ids = np.asarray(self.link_ids, dtype=np.int64)
        support = np.asarray(self.support_configs, dtype=np.int64)
        amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        counts = np.asarray(self.site_counts_by_support, dtype=np.int64)
        if link_ids.ndim != 1 or np.unique(link_ids).size != link_ids.size:
            raise ValueError("link_ids must be a one-dimensional unique array.")
        if support.ndim != 2 or support.shape[1] != link_ids.size:
            raise ValueError("support_configs must align with link_ids.")
        if amplitudes.ndim != 1 or amplitudes.size != support.shape[0]:
            raise ValueError("amplitudes must have one entry per support configuration.")
        if counts.ndim != 2 or counts.shape[0] != support.shape[0]:
            raise ValueError("site_counts_by_support must have one row per support state.")
        norm = float(np.linalg.norm(amplitudes))
        if norm == 0.0:
            raise ValueError("block amplitudes must have nonzero norm.")
        object.__setattr__(self, "link_ids", link_ids.copy())
        object.__setattr__(self, "support_configs", support.copy())
        object.__setattr__(self, "amplitudes", amplitudes / norm)
        object.__setattr__(self, "site_counts_by_support", counts.copy())
        object.__setattr__(
            self,
            "active_plaquette_ids",
            np.unique(np.asarray(self.active_plaquette_ids, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "guard_plaquette_ids",
            np.unique(np.asarray(self.guard_plaquette_ids, dtype=np.int64)),
        )

    @property
    def support_size(self) -> int:
        return int(self.support_configs.shape[0])

    @property
    def has_support_independent_site_counts(self) -> bool:
        return bool(
            self.support_size <= 1
            or np.all(self.site_counts_by_support == self.site_counts_by_support[:1])
        )


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicProductInstance:
    """One finite member of a two-directionally repeated product-tile family."""

    model: SquareQDMModel
    blocks: tuple[QDMBiperiodicPlacedBlock, ...]
    padding: FactorizedLocalQDMPadding
    repeats_x: int
    repeats_y: int
    tile_lx: int
    tile_ly: int

    def __post_init__(self) -> None:
        if self.repeats_x <= 0 or self.repeats_y <= 0:
            raise ValueError("repeats_x and repeats_y must be positive.")
        if self.tile_lx <= 0 or self.tile_ly <= 0:
            raise ValueError("tile_lx and tile_ly must be positive.")
        _validate_complete_product_assignment(self.model, self.blocks, self.padding)

    @property
    def n_tiles(self) -> int:
        return int(self.repeats_x * self.repeats_y)

    @property
    def formal_support_size(self) -> int:
        return int(np.prod([block.support_size for block in self.blocks], dtype=object))


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicProductTile:
    """A finite square-QDM tile proposed for independent x/y repetition.

    The tile stores coherent local blocks and one fixed exterior on a periodic
    reference torus.  Repetition is performed by coordinate translation into a
    larger torus.  Exactness is *not* assumed: the bi-periodic certificate checks
    all dimer constraints and all seam/corner plaquette environments explicitly.
    """

    model: SquareQDMModel
    blocks: tuple[LocalQDMCageBlock, ...]
    padding: FactorizedLocalQDMPadding
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, SquareQDMModel):
            raise TypeError("model must be a SquareQDMModel.")
        if not isinstance(self.model.lattice, SquareLattice):
            raise TypeError("model must use SquareLattice geometry.")
        if self.model.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("bi-periodic product tiles require periodic boundaries.")
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
        metadata: Mapping[str, object] | None = None,
    ) -> SquareQDMBiperiodicProductTile:
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
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def support_size_per_tile(self) -> int:
        return int(np.prod([block.support_size for block in self.blocks], dtype=object))

    @property
    def sites_per_tile(self) -> int:
        return int(self.model.lattice.num_sites)

    @property
    def active_plaquette_density(self) -> float:
        active = {
            int(plaquette_id)
            for block in self.blocks
            for plaquette_id in block.active_plaquette_ids
        }
        return float(len(active) / max(1, int(self.model.lattice.num_plaquettes)))

    def with_couplings(
        self,
        *,
        coup_kin: object | None = None,
        coup_pot: object | None = None,
    ) -> SquareQDMBiperiodicProductTile:
        updates: dict[str, object] = {}
        if coup_kin is not None:
            updates["coup_kin"] = coup_kin
        if coup_pot is not None:
            updates["coup_pot"] = coup_pot
        if not updates:
            return self
        return replace(self, model=replace(self.model, **updates))

    def instantiate(self, repeats_x: int, repeats_y: int) -> SquareQDMBiperiodicProductInstance:
        """Translate the tile over an ``repeats_x`` by ``repeats_y`` torus."""
        if repeats_x <= 0 or repeats_y <= 0:
            raise ValueError("repeats_x and repeats_y must be positive.")
        if not _uniform_couplings_are_repeatable(self.model):
            raise ValueError(
                "bi-periodic repetition currently requires scalar, translation-invariant "
                "kinetic and potential couplings."
            )

        target_model = replace(
            self.model,
            lx=int(self.model.lx * repeats_x),
            ly=int(self.model.ly * repeats_y),
            winding_x=None,
            winding_y=None,
        )
        target_lookup = _link_lookup(target_model)
        translated_blocks: list[QDMBiperiodicPlacedBlock] = []

        for tile_y in range(repeats_y):
            for tile_x in range(repeats_x):
                offset_x = int(tile_x * self.model.lx)
                offset_y = int(tile_y * self.model.ly)
                for source_block in self.blocks:
                    link_ids = np.asarray(
                        [
                            target_lookup[
                                _translate_link_coordinate_2d(
                                    _link_coordinate(self.model, int(link_id)),
                                    offset_x=offset_x,
                                    offset_y=offset_y,
                                )
                            ]
                            for link_id in source_block.link_ids
                        ],
                        dtype=np.int64,
                    )
                    local_position = {
                        int(link_id): position for position, link_id in enumerate(link_ids)
                    }
                    site_counts = np.zeros(
                        (source_block.support_size, int(target_model.lattice.num_sites)),
                        dtype=np.int64,
                    )
                    for site_id in range(int(target_model.lattice.num_sites)):
                        positions = [
                            local_position[int(link_id)]
                            for link_id in target_model.lattice.incident_links(site_id)
                            if int(link_id) in local_position
                        ]
                        if positions:
                            site_counts[:, site_id] = np.sum(
                                source_block.support_configs[:, positions],
                                axis=1,
                            )
                    translated_blocks.append(
                        QDMBiperiodicPlacedBlock(
                            block_id=len(translated_blocks),
                            source_block_id=int(source_block.block_id),
                            link_ids=link_ids,
                            active_plaquette_ids=_translate_plaquette_ids_2d(
                                self.model,
                                target_model,
                                source_block.active_plaquette_ids,
                                offset_x=offset_x,
                                offset_y=offset_y,
                            ),
                            guard_plaquette_ids=_translate_plaquette_ids_2d(
                                self.model,
                                target_model,
                                source_block.guard_plaquette_ids,
                                offset_x=offset_x,
                                offset_y=offset_y,
                            ),
                            support_configs=source_block.support_configs.copy(),
                            amplitudes=source_block.amplitudes.copy(),
                            site_counts_by_support=site_counts,
                        )
                    )

        exterior_link_ids: list[int] = []
        exterior_config: list[int] = []
        for tile_y in range(repeats_y):
            for tile_x in range(repeats_x):
                offset_x = int(tile_x * self.model.lx)
                offset_y = int(tile_y * self.model.ly)
                for link_id, value in zip(
                    self.padding.exterior_link_ids,
                    self.padding.exterior_config,
                    strict=True,
                ):
                    coordinate = _translate_link_coordinate_2d(
                        _link_coordinate(self.model, int(link_id)),
                        offset_x=offset_x,
                        offset_y=offset_y,
                    )
                    exterior_link_ids.append(int(target_lookup[coordinate]))
                    exterior_config.append(int(value))

        padding = FactorizedLocalQDMPadding(
            block_ids=tuple(block.block_id for block in translated_blocks),
            exterior_link_ids=np.asarray(exterior_link_ids, dtype=np.int64),
            exterior_config=np.asarray(exterior_config, dtype=np.int64),
        )
        return SquareQDMBiperiodicProductInstance(
            model=target_model,
            blocks=tuple(translated_blocks),
            padding=padding,
            repeats_x=int(repeats_x),
            repeats_y=int(repeats_y),
            tile_lx=int(self.model.lx),
            tile_ly=int(self.model.ly),
        )


@dataclass(frozen=True, slots=True)
class QDMBiperiodicSeamDiagnostics:
    """Plaquette and dimer-constraint diagnostics grouped by tile boundary class."""

    plaquette_counts: dict[str, int]
    inert_pattern_checks: dict[str, int]
    flippable_inert_patterns: dict[str, int]
    multi_block_plaquettes: dict[str, int]
    max_site_constraint_residuals: dict[str, int]

    @property
    def total_flippable_inert_patterns(self) -> int:
        return int(sum(self.flippable_inert_patterns.values()))

    @property
    def max_site_constraint_residual(self) -> int:
        return int(max(self.max_site_constraint_residuals.values(), default=0))

    @property
    def first_failure_class(self) -> str | None:
        for name in ("corner", "x_seam", "y_seam", "internal"):
            if self.max_site_constraint_residuals.get(name, 0) != 0:
                return f"{name}_dimer_constraint"
            if self.flippable_inert_patterns.get(name, 0) != 0:
                return f"{name}_kinetic_leakage"
        return None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "plaquette_counts": dict(self.plaquette_counts),
            "inert_pattern_checks": dict(self.inert_pattern_checks),
            "flippable_inert_patterns": dict(self.flippable_inert_patterns),
            "multi_block_plaquettes": dict(self.multi_block_plaquettes),
            "max_site_constraint_residuals": dict(self.max_site_constraint_residuals),
            "total_flippable_inert_patterns": self.total_flippable_inert_patterns,
            "max_site_constraint_residual": self.max_site_constraint_residual,
            "first_failure_class": self.first_failure_class,
        }


@dataclass(frozen=True, slots=True)
class QDMBiperiodicInstanceCertificate:
    """Exact local-decomposition check for one finite tile array."""

    repeats_x: int
    repeats_y: int
    block_certificates: tuple[QDMIndependentBlockCertificate, ...]
    seam_diagnostics: QDMBiperiodicSeamDiagnostics
    energy: complex
    winding_sector: tuple[int, int] | None
    tolerance: float

    @property
    def is_certified(self) -> bool:
        return bool(
            self.seam_diagnostics.max_site_constraint_residual == 0
            and self.seam_diagnostics.total_flippable_inert_patterns == 0
            and self.winding_sector is not None
            and all(
                certificate.is_certified(tolerance=self.tolerance)
                for certificate in self.block_certificates
            )
        )

    @property
    def failure_reason(self) -> str | None:
        seam_failure = self.seam_diagnostics.first_failure_class
        if seam_failure is not None:
            return seam_failure
        for certificate in self.block_certificates:
            if certificate.leakage_residual > self.tolerance:
                return "single_block_leakage"
            if certificate.kinetic_residual > self.tolerance:
                return "single_block_kinetic_nonclosure"
            if certificate.potential_residual > self.tolerance:
                return "single_block_potential_nonuniformity"
        if self.winding_sector is None:
            return "support_dependent_winding_sector"
        return None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeats_x": self.repeats_x,
            "repeats_y": self.repeats_y,
            "n_blocks": len(self.block_certificates),
            "energy": self.energy,
            "winding_sector": self.winding_sector,
            "tolerance": self.tolerance,
            "is_certified": self.is_certified,
            "failure_reason": self.failure_reason,
            "seam_diagnostics": self.seam_diagnostics.to_summary_dict(),
            "block_certificates": tuple(
                certificate.to_summary_dict() for certificate in self.block_certificates
            ),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicSequenceCertificate:
    """Certificate for an exact two-parameter square-QDM product-tile family."""

    tile: SquareQDMBiperiodicProductTile
    finite_checks: tuple[QDMBiperiodicInstanceCertificate, ...]
    verification_repeats: int
    minimum_proven_repeats: tuple[int, int]
    tile_energy: complex
    energy_density: float
    tile_winding_sector: tuple[int, int] | None
    tolerance: float
    proof_statement: str

    @property
    def generic_environment_check(self) -> QDMBiperiodicInstanceCertificate | None:
        return next(
            (
                report
                for report in self.finite_checks
                if report.repeats_x == self.verification_repeats
                and report.repeats_y == self.verification_repeats
            ),
            None,
        )

    @property
    def is_certified(self) -> bool:
        generic = self.generic_environment_check
        return bool(
            self.verification_repeats >= 3
            and generic is not None
            and generic.is_certified
            and _uniform_couplings_are_repeatable(self.tile.model)
        )

    @property
    def is_certified_for_all_positive_repeats(self) -> bool:
        expected_pairs = {
            (repeats_x, repeats_y)
            for repeats_x in range(1, self.verification_repeats + 1)
            for repeats_y in range(1, self.verification_repeats + 1)
        }
        reports_by_pair = {
            (report.repeats_x, report.repeats_y): report for report in self.finite_checks
        }
        return bool(
            self.is_certified
            and expected_pairs.issubset(reports_by_pair)
            and all(reports_by_pair[pair].is_certified for pair in expected_pairs)
        )

    @property
    def is_true_2d_sequence(self) -> bool:
        return self.is_certified

    @property
    def support_size_per_tile(self) -> int:
        return self.tile.support_size_per_tile

    def formal_support_size(self, repeats_x: int, repeats_y: int) -> int:
        if repeats_x <= 0 or repeats_y <= 0:
            raise ValueError("repeats_x and repeats_y must be positive.")
        return int(self.support_size_per_tile ** (repeats_x * repeats_y))

    def energy_for_repeats(self, repeats_x: int, repeats_y: int) -> complex:
        if repeats_x <= 0 or repeats_y <= 0:
            raise ValueError("repeats_x and repeats_y must be positive.")
        return complex(repeats_x * repeats_y * self.tile_energy)

    def winding_sector_for_repeats(
        self,
        repeats_x: int,
        repeats_y: int,
    ) -> tuple[int, int] | None:
        if repeats_x <= 0 or repeats_y <= 0:
            raise ValueError("repeats_x and repeats_y must be positive.")
        if self.tile_winding_sector is None:
            return None
        winding_x, winding_y = self.tile_winding_sector
        return int(repeats_y * winding_x), int(repeats_x * winding_y)

    @property
    def failed_checks(self) -> tuple[QDMBiperiodicInstanceCertificate, ...]:
        return tuple(report for report in self.finite_checks if not report.is_certified)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "is_certified": self.is_certified,
            "is_true_2d_sequence": self.is_true_2d_sequence,
            "is_certified_for_all_positive_repeats": (self.is_certified_for_all_positive_repeats),
            "verification_repeats": self.verification_repeats,
            "minimum_proven_repeats": self.minimum_proven_repeats,
            "tile_energy": self.tile_energy,
            "energy_density": self.energy_density,
            "tile_winding_sector": self.tile_winding_sector,
            "support_size_per_tile": self.support_size_per_tile,
            "active_plaquette_density": self.tile.active_plaquette_density,
            "proof_statement": self.proof_statement,
            "finite_checks": tuple(report.to_summary_dict() for report in self.finite_checks),
        }


def _plaquette_class(
    instance: SquareQDMBiperiodicProductInstance,
    plaquette_id: int,
) -> _PlaquetteClass:
    x, y = _plaquette_coordinate(instance.model, plaquette_id)
    crosses_x = x % instance.tile_lx == instance.tile_lx - 1
    crosses_y = y % instance.tile_ly == instance.tile_ly - 1
    if crosses_x and crosses_y:
        return "corner"
    if crosses_x:
        return "x_seam"
    if crosses_y:
        return "y_seam"
    return "internal"


def _site_class(
    instance: SquareQDMBiperiodicProductInstance,
    site_id: int,
) -> _SiteClass:
    x, y = instance.model.lattice.sites[int(site_id)].cell
    on_x = int(x) % instance.tile_lx == 0
    on_y = int(y) % instance.tile_ly == 0
    if on_x and on_y:
        return "corner"
    if on_x:
        return "x_seam"
    if on_y:
        return "y_seam"
    return "internal"


def _constraint_residuals_by_site_class(
    instance: SquareQDMBiperiodicProductInstance,
) -> dict[str, int]:
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    residuals = {name: 0 for name in ("internal", "x_seam", "y_seam", "corner")}
    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(instance.blocks)
        for link_id in block.link_ids
    }
    for site_id in range(int(instance.model.lattice.num_sites)):
        incident_blocks = tuple(
            sorted(
                {
                    owner_by_link[int(link_id)]
                    for link_id in instance.model.lattice.incident_links(site_id)
                    if int(link_id) in owner_by_link
                }
            )
        )
        support_ranges = [range(instance.blocks[index].support_size) for index in incident_blocks]
        combinations = itertools.product(*support_ranges) if support_ranges else [()]
        exterior_count = sum(
            exterior.get(int(link_id), 0)
            for link_id in instance.model.lattice.incident_links(site_id)
        )
        name = _site_class(instance, site_id)
        for support_indices in combinations:
            count = exterior_count + sum(
                int(instance.blocks[index].site_counts_by_support[support_index, site_id])
                for index, support_index in zip(
                    incident_blocks,
                    support_indices,
                    strict=True,
                )
            )
            residuals[name] = max(
                residuals[name],
                abs(count - int(instance.model.required_count)),
            )
    return residuals


def certify_square_qdm_biperiodic_product_instance(
    instance: SquareQDMBiperiodicProductInstance,
    *,
    tolerance: float = 1.0e-9,
) -> QDMBiperiodicInstanceCertificate:
    """Certify one finite bi-periodic product-tile array by local identities."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    owner_by_link = {
        int(link_id): block_index
        for block_index, block in enumerate(instance.blocks)
        for link_id in block.link_ids
    }
    actions_by_block: dict[int, list[object]] = {index: [] for index in range(len(instance.blocks))}
    inert_by_class: dict[str, list[tuple[object, tuple[int, ...]]]] = {
        name: [] for name in ("internal", "x_seam", "y_seam", "corner")
    }
    plaquette_counts = {name: 0 for name in inert_by_class}
    multi_block = {name: 0 for name in inert_by_class}

    for action in _qdm_global_plaquette_actions(instance.model):
        name = _plaquette_class(instance, int(action.plaquette_id))
        plaquette_counts[name] += 1
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
            inert_by_class[name].append((action, owners))
            if owners:
                multi_block[name] += 1

    block_certificates = tuple(
        _block_local_action_certificate(instance, index, actions_by_block[index])
        for index in range(len(instance.blocks))
    )
    checks: dict[str, int] = {}
    flippable: dict[str, int] = {}
    for name, inert_actions in inert_by_class.items():
        checks[name], flippable[name] = _inert_plaquette_pattern_counts(
            instance,
            inert_actions,
        )

    diagnostics = QDMBiperiodicSeamDiagnostics(
        plaquette_counts=plaquette_counts,
        inert_pattern_checks=checks,
        flippable_inert_patterns=flippable,
        multi_block_plaquettes=multi_block,
        max_site_constraint_residuals=_constraint_residuals_by_site_class(instance),
    )
    energy = sum(
        (certificate.energy for certificate in block_certificates),
        start=0.0 + 0.0j,
    )
    return QDMBiperiodicInstanceCertificate(
        repeats_x=int(instance.repeats_x),
        repeats_y=int(instance.repeats_y),
        block_certificates=block_certificates,
        seam_diagnostics=diagnostics,
        energy=complex(energy),
        winding_sector=_constant_product_winding_sector(instance),
        tolerance=float(tolerance),
    )


def diagnose_square_qdm_biperiodic_repeatability(
    tile: SquareQDMBiperiodicProductTile,
    *,
    verification_repeats: int = 3,
    check_smaller_repeats: bool = True,
    tolerance: float = 1.0e-9,
) -> SquareQDMBiperiodicSequenceCertificate:
    """Diagnose whether a tile generates an exact family for arbitrary ``nx, ny``.

    A ``3 x 3`` tile array contains an interior tile, both one-direction seams,
    and four-tile corner environments.  Because square-QDM terms have range one,
    exact coordinate repetition with uniform couplings then propagates these
    identities to every larger torus.  Rings of size one and two are checked
    separately by default because periodic identifications create short-ring
    environments not present in the generic ``3 x 3`` array.
    """
    if verification_repeats < 3:
        raise ValueError("verification_repeats must be at least three.")
    if not _uniform_couplings_are_repeatable(tile.model):
        raise ValueError("bi-periodic certification requires scalar uniform couplings.")

    repeat_values = (
        tuple(range(1, verification_repeats + 1))
        if check_smaller_repeats
        else (verification_repeats,)
    )
    checks = tuple(
        certify_square_qdm_biperiodic_product_instance(
            tile.instantiate(repeats_x, repeats_y),
            tolerance=tolerance,
        )
        for repeats_y in repeat_values
        for repeats_x in repeat_values
    )
    verification = next(
        report
        for report in checks
        if report.repeats_x == verification_repeats and report.repeats_y == verification_repeats
    )
    tile_energy = complex(verification.energy / (verification_repeats**2))
    if abs(tile_energy.imag) > max(tolerance, 1.0e-12):
        raise ValueError("bi-periodic candidate has a non-real tile energy.")
    energy_density = float(tile_energy.real / tile.sites_per_tile)

    tile_winding = _constant_product_winding_sector(tile.instantiate(1, 1))
    if tile.model.lx % 2 != 0 or tile.model.ly % 2 != 0:
        tile_winding = None
    if tile_winding is not None:
        winding_x, winding_y = tile_winding
        for report in checks:
            predicted = (
                int(report.repeats_y * winding_x),
                int(report.repeats_x * winding_y),
            )
            if report.winding_sector != predicted:
                tile_winding = None
                break

    all_small = bool(
        check_smaller_repeats
        and len(checks) == verification_repeats**2
        and all(report.is_certified for report in checks)
    )
    minimum = (1, 1) if all_small else (verification_repeats, verification_repeats)
    statement = (
        "The 3 x 3 tile array checks all range-one local environments: tile interiors, "
        "x seams, y seams, and four-tile corners.  Support-independent dimer constraints, "
        "inactive multi-factor plaquettes, and fixed local block eigenvalue equations "
        "therefore propagate to arbitrary independent repeat counts nx, ny >= 3."
    )
    if all_small:
        statement += (
            " Every short torus with 1 <= nx, ny < 3 was checked separately, so the "
            "family is certified for all positive nx and ny."
        )
    return SquareQDMBiperiodicSequenceCertificate(
        tile=tile,
        finite_checks=checks,
        verification_repeats=int(verification_repeats),
        minimum_proven_repeats=minimum,
        tile_energy=tile_energy,
        energy_density=energy_density,
        tile_winding_sector=tile_winding,
        tolerance=float(tolerance),
        proof_statement=statement,
    )


def certify_square_qdm_biperiodic_product_sequence(
    tile: SquareQDMBiperiodicProductTile,
    *,
    verification_repeats: int = 3,
    check_smaller_repeats: bool = True,
    tolerance: float = 1.0e-9,
) -> SquareQDMBiperiodicSequenceCertificate:
    """Return the exact two-dimensional sequence certificate for ``tile``.

    The function returns a diagnostic object even when certification fails; use
    ``certificate.is_certified`` and ``certificate.failed_checks`` to inspect the
    obstruction rather than relying on an exception for ordinary search failure.
    """
    return diagnose_square_qdm_biperiodic_repeatability(
        tile,
        verification_repeats=verification_repeats,
        check_smaller_repeats=check_smaller_repeats,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicTileSearchConfig:
    """Budget and acceptance rules for direct periodic product-tile search."""

    min_blocks: int = 1
    max_blocks: int | None = None
    max_padding_attempts: int = 256
    max_paddings_per_packing: int = 4
    max_results: int = 64
    max_certified_results: int | None = None
    max_tile_support_size: int | None = 4096
    verification_repeats: int = 3
    check_smaller_repeats: bool = True
    require_kinetic_separation: bool = True
    require_static_exterior: bool = False
    include_sectors: bool = True
    tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be positive.")
        if self.max_blocks is not None and self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be None or at least min_blocks.")
        if self.max_padding_attempts < 0:
            raise ValueError("max_padding_attempts must be non-negative.")
        if self.max_paddings_per_packing < 0:
            raise ValueError("max_paddings_per_packing must be non-negative.")
        if self.max_results < 0:
            raise ValueError("max_results must be non-negative.")
        if self.max_certified_results is not None and self.max_certified_results < 0:
            raise ValueError("max_certified_results must be non-negative or None.")
        if self.max_tile_support_size is not None and self.max_tile_support_size < 1:
            raise ValueError("max_tile_support_size must be positive or None.")
        if self.verification_repeats < 3:
            raise ValueError("verification_repeats must be at least three.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")

    def padding_config(self) -> LocalQDMMultiPaddingConfig:
        return LocalQDMMultiPaddingConfig(
            min_blocks=self.min_blocks,
            max_blocks=self.max_blocks,
            max_paddings=self.max_results,
            max_padding_attempts=self.max_padding_attempts,
            max_paddings_per_packing=self.max_paddings_per_packing,
            include_sectors=self.include_sectors,
            require_static_exterior=self.require_static_exterior,
            tolerance=self.tolerance,
            max_product_support_size=self.max_tile_support_size,
            require_kinetic_separation=self.require_kinetic_separation,
            store_full_states=False,
        )


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicTileSearchRecord:
    """One periodic exterior completion and its two-dimensional diagnosis."""

    tile: SquareQDMBiperiodicProductTile
    certificate: SquareQDMBiperiodicSequenceCertificate
    score: float

    @property
    def block_ids(self) -> tuple[int, ...]:
        return self.tile.padding.block_ids

    @property
    def is_certified(self) -> bool:
        return self.certificate.is_certified

    @property
    def failure_reason(self) -> str | None:
        failed = self.certificate.failed_checks
        if not failed:
            return None
        return failed[0].failure_reason

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "block_ids": self.block_ids,
            "score": self.score,
            "is_certified": self.is_certified,
            "failure_reason": self.failure_reason,
            "active_plaquette_density": self.tile.active_plaquette_density,
            "support_size_per_tile": self.tile.support_size_per_tile,
        }


@dataclass(frozen=True, slots=True)
class SquareQDMBiperiodicTileSearchResult:
    """Direct product-tile search output, including informative failures."""

    records: tuple[SquareQDMBiperiodicTileSearchRecord, ...]
    config: SquareQDMBiperiodicTileSearchConfig
    n_padding_candidates_examined: int

    @property
    def certified_records(self) -> tuple[SquareQDMBiperiodicTileSearchRecord, ...]:
        return tuple(record for record in self.records if record.is_certified)

    @property
    def failed_records(self) -> tuple[SquareQDMBiperiodicTileSearchRecord, ...]:
        return tuple(record for record in self.records if not record.is_certified)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.failed_records:
            name = record.failure_reason or "unknown"
            counts[name] = counts.get(name, 0) + 1
        return counts

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_padding_candidates_examined": self.n_padding_candidates_examined,
            "n_records": len(self.records),
            "n_certified": len(self.certified_records),
            "n_failed": len(self.failed_records),
            "failure_counts": self.failure_counts,
            "records": tuple(record.to_summary_dict() for record in self.records),
        }


def _tile_search_score(certificate: SquareQDMBiperiodicSequenceCertificate) -> float:
    tile = certificate.tile
    failures = certificate.failed_checks
    seam_penalty = sum(
        report.seam_diagnostics.total_flippable_inert_patterns
        + report.seam_diagnostics.max_site_constraint_residual
        for report in failures
    )
    block_penalty = max((block.support_size for block in tile.blocks), default=1)
    certification_bonus = 1000.0 if certificate.is_certified else 0.0
    return float(
        certification_bonus
        + 100.0 * tile.active_plaquette_density
        - seam_penalty
        - 0.01 * block_penalty
    )


def search_square_qdm_biperiodic_product_tiles(
    model: SquareQDMModel,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: SquareQDMBiperiodicTileSearchConfig | None = None,
) -> SquareQDMBiperiodicTileSearchResult:
    """Search periodic static exteriors and certify two-direction repetition.

    The search only solves and materializes the finite reference-tile product
    support, bounded by ``max_tile_support_size``.  Every completion is then
    diagnosed on tile arrays through ``3 x 3`` without constructing the support
    of the repeated two-dimensional system.
    """
    search_config = SquareQDMBiperiodicTileSearchConfig() if config is None else config
    records: list[SquareQDMBiperiodicTileSearchRecord] = []
    examined = 0
    certified_count = 0

    for materialized_padding in iter_multi_qdm_block_paddings(
        model,
        block_pool,
        config=search_config.padding_config(),
        max_yielded=search_config.max_padding_attempts,
    ):
        examined += 1
        padding = factorized_qdm_padding_from_multi_padding(materialized_padding)
        tile = SquareQDMBiperiodicProductTile.from_padding(
            model,
            block_pool,
            padding,
            metadata={
                "search_candidate_index": examined - 1,
                "finite_tile_support_size": int(
                    materialized_padding.global_support_configs.shape[0]
                ),
            },
        )
        certificate = diagnose_square_qdm_biperiodic_repeatability(
            tile,
            verification_repeats=search_config.verification_repeats,
            check_smaller_repeats=search_config.check_smaller_repeats,
            tolerance=search_config.tolerance,
        )
        records.append(
            SquareQDMBiperiodicTileSearchRecord(
                tile=tile,
                certificate=certificate,
                score=_tile_search_score(certificate),
            )
        )
        if certificate.is_certified:
            certified_count += 1
            if (
                search_config.max_certified_results is not None
                and certified_count >= search_config.max_certified_results
            ):
                break
        if len(records) >= search_config.max_results:
            break

    records.sort(key=lambda record: record.score, reverse=True)
    return SquareQDMBiperiodicTileSearchResult(
        records=tuple(records),
        config=search_config,
        n_padding_candidates_examined=examined,
    )
