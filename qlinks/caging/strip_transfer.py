from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.caging.classification import CageClassificationReport
from qlinks.caging.thermodynamic import (
    LocalWitness,
    LocalWitnessFamily,
    LocalWitnessTemplate,
    WitnessNormalization,
    local_witnesses_from_classification_report,
)
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models import SquareQDMModel
from qlinks.variables import VariableKind

SquareQDMLinkKind = Literal["x", "y"]
StripBoundaryCondition = Literal["open", "periodic"]
SquareQDMWindingConvention = Literal["electric"]
WindingProjectionMethod = Literal["auto", "dynamic_programming", "fourier"]


def _bit(mask: int, index: int) -> int:
    return (int(mask) >> int(index)) & 1


def _rotate_left(mask: int, *, width: int) -> int:
    all_bits = (1 << width) - 1
    return ((int(mask) << 1) & all_bits) | (int(mask) >> (width - 1))


def _submasks(mask: int):
    current = int(mask)
    while True:
        yield current
        if current == 0:
            break
        current = (current - 1) & int(mask)


def _safe_exp(log_value: float) -> float:
    if not isfinite(log_value):
        return 0.0 if log_value < 0.0 else float("inf")
    if log_value > log(np.finfo(np.float64).max):
        return float("inf")
    return float(exp(log_value))


@dataclass(frozen=True, slots=True, order=True)
class SquareQDMStripWindingSector:
    """One electric winding sector of a periodic square-QDM strip.

    The labels use the same convention as
    :class:`qlinks.constraints.SquareQDMElectricWindingSector`.  The x label is
    measured on the x-wrapping horizontal links, while the y label is measured
    on the y-wrapping vertical links.
    """

    winding_x: int
    winding_y: int
    convention: SquareQDMWindingConvention = "electric"

    def __post_init__(self) -> None:
        if self.convention != "electric":
            raise ValueError("only the square-QDM electric winding convention is supported.")
        object.__setattr__(self, "winding_x", int(self.winding_x))
        object.__setattr__(self, "winding_y", int(self.winding_y))

    @property
    def label(self) -> tuple[int, int]:
        return (self.winding_x, self.winding_y)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "winding_x": self.winding_x,
            "winding_y": self.winding_y,
            "convention": self.convention,
        }


@dataclass(frozen=True, slots=True, order=True)
class SquareQDMLinkCoordinate:
    """One square-QDM link in strip coordinates.

    ``x`` labels the site column anchoring the link.  An ``x`` link joins
    ``(x, y)`` to ``(x + 1, y)``; a ``y`` link joins ``(x, y)`` to
    ``(x, y + 1 mod circumference)``.
    """

    x: int
    y: int
    kind: SquareQDMLinkKind

    def __post_init__(self) -> None:
        if self.x < 0:
            raise ValueError("x must be non-negative.")
        if self.y < 0:
            raise ValueError("y must be non-negative.")
        if self.kind not in ("x", "y"):
            raise ValueError("kind must be 'x' or 'y'.")


@dataclass(frozen=True, slots=True)
class SquareQDMWitnessPlacement:
    """Place a local witness on an infinite square-QDM cylinder.

    The local-variable ordering is exactly the ordering used by
    :class:`LocalWitnessTemplate`.  ``link_coordinates`` are normalized so the
    first affected site column is zero.  This makes the placement independent
    of the finite reference system from which the witness was extracted.
    """

    template: LocalWitnessTemplate
    circumference: int
    link_coordinates: tuple[SquareQDMLinkCoordinate, ...]
    reference_origin_x: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.circumference < 2:
            raise ValueError("circumference must be at least two.")

        coordinates = tuple(self.link_coordinates)
        if len(coordinates) != self.template.n_variables:
            raise ValueError(
                "link_coordinates length must match the witness width: "
                f"{len(coordinates)} != {self.template.n_variables}."
            )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("link_coordinates must not contain duplicates.")
        if min(coordinate.x for coordinate in coordinates) != 0:
            raise ValueError("link_coordinates must be normalized to start at x=0.")
        if any(coordinate.y >= self.circumference for coordinate in coordinates):
            raise ValueError("link-coordinate y values must lie inside the circumference.")

        for pattern in self.template.local_patterns:
            if any(value not in (0, 1) for value in pattern):
                raise ValueError("square-QDM strip witnesses require binary local patterns.")

        object.__setattr__(self, "link_coordinates", coordinates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def affected_sites(self) -> tuple[tuple[int, int], ...]:
        sites: set[tuple[int, int]] = set()
        for coordinate in self.link_coordinates:
            sites.add((coordinate.x, coordinate.y))
            if coordinate.kind == "x":
                sites.add((coordinate.x + 1, coordinate.y))
            else:
                sites.add((coordinate.x, (coordinate.y + 1) % self.circumference))
        return tuple(sorted(sites))

    @property
    def window_width(self) -> int:
        return 1 + max(site_x for site_x, _site_y in self.affected_sites)

    def with_circumference(self, circumference: int) -> SquareQDMWitnessPlacement:
        """Reuse the same local operator and link coordinates on a wider cylinder."""
        return SquareQDMWitnessPlacement(
            template=self.template,
            circumference=int(circumference),
            link_coordinates=self.link_coordinates,
            reference_origin_x=self.reference_origin_x,
            metadata={
                **self.metadata,
                "rescaled_from_circumference": self.circumference,
            },
        )

    def instantiate_on_model(
        self,
        model: SquareQDMModel,
        *,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> LocalWitness:
        """Embed the normalized strip placement in a finite square-QDM model."""
        lattice = model.lattice
        if not isinstance(lattice, SquareLattice):
            raise TypeError("model must use SquareLattice geometry.")
        if int(lattice.ly) != int(self.circumference):
            raise ValueError("the finite model circumference must match the witness placement.")
        periodic = lattice.boundary_condition == BoundaryCondition.PERIODIC
        lookup: dict[tuple[int, int, str], int] = {}
        for link in lattice.links:
            source_cell = lattice.sites[int(link.source)].cell
            lookup[(int(source_cell[0]), int(source_cell[1]), str(link.kind))] = int(link.id)

        variable_indices: list[int] = []
        for coordinate in self.link_coordinates:
            x = int(origin_x) + int(coordinate.x)
            y = int(origin_y) + int(coordinate.y)
            if periodic:
                x %= int(lattice.lx)
                y %= int(lattice.ly)
            key = (x, y, str(coordinate.kind))
            try:
                link_id = lookup[key]
            except KeyError as exc:
                raise ValueError(
                    f"witness coordinate {key!r} does not exist in the finite model."
                ) from exc
            variable_indices.append(int(model.layout.link_variable_index(link_id)))
        return self.template.instantiate(tuple(variable_indices))

    @classmethod
    def from_local_witness(
        cls,
        model: SquareQDMModel,
        witness: LocalWitness,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> SquareQDMWitnessPlacement:
        """Extract strip link coordinates from a finite square-QDM embedding.

        For a periodic reference lattice, the shortest unwrapped interval is
        selected automatically.  A witness crossing the finite-system x seam
        therefore becomes an ordinary bounded placement on the infinite strip.
        """
        lattice = model.lattice
        if not isinstance(lattice, SquareLattice):
            raise TypeError("model must use SquareLattice geometry.")
        if int(lattice.ly) < 2:
            raise ValueError("the reference square lattice must have ly >= 2.")

        raw_coordinates: list[SquareQDMLinkCoordinate] = []
        for variable_index in witness.variable_indices:
            spec = model.layout.spec(int(variable_index))
            if spec.kind != VariableKind.LINK:
                raise ValueError("square-QDM strip witnesses must act only on link variables.")

            link = lattice.links[int(spec.geometry_index)]
            if link.kind not in ("x", "y"):
                raise ValueError(f"unsupported square-lattice link kind: {link.kind!r}.")
            source_cell = lattice.sites[int(link.source)].cell
            raw_coordinates.append(
                SquareQDMLinkCoordinate(
                    x=int(source_cell[0]),
                    y=int(source_cell[1]) % int(lattice.ly),
                    kind=link.kind,
                )
            )

        normalized, origin_x = _normalize_square_qdm_x_coordinates(
            raw_coordinates,
            lx=int(lattice.lx),
            periodic=lattice.boundary_condition == BoundaryCondition.PERIODIC,
        )
        placement_metadata = {
            "reference_lx": int(lattice.lx),
            "reference_ly": int(lattice.ly),
            "reference_boundary_condition": lattice.boundary_condition.value,
            "reference_variable_indices": tuple(witness.variable_indices),
        }
        if metadata is not None:
            placement_metadata.update(dict(metadata))

        return cls(
            template=witness.template,
            circumference=int(lattice.ly),
            link_coordinates=normalized,
            reference_origin_x=origin_x,
            metadata=placement_metadata,
        )


@dataclass(frozen=True, slots=True)
class SquareQDMColumnTransition:
    """One allowed column of a fully packed square-lattice dimer covering."""

    incoming_mask: int
    outgoing_mask: int
    vertical_mask: int


@dataclass(frozen=True, slots=True)
class SquareQDMStripWitnessEvaluation:
    """Infinite-temperature expectation from an exact strip contraction."""

    circumference: int
    length: int
    boundary_x: StripBoundaryCondition
    insertion_x: int | None
    window_width: int
    expectation: float
    log_partition_count: float
    log_weighted_count: float
    partition_count: float
    weighted_count: float
    winding_sector: SquareQDMStripWindingSector | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "circumference": self.circumference,
            "length": self.length,
            "boundary_x": self.boundary_x,
            "insertion_x": self.insertion_x,
            "window_width": self.window_width,
            "expectation": self.expectation,
            "log_partition_count": self.log_partition_count,
            "log_weighted_count": self.log_weighted_count,
            "partition_count": self.partition_count,
            "weighted_count": self.weighted_count,
            "winding_sector": (
                None if self.winding_sector is None else self.winding_sector.to_summary_dict()
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMStripScalingReport:
    """A fixed-circumference sequence of strip witness expectations."""

    placement: SquareQDMWitnessPlacement
    evaluations: tuple[SquareQDMStripWitnessEvaluation, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.evaluations, key=lambda evaluation: evaluation.length))
        if not ordered:
            raise ValueError("evaluations must not be empty.")
        if len({evaluation.length for evaluation in ordered}) != len(ordered):
            raise ValueError("evaluations must have distinct lengths.")
        if len({evaluation.boundary_x for evaluation in ordered}) != 1:
            raise ValueError("all evaluations must use the same x boundary condition.")
        if len({evaluation.winding_sector for evaluation in ordered}) != 1:
            raise ValueError("all evaluations must use the same winding sector.")
        if any(evaluation.circumference != self.placement.circumference for evaluation in ordered):
            raise ValueError("evaluation circumferences must match the placement.")
        object.__setattr__(self, "evaluations", ordered)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(evaluation.length for evaluation in self.evaluations)

    @property
    def expectations(self) -> tuple[float, ...]:
        return tuple(evaluation.expectation for evaluation in self.evaluations)

    @property
    def boundary_x(self) -> StripBoundaryCondition:
        return self.evaluations[0].boundary_x

    @property
    def winding_sector(self) -> SquareQDMStripWindingSector | None:
        return self.evaluations[0].winding_sector

    def tail_estimate(self, *, tail_points: int = 3) -> dict[str, object]:
        """Return a descriptive tail mean and spread.

        The result is deliberately not named a thermodynamic-limit fit.  Strip
        transfer sequences can retain parity oscillations, especially at small
        circumference, so the tail spread should be inspected explicitly.
        """
        if tail_points <= 0:
            raise ValueError("tail_points must be positive.")
        tail = self.evaluations[-min(tail_points, len(self.evaluations)) :]
        values = np.asarray([evaluation.expectation for evaluation in tail], dtype=float)
        return {
            "lengths": tuple(evaluation.length for evaluation in tail),
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "spread": float(np.max(values) - np.min(values)),
        }

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "circumference": self.placement.circumference,
            "boundary_x": self.boundary_x,
            "winding_sector": (
                None if self.winding_sector is None else self.winding_sector.to_summary_dict()
            ),
            "lengths": self.lengths,
            "expectations": self.expectations,
            "evaluations": tuple(evaluation.to_summary_dict() for evaluation in self.evaluations),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMWitnessFamilyStripRecord:
    """One finite-system embedding of a common witness family on a strip."""

    system_label: Hashable
    embedding_index: int
    placement: SquareQDMWitnessPlacement
    scaling_report: SquareQDMStripScalingReport

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "system_label": self.system_label,
            "embedding_index": self.embedding_index,
            "placement": {
                "circumference": self.placement.circumference,
                "window_width": self.placement.window_width,
                "link_coordinates": self.placement.link_coordinates,
                "metadata": dict(self.placement.metadata),
            },
            "scaling_report": self.scaling_report.to_summary_dict(),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMWitnessFamilyStripReport:
    """Strip evaluations of one cage-derived local witness family."""

    family: LocalWitnessFamily
    records: tuple[SquareQDMWitnessFamilyStripRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("records must not be empty.")
        if len({record.system_label for record in self.records}) != len(self.records):
            raise ValueError("records must have distinct system labels.")

    @property
    def system_labels(self) -> tuple[Hashable, ...]:
        return tuple(record.system_label for record in self.records)

    def record_for(self, system_label: Hashable) -> SquareQDMWitnessFamilyStripRecord:
        for record in self.records:
            if record.system_label == system_label:
                return record
        raise KeyError(system_label)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "system_labels": self.system_labels,
            "template": self.family.template.to_summary_dict(),
            "records": tuple(record.to_summary_dict() for record in self.records),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMClassificationWitnessStripRecord:
    """One actual reduced-IZ witness extracted from a cage classification."""

    witness_index: int
    witness: LocalWitness
    placement: SquareQDMWitnessPlacement
    scaling_report: SquareQDMStripScalingReport

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "witness_index": self.witness_index,
            "q_operator_norm": self.witness.q_operator_norm,
            "source_zero_indices": self.witness.template.source_zero_indices,
            "placement": {
                "circumference": self.placement.circumference,
                "window_width": self.placement.window_width,
                "link_coordinates": self.placement.link_coordinates,
            },
            "scaling_report": self.scaling_report.to_summary_dict(),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMClassificationWitnessStripReport:
    """Transfer results for every trusted reduced-IZ witness of one cage."""

    records: tuple[SquareQDMClassificationWitnessStripRecord, ...]
    normalization: WitnessNormalization

    @property
    def minimum_tail_expectation(self) -> float:
        if not self.records:
            raise ValueError("records must not be empty.")
        return float(
            min(record.scaling_report.tail_estimate()["minimum"] for record in self.records)
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "normalization": self.normalization,
            "n_witnesses": len(self.records),
            "minimum_tail_expectation": (
                None if not self.records else self.minimum_tail_expectation
            ),
            "records": tuple(record.to_summary_dict() for record in self.records),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMBetaZeroEnergyDensityEvaluation:
    """Infinite-temperature square-QDM energy-density estimate."""

    circumference: int
    length: int
    horizontal_flippability: float
    vertical_flippability: float
    potential_coupling: float
    winding_sector: SquareQDMStripWindingSector | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_flippability(self) -> float:
        return float(self.horizontal_flippability + self.vertical_flippability)

    @property
    def kinetic_energy_density(self) -> float:
        return 0.0

    @property
    def potential_energy_density(self) -> float:
        return float(self.potential_coupling * self.total_flippability)

    @property
    def energy_density(self) -> float:
        return self.potential_energy_density

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "circumference": self.circumference,
            "length": self.length,
            "horizontal_flippability": self.horizontal_flippability,
            "vertical_flippability": self.vertical_flippability,
            "total_flippability": self.total_flippability,
            "kinetic_energy_density": self.kinetic_energy_density,
            "potential_energy_density": self.potential_energy_density,
            "energy_density": self.energy_density,
            "potential_coupling": self.potential_coupling,
            "winding_sector": (
                None if self.winding_sector is None else self.winding_sector.to_summary_dict()
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SquareQDMBetaZeroEnergyDensityScalingReport:
    """Two-dimensional finite-size sequence for the beta-zero energy density."""

    evaluations: tuple[SquareQDMBetaZeroEnergyDensityEvaluation, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.evaluations, key=lambda item: (item.circumference, item.length))
        )
        if not ordered:
            raise ValueError("evaluations must not be empty.")
        object.__setattr__(self, "evaluations", ordered)

    @property
    def energy_densities(self) -> tuple[float, ...]:
        return tuple(item.energy_density for item in self.evaluations)

    def tail_estimate(self, *, tail_points: int = 3) -> dict[str, object]:
        if tail_points <= 0:
            raise ValueError("tail_points must be positive.")
        tail = self.evaluations[-min(tail_points, len(self.evaluations)) :]
        values = np.asarray([item.energy_density for item in tail], dtype=float)
        return {
            "sizes": tuple((item.length, item.circumference) for item in tail),
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "spread": float(np.max(values) - np.min(values)),
        }

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "energy_densities": self.energy_densities,
            "tail_estimate": self.tail_estimate(),
            "evaluations": tuple(item.to_summary_dict() for item in self.evaluations),
        }


@dataclass(frozen=True, slots=True)
class _PeriodicTransferSpectrum:
    eigenvalues: npt.NDArray[np.float64]
    insertion_diagonal: npt.NDArray[np.float64]
    scale: float


@dataclass(frozen=True, slots=True)
class _ChargeResolvedTrace:
    log_counts: dict[int, float]
    counts: dict[int, float]


@dataclass(frozen=True, slots=True)
class SquareQDMStripTransferMatrix:
    """Exact transfer matrix for fully packed dimers on a square cylinder.

    The y direction is periodic and has fixed ``circumference``.  Boundary
    states are bit masks of horizontal dimers entering a site column.  A column
    transition stores the outgoing horizontal mask and the occupied vertical
    links inside that column.
    """

    circumference: int
    transitions: tuple[SquareQDMColumnTransition, ...] = field(init=False, repr=False)
    transfer_matrix: sp.csr_array = field(init=False, repr=False)
    _transitions_by_incoming: tuple[tuple[SquareQDMColumnTransition, ...], ...] = field(
        init=False,
        repr=False,
    )
    _electric_y_transfer_by_parity: tuple[dict[int, sp.csr_array], ...] = field(
        init=False,
        repr=False,
    )
    _transfer_scale: float | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.circumference < 2:
            raise ValueError("circumference must be at least two.")
        if self.circumference > 16:
            raise ValueError(
                "circumference above 16 is disabled because the boundary-state space is 2^Ly."
            )

        transitions = _square_qdm_column_transitions(self.circumference)
        n_states = 1 << self.circumference
        rows = np.fromiter(
            (transition.incoming_mask for transition in transitions),
            dtype=np.int64,
        )
        cols = np.fromiter(
            (transition.outgoing_mask for transition in transitions),
            dtype=np.int64,
        )
        data = np.ones(len(transitions), dtype=np.float64)
        matrix = sp.coo_array(
            (data, (rows, cols)),
            shape=(n_states, n_states),
            dtype=np.float64,
        ).tocsr()

        by_incoming: list[list[SquareQDMColumnTransition]] = [[] for _ in range(n_states)]
        for transition in transitions:
            by_incoming[transition.incoming_mask].append(transition)

        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "transfer_matrix", matrix)
        object.__setattr__(
            self,
            "_transitions_by_incoming",
            tuple(tuple(group) for group in by_incoming),
        )
        object.__setattr__(
            self,
            "_electric_y_transfer_by_parity",
            tuple(
                _electric_y_charge_resolved_transfer_matrices(
                    transitions,
                    circumference=self.circumference,
                    column_parity=parity,
                )
                for parity in (0, 1)
            ),
        )

    @property
    def n_boundary_states(self) -> int:
        return 1 << self.circumference

    def witness_insertion_matrix(
        self,
        placement: SquareQDMWitnessPlacement,
    ) -> sp.csr_array:
        """Contract the exact local ``Q_R`` weight into one strip insertion."""
        resolved = self._witness_insertion_matrices(
            placement,
            insertion_x=0,
            resolve_electric_winding_y=False,
        )
        return resolved[0]

    def witness_insertion_matrices_by_winding_y(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        insertion_x: int = 0,
    ) -> dict[int, sp.csr_array]:
        """Return insertion matrices resolved by electric y-winding charge.

        The dictionary key is the contribution of the witness window to the
        electric winding measured across the y-wrapping links.  ``insertion_x``
        fixes the checkerboard parity of the first column in the window.
        """
        return self._witness_insertion_matrices(
            placement,
            insertion_x=int(insertion_x),
            resolve_electric_winding_y=True,
        )

    def _witness_insertion_matrices(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        insertion_x: int,
        resolve_electric_winding_y: bool,
    ) -> dict[int, sp.csr_array]:
        self._validate_placement(placement)
        coordinate_to_index = {
            coordinate: index for index, coordinate in enumerate(placement.link_coordinates)
        }
        pattern_to_index = {
            pattern: index for index, pattern in enumerate(placement.template.local_patterns)
        }
        operator = placement.template.local_operator
        affected_sites = placement.affected_sites
        window_width = placement.window_width

        entries_by_charge: dict[int, dict[tuple[int, int], float]] = {}

        def source_link_value(
            path: Sequence[SquareQDMColumnTransition],
            coordinate: SquareQDMLinkCoordinate,
        ) -> int:
            transition = path[coordinate.x]
            mask = transition.outgoing_mask if coordinate.kind == "x" else transition.vertical_mask
            return _bit(mask, coordinate.y)

        def target_site_is_valid(
            path: Sequence[SquareQDMColumnTransition],
            *,
            site: tuple[int, int],
            target_pattern: tuple[int, ...],
        ) -> bool:
            site_x, site_y = site
            transition = path[site_x]
            incident_coordinates = (
                SquareQDMLinkCoordinate(site_x - 1, site_y, "x") if site_x > 0 else None,
                SquareQDMLinkCoordinate(site_x, site_y, "x"),
                SquareQDMLinkCoordinate(
                    site_x,
                    (site_y - 1) % self.circumference,
                    "y",
                ),
                SquareQDMLinkCoordinate(site_x, site_y, "y"),
            )

            occupied = 0
            for position, coordinate in enumerate(incident_coordinates):
                if coordinate is not None and coordinate in coordinate_to_index:
                    occupied += int(target_pattern[coordinate_to_index[coordinate]])
                    continue

                if position == 0:
                    occupied += _bit(transition.incoming_mask, site_y)
                elif position == 1:
                    occupied += _bit(transition.outgoing_mask, site_y)
                elif position == 2:
                    occupied += _bit(
                        transition.vertical_mask,
                        (site_y - 1) % self.circumference,
                    )
                else:
                    occupied += _bit(transition.vertical_mask, site_y)

            return occupied == 1

        def path_q_weight(path: Sequence[SquareQDMColumnTransition]) -> float:
            source_pattern = tuple(
                source_link_value(path, coordinate) for coordinate in placement.link_coordinates
            )
            source_index = pattern_to_index.get(source_pattern)
            if source_index is None:
                return 0.0

            q_weight = 0.0
            for target_index, target_pattern in enumerate(placement.template.local_patterns):
                coefficient = operator[target_index, source_index]
                if coefficient == 0.0:
                    continue
                if all(
                    target_site_is_valid(
                        path,
                        site=site,
                        target_pattern=target_pattern,
                    )
                    for site in affected_sites
                ):
                    q_weight += float(abs(coefficient) ** 2)
            return q_weight

        def path_winding_y_charge(path: Sequence[SquareQDMColumnTransition]) -> int:
            if not resolve_electric_winding_y:
                return 0
            return sum(
                _electric_winding_y_transition_value(
                    transition,
                    circumference=self.circumference,
                    column_x=insertion_x + offset,
                )
                for offset, transition in enumerate(path)
            )

        def extend_path(
            path: list[SquareQDMColumnTransition],
            *,
            incoming_mask: int,
        ) -> None:
            if len(path) == window_width:
                weight = path_q_weight(path)
                if weight != 0.0:
                    charge = path_winding_y_charge(path)
                    entries = entries_by_charge.setdefault(charge, {})
                    key = (path[0].incoming_mask, path[-1].outgoing_mask)
                    entries[key] = entries.get(key, 0.0) + weight
                return

            for transition in self._transitions_by_incoming[incoming_mask]:
                path.append(transition)
                extend_path(path, incoming_mask=transition.outgoing_mask)
                path.pop()

        for incoming_mask in range(self.n_boundary_states):
            extend_path([], incoming_mask=incoming_mask)

        if not entries_by_charge:
            return {
                0: sp.csr_array(
                    (self.n_boundary_states, self.n_boundary_states),
                    dtype=np.float64,
                )
            }

        result: dict[int, sp.csr_array] = {}
        for charge, entries in entries_by_charge.items():
            rows = np.fromiter((key[0] for key in entries), dtype=np.int64)
            cols = np.fromiter((key[1] for key in entries), dtype=np.int64)
            data = np.fromiter(entries.values(), dtype=np.float64)
            result[int(charge)] = sp.coo_array(
                (data, (rows, cols)),
                shape=(self.n_boundary_states, self.n_boundary_states),
                dtype=np.float64,
            ).tocsr()
        return result

    def evaluate_witness(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        length: int,
        boundary_x: StripBoundaryCondition = "open",
        insertion_x: int | None = None,
        winding_sector: SquareQDMStripWindingSector | tuple[int, int] | None = None,
        winding_projection: WindingProjectionMethod = "auto",
        fourier_points: int | None = None,
    ) -> SquareQDMStripWitnessEvaluation:
        """Evaluate ``Tr(Q_R)/dim(H)`` without enumerating dimer coverings."""
        self._validate_placement(placement)
        sector = _normalize_square_qdm_strip_winding_sector(winding_sector)
        if length < placement.window_width:
            raise ValueError("length must be at least the witness window width.")
        if boundary_x not in ("open", "periodic"):
            raise ValueError("boundary_x must be 'open' or 'periodic'.")
        if sector is not None and boundary_x != "periodic":
            raise ValueError("winding-sector resolution requires periodic x boundaries.")

        if boundary_x == "open":
            insertion = self.witness_insertion_matrix(placement)
            if insertion_x is None:
                insertion_x = (length - placement.window_width) // 2
            return self._evaluate_open(
                placement,
                insertion=insertion,
                length=length,
                insertion_x=int(insertion_x),
            )

        if sector is not None:
            periodic_insertion_x = 0 if insertion_x is None else int(insertion_x)
            projection = self._resolve_winding_projection_method(winding_projection)
            if projection == "dynamic_programming":
                return self._evaluate_periodic_sector(
                    placement,
                    length=length,
                    insertion_x=periodic_insertion_x,
                    winding_sector=sector,
                )
            return self._evaluate_periodic_sector_fourier(
                placement,
                length=length,
                insertion_x=periodic_insertion_x,
                winding_sector=sector,
                fourier_points=fourier_points,
            )

        if insertion_x is not None:
            raise ValueError(
                "insertion_x is only used for open boundaries or winding-resolved periodic "
                "contractions."
            )
        insertion = self.witness_insertion_matrix(placement)
        return self._evaluate_periodic(
            placement,
            insertion=insertion,
            length=length,
        )

    def scan_witness(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        lengths: Sequence[int],
        boundary_x: StripBoundaryCondition = "open",
        centered: bool = True,
        winding_sector: SquareQDMStripWindingSector | tuple[int, int] | None = None,
        periodic_insertion_x: int = 0,
        winding_projection: WindingProjectionMethod = "auto",
        fourier_points: int | None = None,
    ) -> SquareQDMStripScalingReport:
        """Evaluate a fixed witness over several strip lengths."""
        self._validate_placement(placement)
        sector = _normalize_square_qdm_strip_winding_sector(winding_sector)
        projection = self._resolve_winding_projection_method(winding_projection)
        if boundary_x not in ("open", "periodic"):
            raise ValueError("boundary_x must be 'open' or 'periodic'.")
        if sector is not None and boundary_x != "periodic":
            raise ValueError("winding-sector resolution requires periodic x boundaries.")

        insertion = self.witness_insertion_matrix(placement) if sector is None else None
        periodic_spectrum = (
            self._periodic_spectrum(insertion)
            if boundary_x == "periodic" and sector is None and insertion is not None
            else None
        )
        evaluations: list[SquareQDMStripWitnessEvaluation] = []
        for raw_length in lengths:
            length = int(raw_length)
            if length < placement.window_width:
                raise ValueError("all lengths must be at least the witness window width.")
            if boundary_x == "periodic":
                if sector is None:
                    assert insertion is not None
                    evaluation = self._evaluate_periodic(
                        placement,
                        insertion=insertion,
                        length=length,
                        spectrum=periodic_spectrum,
                    )
                elif projection == "dynamic_programming":
                    evaluation = self._evaluate_periodic_sector(
                        placement,
                        length=length,
                        insertion_x=int(periodic_insertion_x),
                        winding_sector=sector,
                    )
                else:
                    evaluation = self._evaluate_periodic_sector_fourier(
                        placement,
                        length=length,
                        insertion_x=int(periodic_insertion_x),
                        winding_sector=sector,
                        fourier_points=fourier_points,
                    )
            else:
                assert insertion is not None
                insertion_x = (length - placement.window_width) // 2 if centered else 0
                evaluation = self._evaluate_open(
                    placement,
                    insertion=insertion,
                    length=length,
                    insertion_x=insertion_x,
                )
            evaluations.append(evaluation)

        return SquareQDMStripScalingReport(
            placement=placement,
            evaluations=tuple(evaluations),
        )

    def periodic_winding_sector_counts(
        self,
        *,
        length: int,
        winding_projection: WindingProjectionMethod = "auto",
        fourier_points: int | None = None,
    ) -> dict[SquareQDMStripWindingSector, float]:
        """Count periodic dimer coverings in every electric winding sector."""
        if length <= 0:
            raise ValueError("length must be positive.")
        projection = self._resolve_winding_projection_method(winding_projection)
        if projection == "fourier":
            return self._periodic_winding_sector_counts_fourier(
                length=length,
                fourier_points=fourier_points,
            )
        if self.n_boundary_states > 512:
            raise ValueError(
                "dynamic-programming winding resolution is limited to at most "
                "512 boundary states (Ly <= 9); use winding_projection='fourier'."
            )

        factors = tuple(
            self._electric_y_transfer_by_parity[column_x % 2] for column_x in range(length)
        )
        result: dict[SquareQDMStripWindingSector, float] = {}
        sectors_by_x = self._boundary_states_grouped_by_winding_x(length=length)
        for winding_x, boundary_states in sectors_by_x.items():
            trace = _charge_resolved_trace(
                factors,
                start_states=boundary_states,
                n_boundary_states=self.n_boundary_states,
            )
            for winding_y, count in trace.counts.items():
                if count <= 0.0:
                    continue
                result[
                    SquareQDMStripWindingSector(
                        winding_x=winding_x,
                        winding_y=winding_y,
                    )
                ] = count
        return result

    def _evaluate_open(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        insertion: sp.csr_array,
        length: int,
        insertion_x: int,
    ) -> SquareQDMStripWitnessEvaluation:
        if insertion_x < 0 or insertion_x + placement.window_width > length:
            raise ValueError("insertion_x places the witness outside the open strip.")

        left = np.zeros(self.n_boundary_states, dtype=np.float64)
        left[0] = 1.0
        right = left.copy()
        left, log_left_scale = _propagate_normalized(
            left,
            matrix=self.transfer_matrix,
            steps=insertion_x,
            side="left",
        )
        right, log_right_scale = _propagate_normalized(
            right,
            matrix=self.transfer_matrix,
            steps=length - insertion_x - placement.window_width,
            side="right",
        )

        middle_right = right.copy()
        for _ in range(placement.window_width):
            middle_right = np.asarray(self.transfer_matrix @ middle_right).reshape(-1)

        denominator_scaled = float(np.dot(left, middle_right))
        numerator_scaled = float(np.dot(left, np.asarray(insertion @ right).reshape(-1)))
        if denominator_scaled <= 0.0:
            raise ValueError("the requested open strip has no dimer coverings.")

        expectation = float(numerator_scaled / denominator_scaled)
        common_log_scale = log_left_scale + log_right_scale
        log_partition_count = common_log_scale + log(denominator_scaled)
        log_weighted_count = (
            float("-inf") if numerator_scaled <= 0.0 else common_log_scale + log(numerator_scaled)
        )
        return SquareQDMStripWitnessEvaluation(
            circumference=self.circumference,
            length=length,
            boundary_x="open",
            insertion_x=insertion_x,
            window_width=placement.window_width,
            expectation=expectation,
            log_partition_count=log_partition_count,
            log_weighted_count=log_weighted_count,
            partition_count=_safe_exp(log_partition_count),
            weighted_count=_safe_exp(log_weighted_count),
            metadata={"contraction": "normalized_open_boundary_vectors"},
        )

    def _evaluate_periodic(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        insertion: sp.csr_array,
        length: int,
        spectrum: _PeriodicTransferSpectrum | None = None,
    ) -> SquareQDMStripWitnessEvaluation:
        periodic_spectrum = spectrum or self._periodic_spectrum(insertion)
        eigenvalues = periodic_spectrum.eigenvalues
        insertion_diagonal = periodic_spectrum.insertion_diagonal
        scale = periodic_spectrum.scale
        ratios = eigenvalues / scale
        denominator_scaled = float(np.sum(ratios**length))
        if denominator_scaled <= 0.0:
            raise ValueError("the requested periodic strip has no dimer coverings.")

        numerator_scaled = float(
            np.real(np.sum(ratios ** (length - placement.window_width) * insertion_diagonal))
        )
        expectation = float(numerator_scaled / (scale**placement.window_width * denominator_scaled))
        log_partition_count = length * log(scale) + log(denominator_scaled)
        log_weighted_count = (
            float("-inf")
            if numerator_scaled <= 0.0
            else ((length - placement.window_width) * log(scale) + log(numerator_scaled))
        )
        return SquareQDMStripWitnessEvaluation(
            circumference=self.circumference,
            length=length,
            boundary_x="periodic",
            insertion_x=None,
            window_width=placement.window_width,
            expectation=expectation,
            log_partition_count=log_partition_count,
            log_weighted_count=log_weighted_count,
            partition_count=_safe_exp(log_partition_count),
            weighted_count=_safe_exp(log_weighted_count),
            metadata={"contraction": "symmetric_transfer_eigendecomposition"},
        )

    def _evaluate_periodic_sector(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        length: int,
        insertion_x: int,
        winding_sector: SquareQDMStripWindingSector,
    ) -> SquareQDMStripWitnessEvaluation:
        if insertion_x < 0 or insertion_x + placement.window_width > length:
            raise ValueError(
                "winding-resolved insertion must not cross the canonical x seam; "
                "choose insertion_x so the local window lies inside [0, length)."
            )
        if self.n_boundary_states > 512:
            raise ValueError(
                "winding-resolved periodic contraction is currently limited to at most "
                "512 boundary states (Ly <= 9)."
            )

        boundary_states = self._boundary_states_grouped_by_winding_x(length=length).get(
            winding_sector.winding_x,
            (),
        )
        if not boundary_states:
            raise ValueError(
                "the requested winding_x sector has no compatible transfer boundary states."
            )

        denominator_factors = tuple(
            self._electric_y_transfer_by_parity[column_x % 2] for column_x in range(length)
        )
        denominator = _charge_resolved_trace(
            denominator_factors,
            start_states=boundary_states,
            n_boundary_states=self.n_boundary_states,
        )
        log_partition_count = denominator.log_counts.get(
            winding_sector.winding_y,
            float("-inf"),
        )
        if not isfinite(log_partition_count):
            raise ValueError("the requested periodic winding sector has no dimer coverings.")

        insertion = self.witness_insertion_matrices_by_winding_y(
            placement,
            insertion_x=insertion_x,
        )
        numerator_factors: list[dict[int, sp.csr_array]] = []
        numerator_factors.extend(
            self._electric_y_transfer_by_parity[column_x % 2] for column_x in range(insertion_x)
        )
        numerator_factors.append(insertion)
        numerator_factors.extend(
            self._electric_y_transfer_by_parity[column_x % 2]
            for column_x in range(
                insertion_x + placement.window_width,
                length,
            )
        )
        numerator = _charge_resolved_trace(
            tuple(numerator_factors),
            start_states=boundary_states,
            n_boundary_states=self.n_boundary_states,
        )
        log_weighted_count = numerator.log_counts.get(
            winding_sector.winding_y,
            float("-inf"),
        )
        expectation = (
            0.0
            if not isfinite(log_weighted_count)
            else float(exp(log_weighted_count - log_partition_count))
        )
        return SquareQDMStripWitnessEvaluation(
            circumference=self.circumference,
            length=length,
            boundary_x="periodic",
            insertion_x=insertion_x,
            window_width=placement.window_width,
            expectation=expectation,
            log_partition_count=log_partition_count,
            log_weighted_count=log_weighted_count,
            partition_count=_safe_exp(log_partition_count),
            weighted_count=_safe_exp(log_weighted_count),
            winding_sector=winding_sector,
            metadata={
                "contraction": "electric_winding_charge_resolved_dynamic_programming",
                "n_x_boundary_states": len(boundary_states),
            },
        )

    def _resolve_winding_projection_method(
        self,
        method: WindingProjectionMethod,
    ) -> Literal["dynamic_programming", "fourier"]:
        if method not in ("auto", "dynamic_programming", "fourier"):
            raise ValueError(
                "winding_projection must be 'auto', 'dynamic_programming', or 'fourier'."
            )
        if method == "auto":
            return "dynamic_programming" if self.n_boundary_states <= 512 else "fourier"
        return method

    def _evaluate_periodic_sector_fourier(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        length: int,
        insertion_x: int,
        winding_sector: SquareQDMStripWindingSector,
        fourier_points: int | None,
    ) -> SquareQDMStripWitnessEvaluation:
        self._validate_fourier_winding_problem(length=length)
        transfer_scale = self._fourier_transfer_scale()
        points = _normalize_fourier_points(length=length, fourier_points=fourier_points)
        target_degree = _winding_y_to_positive_charge_count(
            winding_sector.winding_y,
            length=length,
        )
        if target_degree is None:
            raise ValueError("the requested periodic winding sector has no dimer coverings.")

        groups = self._boundary_states_grouped_by_winding_x(length=length)
        boundary_states = groups.get(winding_sector.winding_x, ())
        if not boundary_states:
            raise ValueError(
                "the requested winding_x sector has no compatible transfer boundary states."
            )
        self._validate_fourier_sector_size(groups, winding_sector.winding_x)

        insertion_x_mod = int(insertion_x) % length
        regular_blocks: dict[tuple[int, int], dict[int, npt.NDArray[np.complex128]]] = {}

        def regular_charge_blocks(
            *,
            column_parity: int,
            source_sector: int,
        ) -> dict[int, npt.NDArray[np.complex128]]:
            key = (int(column_parity) % 2, int(source_sector))
            cached = regular_blocks.get(key)
            if cached is not None:
                return cached
            source_states = groups.get(int(source_sector), ())
            target_states = groups.get(-int(source_sector), ())
            if not source_states or not target_states:
                raise ValueError("the requested winding_x sector has no transfer path.")
            blocks: dict[int, npt.NDArray[np.complex128]] = {}
            for charge, matrix in self._electric_y_transfer_by_parity[key[0]].items():
                degree = (int(charge) + 1) // 2
                blocks[degree] = (
                    matrix[np.ix_(source_states, target_states)].toarray().astype(np.complex128)
                    / transfer_scale
                )
            regular_blocks[key] = blocks
            return blocks

        start_sector = (
            winding_sector.winding_x if insertion_x_mod % 2 == 0 else -winding_sector.winding_x
        )
        insertion_end_sector = start_sector if placement.window_width % 2 == 0 else -start_sector
        insertion_source_states = groups.get(start_sector, ())
        insertion_target_states = groups.get(insertion_end_sector, ())
        if not insertion_source_states or not insertion_target_states:
            raise ValueError("the witness insertion is incompatible with the winding_x sector.")

        insertion_by_charge = self.witness_insertion_matrices_by_winding_y(
            placement,
            insertion_x=insertion_x_mod,
        )
        insertion_blocks: dict[int, npt.NDArray[np.complex128]] = {}
        insertion_scale = transfer_scale**placement.window_width
        for charge, matrix in insertion_by_charge.items():
            shifted = int(charge) + placement.window_width
            if shifted % 2 != 0:
                raise ValueError("witness insertion carries an invalid winding-charge parity.")
            degree = shifted // 2
            insertion_blocks[degree] = (
                matrix[np.ix_(insertion_source_states, insertion_target_states)]
                .toarray()
                .astype(np.complex128)
                / insertion_scale
            )

        denominator_samples = np.empty(points, dtype=np.complex128)
        numerator_samples = np.empty(points, dtype=np.complex128)
        for phase_index in range(points):
            root = np.exp(2.0j * np.pi * phase_index / points)
            denominator_matrix = _twisted_regular_segment_matrix(
                start_column=0,
                steps=length,
                start_sector=winding_sector.winding_x,
                root=root,
                regular_charge_blocks=regular_charge_blocks,
            )
            denominator_samples[phase_index] = np.trace(denominator_matrix)

            insertion_matrix = _evaluate_twisted_blocks(insertion_blocks, root=root)
            environment_matrix = _twisted_regular_segment_matrix(
                start_column=insertion_x_mod + placement.window_width,
                steps=length - placement.window_width,
                start_sector=insertion_end_sector,
                root=root,
                regular_charge_blocks=regular_charge_blocks,
            )
            numerator_samples[phase_index] = np.trace(insertion_matrix @ environment_matrix)

        partition_scaled = _fourier_project_coefficient(
            denominator_samples,
            target_degree=target_degree,
        )
        if partition_scaled <= 0.0:
            raise ValueError("the requested periodic winding sector has no dimer coverings.")
        weighted_scaled = _fourier_project_coefficient(
            numerator_samples,
            target_degree=target_degree,
        )

        common_log_scale = length * log(transfer_scale)
        log_partition_count = common_log_scale + log(partition_scaled)
        log_weighted_count = (
            float("-inf") if weighted_scaled <= 0.0 else common_log_scale + log(weighted_scaled)
        )
        expectation = float(weighted_scaled / partition_scaled)
        exact_projection = points >= length + 1
        return SquareQDMStripWitnessEvaluation(
            circumference=self.circumference,
            length=length,
            boundary_x="periodic",
            insertion_x=insertion_x_mod,
            window_width=placement.window_width,
            expectation=expectation,
            log_partition_count=log_partition_count,
            log_weighted_count=log_weighted_count,
            partition_count=_safe_exp(log_partition_count),
            weighted_count=_safe_exp(log_weighted_count),
            winding_sector=winding_sector,
            metadata={
                "contraction": "electric_winding_fourier_projected_dense_transfer",
                "fourier_points": points,
                "exact_fourier_projection": exact_projection,
                "aliased_fourier_projection": not exact_projection,
                "n_x_boundary_states": len(boundary_states),
                "transfer_scale": transfer_scale,
                "insertion_crosses_canonical_seam": (
                    insertion_x_mod + placement.window_width > length
                ),
            },
        )

    def _periodic_winding_sector_counts_fourier(
        self,
        *,
        length: int,
        fourier_points: int | None,
    ) -> dict[SquareQDMStripWindingSector, float]:
        self._validate_fourier_winding_problem(length=length)
        transfer_scale = self._fourier_transfer_scale()
        points = _normalize_fourier_points(length=length, fourier_points=fourier_points)
        if points < length + 1:
            raise ValueError(
                "periodic_winding_sector_counts requires at least length + 1 Fourier points "
                "so that all winding_y sectors are resolved without aliasing."
            )

        groups = self._boundary_states_grouped_by_winding_x(length=length)
        result: dict[SquareQDMStripWindingSector, float] = {}
        for winding_x, _boundary_states in groups.items():
            self._validate_fourier_sector_size(groups, winding_x)
            regular_blocks: dict[
                tuple[int, int],
                dict[int, npt.NDArray[np.complex128]],
            ] = {}

            def regular_charge_blocks(
                *,
                column_parity: int,
                source_sector: int,
                cache: dict[
                    tuple[int, int],
                    dict[int, npt.NDArray[np.complex128]],
                ] = regular_blocks,
            ) -> dict[int, npt.NDArray[np.complex128]]:
                key = (int(column_parity) % 2, int(source_sector))
                cached = cache.get(key)
                if cached is not None:
                    return cached
                source_states = groups.get(int(source_sector), ())
                target_states = groups.get(-int(source_sector), ())
                blocks = {
                    (int(charge) + 1)
                    // 2: (
                        matrix[np.ix_(source_states, target_states)].toarray().astype(np.complex128)
                        / transfer_scale
                    )
                    for charge, matrix in self._electric_y_transfer_by_parity[key[0]].items()
                }
                cache[key] = blocks
                return blocks

            samples = np.empty(points, dtype=np.complex128)
            for phase_index in range(points):
                root = np.exp(2.0j * np.pi * phase_index / points)
                matrix = _twisted_regular_segment_matrix(
                    start_column=0,
                    steps=length,
                    start_sector=winding_x,
                    root=root,
                    regular_charge_blocks=regular_charge_blocks,
                )
                samples[phase_index] = np.trace(matrix)

            coefficients = np.fft.fft(samples) / points
            for positive_charge_count in range(length + 1):
                scaled_count = _nonnegative_real_fourier_value(
                    coefficients[positive_charge_count],
                    reference_scale=float(np.max(np.abs(coefficients))),
                )
                if scaled_count <= 0.0:
                    continue
                winding_y = 2 * positive_charge_count - length
                log_count = length * log(transfer_scale) + log(scaled_count)
                result[
                    SquareQDMStripWindingSector(
                        winding_x=winding_x,
                        winding_y=winding_y,
                    )
                ] = _safe_exp(log_count)
        return result

    def _fourier_transfer_scale(self) -> float:
        cached = self._transfer_scale
        if cached is not None:
            return cached
        if self.n_boundary_states <= 8:
            scale = float(np.max(np.linalg.eigvalsh(self.transfer_matrix.toarray())))
        else:
            scale = float(
                sp.linalg.eigsh(
                    self.transfer_matrix,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                )[0]
            )
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("transfer matrix has no positive spectral radius.")
        object.__setattr__(self, "_transfer_scale", scale)
        return scale

    def _validate_fourier_winding_problem(self, *, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive.")
        if length % 2 != 0 or self.circumference % 2 != 0:
            raise ValueError(
                "Fourier-projected electric winding currently requires even Lx and Ly."
            )

    def _validate_fourier_sector_size(
        self,
        groups: Mapping[int, Sequence[int]],
        winding_x: int,
    ) -> None:
        sector_size = max(
            len(groups.get(int(winding_x), ())),
            len(groups.get(-int(winding_x), ())),
        )
        if sector_size > 1024:
            raise ValueError(
                "dense Fourier winding projection currently supports at most 1024 states "
                "in one winding_x block (typically Ly <= 12)."
            )

    def _boundary_states_grouped_by_winding_x(
        self,
        *,
        length: int,
    ) -> dict[int, tuple[int, ...]]:
        grouped: dict[int, list[int]] = {}
        for mask in range(self.n_boundary_states):
            winding_x = _electric_winding_x_boundary_value(
                mask,
                circumference=self.circumference,
                length=length,
            )
            grouped.setdefault(winding_x, []).append(mask)
        return {value: tuple(states) for value, states in grouped.items()}

    def _periodic_spectrum(
        self,
        insertion: sp.csr_array,
    ) -> _PeriodicTransferSpectrum:
        if self.n_boundary_states > 2048:
            raise ValueError(
                "periodic contraction currently requires dense diagonalization and is limited "
                "to at most 2048 boundary states; use boundary_x='open' for larger widths."
            )

        transfer_dense = self.transfer_matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(transfer_dense)
        scale = float(np.max(np.abs(eigenvalues)))
        if scale == 0.0:
            raise ValueError("transfer matrix has zero spectral radius.")

        insertion_dense = insertion.toarray()
        insertion_diagonal = np.einsum(
            "ij,ij->j",
            eigenvectors,
            insertion_dense @ eigenvectors,
        ).real
        return _PeriodicTransferSpectrum(
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            insertion_diagonal=np.asarray(insertion_diagonal, dtype=np.float64),
            scale=scale,
        )

    def _validate_placement(self, placement: SquareQDMWitnessPlacement) -> None:
        if placement.circumference != self.circumference:
            raise ValueError(
                "placement circumference does not match the transfer matrix: "
                f"{placement.circumference} != {self.circumference}."
            )


def evaluate_square_qdm_witness_family_on_strips(
    family: LocalWitnessFamily,
    *,
    models: Mapping[Hashable, SquareQDMModel],
    lengths: Sequence[int] | Mapping[Hashable, Sequence[int]],
    boundary_x: StripBoundaryCondition = "periodic",
    winding_sector: (
        SquareQDMStripWindingSector
        | tuple[int, int]
        | Mapping[Hashable, SquareQDMStripWindingSector | tuple[int, int] | None]
        | None
    ) = None,
    embedding_index: int = 0,
    winding_projection: WindingProjectionMethod = "auto",
    fourier_points: int | None = None,
) -> SquareQDMWitnessFamilyStripReport:
    """Evaluate one common cage-derived witness family on square-QDM strips.

    ``family`` is usually produced by :func:`common_local_witness_families`.
    Each family embedding is paired with the model carrying the same system
    label, converted to strip coordinates, and evaluated without rebuilding a
    global dimer basis.
    """
    records: list[SquareQDMWitnessFamilyStripRecord] = []
    for embedding in family.embeddings:
        system_label = embedding.system_label
        if system_label not in models:
            raise KeyError(f"missing square-QDM model for system label {system_label!r}.")
        if embedding_index < 0 or embedding_index >= len(embedding.witnesses):
            raise IndexError(
                f"embedding_index={embedding_index} is unavailable for system label "
                f"{system_label!r}; found {len(embedding.witnesses)} embeddings."
            )

        model = models[system_label]
        witness = embedding.witnesses[embedding_index]
        placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
        transfer = SquareQDMStripTransferMatrix(circumference=placement.circumference)
        local_lengths = lengths[system_label] if isinstance(lengths, Mapping) else lengths
        local_sector = (
            winding_sector.get(system_label)
            if isinstance(winding_sector, Mapping)
            else winding_sector
        )
        scaling = transfer.scan_witness(
            placement,
            lengths=local_lengths,
            boundary_x=boundary_x,
            winding_sector=local_sector,
            winding_projection=winding_projection,
            fourier_points=fourier_points,
        )
        records.append(
            SquareQDMWitnessFamilyStripRecord(
                system_label=system_label,
                embedding_index=embedding_index,
                placement=placement,
                scaling_report=scaling,
            )
        )

    return SquareQDMWitnessFamilyStripReport(
        family=family,
        records=tuple(records),
    )


def evaluate_square_qdm_classification_witnesses_on_strips(
    report: CageClassificationReport,
    *,
    model: SquareQDMModel,
    lengths: Sequence[int],
    boundary_x: StripBoundaryCondition = "periodic",
    winding_sector: SquareQDMStripWindingSector | tuple[int, int] | None = None,
    normalization: WitnessNormalization = "operator_norm",
    include_projector_like: bool = True,
    winding_projection: WindingProjectionMethod = "auto",
    fourier_points: int | None = None,
) -> SquareQDMClassificationWitnessStripReport:
    """Evaluate the actual cage-derived reduced-IZ witnesses on strips.

    The default operator-norm normalization fixes ``||Q_R|| = 1``.  This
    removes the arbitrary coefficient scale from comparisons between distinct
    interference-zero rows and between different system sizes.
    """
    witnesses = local_witnesses_from_classification_report(
        report,
        include_projector_like=include_projector_like,
        normalization=normalization,
    )
    records: list[SquareQDMClassificationWitnessStripRecord] = []
    for witness_index, witness in enumerate(witnesses):
        placement = SquareQDMWitnessPlacement.from_local_witness(model, witness)
        transfer = SquareQDMStripTransferMatrix(circumference=placement.circumference)
        scaling = transfer.scan_witness(
            placement,
            lengths=lengths,
            boundary_x=boundary_x,
            winding_sector=winding_sector,
            winding_projection=winding_projection,
            fourier_points=fourier_points,
        )
        records.append(
            SquareQDMClassificationWitnessStripRecord(
                witness_index=witness_index,
                witness=witness,
                placement=placement,
                scaling_report=scaling,
            )
        )
    return SquareQDMClassificationWitnessStripReport(
        records=tuple(records),
        normalization=normalization,
    )


def square_qdm_directed_plaquette_witness_placement(
    *,
    circumference: int,
    orientation: Literal["horizontal", "vertical"],
) -> SquareQDMWitnessPlacement:
    """Return a unit-normalized directed plaquette-flip witness.

    ``Q_R`` is the projector onto one of the two flippable plaquette
    orientations.  Summing the two expectations gives the total local
    flippability probability and therefore the beta-zero potential-energy
    density for a uniform square QDM.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")
    horizontal = (1, 0, 1, 0)
    vertical = (0, 1, 0, 1)
    source = horizontal if orientation == "horizontal" else vertical
    target = vertical if orientation == "horizontal" else horizontal
    patterns = (horizontal, vertical)
    pattern_to_index = {pattern: index for index, pattern in enumerate(patterns)}
    operator = np.zeros((2, 2), dtype=np.complex128)
    operator[pattern_to_index[target], pattern_to_index[source]] = 1.0
    template = LocalWitnessTemplate(
        pattern_key=((source, target, (1.0, 0.0)),),
        local_patterns=patterns,
        local_operator=operator,
        metadata={
            "source": "square_qdm_directed_plaquette",
            "orientation": orientation,
            "normalization": "operator_norm",
        },
    )
    return SquareQDMWitnessPlacement(
        template=template,
        circumference=int(circumference),
        link_coordinates=(
            SquareQDMLinkCoordinate(x=0, y=0, kind="x"),
            SquareQDMLinkCoordinate(x=1, y=0, kind="y"),
            SquareQDMLinkCoordinate(x=0, y=1, kind="x"),
            SquareQDMLinkCoordinate(x=0, y=0, kind="y"),
        ),
        metadata={"orientation": orientation},
    )


def evaluate_square_qdm_beta_zero_energy_density(
    *,
    circumference: int,
    length: int,
    potential_coupling: float = 1.0,
    winding_sector: SquareQDMStripWindingSector | tuple[int, int] | None = None,
    winding_projection: WindingProjectionMethod = "auto",
    fourier_points: int | None = None,
) -> SquareQDMBetaZeroEnergyDensityEvaluation:
    """Evaluate the beta-zero energy density of a uniform square QDM.

    The kinetic term is off-diagonal in the dimer basis and has zero trace.
    The potential part is the coupling times the probability that a plaquette
    is flippable in either orientation.
    """
    if not np.isfinite(potential_coupling):
        raise ValueError("potential_coupling must be finite.")
    transfer = SquareQDMStripTransferMatrix(circumference=int(circumference))
    horizontal = transfer.evaluate_witness(
        square_qdm_directed_plaquette_witness_placement(
            circumference=circumference,
            orientation="horizontal",
        ),
        length=int(length),
        boundary_x="periodic",
        winding_sector=winding_sector,
        winding_projection=winding_projection,
        fourier_points=fourier_points,
    )
    vertical = transfer.evaluate_witness(
        square_qdm_directed_plaquette_witness_placement(
            circumference=circumference,
            orientation="vertical",
        ),
        length=int(length),
        boundary_x="periodic",
        winding_sector=winding_sector,
        winding_projection=winding_projection,
        fourier_points=fourier_points,
    )
    sector = _normalize_square_qdm_strip_winding_sector(winding_sector)
    return SquareQDMBetaZeroEnergyDensityEvaluation(
        circumference=int(circumference),
        length=int(length),
        horizontal_flippability=float(horizontal.expectation),
        vertical_flippability=float(vertical.expectation),
        potential_coupling=float(potential_coupling),
        winding_sector=sector,
        metadata={
            "winding_projection": winding_projection,
            "fourier_points": fourier_points,
            "kinetic_trace_argument": "off_diagonal_in_dimer_basis",
        },
    )


def scan_square_qdm_beta_zero_energy_density(
    sizes: Sequence[int] | Sequence[tuple[int, int]],
    *,
    potential_coupling: float = 1.0,
    winding_sector: SquareQDMStripWindingSector | tuple[int, int] | None = (0, 0),
    winding_projection: WindingProjectionMethod = "auto",
    fourier_points: int | None = None,
) -> SquareQDMBetaZeroEnergyDensityScalingReport:
    """Evaluate beta-zero energy densities for square tori or ``(Lx, Ly)`` pairs."""
    evaluations: list[SquareQDMBetaZeroEnergyDensityEvaluation] = []
    for size in sizes:
        if isinstance(size, tuple):
            length, circumference = int(size[0]), int(size[1])
        else:
            length = circumference = int(size)
        evaluations.append(
            evaluate_square_qdm_beta_zero_energy_density(
                circumference=circumference,
                length=length,
                potential_coupling=potential_coupling,
                winding_sector=winding_sector,
                winding_projection=winding_projection,
                fourier_points=fourier_points,
            )
        )
    return SquareQDMBetaZeroEnergyDensityScalingReport(tuple(evaluations))


def _normalize_fourier_points(
    *,
    length: int,
    fourier_points: int | None,
) -> int:
    points = length + 1 if fourier_points is None else int(fourier_points)
    if points <= 0:
        raise ValueError("fourier_points must be positive.")
    return points


def _winding_y_to_positive_charge_count(
    winding_y: int,
    *,
    length: int,
) -> int | None:
    shifted = int(winding_y) + int(length)
    if shifted % 2 != 0:
        return None
    count = shifted // 2
    if count < 0 or count > length:
        return None
    return count


def _evaluate_twisted_blocks(
    blocks: Mapping[int, npt.NDArray[np.complex128]],
    *,
    root: complex,
) -> npt.NDArray[np.complex128]:
    if not blocks:
        raise ValueError("twisted transfer blocks must not be empty.")
    first = next(iter(blocks.values()))
    result = np.zeros_like(first, dtype=np.complex128)
    for degree, block in blocks.items():
        result += (root ** int(degree)) * block
    return result


def _twisted_regular_segment_matrix(
    *,
    start_column: int,
    steps: int,
    start_sector: int,
    root: complex,
    regular_charge_blocks,
) -> npt.NDArray[np.complex128]:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    first_blocks = regular_charge_blocks(
        column_parity=int(start_column) % 2,
        source_sector=int(start_sector),
    )
    first = _evaluate_twisted_blocks(first_blocks, root=root)
    if steps == 0:
        return np.eye(first.shape[0], dtype=np.complex128)
    if steps == 1:
        return first

    second_blocks = regular_charge_blocks(
        column_parity=(int(start_column) + 1) % 2,
        source_sector=-int(start_sector),
    )
    second = _evaluate_twisted_blocks(second_blocks, root=root)
    pair = first @ second
    result = np.linalg.matrix_power(pair, steps // 2)
    if steps % 2 != 0:
        result = result @ first
    return result


def _nonnegative_real_fourier_value(
    value: complex,
    *,
    reference_scale: float,
) -> float:
    scale = max(1.0, float(reference_scale))
    tolerance = 1.0e-9 * scale
    if abs(float(np.imag(value))) > tolerance:
        raise ValueError(
            "Fourier projection retained a non-negligible imaginary component; "
            "increase fourier_points or reduce the requested strip size."
        )
    real_value = float(np.real(value))
    if real_value < -tolerance:
        raise ValueError(
            "Fourier projection produced a negative sector weight beyond numerical tolerance."
        )
    if abs(real_value) <= tolerance:
        return 0.0
    return real_value


def _fourier_project_coefficient(
    samples: npt.NDArray[np.complex128],
    *,
    target_degree: int,
) -> float:
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Fourier samples must be a non-empty vector.")
    coefficients = np.fft.fft(samples) / samples.size
    value = coefficients[int(target_degree) % samples.size]
    return _nonnegative_real_fourier_value(
        value,
        reference_scale=float(np.max(np.abs(coefficients))),
    )


def _normalize_square_qdm_strip_winding_sector(
    sector: SquareQDMStripWindingSector | tuple[int, int] | None,
) -> SquareQDMStripWindingSector | None:
    if sector is None:
        return None
    if isinstance(sector, SquareQDMStripWindingSector):
        return sector
    if len(sector) != 2:
        raise ValueError("winding sector tuples must have the form (winding_x, winding_y).")
    return SquareQDMStripWindingSector(
        winding_x=int(sector[0]),
        winding_y=int(sector[1]),
    )


def _electric_winding_x_boundary_value(
    mask: int,
    *,
    circumference: int,
    length: int,
) -> int:
    source_x = length - 1
    return sum(
        (1 if (source_x + y) % 2 == 0 else -1) * (2 * _bit(mask, y) - 1)
        for y in range(circumference)
    )


def _electric_winding_y_transition_value(
    transition: SquareQDMColumnTransition,
    *,
    circumference: int,
    column_x: int,
) -> int:
    source_y = circumference - 1
    staggered_sign = 1 if (column_x + source_y) % 2 == 0 else -1
    wrapping_occupation = _bit(transition.vertical_mask, source_y)
    return staggered_sign * (2 * wrapping_occupation - 1)


def _electric_y_charge_resolved_transfer_matrices(
    transitions: Sequence[SquareQDMColumnTransition],
    *,
    circumference: int,
    column_parity: int,
) -> dict[int, sp.csr_array]:
    n_states = 1 << circumference
    entries_by_charge: dict[int, list[SquareQDMColumnTransition]] = {}
    for transition in transitions:
        charge = _electric_winding_y_transition_value(
            transition,
            circumference=circumference,
            column_x=int(column_parity),
        )
        entries_by_charge.setdefault(charge, []).append(transition)

    result: dict[int, sp.csr_array] = {}
    for charge, selected in entries_by_charge.items():
        rows = np.fromiter(
            (transition.incoming_mask for transition in selected),
            dtype=np.int64,
        )
        cols = np.fromiter(
            (transition.outgoing_mask for transition in selected),
            dtype=np.int64,
        )
        data = np.ones(len(selected), dtype=np.float64)
        result[int(charge)] = sp.coo_array(
            (data, (rows, cols)),
            shape=(n_states, n_states),
            dtype=np.float64,
        ).tocsr()
    return result


def _charge_resolved_trace(
    factors: Sequence[Mapping[int, sp.csr_array]],
    *,
    start_states: Sequence[int],
    n_boundary_states: int,
) -> _ChargeResolvedTrace:
    states = tuple(int(state) for state in start_states)
    if not states:
        return _ChargeResolvedTrace(log_counts={}, counts={})

    initial = np.zeros((len(states), n_boundary_states), dtype=np.float64)
    initial[np.arange(len(states), dtype=np.int64), np.asarray(states, dtype=np.int64)] = 1.0
    rows_by_charge: dict[int, npt.NDArray[np.float64]] = {0: initial}
    log_scale = 0.0

    for factor in factors:
        next_rows: dict[int, npt.NDArray[np.float64]] = {}
        for accumulated_charge, rows in rows_by_charge.items():
            for factor_charge, matrix in factor.items():
                charge = int(accumulated_charge + factor_charge)
                contribution = np.asarray(rows @ matrix, dtype=np.float64)
                if not np.any(contribution):
                    continue
                existing = next_rows.get(charge)
                if existing is None:
                    next_rows[charge] = contribution
                else:
                    existing += contribution

        if not next_rows:
            return _ChargeResolvedTrace(log_counts={}, counts={})

        scale = float(sum(np.sum(rows) for rows in next_rows.values()))
        if scale <= 0.0 or not isfinite(scale):
            raise ValueError("charge-resolved transfer contraction became non-finite.")
        for rows in next_rows.values():
            rows /= scale
        rows_by_charge = next_rows
        log_scale += log(scale)

    state_array = np.asarray(states, dtype=np.int64)
    log_counts: dict[int, float] = {}
    counts: dict[int, float] = {}
    for charge, rows in rows_by_charge.items():
        coefficient = float(np.sum(rows[np.arange(len(states)), state_array]))
        if coefficient <= 0.0:
            continue
        log_count = log_scale + log(coefficient)
        log_counts[int(charge)] = log_count
        counts[int(charge)] = _safe_exp(log_count)
    return _ChargeResolvedTrace(log_counts=log_counts, counts=counts)


def _normalize_square_qdm_x_coordinates(
    coordinates: Sequence[SquareQDMLinkCoordinate],
    *,
    lx: int,
    periodic: bool,
) -> tuple[tuple[SquareQDMLinkCoordinate, ...], int]:
    if not coordinates:
        raise ValueError("coordinates must not be empty.")
    if lx <= 0:
        raise ValueError("lx must be positive.")

    if not periodic:
        origin = min(coordinate.x for coordinate in coordinates)
        return (
            tuple(
                SquareQDMLinkCoordinate(
                    x=coordinate.x - origin,
                    y=coordinate.y,
                    kind=coordinate.kind,
                )
                for coordinate in coordinates
            ),
            origin,
        )

    candidates: list[tuple[int, tuple[SquareQDMLinkCoordinate, ...], int]] = []
    for cut in range(lx):
        unwrapped = tuple(
            SquareQDMLinkCoordinate(
                x=(coordinate.x - cut) % lx,
                y=coordinate.y,
                kind=coordinate.kind,
            )
            for coordinate in coordinates
        )
        minimum = min(coordinate.x for coordinate in unwrapped)
        normalized = tuple(
            SquareQDMLinkCoordinate(
                x=coordinate.x - minimum,
                y=coordinate.y,
                kind=coordinate.kind,
            )
            for coordinate in unwrapped
        )
        span = max(coordinate.x + (1 if coordinate.kind == "x" else 0) for coordinate in normalized)
        origin = (cut + minimum) % lx
        candidates.append((span, normalized, origin))

    _span, normalized, origin = min(
        candidates,
        key=lambda item: (
            item[0],
            tuple((coordinate.x, coordinate.y, coordinate.kind) for coordinate in item[1]),
        ),
    )
    return normalized, origin


def _square_qdm_column_transitions(
    circumference: int,
) -> tuple[SquareQDMColumnTransition, ...]:
    all_bits = (1 << circumference) - 1
    transitions: list[SquareQDMColumnTransition] = []

    for vertical_mask in range(1 << circumference):
        shifted = _rotate_left(vertical_mask, width=circumference)
        if vertical_mask & shifted:
            continue

        vertically_covered = vertical_mask | shifted
        horizontally_covered = all_bits ^ vertically_covered
        for incoming_mask in _submasks(horizontally_covered):
            outgoing_mask = horizontally_covered ^ incoming_mask
            transitions.append(
                SquareQDMColumnTransition(
                    incoming_mask=int(incoming_mask),
                    outgoing_mask=int(outgoing_mask),
                    vertical_mask=int(vertical_mask),
                )
            )

    transitions.sort(
        key=lambda transition: (
            transition.incoming_mask,
            transition.outgoing_mask,
            transition.vertical_mask,
        )
    )
    return tuple(transitions)


def _propagate_normalized(
    vector: npt.NDArray[np.float64],
    *,
    matrix: sp.csr_array,
    steps: int,
    side: Literal["left", "right"],
) -> tuple[npt.NDArray[np.float64], float]:
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    result = np.asarray(vector, dtype=np.float64).reshape(-1)
    log_scale = 0.0
    for _ in range(steps):
        if side == "left":
            result = np.asarray(result @ matrix).reshape(-1)
        else:
            result = np.asarray(matrix @ result).reshape(-1)

        scale = float(np.sum(result))
        if scale <= 0.0:
            return result, float("-inf")
        result /= scale
        log_scale += log(scale)

    return result, log_scale
