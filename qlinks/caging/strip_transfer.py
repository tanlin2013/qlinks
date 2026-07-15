from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.caging.thermodynamic import LocalWitness, LocalWitnessTemplate
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models import SquareQDMModel
from qlinks.variables import VariableKind

SquareQDMLinkKind = Literal["x", "y"]
StripBoundaryCondition = Literal["open", "periodic"]


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
            "lengths": self.lengths,
            "expectations": self.expectations,
            "evaluations": tuple(evaluation.to_summary_dict() for evaluation in self.evaluations),
        }


@dataclass(frozen=True, slots=True)
class _PeriodicTransferSpectrum:
    eigenvalues: npt.NDArray[np.float64]
    insertion_diagonal: npt.NDArray[np.float64]
    scale: float


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

    @property
    def n_boundary_states(self) -> int:
        return 1 << self.circumference

    def witness_insertion_matrix(
        self,
        placement: SquareQDMWitnessPlacement,
    ) -> sp.csr_array:
        """Contract the exact local ``Q_R`` weight into one strip insertion."""
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

        entries: dict[tuple[int, int], float] = {}

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

        def extend_path(
            path: list[SquareQDMColumnTransition],
            *,
            incoming_mask: int,
        ) -> None:
            if len(path) == window_width:
                weight = path_q_weight(path)
                if weight != 0.0:
                    key = (path[0].incoming_mask, path[-1].outgoing_mask)
                    entries[key] = entries.get(key, 0.0) + weight
                return

            for transition in self._transitions_by_incoming[incoming_mask]:
                path.append(transition)
                extend_path(path, incoming_mask=transition.outgoing_mask)
                path.pop()

        for incoming_mask in range(self.n_boundary_states):
            extend_path([], incoming_mask=incoming_mask)

        if not entries:
            return sp.csr_array(
                (self.n_boundary_states, self.n_boundary_states),
                dtype=np.float64,
            )

        rows = np.fromiter((key[0] for key in entries), dtype=np.int64)
        cols = np.fromiter((key[1] for key in entries), dtype=np.int64)
        data = np.fromiter(entries.values(), dtype=np.float64)
        return sp.coo_array(
            (data, (rows, cols)),
            shape=(self.n_boundary_states, self.n_boundary_states),
            dtype=np.float64,
        ).tocsr()

    def evaluate_witness(
        self,
        placement: SquareQDMWitnessPlacement,
        *,
        length: int,
        boundary_x: StripBoundaryCondition = "open",
        insertion_x: int | None = None,
    ) -> SquareQDMStripWitnessEvaluation:
        """Evaluate ``Tr(Q_R)/dim(H)`` without enumerating dimer coverings."""
        self._validate_placement(placement)
        if length < placement.window_width:
            raise ValueError("length must be at least the witness window width.")
        if boundary_x not in ("open", "periodic"):
            raise ValueError("boundary_x must be 'open' or 'periodic'.")

        insertion = self.witness_insertion_matrix(placement)
        if boundary_x == "open":
            if insertion_x is None:
                insertion_x = (length - placement.window_width) // 2
            return self._evaluate_open(
                placement,
                insertion=insertion,
                length=length,
                insertion_x=int(insertion_x),
            )

        if insertion_x is not None:
            raise ValueError("insertion_x is not used for periodic x boundaries.")
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
    ) -> SquareQDMStripScalingReport:
        """Evaluate a fixed witness over several strip lengths."""
        self._validate_placement(placement)
        if boundary_x not in ("open", "periodic"):
            raise ValueError("boundary_x must be 'open' or 'periodic'.")

        insertion = self.witness_insertion_matrix(placement)
        periodic_spectrum = self._periodic_spectrum(insertion) if boundary_x == "periodic" else None
        evaluations: list[SquareQDMStripWitnessEvaluation] = []
        for raw_length in lengths:
            length = int(raw_length)
            if length < placement.window_width:
                raise ValueError("all lengths must be at least the witness window width.")
            if boundary_x == "periodic":
                evaluation = self._evaluate_periodic(
                    placement,
                    insertion=insertion,
                    length=length,
                    spectrum=periodic_spectrum,
                )
            else:
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
