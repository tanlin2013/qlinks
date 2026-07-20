"""Model-independent numerical evidence helpers for caged eigenstates.

This module collects reporting and optimization routines that are useful for
manuscript-facing numerical diagnostics without assuming a particular lattice
model.  Model-specific construction of candidate states, local operators, or
real-space coordinates remains in the corresponding model/notebook layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as sp

from qlinks.caging.spectral import MatrixLike, diagnose_eigenpair
from qlinks.caging.stability import partition_cage_hamiltonian

WindowMetric = Literal["chebyshev", "manhattan"]


def _as_state(vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise ValueError("state must not be the zero vector.")
    return state / norm


def _dense(matrix: MatrixLike) -> npt.NDArray[np.complex128]:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.complex128)
    return np.asarray(matrix, dtype=np.complex128)


def _svd_rank_and_gap(
    matrix: MatrixLike,
    *,
    tolerance: float,
) -> tuple[int, int, float, npt.NDArray[np.float64]]:
    array = _dense(matrix)
    if array.size == 0:
        singular_values = np.empty(0, dtype=np.float64)
        return 0, int(array.shape[1]), float("inf"), singular_values
    singular_values = np.asarray(
        scipy_linalg.svdvals(array),
        dtype=np.float64,
    )
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(array.shape[1] - rank)
    positive = singular_values[singular_values > tolerance]
    gap = float(np.min(positive)) if positive.size else float("inf")
    return rank, nullity, gap, singular_values


@dataclass(frozen=True, slots=True)
class CageFiniteSizeScorecard:
    """Reproducible finite-size certificate for one candidate caged state."""

    hilbert_dimension: int
    candidate_shell_size: int
    actual_support_size: int
    boundary_shape: tuple[int, int]
    boundary_rank: int
    boundary_nullity: int
    boundary_singular_gap: float
    internal_residual: float
    boundary_residual: float
    eigenpair_residual: float
    relative_eigenpair_residual: float
    energy: complex
    tolerance: float
    metadata: dict[str, object] = field(default_factory=dict)
    singular_values: tuple[float, ...] = ()

    @property
    def is_certified(self) -> bool:
        return (
            self.internal_residual <= self.tolerance
            and self.boundary_residual <= self.tolerance
            and self.relative_eigenpair_residual <= self.tolerance
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "candidate_shell_size": self.candidate_shell_size,
            "actual_support_size": self.actual_support_size,
            "boundary_rows": self.boundary_shape[0],
            "boundary_columns": self.boundary_shape[1],
            "boundary_rank": self.boundary_rank,
            "boundary_nullity": self.boundary_nullity,
            "boundary_singular_gap": self.boundary_singular_gap,
            "internal_residual": self.internal_residual,
            "boundary_residual": self.boundary_residual,
            "eigenpair_residual": self.eigenpair_residual,
            "relative_eigenpair_residual": self.relative_eigenpair_residual,
            "energy": self.energy,
            "tolerance": self.tolerance,
            "is_certified": self.is_certified,
            **dict(self.metadata),
        }


def cage_finite_size_scorecard(
    hamiltonian: MatrixLike,
    candidate_shell: Sequence[int],
    state: npt.ArrayLike,
    *,
    kinetic: MatrixLike | None = None,
    actual_support: Sequence[int] | None = None,
    amplitude_tolerance: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
    metadata: dict[str, object] | None = None,
) -> CageFiniteSizeScorecard:
    """Build the manuscript scorecard for one finite-size cage.

    The boundary matrix is evaluated on the complete candidate shell, while
    the internal/boundary residuals are evaluated on the actual nonzero support.
    This keeps the shell nullity separate from a postselected localized basis.
    """
    if amplitude_tolerance <= 0.0 or rank_tolerance <= 0.0:
        raise ValueError("tolerances must be positive.")
    shell = tuple(int(index) for index in candidate_shell)
    if not shell:
        raise ValueError("candidate_shell must not be empty.")
    vector = _as_state(state)
    if vector.size != hamiltonian.shape[0]:
        raise ValueError("hamiltonian and state have incompatible dimensions.")
    if actual_support is None:
        support = tuple(int(i) for i in np.flatnonzero(np.abs(vector) > amplitude_tolerance))
    else:
        support = tuple(int(index) for index in actual_support)
    if not support:
        raise ValueError("actual support must not be empty.")

    leakage_operator = hamiltonian if kinetic is None else kinetic
    shell_blocks = partition_cage_hamiltonian(leakage_operator, shell)
    rank, nullity, gap, singular_values = _svd_rank_and_gap(
        shell_blocks.boundary,
        tolerance=rank_tolerance,
    )

    support_blocks = partition_cage_hamiltonian(hamiltonian, support)
    local = vector[np.asarray(support, dtype=np.int64)]
    local = local / np.linalg.norm(local)
    action_inside = np.asarray(support_blocks.internal @ local, dtype=np.complex128).reshape(-1)
    energy = complex(np.vdot(local, action_inside))
    internal_residual = float(np.linalg.norm(action_inside - energy * local))
    boundary_residual = float(
        np.linalg.norm(np.asarray(support_blocks.boundary @ local, dtype=np.complex128).reshape(-1))
    )
    eigenpair = diagnose_eigenpair(hamiltonian, vector)

    return CageFiniteSizeScorecard(
        hilbert_dimension=int(hamiltonian.shape[0]),
        candidate_shell_size=len(shell),
        actual_support_size=len(support),
        boundary_shape=(
            int(shell_blocks.boundary.shape[0]),
            int(shell_blocks.boundary.shape[1]),
        ),
        boundary_rank=rank,
        boundary_nullity=nullity,
        boundary_singular_gap=gap,
        internal_residual=internal_residual,
        boundary_residual=boundary_residual,
        eigenpair_residual=eigenpair.residual_norm,
        relative_eigenpair_residual=eigenpair.relative_residual_norm,
        energy=eigenpair.energy,
        tolerance=max(amplitude_tolerance, rank_tolerance),
        metadata={} if metadata is None else dict(metadata),
        singular_values=tuple(float(value) for value in singular_values),
    )


@dataclass(frozen=True, slots=True)
class WindowedAnnihilatorPoint:
    """Best coefficient-normalized annihilator inside one spatial window."""

    radius: float
    center: tuple[float, ...]
    selected_operator_indices: tuple[int, ...]
    active_operator_indices: tuple[int, ...]
    minimum_residual: float
    rank: int
    nullity: int
    coefficient_support_size: int
    best_coefficients: tuple[complex, ...]
    singular_values: tuple[float, ...]

    @property
    def n_selected(self) -> int:
        return len(self.selected_operator_indices)

    @property
    def n_active(self) -> int:
        return len(self.active_operator_indices)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "center": self.center,
            "n_selected": self.n_selected,
            "n_active": self.n_active,
            "selected_operator_indices": self.selected_operator_indices,
            "active_operator_indices": self.active_operator_indices,
            "minimum_residual": self.minimum_residual,
            "rank": self.rank,
            "nullity": self.nullity,
            "coefficient_support_size": self.coefficient_support_size,
        }


@dataclass(frozen=True, slots=True)
class WindowedAnnihilatorScan:
    """Minimum local-annihilation residual as a function of allowed radius."""

    points: tuple[WindowedAnnihilatorPoint, ...]
    metric: WindowMetric
    periodic_box: tuple[float, ...] | None
    action_normalization: str
    action_tolerance: float
    rank_tolerance: float

    @property
    def bounded_annihilation_radius(self) -> float:
        for point in self.points:
            if point.minimum_residual <= self.rank_tolerance:
                return point.radius
        return float("inf")

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_points": len(self.points),
            "metric": self.metric,
            "periodic_box": self.periodic_box,
            "action_normalization": self.action_normalization,
            "action_tolerance": self.action_tolerance,
            "rank_tolerance": self.rank_tolerance,
            "bounded_annihilation_radius": self.bounded_annihilation_radius,
        }


def _periodic_displacement(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    periodic_box: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    delta = np.abs(left - right)
    if periodic_box is not None:
        delta = np.minimum(delta, periodic_box - delta)
    return delta


def _distance(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    *,
    metric: WindowMetric,
    periodic_box: npt.NDArray[np.float64] | None,
) -> float:
    delta = _periodic_displacement(left, right, periodic_box)
    if metric == "chebyshev":
        return float(np.max(delta))
    if metric == "manhattan":
        return float(np.sum(delta))
    raise ValueError(f"unsupported metric: {metric}")


def _smallest_singular_vector(
    action_matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[float, int, int, npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    rows, columns = action_matrix.shape
    if columns == 0:
        return float("inf"), 0, 0, np.empty(0, dtype=np.complex128), np.empty(0)
    _left, singular_values, vh = scipy_linalg.svd(action_matrix, full_matrices=True)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(columns - rank)
    if nullity > 0:
        coefficients = np.asarray(vh.conj().T[:, rank], dtype=np.complex128)
        residual = float(np.linalg.norm(action_matrix @ coefficients))
    else:
        coefficients = np.asarray(vh.conj().T[:, -1], dtype=np.complex128)
        residual = float(singular_values[-1])
    coefficients /= np.linalg.norm(coefficients)
    return residual, rank, nullity, coefficients, singular_values


def scan_windowed_operator_annihilators(
    operators: Sequence[MatrixLike],
    state: npt.ArrayLike,
    operator_centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    *,
    periodic_box: Sequence[float] | None = None,
    metric: WindowMetric = "chebyshev",
    normalize_actions: bool = True,
    action_tolerance: float = 1.0e-12,
    rank_tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-8,
) -> WindowedAnnihilatorScan:
    """Optimize a collective local annihilator in growing spatial windows.

    For each candidate center and radius, columns of the action matrix are
    ``O_j |psi>`` for operators whose real-space centers lie in the window.
    The smallest singular value is the minimum residual over coefficient
    vectors of unit Euclidean norm.  Normalizing individual action columns is
    useful when comparing locality rather than bare operator conventions.
    """
    if len(operators) != len(operator_centers):
        raise ValueError("operators and operator_centers must have the same length.")
    if not operators:
        raise ValueError("operators must not be empty.")
    if action_tolerance <= 0.0 or rank_tolerance <= 0.0:
        raise ValueError("tolerances must be positive.")
    normalized_state = _as_state(state)
    centers = np.asarray(operator_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] != len(operators):
        raise ValueError("operator_centers must have shape (n_operators, dimension).")
    box = None if periodic_box is None else np.asarray(periodic_box, dtype=np.float64)
    if box is not None:
        if box.shape != (centers.shape[1],) or np.any(box <= 0.0):
            raise ValueError("periodic_box must have one positive length per dimension.")

    actions: list[npt.NDArray[np.complex128]] = []
    action_norms: list[float] = []
    for operator in operators:
        if operator.shape != (normalized_state.size, normalized_state.size):
            raise ValueError("every operator must act on the state Hilbert space.")
        action = np.asarray(operator @ normalized_state, dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(action))
        action_norms.append(norm)
        actions.append(action / norm if normalize_actions and norm > action_tolerance else action)

    active_mask = np.asarray(action_norms) > action_tolerance
    unique_centers = np.unique(centers, axis=0)
    points: list[WindowedAnnihilatorPoint] = []
    for radius_raw in radii:
        radius = float(radius_raw)
        if radius < 0.0:
            raise ValueError("radii must be non-negative.")
        best: WindowedAnnihilatorPoint | None = None
        for center in unique_centers:
            selected = tuple(
                index
                for index, operator_center in enumerate(centers)
                if _distance(
                    operator_center,
                    center,
                    metric=metric,
                    periodic_box=box,
                )
                <= radius + 1.0e-12
            )
            active = tuple(index for index in selected if active_mask[index])
            if not active:
                candidate = WindowedAnnihilatorPoint(
                    radius=radius,
                    center=tuple(float(value) for value in center),
                    selected_operator_indices=selected,
                    active_operator_indices=active,
                    minimum_residual=float("inf"),
                    rank=0,
                    nullity=0,
                    coefficient_support_size=0,
                    best_coefficients=(),
                    singular_values=(),
                )
            else:
                action_matrix = np.column_stack([actions[index] for index in active])
                residual, rank, nullity, coefficients, singular_values = _smallest_singular_vector(
                    action_matrix, tolerance=rank_tolerance
                )
                coefficient_support_size = int(np.sum(np.abs(coefficients) > coefficient_tolerance))
                candidate = WindowedAnnihilatorPoint(
                    radius=radius,
                    center=tuple(float(value) for value in center),
                    selected_operator_indices=selected,
                    active_operator_indices=active,
                    minimum_residual=residual,
                    rank=rank,
                    nullity=nullity,
                    coefficient_support_size=coefficient_support_size,
                    best_coefficients=tuple(complex(value) for value in coefficients),
                    singular_values=tuple(float(value) for value in singular_values),
                )
            if best is None or (
                candidate.minimum_residual,
                candidate.n_active,
                candidate.center,
            ) < (
                best.minimum_residual,
                best.n_active,
                best.center,
            ):
                best = candidate
        assert best is not None
        points.append(best)

    return WindowedAnnihilatorScan(
        points=tuple(points),
        metric=metric,
        periodic_box=None if box is None else tuple(float(value) for value in box),
        action_normalization="column_norm" if normalize_actions else "operator_convention",
        action_tolerance=action_tolerance,
        rank_tolerance=rank_tolerance,
    )


@dataclass(frozen=True, slots=True)
class BetaZeroMatchingReport:
    """Linear coefficient space matching a scar energy to the beta-zero mean."""

    scar_term_expectations: tuple[float, ...]
    thermal_term_expectations: tuple[float, ...]
    mismatch_vector: tuple[float, ...]
    constraint_rank: int
    compatible_dimension: int
    compatible_basis: npt.NDArray[np.float64]
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatible_basis",
            np.asarray(self.compatible_basis, dtype=np.float64).copy(),
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_terms": len(self.mismatch_vector),
            "constraint_rank": self.constraint_rank,
            "compatible_dimension": self.compatible_dimension,
            "mismatch_norm": float(np.linalg.norm(self.mismatch_vector)),
            "tolerance": self.tolerance,
        }


def beta_zero_matching_subspace(
    scar_term_expectations: npt.ArrayLike,
    thermal_term_expectations: npt.ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> BetaZeroMatchingReport:
    """Return coefficients ``c`` satisfying ``c·(scar-thermal)=0``."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    scar = np.asarray(scar_term_expectations, dtype=np.float64).reshape(-1)
    thermal = np.asarray(thermal_term_expectations, dtype=np.float64).reshape(-1)
    if scar.shape != thermal.shape or scar.size == 0:
        raise ValueError("expectation arrays must have the same nonzero length.")
    mismatch = scar - thermal
    row = mismatch.reshape(1, -1)
    singular_values = scipy_linalg.svdvals(row)
    rank = int(np.sum(singular_values > tolerance))
    basis = scipy_linalg.null_space(row, rcond=tolerance)
    return BetaZeroMatchingReport(
        scar_term_expectations=tuple(float(value) for value in scar),
        thermal_term_expectations=tuple(float(value) for value in thermal),
        mismatch_vector=tuple(float(value) for value in mismatch),
        constraint_rank=rank,
        compatible_dimension=int(basis.shape[1]),
        compatible_basis=np.asarray(basis, dtype=np.float64),
        tolerance=tolerance,
    )


def project_coefficients_to_beta_zero_match(
    coefficients: npt.ArrayLike,
    report: BetaZeroMatchingReport,
    *,
    normalize: bool = False,
) -> npt.NDArray[np.float64]:
    """Orthogonally project coefficients onto a beta-zero matching space."""
    vector = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if vector.size != len(report.mismatch_vector):
        raise ValueError("coefficient vector has incompatible length.")
    basis = report.compatible_basis
    projected = basis @ (basis.T @ vector) if basis.shape[1] else np.zeros_like(vector)
    if normalize:
        norm = float(np.linalg.norm(projected))
        if norm == 0.0:
            raise ValueError("projected coefficient vector vanishes.")
        projected = projected / norm
    return np.asarray(projected, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Quasi1DSequencePoint:
    """One system-size point entering a fixed-width thermodynamic audit."""

    length: int
    width: int
    exact_residual: float
    witness_radius: float
    transverse_witness_span: float | None = None
    thermal_second_moment: float | None = None
    interference_gap: float | None = None
    compatibility_rank: int | None = None
    local_parameter_count: int | None = None
    support_size: int | None = None
    sector_dimension: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "length": self.length,
            "width": self.width,
            "exact_residual": self.exact_residual,
            "witness_radius": self.witness_radius,
            "transverse_witness_span": self.transverse_witness_span,
            "thermal_second_moment": self.thermal_second_moment,
            "interference_gap": self.interference_gap,
            "compatibility_rank": self.compatibility_rank,
            "local_parameter_count": self.local_parameter_count,
            "support_size": self.support_size,
            "sector_dimension": self.sector_dimension,
        }
        if self.compatibility_rank is not None and self.local_parameter_count:
            result["compatibility_codimension_fraction"] = (
                self.compatibility_rank / self.local_parameter_count
            )
        if self.support_size is not None and self.sector_dimension:
            result["support_fraction"] = self.support_size / self.sector_dimension
        result.update(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class Quasi1DAuditReport:
    """Conservative assessment of what a fixed-width sequence establishes."""

    points: tuple[Quasi1DSequencePoint, ...]
    energy_density_mismatch: float | None
    level_gap_ratio: float | None
    zero_mode_fraction: float | None
    issues: tuple[str, ...]
    established: tuple[str, ...]
    tolerance: float

    @property
    def is_exact_sequence(self) -> bool:
        return (
            bool(self.points)
            and max(point.exact_residual for point in self.points) <= self.tolerance
        )

    @property
    def has_bounded_witness(self) -> bool:
        return bool(self.points) and np.isfinite(max(point.witness_radius for point in self.points))

    @property
    def has_positive_thermal_activity(self) -> bool:
        values = [
            point.thermal_second_moment
            for point in self.points
            if point.thermal_second_moment is not None
        ]
        return bool(values) and min(values) > self.tolerance

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_points": len(self.points),
            "is_exact_sequence": self.is_exact_sequence,
            "has_bounded_witness": self.has_bounded_witness,
            "has_positive_thermal_activity": self.has_positive_thermal_activity,
            "energy_density_mismatch": self.energy_density_mismatch,
            "level_gap_ratio": self.level_gap_ratio,
            "zero_mode_fraction": self.zero_mode_fraction,
            "issues": self.issues,
            "established": self.established,
            "tolerance": self.tolerance,
        }


def audit_quasi_1d_sequence(
    points: Sequence[Quasi1DSequencePoint],
    *,
    energy_density_mismatch: float | None = None,
    level_gap_ratio: float | None = None,
    zero_mode_fraction: float | None = None,
    tolerance: float = 1.0e-9,
) -> Quasi1DAuditReport:
    """Separate exact quasi-1D results from unresolved ETH/topology claims."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    ordered = tuple(sorted(points, key=lambda point: point.length))
    if not ordered:
        raise ValueError("points must not be empty.")
    widths = {point.width for point in ordered}
    issues: list[str] = []
    established: list[str] = []

    if len(widths) == 1:
        established.append("fixed-width thermodynamic sequence")
        issues.append(
            "The transverse width is fixed, so the limit is quasi-one-dimensional "
            "and does not establish a two-dimensional thermodynamic limit."
        )
    else:
        issues.append("The supplied points do not define one fixed-width sequence.")

    if max(point.exact_residual for point in ordered) <= tolerance:
        established.append("exact caged eigenstates along all tested sizes")
    else:
        issues.append("At least one tested size fails the exact-state residual tolerance.")

    radii = np.asarray([point.witness_radius for point in ordered], dtype=np.float64)
    if np.all(np.isfinite(radii)) and np.max(radii) - np.min(radii) <= tolerance:
        established.append("size-independent tested witness radius")
    elif np.all(np.isfinite(radii)):
        issues.append(
            "The minimal tested witness radius changes with length; "
            "boundedness requires further scaling."
        )
    else:
        issues.append("No finite local witness was found at one or more sizes.")

    transverse_fractions = [
        point.transverse_witness_span / point.width
        for point in ordered
        if point.transverse_witness_span is not None and point.width > 0
    ]
    if transverse_fractions and max(transverse_fractions) >= 0.5:
        issues.append(
            "The fixed witness occupies a macroscopic fraction of the transverse circumference; "
            "its bounded strip support does not show that the same local diagnostic "
            "remains independent of width."
        )

    thermal_values = [
        point.thermal_second_moment for point in ordered if point.thermal_second_moment is not None
    ]
    if thermal_values and min(thermal_values) > tolerance:
        established.append("positive thermal witness activity on all reported sizes")
    else:
        issues.append(
            "A positive thermodynamic lower bound on the thermal second moment "
            "has not been established."
        )

    compatibility = [
        (point.compatibility_rank, point.local_parameter_count)
        for point in ordered
        if point.compatibility_rank is not None and point.local_parameter_count
    ]
    if compatibility:
        fractions = np.asarray([rank / count for rank, count in compatibility], dtype=float)
        if np.min(fractions) > tolerance:
            issues.append(
                "The preserving deformation class has an extensive local compatibility codimension;"
                " this is structural fine tuning rather than finite-codimension protection."
            )

    if energy_density_mismatch is not None:
        if abs(energy_density_mismatch) <= tolerance:
            established.append("scar energy density matched to the beta-zero ensemble")
        else:
            issues.append(
                "The scar energy density is not at beta zero; "
                "a finite-temperature microcanonical comparison is required."
            )

    if zero_mode_fraction is not None and zero_mode_fraction > 0.05:
        issues.append(
            "A large zero-mode manifold contaminates level statistics and microcanonical windows "
            "near the cage energy."
        )
    if level_gap_ratio is None:
        issues.append(
            "Thermal level statistics have not been supplied for a fully desymmetrized "
            "surrounding spectrum."
        )
    elif abs(level_gap_ratio - 0.5307) < abs(level_gap_ratio - (2.0 * np.log(2.0) - 1.0)):
        established.append("finite-size level statistics closer to GOE than Poisson")
    else:
        issues.append("Finite-size level statistics are not closer to GOE than to Poisson.")

    issues.append(
        "A width-four strip can realize phases and constraints different from "
        "the two-dimensional square QDM; quasi-1D thermal evidence cannot be "
        "extrapolated to the 2D phase diagram."
    )
    return Quasi1DAuditReport(
        points=ordered,
        energy_density_mismatch=energy_density_mismatch,
        level_gap_ratio=level_gap_ratio,
        zero_mode_fraction=zero_mode_fraction,
        issues=tuple(issues),
        established=tuple(established),
        tolerance=tolerance,
    )


CompatibilityTargetMode = Literal["fixed_vectors", "invariant_subspace"]


@dataclass(frozen=True, slots=True)
class OperatorCoefficientCompatibilityReport:
    """Linear coefficient space preserving vectors or their common subspace."""

    n_operators: int
    target_dimension: int
    mode: CompatibilityTargetMode
    rank: int
    compatible_dimension: int
    singular_gap: float
    constraint_matrix: npt.NDArray[np.complex128]
    compatible_basis: npt.NDArray[np.complex128]
    singular_values: tuple[float, ...]
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_matrix",
            np.asarray(self.constraint_matrix, dtype=np.complex128).copy(),
        )
        object.__setattr__(
            self,
            "compatible_basis",
            np.asarray(self.compatible_basis, dtype=np.complex128).copy(),
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_operators": self.n_operators,
            "target_dimension": self.target_dimension,
            "mode": self.mode,
            "rank": self.rank,
            "compatible_dimension": self.compatible_dimension,
            "codimension": self.rank,
            "singular_gap": self.singular_gap,
            "tolerance": self.tolerance,
        }


def operator_coefficient_compatibility(
    operators: Sequence[MatrixLike],
    target_states: npt.ArrayLike,
    *,
    mode: CompatibilityTargetMode = "fixed_vectors",
    tolerance: float = 1.0e-10,
) -> OperatorCoefficientCompatibilityReport:
    """Find local-operator coefficients preserving target vectors or a subspace.

    For ``fixed_vectors``, every supplied orthonormal target vector must remain
    an eigenvector separately, allowing a different energy shift for each.
    For ``invariant_subspace``, only the span must remain invariant.
    """
    if not operators:
        raise ValueError("operators must not be empty.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    states = np.asarray(target_states, dtype=np.complex128)
    if states.ndim == 1:
        states = states[:, None]
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("target_states must contain at least one vector.")
    q, r = np.linalg.qr(states)
    diagonal = np.abs(np.diag(r))
    target_rank = int(np.sum(diagonal > tolerance))
    if target_rank != states.shape[1]:
        raise ValueError("target_states must be linearly independent.")
    targets = np.asarray(q[:, :target_rank], dtype=np.complex128)
    dimension = targets.shape[0]
    if any(operator.shape != (dimension, dimension) for operator in operators):
        raise ValueError("every operator must act on the target Hilbert space.")

    columns: list[npt.NDArray[np.complex128]] = []
    if mode == "fixed_vectors":
        for operator in operators:
            pieces: list[npt.NDArray[np.complex128]] = []
            for index in range(target_rank):
                vector = targets[:, index]
                action = np.asarray(operator @ vector, dtype=np.complex128).reshape(-1)
                action -= vector * np.vdot(vector, action)
                pieces.append(action)
            columns.append(np.concatenate(pieces))
    elif mode == "invariant_subspace":
        projector = targets @ targets.conj().T
        complement = np.eye(dimension, dtype=np.complex128) - projector
        for operator in operators:
            action = complement @ np.asarray(operator @ targets, dtype=np.complex128)
            columns.append(np.asarray(action, dtype=np.complex128).reshape(-1, order="F"))
    else:
        raise ValueError(f"unsupported compatibility mode: {mode}")

    constraint = np.column_stack(columns)
    singular_values = np.asarray(scipy_linalg.svdvals(constraint), dtype=np.float64)
    rank = int(np.sum(singular_values > tolerance))
    positive = singular_values[singular_values > tolerance]
    gap = float(np.min(positive)) if positive.size else float("inf")
    compatible_basis = scipy_linalg.null_space(constraint, rcond=tolerance)
    return OperatorCoefficientCompatibilityReport(
        n_operators=len(operators),
        target_dimension=target_rank,
        mode=mode,
        rank=rank,
        compatible_dimension=int(compatible_basis.shape[1]),
        singular_gap=gap,
        constraint_matrix=constraint,
        compatible_basis=np.asarray(compatible_basis, dtype=np.complex128),
        singular_values=tuple(float(value) for value in singular_values),
        tolerance=tolerance,
    )
