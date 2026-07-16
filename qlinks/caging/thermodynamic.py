from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.caging.classification import CageClassificationReport, IZProbeMechanismLabel
from qlinks.caging.support import (
    ReducedIZPatternSupport,
    distinct_reduced_iz_pattern_supports,
)
from qlinks.open_system.local_recycling import embed_local_pattern_operator

ReducedIZPatternKey = tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, float]], ...]
WitnessNormalization = Literal["none", "operator_norm", "frobenius_norm"]


def _complex_from_key(value: tuple[float, float]) -> complex:
    return complex(float(value[0]), float(value[1]))


def _normalize_weights(weights: npt.ArrayLike | None, *, size: int) -> npt.NDArray[np.float64]:
    if size <= 0:
        raise ValueError("size must be positive.")
    if weights is None:
        return np.full(size, 1.0 / float(size), dtype=np.float64)

    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(f"weights must have shape ({size},).")
    if np.any(arr < 0.0):
        raise ValueError("weights must be non-negative.")
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a finite positive sum.")
    return arr / total


def _state_matrix(
    states: npt.ArrayLike,
    *,
    dim: int,
    normalize_columns: bool = True,
) -> npt.NDArray[np.complex128]:
    matrix = np.asarray(states, dtype=np.complex128)
    if matrix.ndim == 1:
        if matrix.size != dim:
            raise ValueError("state vector has incompatible dimension.")
        matrix = matrix.reshape(dim, 1)
    elif matrix.ndim == 2:
        if matrix.shape[0] == dim:
            pass
        elif matrix.shape[1] == dim:
            matrix = matrix.T
        else:
            raise ValueError("states must have shape (dim, n_states) or (n_states, dim).")
    else:
        raise ValueError("states must be a vector or a two-dimensional matrix.")

    if matrix.shape[1] == 0:
        raise ValueError("states must contain at least one state.")

    result = matrix.astype(np.complex128, copy=True)
    if normalize_columns:
        norms = np.linalg.norm(result, axis=0)
        if np.any(norms == 0.0):
            raise ValueError("states must not contain a zero vector.")
        result /= norms[np.newaxis, :]
    return result


@dataclass(frozen=True, slots=True)
class LocalWitnessTemplate:
    """Size-independent local row operator reconstructed from a reduced-IZ pattern.

    The template stores only local configurations and matrix elements.  It does
    not store global variable indices, so the same object can be embedded in
    several system sizes or translated to several locations.
    """

    pattern_key: ReducedIZPatternKey
    local_patterns: tuple[tuple[int, ...], ...]
    local_operator: npt.NDArray[np.complex128]
    source_zero_indices: tuple[int, ...] = ()
    mechanism_labels: tuple[IZProbeMechanismLabel, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patterns = tuple(tuple(int(value) for value in pattern) for pattern in self.local_patterns)
        if not patterns:
            raise ValueError("local_patterns must not be empty.")
        n_variables = len(patterns[0])
        if any(len(pattern) != n_variables for pattern in patterns):
            raise ValueError("all local patterns must have the same width.")
        if len(set(patterns)) != len(patterns):
            raise ValueError("local_patterns must not contain duplicates.")

        operator = np.asarray(self.local_operator, dtype=np.complex128)
        expected_shape = (len(patterns), len(patterns))
        if operator.shape != expected_shape:
            raise ValueError(
                f"local_operator has shape {operator.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(operator)):
            raise ValueError("local_operator must contain finite values.")

        object.__setattr__(self, "local_patterns", patterns)
        object.__setattr__(self, "local_operator", operator.copy())
        object.__setattr__(
            self,
            "source_zero_indices",
            tuple(int(index) for index in self.source_zero_indices),
        )
        object.__setattr__(self, "mechanism_labels", tuple(self.mechanism_labels))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_variables(self) -> int:
        return len(self.local_patterns[0])

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)

    @property
    def q_operator(self) -> npt.NDArray[np.complex128]:
        return self.local_operator.conj().T @ self.local_operator

    @property
    def operator_norm(self) -> float:
        return float(np.linalg.norm(self.local_operator, ord=2))

    @property
    def q_operator_norm(self) -> float:
        return float(self.operator_norm**2)

    @property
    def frobenius_norm(self) -> float:
        return float(np.linalg.norm(self.local_operator, ord="fro"))

    def normalized(
        self,
        normalization: WitnessNormalization = "operator_norm",
    ) -> LocalWitnessTemplate:
        """Return a canonically normalized copy of the local row operator.

        ``operator_norm`` is the preferred ETH convention because it fixes
        ``||Q_R|| = ||L_R||^2 = 1``.  Thermal expectations are then directly
        comparable between witnesses and system sizes.
        """
        if normalization == "none":
            return self
        if normalization == "operator_norm":
            scale = self.operator_norm
        elif normalization == "frobenius_norm":
            scale = self.frobenius_norm
        else:
            raise ValueError(f"Unsupported witness normalization: {normalization!r}.")
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("Cannot normalize a zero or non-finite local witness.")

        metadata = dict(self.metadata)
        metadata.update(
            {
                "normalization": normalization,
                "normalization_divisor": float(scale),
                "unnormalized_operator_norm": self.operator_norm,
                "unnormalized_frobenius_norm": self.frobenius_norm,
            }
        )
        return LocalWitnessTemplate(
            pattern_key=self.pattern_key,
            local_patterns=self.local_patterns,
            local_operator=self.local_operator / scale,
            source_zero_indices=self.source_zero_indices,
            mechanism_labels=self.mechanism_labels,
            metadata=metadata,
        )

    def instantiate(self, variable_indices: Sequence[int]) -> LocalWitness:
        return LocalWitness(
            template=self,
            variable_indices=tuple(int(index) for index in variable_indices),
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "operator_norm": self.operator_norm,
            "q_operator_norm": self.q_operator_norm,
            "frobenius_norm": self.frobenius_norm,
            "source_zero_indices": self.source_zero_indices,
            "mechanism_labels": self.mechanism_labels,
            "local_patterns": self.local_patterns,
            "pattern_key": self.pattern_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LocalWitness:
    """One embedding of a size-independent local witness template."""

    template: LocalWitnessTemplate
    variable_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.variable_indices)
        if len(indices) != self.template.n_variables:
            raise ValueError(
                "variable_indices length must match the template width: "
                f"{len(indices)} != {self.template.n_variables}."
            )
        if len(set(indices)) != len(indices):
            raise ValueError("variable_indices must not contain duplicates.")
        if any(index < 0 for index in indices):
            raise ValueError("variable_indices must be non-negative.")
        object.__setattr__(self, "variable_indices", indices)

    @property
    def local_patterns(self) -> tuple[tuple[int, ...], ...]:
        return self.template.local_patterns

    @property
    def local_operator(self) -> npt.NDArray[np.complex128]:
        return self.template.local_operator

    @property
    def q_operator_norm(self) -> float:
        return self.template.q_operator_norm

    def embed(self, basis_configs: npt.NDArray[np.integer]) -> sp.csr_array:
        return embed_local_pattern_operator(
            basis_configs=basis_configs,
            variable_indices=self.variable_indices,
            local_patterns=self.local_patterns,
            local_operator=self.local_operator,
        )


@dataclass(frozen=True, slots=True)
class LocalWitnessEvaluation:
    """Expectation and variance of ``Q_R = L_R^dagger L_R``."""

    expectation: float
    second_moment: float
    variance: float
    annihilation_residual: float
    normalized_expectation: float
    n_states: int
    effective_state_count: float
    per_state_expectations: tuple[float, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_annihilated(self) -> bool:
        return self.annihilation_residual <= 1.0e-10

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "expectation": self.expectation,
            "second_moment": self.second_moment,
            "variance": self.variance,
            "annihilation_residual": self.annihilation_residual,
            "normalized_expectation": self.normalized_expectation,
            "n_states": self.n_states,
            "effective_state_count": self.effective_state_count,
            "per_state_expectations": self.per_state_expectations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MicrocanonicalWitnessEvaluation:
    """Local-witness evaluation in a finite-size microcanonical shell."""

    evaluation: LocalWitnessEvaluation
    shell_indices: tuple[int, ...]
    energy_center: float
    half_width: float
    shell_energy_min: float
    shell_energy_max: float
    mean_energy: float
    mean_energy_density: float | None

    @property
    def n_shell_states(self) -> int:
        return len(self.shell_indices)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_shell_states": self.n_shell_states,
            "shell_indices": self.shell_indices,
            "energy_center": self.energy_center,
            "half_width": self.half_width,
            "shell_energy_min": self.shell_energy_min,
            "shell_energy_max": self.shell_energy_max,
            "mean_energy": self.mean_energy,
            "mean_energy_density": self.mean_energy_density,
            "evaluation": self.evaluation.to_summary_dict(),
        }


@dataclass(frozen=True, slots=True)
class LocalWitnessEmbeddingRecord:
    """All exact embeddings of one template in one finite system."""

    system_label: Hashable
    witnesses: tuple[LocalWitness, ...]


@dataclass(frozen=True, slots=True)
class LocalWitnessFamily:
    """One reduced-IZ local pattern found in several finite systems."""

    template: LocalWitnessTemplate
    embeddings: tuple[LocalWitnessEmbeddingRecord, ...]

    @property
    def system_labels(self) -> tuple[Hashable, ...]:
        return tuple(record.system_label for record in self.embeddings)

    @property
    def n_systems(self) -> int:
        return len(self.embeddings)

    def witnesses_for(self, system_label: Hashable) -> tuple[LocalWitness, ...]:
        for record in self.embeddings:
            if record.system_label == system_label:
                return record.witnesses
        raise KeyError(system_label)


@dataclass(frozen=True, slots=True)
class ETHScalingPoint:
    """One finite-size comparison between a cage state and a thermal ensemble."""

    system_size: int
    cage: LocalWitnessEvaluation
    thermal: LocalWitnessEvaluation
    energy: float | None = None
    energy_density: float | None = None
    system_label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.system_size <= 0:
            raise ValueError("system_size must be positive.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def expectation_gap(self) -> float:
        return float(self.thermal.expectation - self.cage.expectation)

    @property
    def absolute_expectation_gap(self) -> float:
        return abs(self.expectation_gap)

    @property
    def normalized_gap(self) -> float:
        scale = max(
            abs(self.thermal.expectation),
            abs(self.cage.expectation),
            1.0e-15,
        )
        return float(self.expectation_gap / scale)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "system_label": self.system_label,
            "system_size": self.system_size,
            "energy": self.energy,
            "energy_density": self.energy_density,
            "cage_expectation": self.cage.expectation,
            "cage_variance": self.cage.variance,
            "thermal_expectation": self.thermal.expectation,
            "thermal_variance": self.thermal.variance,
            "expectation_gap": self.expectation_gap,
            "absolute_expectation_gap": self.absolute_expectation_gap,
            "normalized_gap": self.normalized_gap,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class InverseSizeFit:
    """Descriptive fit ``y(N) = c_0 + c_1/N + ...``."""

    order: int
    coefficients: tuple[float, ...]
    thermodynamic_limit: float
    root_mean_square_residual: float
    system_sizes: tuple[int, ...]
    observed_values: tuple[float, ...]
    fitted_values: tuple[float, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "coefficients": self.coefficients,
            "thermodynamic_limit": self.thermodynamic_limit,
            "root_mean_square_residual": self.root_mean_square_residual,
            "system_sizes": self.system_sizes,
            "observed_values": self.observed_values,
            "fitted_values": self.fitted_values,
        }


@dataclass(frozen=True, slots=True)
class ETHScalingReport:
    """Finite-size scaling data for one fixed local witness template."""

    template: LocalWitnessTemplate
    points: tuple[ETHScalingPoint, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.points, key=lambda point: point.system_size))
        sizes = tuple(point.system_size for point in ordered)
        if len(set(sizes)) != len(sizes):
            raise ValueError("points must have distinct system_size values.")
        object.__setattr__(self, "points", ordered)

    @property
    def system_sizes(self) -> tuple[int, ...]:
        return tuple(point.system_size for point in self.points)

    @property
    def expectation_gaps(self) -> tuple[float, ...]:
        return tuple(point.expectation_gap for point in self.points)

    @property
    def thermal_expectations(self) -> tuple[float, ...]:
        return tuple(point.thermal.expectation for point in self.points)

    @property
    def cage_expectations(self) -> tuple[float, ...]:
        return tuple(point.cage.expectation for point in self.points)

    def tail_liminf_lower_bound(self, *, tail_points: int = 2) -> float:
        """Return the minimum absolute gap among the largest available sizes.

        This is a finite-data lower bound, not a proof of the mathematical
        liminf.  Its explicit name is intended to prevent overinterpretation.
        """
        if tail_points <= 0:
            raise ValueError("tail_points must be positive.")
        if not self.points:
            raise ValueError("at least one scaling point is required.")
        tail = self.points[-min(tail_points, len(self.points)) :]
        return float(min(point.absolute_expectation_gap for point in tail))

    def fit_expectation_gap(self, *, order: int = 1) -> InverseSizeFit:
        return _inverse_size_fit(
            system_sizes=self.system_sizes,
            values=self.expectation_gaps,
            order=order,
        )

    def fit_thermal_expectation(self, *, order: int = 1) -> InverseSizeFit:
        return _inverse_size_fit(
            system_sizes=self.system_sizes,
            values=self.thermal_expectations,
            order=order,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "template": self.template.to_summary_dict(),
            "n_points": len(self.points),
            "system_sizes": self.system_sizes,
            "expectation_gaps": self.expectation_gaps,
            "points": tuple(point.to_summary_dict() for point in self.points),
        }


@dataclass(frozen=True, slots=True)
class EnergyDensityMatchReport:
    """Compare a cage-family energy density with a thermal comparator."""

    cage_energy_density: float
    thermal_energy_density: float
    tolerance: float = 1.0e-8
    comparator: str = "beta_zero"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        if not np.isfinite(self.cage_energy_density):
            raise ValueError("cage_energy_density must be finite.")
        if not np.isfinite(self.thermal_energy_density):
            raise ValueError("thermal_energy_density must be finite.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def difference(self) -> float:
        return float(self.cage_energy_density - self.thermal_energy_density)

    @property
    def absolute_difference(self) -> float:
        return abs(self.difference)

    @property
    def is_matched(self) -> bool:
        return self.absolute_difference <= self.tolerance

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "cage_energy_density": self.cage_energy_density,
            "thermal_energy_density": self.thermal_energy_density,
            "difference": self.difference,
            "absolute_difference": self.absolute_difference,
            "tolerance": self.tolerance,
            "is_matched": self.is_matched,
            "comparator": self.comparator,
            "metadata": dict(self.metadata),
        }


def local_witness_template_from_pattern_support(
    pattern_support: ReducedIZPatternSupport,
    *,
    normalization: WitnessNormalization = "none",
    metadata: Mapping[str, object] | None = None,
) -> LocalWitnessTemplate:
    """Reconstruct a local row operator from a reduced-IZ transition pattern."""
    patterns = sorted(
        {
            tuple(int(value) for value in pattern)
            for source, target, _coefficient in pattern_support.pattern_key
            for pattern in (source, target)
        }
    )
    if not patterns:
        raise ValueError("pattern_support does not contain any local transitions.")

    pattern_to_index = {pattern: index for index, pattern in enumerate(patterns)}
    operator = np.zeros((len(patterns), len(patterns)), dtype=np.complex128)
    for source, target, coefficient_key in pattern_support.pattern_key:
        operator[
            pattern_to_index[tuple(target)],
            pattern_to_index[tuple(source)],
        ] += _complex_from_key(coefficient_key)

    template_metadata = {
        "source_variable_indices": tuple(pattern_support.variable_indices),
    }
    if metadata is not None:
        template_metadata.update(dict(metadata))

    template = LocalWitnessTemplate(
        pattern_key=pattern_support.pattern_key,
        local_patterns=tuple(patterns),
        local_operator=operator,
        source_zero_indices=pattern_support.source_zero_indices,
        mechanism_labels=pattern_support.mechanism_labels,
        metadata=template_metadata,
    )
    return template.normalized(normalization)


def local_witnesses_from_classification_report(
    report: CageClassificationReport,
    *,
    include_projector_like: bool = True,
    normalization: WitnessNormalization = "none",
) -> tuple[LocalWitness, ...]:
    """Return all trusted reduced-IZ witness embeddings in one finite system."""
    witnesses: list[LocalWitness] = []
    for pattern_support in distinct_reduced_iz_pattern_supports(
        report,
        include_projector_like=include_projector_like,
    ):
        template = local_witness_template_from_pattern_support(
            pattern_support,
            normalization=normalization,
        )
        witnesses.append(template.instantiate(pattern_support.variable_indices))
    return tuple(witnesses)


def common_local_witness_families(
    reports: Mapping[Hashable, CageClassificationReport],
    *,
    include_projector_like: bool = True,
    require_all_systems: bool = True,
    normalization: WitnessNormalization = "none",
) -> tuple[LocalWitnessFamily, ...]:
    """Match identical reduced-IZ local patterns across system sizes.

    Matching is exact in the ordered local pattern basis.  Translations are
    automatically matched because global variable indices are not part of the
    template key.  Rotations or reflections require the caller to relabel local
    variables consistently before classification.
    """
    if not reports:
        return ()

    grouped: dict[
        ReducedIZPatternKey,
        dict[Hashable, list[tuple[ReducedIZPatternSupport, LocalWitness]]],
    ] = {}
    for system_label, report in reports.items():
        for pattern_support in distinct_reduced_iz_pattern_supports(
            report,
            include_projector_like=include_projector_like,
        ):
            template = local_witness_template_from_pattern_support(
                pattern_support,
                normalization=normalization,
            )
            witness = template.instantiate(pattern_support.variable_indices)
            grouped.setdefault(template.pattern_key, {}).setdefault(system_label, []).append(
                (pattern_support, witness)
            )

    required_labels = set(reports)
    families: list[LocalWitnessFamily] = []
    for _pattern_key, by_system in grouped.items():
        if require_all_systems and set(by_system) != required_labels:
            continue

        first_support, _first_witness = next(iter(next(iter(by_system.values()))))
        template = local_witness_template_from_pattern_support(
            first_support,
            normalization=normalization,
        )
        embeddings = tuple(
            LocalWitnessEmbeddingRecord(
                system_label=system_label,
                witnesses=tuple(witness for _support, witness in by_system[system_label]),
            )
            for system_label in reports
            if system_label in by_system
        )
        families.append(LocalWitnessFamily(template=template, embeddings=embeddings))

    families.sort(key=lambda family: (family.template.n_variables, family.template.pattern_key))
    return tuple(families)


def evaluate_local_witness_on_states(
    witness: LocalWitness,
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    normalize_columns: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> LocalWitnessEvaluation:
    """Evaluate ``Q_R`` on a pure state or a weighted ensemble of states.

    The constrained-basis embedding of ``L_R`` is sparse.  The variance is the
    quantum variance of ``Q_R`` in the mixed state represented by the supplied
    weights, not the variance of the per-eigenstate expectation values.
    """
    basis = np.asarray(basis_configs)
    if basis.ndim != 2:
        raise ValueError("basis_configs must be two-dimensional.")
    matrix = _state_matrix(states, dim=basis.shape[0], normalize_columns=normalize_columns)
    probabilities = _normalize_weights(weights, size=matrix.shape[1])

    local_operator = witness.embed(basis)
    actions = np.asarray(local_operator @ matrix, dtype=np.complex128)
    q_actions = np.asarray(local_operator.conj().T @ actions, dtype=np.complex128)

    per_state_q = np.sum(np.abs(actions) ** 2, axis=0).real
    per_state_q2 = np.sum(np.abs(q_actions) ** 2, axis=0).real
    expectation = float(np.dot(probabilities, per_state_q))
    second_moment = float(np.dot(probabilities, per_state_q2))
    variance = float(max(second_moment - expectation**2, 0.0))
    operator_scale = witness.q_operator_norm
    normalized_expectation = expectation / operator_scale if operator_scale > 0.0 else 0.0
    effective_count = float(1.0 / np.sum(probabilities**2))

    return LocalWitnessEvaluation(
        expectation=expectation,
        second_moment=second_moment,
        variance=variance,
        annihilation_residual=float(np.sqrt(max(expectation, 0.0))),
        normalized_expectation=float(normalized_expectation),
        n_states=int(matrix.shape[1]),
        effective_state_count=effective_count,
        per_state_expectations=tuple(float(value) for value in per_state_q),
        metadata={} if metadata is None else dict(metadata),
    )


def evaluate_local_witness_on_diagonal_ensemble(
    witness: LocalWitness,
    *,
    basis_configs: npt.NDArray[np.integer],
    probabilities: npt.ArrayLike | None = None,
    metadata: Mapping[str, object] | None = None,
) -> LocalWitnessEvaluation:
    """Evaluate ``Q_R`` in an ensemble diagonal in the constrained basis.

    With ``probabilities=None`` this is the exact infinite-temperature trace in
    the supplied constrained basis and symmetry sector.
    """
    basis = np.asarray(basis_configs)
    if basis.ndim != 2:
        raise ValueError("basis_configs must be two-dimensional.")
    weights = _normalize_weights(probabilities, size=basis.shape[0])
    local_operator = witness.embed(basis)
    q_operator = (local_operator.conj().T @ local_operator).tocsr()

    q_diagonal = np.asarray(q_operator.diagonal(), dtype=np.complex128).real
    q2_diagonal = np.asarray(q_operator.multiply(q_operator.conj()).sum(axis=1)).ravel().real
    expectation = float(np.dot(weights, q_diagonal))
    second_moment = float(np.dot(weights, q2_diagonal))
    variance = float(max(second_moment - expectation**2, 0.0))
    operator_scale = witness.q_operator_norm
    normalized_expectation = expectation / operator_scale if operator_scale > 0.0 else 0.0

    ensemble_metadata = {
        "ensemble": "uniform_constrained_basis" if probabilities is None else "diagonal",
        "basis_dimension": int(basis.shape[0]),
    }
    if metadata is not None:
        ensemble_metadata.update(dict(metadata))

    return LocalWitnessEvaluation(
        expectation=expectation,
        second_moment=second_moment,
        variance=variance,
        annihilation_residual=float(np.sqrt(max(expectation, 0.0))),
        normalized_expectation=float(normalized_expectation),
        n_states=int(basis.shape[0]),
        effective_state_count=float(1.0 / np.sum(weights**2)),
        per_state_expectations=tuple(float(value) for value in q_diagonal),
        metadata=ensemble_metadata,
    )


def evaluate_local_witness_microcanonical(
    witness: LocalWitness,
    *,
    basis_configs: npt.NDArray[np.integer],
    eigenvectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energy_center: float,
    half_width: float,
    system_size: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MicrocanonicalWitnessEvaluation:
    """Evaluate ``Q_R`` in an equal-weight finite-size energy shell."""
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative.")
    energies = np.asarray(eigenvalues, dtype=np.float64)
    if energies.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional.")

    basis = np.asarray(basis_configs)
    vectors = _state_matrix(eigenvectors, dim=basis.shape[0], normalize_columns=True)
    if vectors.shape[1] != energies.size:
        raise ValueError("eigenvectors and eigenvalues contain different numbers of states.")

    mask = np.abs(energies - float(energy_center)) <= float(half_width)
    shell_indices = np.flatnonzero(mask)
    if shell_indices.size == 0:
        raise ValueError("the requested microcanonical shell is empty.")

    shell_energies = energies[shell_indices]
    shell_vectors = vectors[:, shell_indices]
    evaluation_metadata = {
        "ensemble": "microcanonical",
        "energy_center": float(energy_center),
        "half_width": float(half_width),
    }
    if metadata is not None:
        evaluation_metadata.update(dict(metadata))

    evaluation = evaluate_local_witness_on_states(
        witness,
        basis_configs=basis,
        states=shell_vectors,
        weights=None,
        normalize_columns=False,
        metadata=evaluation_metadata,
    )
    mean_energy = float(np.mean(shell_energies))
    mean_density = None if system_size is None else mean_energy / float(system_size)

    return MicrocanonicalWitnessEvaluation(
        evaluation=evaluation,
        shell_indices=tuple(int(index) for index in shell_indices),
        energy_center=float(energy_center),
        half_width=float(half_width),
        shell_energy_min=float(np.min(shell_energies)),
        shell_energy_max=float(np.max(shell_energies)),
        mean_energy=mean_energy,
        mean_energy_density=mean_density,
    )


def make_eth_scaling_point(
    *,
    system_size: int,
    witness: LocalWitness,
    basis_configs: npt.NDArray[np.integer],
    cage_state: npt.ArrayLike,
    thermal_states: npt.ArrayLike | None = None,
    thermal_weights: npt.ArrayLike | None = None,
    diagonal_probabilities: npt.ArrayLike | None = None,
    energy: float | None = None,
    energy_density: float | None = None,
    system_label: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ETHScalingPoint:
    """Build one finite-size ETH comparison for a fixed witness embedding.

    If ``thermal_states`` is omitted, the thermal side is evaluated in a basis-
    diagonal ensemble.  Passing neither thermal states nor diagonal
    probabilities selects the exact infinite-temperature constrained ensemble.
    """
    cage = evaluate_local_witness_on_states(
        witness,
        basis_configs=basis_configs,
        states=cage_state,
        metadata={"ensemble": "cage"},
    )
    if thermal_states is None:
        thermal = evaluate_local_witness_on_diagonal_ensemble(
            witness,
            basis_configs=basis_configs,
            probabilities=diagonal_probabilities,
        )
    else:
        if diagonal_probabilities is not None:
            raise ValueError("diagonal_probabilities cannot be combined with thermal_states.")
        thermal = evaluate_local_witness_on_states(
            witness,
            basis_configs=basis_configs,
            states=thermal_states,
            weights=thermal_weights,
            metadata={"ensemble": "state_ensemble"},
        )

    density = energy_density
    if density is None and energy is not None:
        density = float(energy) / float(system_size)

    return ETHScalingPoint(
        system_size=int(system_size),
        cage=cage,
        thermal=thermal,
        energy=None if energy is None else float(energy),
        energy_density=None if density is None else float(density),
        system_label=system_label,
        metadata={} if metadata is None else dict(metadata),
    )


def _inverse_size_fit(
    *,
    system_sizes: Sequence[int],
    values: Sequence[float],
    order: int,
) -> InverseSizeFit:
    if order < 0:
        raise ValueError("order must be non-negative.")
    sizes = np.asarray(system_sizes, dtype=np.float64)
    observed = np.asarray(values, dtype=np.float64)
    if sizes.ndim != 1 or observed.ndim != 1 or sizes.size != observed.size:
        raise ValueError("system_sizes and values must be one-dimensional and equally sized.")
    if sizes.size < order + 1:
        raise ValueError(f"at least {order + 1} points are required for an order-{order} fit.")
    if np.any(sizes <= 0.0):
        raise ValueError("system_sizes must be positive.")

    inverse_sizes = 1.0 / sizes
    design = np.column_stack([inverse_sizes**power for power in range(order + 1)])
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    fitted = design @ coefficients
    rms = float(np.sqrt(np.mean((observed - fitted) ** 2)))

    return InverseSizeFit(
        order=int(order),
        coefficients=tuple(float(value) for value in coefficients),
        thermodynamic_limit=float(coefficients[0]),
        root_mean_square_residual=rms,
        system_sizes=tuple(int(value) for value in system_sizes),
        observed_values=tuple(float(value) for value in observed),
        fitted_values=tuple(float(value) for value in fitted),
    )
