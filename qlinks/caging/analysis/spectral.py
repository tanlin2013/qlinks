from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as sp

MatrixLike = npt.NDArray[np.number] | sp.sparray | sp.spmatrix
LevelStatisticsClass = Literal["poisson", "goe", "gue"]


@dataclass(frozen=True, slots=True)
class EigenpairResidualReport:
    """Finite-size residual data for a proposed eigenpair."""

    energy: complex
    residual_norm: float
    relative_residual_norm: float
    variance: float
    state_norm: float

    @property
    def is_exact(self) -> bool:
        return self.relative_residual_norm <= 1.0e-10

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "energy": self.energy,
            "residual_norm": self.residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
            "variance": self.variance,
            "state_norm": self.state_norm,
            "is_exact": self.is_exact,
        }


@dataclass(frozen=True, slots=True)
class SymmetrySectorBasis:
    """Orthonormal basis for a symmetry-resolved subspace.

    ``basis`` has shape ``(full_dimension, sector_dimension)`` and orthonormal
    columns.  It may be sparse for cyclic sectors and dense after further
    refinement by an involution.
    """

    basis: MatrixLike
    labels: dict[str, object] = field(default_factory=dict)
    unitarity_residual: float = 0.0

    @property
    def full_dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def sector_dimension(self) -> int:
        return int(self.basis.shape[1])

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "full_dimension": self.full_dimension,
            "sector_dimension": self.sector_dimension,
            "unitarity_residual": self.unitarity_residual,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True, slots=True)
class MicrocanonicalWindowSelection:
    """A finite-size energy window selected around a target energy."""

    indices: tuple[int, ...]
    target_energy: float
    half_width: float
    energy_min: float
    energy_max: float
    mean_energy: float
    center_offset: float

    @property
    def n_states(self) -> int:
        return len(self.indices)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "indices": self.indices,
            "n_states": self.n_states,
            "target_energy": self.target_energy,
            "half_width": self.half_width,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "mean_energy": self.mean_energy,
            "center_offset": self.center_offset,
        }


@dataclass(frozen=True, slots=True)
class SpectralMicrocanonicalEnsemble:
    """Low-rank representation of a finite-size microcanonical ensemble.

    The selected eigenvectors are stored as columns and carry equal weights.
    This avoids materializing a dense density matrix unless explicitly
    requested, while still exposing the ensemble used for observable traces.
    """

    selection: MicrocanonicalWindowSelection
    eigenvalues: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.complex128]
    weights: npt.NDArray[np.float64]
    volume: int | None = None

    @property
    def n_states(self) -> int:
        return int(self.eigenvectors.shape[1])

    @property
    def hilbert_dimension(self) -> int:
        return int(self.eigenvectors.shape[0])

    @property
    def energy_density_half_width(self) -> float | None:
        if self.volume is None:
            return None
        return float(self.selection.half_width / self.volume)

    def density_matrix(self) -> npt.NDArray[np.complex128]:
        """Materialize ``rho_mc = sum_a w_a |E_a><E_a|``."""
        weighted = self.eigenvectors * self.weights[np.newaxis, :]
        return np.asarray(weighted @ self.eigenvectors.conj().T, dtype=np.complex128)

    def expectation(self, operator: MatrixLike) -> complex:
        """Return ``Tr(rho_mc operator)`` without materializing ``rho_mc``."""
        actions = np.asarray(operator @ self.eigenvectors, dtype=np.complex128)
        diagonal = np.einsum(
            "ij,ij->j",
            self.eigenvectors.conj(),
            actions,
        )
        return complex(np.dot(self.weights, diagonal))

    def observable_moments(
        self,
        operator: MatrixLike,
        *,
        squared_operator: MatrixLike | None = None,
        hermiticity_tolerance: float = 1.0e-10,
    ) -> "SpectralObservableMoments":
        """Return mean, second moment, and variance in this ensemble."""
        return spectral_observable_moments(
            operator,
            self.eigenvectors,
            squared_operator=squared_operator,
            weights=self.weights,
            hermiticity_tolerance=hermiticity_tolerance,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.selection.to_summary_dict(),
            "hilbert_dimension": self.hilbert_dimension,
            "volume": self.volume,
            "energy_density_half_width": self.energy_density_half_width,
        }


@dataclass(frozen=True, slots=True)
class ThermodynamicEnergyWindowPlan:
    """Size-scaled energy window for a thermodynamic microcanonical sequence.

    A width ``Delta E_L = c * epsilon * volume**alpha`` with ``alpha < 1``
    has a vanishing energy-density width.  The default ``alpha=1/2`` is the
    square-root-volume choice used in the manuscript notebooks.
    """

    volume: int
    energy_density: float
    target_energy: float
    half_width: float
    energy_density_half_width: float
    width_prefactor: float
    local_energy_scale: float
    width_exponent: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "volume": self.volume,
            "energy_density": self.energy_density,
            "target_energy": self.target_energy,
            "half_width": self.half_width,
            "energy_density_half_width": self.energy_density_half_width,
            "width_prefactor": self.width_prefactor,
            "local_energy_scale": self.local_energy_scale,
            "width_exponent": self.width_exponent,
        }


@dataclass(frozen=True, slots=True)
class SpectralObservableMoments:
    """First two measurement moments of one Hermitian observable."""

    mean: float
    second_moment: float
    variance: float
    n_states: int
    effective_state_count: float
    minimum_expectation: float
    maximum_expectation: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "second_moment": self.second_moment,
            "variance": self.variance,
            "n_states": self.n_states,
            "effective_state_count": self.effective_state_count,
            "minimum_expectation": self.minimum_expectation,
            "maximum_expectation": self.maximum_expectation,
        }


@dataclass(frozen=True, slots=True)
class SmoothSpectralFilter:
    """Normalized smooth energy-filter weights and their diagnostics."""

    weights: tuple[float, ...]
    target_energy: float
    sigma: float
    mean_energy: float
    energy_variance: float
    effective_state_count: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "target_energy": self.target_energy,
            "sigma": self.sigma,
            "mean_energy": self.mean_energy,
            "energy_variance": self.energy_variance,
            "effective_state_count": self.effective_state_count,
        }


@dataclass(frozen=True, slots=True)
class AdjacentGapRatioReport:
    """Adjacent-gap-ratio statistics in one fully resolved symmetry sector."""

    mean_ratio: float
    ratios: tuple[float, ...]
    spacings: tuple[float, ...]
    n_levels_input: int
    n_levels_used: int
    trim_fraction: float
    degeneracy_tolerance: float
    expected_poisson: float = 2.0 * np.log(2.0) - 1.0
    expected_goe: float = 0.5307
    expected_gue: float = 0.5996

    def distance_to(self, ensemble: LevelStatisticsClass) -> float:
        target = {
            "poisson": self.expected_poisson,
            "goe": self.expected_goe,
            "gue": self.expected_gue,
        }[ensemble]
        return abs(self.mean_ratio - target)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "mean_ratio": self.mean_ratio,
            "n_ratios": len(self.ratios),
            "n_levels_input": self.n_levels_input,
            "n_levels_used": self.n_levels_used,
            "trim_fraction": self.trim_fraction,
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "expected_poisson": self.expected_poisson,
            "expected_goe": self.expected_goe,
            "expected_gue": self.expected_gue,
        }


def diagnose_eigenpair(
    hamiltonian: MatrixLike,
    state: npt.ArrayLike,
) -> EigenpairResidualReport:
    """Return the Rayleigh energy, residual, and variance of a state."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if hamiltonian.shape != (vector.size, vector.size):
        raise ValueError("hamiltonian and state have incompatible dimensions.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("state must not be the zero vector.")
    normalized = vector / norm
    action = np.asarray(hamiltonian @ normalized, dtype=np.complex128).reshape(-1)
    energy = complex(np.vdot(normalized, action))
    residual = action - energy * normalized
    residual_norm = float(np.linalg.norm(residual))
    scale = max(float(np.linalg.norm(action)), abs(energy), 1.0)
    variance = float(max(np.vdot(residual, residual).real, 0.0))
    return EigenpairResidualReport(
        energy=energy,
        residual_norm=residual_norm,
        relative_residual_norm=residual_norm / scale,
        variance=variance,
        state_norm=norm,
    )


def basis_permutation_from_transform(
    basis_configs: npt.NDArray[np.integer],
    transform: Callable[[npt.NDArray[np.int64]], npt.ArrayLike],
) -> npt.NDArray[np.int64]:
    """Return the basis-index permutation induced by a configuration transform.

    The returned array ``p`` uses the convention ``U |i> = |p[i]>``.
    """
    configs = np.asarray(basis_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_states, n_variables).")
    lookup = {tuple(int(value) for value in config): index for index, config in enumerate(configs)}
    if len(lookup) != configs.shape[0]:
        raise ValueError("basis_configs must not contain duplicate configurations.")

    permutation = np.empty(configs.shape[0], dtype=np.int64)
    for index, config in enumerate(configs):
        transformed = np.asarray(transform(config.copy()), dtype=np.int64).reshape(-1)
        if transformed.shape != config.shape:
            raise ValueError("configuration transform changed the configuration shape.")
        try:
            permutation[index] = lookup[tuple(int(value) for value in transformed)]
        except KeyError as exc:
            raise ValueError(
                "configuration transform does not preserve the supplied basis sector."
            ) from exc

    if np.unique(permutation).size != permutation.size:
        raise ValueError("configuration transform is not a permutation of the supplied basis.")
    return permutation


def basis_permutation_from_variable_permutation(
    basis_configs: npt.NDArray[np.integer],
    variable_permutation: Sequence[int],
) -> npt.NDArray[np.int64]:
    """Return a basis permutation from ``new_config = old_config[permutation]``."""
    configs = np.asarray(basis_configs, dtype=np.int64)
    variables = np.asarray(variable_permutation, dtype=np.int64).reshape(-1)
    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_states, n_variables).")
    if variables.size != configs.shape[1]:
        raise ValueError("variable_permutation has incompatible length.")
    if (
        np.unique(variables).size != variables.size
        or np.any(variables < 0)
        or np.any(variables >= variables.size)
    ):
        raise ValueError("variable_permutation must be a permutation of variable indices.")
    return basis_permutation_from_transform(configs, lambda config: config[variables])


def permutation_matrix(index_permutation: npt.ArrayLike) -> sp.csr_array:
    """Construct the sparse permutation matrix for ``U |i> = |p[i]>``."""
    permutation = np.asarray(index_permutation, dtype=np.int64).reshape(-1)
    n = permutation.size
    if np.unique(permutation).size != n or np.any(permutation < 0) or np.any(permutation >= n):
        raise ValueError("index_permutation must be a permutation of range(n).")
    data = np.ones(n, dtype=np.complex128)
    columns = np.arange(n, dtype=np.int64)
    return sp.csr_array((data, (permutation, columns)), shape=(n, n))


def cyclic_symmetry_sector_basis(
    index_permutation: npt.ArrayLike,
    *,
    order: int,
    momentum_index: int,
    labels: dict[str, object] | None = None,
    tolerance: float = 1.0e-10,
) -> SymmetrySectorBasis:
    """Build a Fourier-orbit basis for one sector of a cyclic permutation.

    ``momentum_index`` labels the eigenvalue ``exp(2π i k / order)``.
    Short orbits are included only when this character is compatible with the
    orbit stabilizer.
    """
    permutation = np.asarray(index_permutation, dtype=np.int64).reshape(-1)
    n = permutation.size
    if order <= 0:
        raise ValueError("order must be positive.")
    if np.unique(permutation).size != n or np.any(permutation < 0) or np.any(permutation >= n):
        raise ValueError("index_permutation must be a permutation of range(n).")

    k_index = int(momentum_index) % int(order)
    momentum = 2.0 * np.pi * k_index / float(order)
    visited = np.zeros(n, dtype=bool)
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    orbit_lengths: list[int] = []
    column = 0

    for seed in range(n):
        if visited[seed]:
            continue
        orbit: list[int] = []
        current = seed
        while not visited[current]:
            visited[current] = True
            orbit.append(current)
            current = int(permutation[current])
        if current != seed:
            raise ValueError("index_permutation orbit did not close on its seed.")
        length = len(orbit)
        if order % length != 0:
            raise ValueError("permutation orbit length does not divide the declared order.")
        compatibility = np.exp(1.0j * momentum * length)
        if abs(compatibility - 1.0) > tolerance:
            continue
        normalization = np.sqrt(float(length))
        for step, basis_index in enumerate(orbit):
            rows.append(int(basis_index))
            columns.append(column)
            data.append(np.exp(-1.0j * momentum * step) / normalization)
        orbit_lengths.append(length)
        column += 1

    basis = sp.csc_array(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(n, column),
    )
    gram = basis.conj().T @ basis
    unitarity_residual = float(np.linalg.norm(gram.toarray() - np.eye(column)))
    sector_labels = {
        "symmetry": "cyclic",
        "order": int(order),
        "momentum_index": k_index,
        "momentum": float(momentum),
        "orbit_lengths": tuple(orbit_lengths),
    }
    if labels is not None:
        sector_labels.update(dict(labels))
    return SymmetrySectorBasis(
        basis=basis,
        labels=sector_labels,
        unitarity_residual=unitarity_residual,
    )


def commuting_cyclic_symmetry_sector_basis(
    index_permutations: Sequence[npt.ArrayLike],
    *,
    orders: Sequence[int],
    momentum_indices: Sequence[int],
    labels: dict[str, object] | None = None,
    tolerance: float = 1.0e-10,
) -> SymmetrySectorBasis:
    """Build a simultaneous character basis for commuting cyclic permutations.

    This is the finite Abelian-group extension of
    :func:`cyclic_symmetry_sector_basis`.  It is particularly useful for the
    two translation generators of a periodic lattice.  One normalized Fourier
    vector is produced for every compatible group orbit.
    """
    permutations = tuple(
        np.asarray(value, dtype=np.int64).reshape(-1) for value in index_permutations
    )
    orders_tuple = tuple(int(value) for value in orders)
    momenta_tuple = tuple(int(value) for value in momentum_indices)
    if not permutations:
        raise ValueError("index_permutations must not be empty.")
    if len(permutations) != len(orders_tuple) or len(permutations) != len(momenta_tuple):
        raise ValueError("permutations, orders, and momentum_indices must have equal length.")
    n = permutations[0].size
    if any(permutation.size != n for permutation in permutations):
        raise ValueError("all permutations must have the same size.")
    for permutation, order in zip(permutations, orders_tuple, strict=True):
        if order <= 0:
            raise ValueError("orders must be positive.")
        if np.unique(permutation).size != n or np.any(permutation < 0) or np.any(permutation >= n):
            raise ValueError("each index permutation must permute range(n).")
        current = np.arange(n, dtype=np.int64)
        for _ in range(order):
            current = permutation[current]
        if not np.array_equal(current, np.arange(n, dtype=np.int64)):
            raise ValueError("a permutation does not close at its declared order.")
    for left in range(len(permutations)):
        for right in range(left + 1, len(permutations)):
            if not np.array_equal(
                permutations[left][permutations[right]],
                permutations[right][permutations[left]],
            ):
                raise ValueError("the cyclic permutations must commute.")

    powers: list[tuple[npt.NDArray[np.int64], ...]] = []
    identity = np.arange(n, dtype=np.int64)
    for permutation, order in zip(permutations, orders_tuple, strict=True):
        local = [identity]
        for _ in range(1, order):
            local.append(permutation[local[-1]])
        powers.append(tuple(local))

    momenta = tuple(
        2.0 * np.pi * (momentum % order) / float(order)
        for momentum, order in zip(momenta_tuple, orders_tuple, strict=True)
    )
    visited = np.zeros(n, dtype=bool)
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    orbit_sizes: list[int] = []
    column = 0
    group_ranges = tuple(range(order) for order in orders_tuple)

    for seed in range(n):
        if visited[seed]:
            continue
        coefficient_by_state: dict[int, complex] = {}
        orbit: set[int] = set()
        for exponents in product(*group_ranges):
            state = int(seed)
            phase_angle = 0.0
            for generator_index, exponent in enumerate(exponents):
                state = int(powers[generator_index][exponent][state])
                phase_angle += momenta[generator_index] * exponent
            orbit.add(state)
            coefficient_by_state[state] = coefficient_by_state.get(state, 0.0j) + np.exp(
                -1.0j * phase_angle
            )
        for state in orbit:
            visited[state] = True
        norm = float(np.sqrt(sum(abs(value) ** 2 for value in coefficient_by_state.values())))
        if norm <= tolerance:
            continue
        for state, coefficient in coefficient_by_state.items():
            if abs(coefficient) <= tolerance:
                continue
            rows.append(int(state))
            columns.append(column)
            data.append(complex(coefficient / norm))
        orbit_sizes.append(len(orbit))
        column += 1

    basis = sp.csr_array(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(n, column),
    )
    gram = basis.conj().T @ basis
    residual = float(sp.linalg.norm(gram - sp.eye(column, dtype=np.complex128, format="csr")))
    sector_labels = {} if labels is None else dict(labels)
    sector_labels.update(
        {
            "cyclic_orders": orders_tuple,
            "momentum_indices": tuple(
                momentum % order
                for momentum, order in zip(momenta_tuple, orders_tuple, strict=True)
            ),
            "orbit_sizes": tuple(int(value) for value in orbit_sizes),
        }
    )
    return SymmetrySectorBasis(
        basis=basis,
        labels=sector_labels,
        unitarity_residual=residual,
    )


def refine_sector_by_involution(
    sector: SymmetrySectorBasis,
    involution_index_permutation: npt.ArrayLike,
    *,
    eigenvalue: int,
    label: str = "parity",
    tolerance: float = 1.0e-9,
) -> SymmetrySectorBasis:
    """Refine a symmetry sector by a commuting involutive permutation."""
    if eigenvalue not in (-1, 1):
        raise ValueError("eigenvalue must be +1 or -1.")
    permutation = np.asarray(involution_index_permutation, dtype=np.int64).reshape(-1)
    if permutation.size != sector.full_dimension:
        raise ValueError("involution and sector basis have incompatible dimensions.")
    if not np.array_equal(permutation[permutation], np.arange(permutation.size)):
        raise ValueError("the supplied basis permutation is not an involution.")

    operator = permutation_matrix(permutation)
    q = sector.basis
    representation_raw = q.conj().T @ (operator @ q)
    representation = (
        representation_raw.toarray()
        if sp.issparse(representation_raw)
        else np.asarray(representation_raw, dtype=np.complex128)
    )
    representation = 0.5 * (representation + representation.conj().T)
    values, vectors = scipy_linalg.eigh(representation)
    mask = np.abs(values - float(eigenvalue)) <= tolerance
    if not np.any(mask):
        raise ValueError(f"requested {label} sector is empty.")
    refined = np.asarray(q @ vectors[:, mask], dtype=np.complex128)
    gram = refined.conj().T @ refined
    unitarity_residual = float(np.linalg.norm(gram - np.eye(refined.shape[1])))
    labels = dict(sector.labels)
    labels[label] = int(eigenvalue)
    labels[f"{label}_representation_residual"] = float(
        np.linalg.norm(representation @ representation - np.eye(representation.shape[0]))
    )
    return SymmetrySectorBasis(
        basis=refined,
        labels=labels,
        unitarity_residual=unitarity_residual,
    )


def project_operator_to_sector(
    operator: MatrixLike,
    sector: SymmetrySectorBasis | MatrixLike,
) -> npt.NDArray[np.complex128]:
    """Project an operator into an orthonormal sector basis."""
    basis = sector.basis if isinstance(sector, SymmetrySectorBasis) else sector
    if operator.shape[0] != operator.shape[1] or operator.shape[0] != basis.shape[0]:
        raise ValueError("operator and sector basis have incompatible dimensions.")
    projected = basis.conj().T @ (operator @ basis)
    if sp.issparse(projected):
        return np.asarray(projected.toarray(), dtype=np.complex128)
    return np.asarray(projected, dtype=np.complex128)


def project_state_to_sector(
    state: npt.ArrayLike,
    sector: SymmetrySectorBasis | MatrixLike,
) -> npt.NDArray[np.complex128]:
    """Return sector coordinates of a full-space state."""
    basis = sector.basis if isinstance(sector, SymmetrySectorBasis) else sector
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size != basis.shape[0]:
        raise ValueError("state and sector basis have incompatible dimensions.")
    return np.asarray(basis.conj().T @ vector, dtype=np.complex128).reshape(-1)


def lift_state_from_sector(
    sector_state: npt.ArrayLike,
    sector: SymmetrySectorBasis | MatrixLike,
) -> npt.NDArray[np.complex128]:
    """Lift sector coordinates into the full basis."""
    basis = sector.basis if isinstance(sector, SymmetrySectorBasis) else sector
    vector = np.asarray(sector_state, dtype=np.complex128).reshape(-1)
    if vector.size != basis.shape[1]:
        raise ValueError("sector state has incompatible dimension.")
    return np.asarray(basis @ vector, dtype=np.complex128).reshape(-1)


def thermodynamic_energy_window_plan(
    *,
    volume: int,
    energy_density: float,
    width_prefactor: float = 1.0,
    local_energy_scale: float = 1.0,
    width_exponent: float = 0.5,
) -> ThermodynamicEnergyWindowPlan:
    """Return a subextensive microcanonical energy-window plan.

    The window is centered at ``E_L = e * volume`` and has half-width
    ``width_prefactor * local_energy_scale * volume**width_exponent``.
    ``0 <= width_exponent < 1`` guarantees a vanishing energy-density width.
    Whether the number of states in the window grows must be checked from the
    spectrum; it is not implied by this kinematic scaling alone.
    """
    if volume <= 0:
        raise ValueError("volume must be positive.")
    if not np.isfinite(energy_density):
        raise ValueError("energy_density must be finite.")
    if not np.isfinite(width_prefactor) or width_prefactor <= 0.0:
        raise ValueError("width_prefactor must be finite and positive.")
    if not np.isfinite(local_energy_scale) or local_energy_scale <= 0.0:
        raise ValueError("local_energy_scale must be finite and positive.")
    if not np.isfinite(width_exponent) or not 0.0 <= width_exponent < 1.0:
        raise ValueError("width_exponent must satisfy 0 <= exponent < 1.")
    target = float(energy_density) * int(volume)
    half_width = (
        float(width_prefactor) * float(local_energy_scale) * float(volume) ** float(width_exponent)
    )
    return ThermodynamicEnergyWindowPlan(
        volume=int(volume),
        energy_density=float(energy_density),
        target_energy=float(target),
        half_width=float(half_width),
        energy_density_half_width=float(half_width / float(volume)),
        width_prefactor=float(width_prefactor),
        local_energy_scale=float(local_energy_scale),
        width_exponent=float(width_exponent),
    )


def select_microcanonical_window_by_width(
    eigenvalues: npt.ArrayLike,
    *,
    target_energy: float,
    half_width: float,
    exclude_indices: Sequence[int] = (),
    degeneracy_tolerance: float = 1.0e-10,
) -> MicrocanonicalWindowSelection:
    """Select every level in a prescribed energy interval.

    Exact degeneracies at the interval boundary are retained by the tolerance.
    The returned ``half_width`` is the largest actual distance of a retained
    level from the target; the requested width remains known to the caller via
    its thermodynamic window plan.
    """
    energies = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if not np.isfinite(target_energy):
        raise ValueError("target_energy must be finite.")
    if not np.isfinite(half_width) or half_width < 0.0:
        raise ValueError("half_width must be finite and non-negative.")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative.")
    excluded = {int(index) for index in exclude_indices}
    if any(index < 0 or index >= energies.size for index in excluded):
        raise IndexError("exclude_indices contains an out-of-range level index.")
    mask = np.abs(energies - float(target_energy)) <= (
        float(half_width) + float(degeneracy_tolerance)
    )
    if excluded:
        mask[np.fromiter(excluded, dtype=np.int64)] = False
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise ValueError("the prescribed microcanonical window contains no states.")
    selected_energies = energies[selected]
    actual_half_width = float(np.max(np.abs(selected_energies - float(target_energy))))
    return MicrocanonicalWindowSelection(
        indices=tuple(int(index) for index in selected),
        target_energy=float(target_energy),
        half_width=actual_half_width,
        energy_min=float(np.min(selected_energies)),
        energy_max=float(np.max(selected_energies)),
        mean_energy=float(np.mean(selected_energies)),
        center_offset=float(np.mean(selected_energies) - float(target_energy)),
    )


def microcanonical_ensemble_from_spectrum(
    eigenvalues: npt.ArrayLike,
    eigenvectors: npt.ArrayLike,
    *,
    target_energy: float,
    half_width: float,
    exclude_indices: Sequence[int] = (),
    degeneracy_tolerance: float = 1.0e-10,
    volume: int | None = None,
) -> SpectralMicrocanonicalEnsemble:
    """Construct an equal-weight microcanonical ensemble from an eigensystem.

    The interval is selected by :func:`select_microcanonical_window_by_width`.
    Degenerate levels intersecting the numerical boundary are retained through
    ``degeneracy_tolerance``.  The returned object stores only the selected
    eigenvectors and therefore remains a low-rank representation of the
    density matrix.
    """
    energies = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[0] != energies.size or vectors.shape[1] != energies.size:
        raise ValueError("eigenvectors must be a square matrix whose columns match eigenvalues.")
    if volume is not None and int(volume) <= 0:
        raise ValueError("volume must be positive when supplied.")
    selection = select_microcanonical_window_by_width(
        energies,
        target_energy=target_energy,
        half_width=half_width,
        exclude_indices=exclude_indices,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    indices = np.asarray(selection.indices, dtype=np.int64)
    selected_vectors = np.asarray(vectors[:, indices], dtype=np.complex128)
    weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    return SpectralMicrocanonicalEnsemble(
        selection=selection,
        eigenvalues=np.asarray(energies[indices], dtype=np.float64),
        eigenvectors=selected_vectors,
        weights=weights,
        volume=None if volume is None else int(volume),
    )


def gaussian_spectral_filter(
    eigenvalues: npt.ArrayLike,
    *,
    target_energy: float,
    sigma: float,
    cutoff_sigma: float | None = 6.0,
) -> SmoothSpectralFilter:
    """Return normalized Gaussian weights centered on ``target_energy``."""
    energies = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if energies.size == 0:
        raise ValueError("eigenvalues must not be empty.")
    if not np.isfinite(target_energy):
        raise ValueError("target_energy must be finite.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    scaled = (energies - float(target_energy)) / float(sigma)
    weights = np.exp(-0.5 * scaled**2)
    if cutoff_sigma is not None:
        if not np.isfinite(cutoff_sigma) or cutoff_sigma <= 0.0:
            raise ValueError("cutoff_sigma must be finite and positive when supplied.")
        weights[np.abs(scaled) > float(cutoff_sigma)] = 0.0
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Gaussian filter has zero numerical weight.")
    weights /= total
    mean_energy = float(np.dot(weights, energies))
    variance = float(np.dot(weights, (energies - mean_energy) ** 2))
    effective = float(1.0 / np.sum(weights**2))
    return SmoothSpectralFilter(
        weights=tuple(float(value) for value in weights),
        target_energy=float(target_energy),
        sigma=float(sigma),
        mean_energy=mean_energy,
        energy_variance=max(variance, 0.0),
        effective_state_count=effective,
    )


def spectral_observable_moments(
    operator: MatrixLike,
    eigenvectors: npt.ArrayLike,
    *,
    squared_operator: MatrixLike | None = None,
    indices: Sequence[int] | None = None,
    weights: npt.ArrayLike | None = None,
    hermiticity_tolerance: float = 1.0e-10,
) -> SpectralObservableMoments:
    """Evaluate ``Tr(rho O)``, ``Tr(rho O^2)``, and measurement variance.

    ``eigenvectors`` are columns. The state is an equal-weight mixture over
    ``indices`` unless explicit non-negative normalized ``weights`` are given.

    ``squared_operator`` is useful after symmetry projection. For a local
    observable that does not preserve the resolved symmetry sector, the
    correct second moment is ``P O^2 P``, not ``(P O P)^2``. Supply the former
    as ``squared_operator``. When omitted, the function uses the action norm
    and therefore evaluates the square of the supplied ``operator``.
    """
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[0] != operator.shape[0]:
        raise ValueError("eigenvectors must have shape (dimension, n_states).")
    if sp.issparse(operator):
        anti = operator - operator.conj().T
        anti_norm = float(sp.linalg.norm(anti))
    else:
        dense = np.asarray(operator, dtype=np.complex128)
        anti_norm = float(np.linalg.norm(dense - dense.conj().T))
    if anti_norm > hermiticity_tolerance:
        raise ValueError("operator must be Hermitian at the stated tolerance.")

    if indices is None:
        selected = np.arange(vectors.shape[1], dtype=np.int64)
    else:
        selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
        if selected.size == 0:
            raise ValueError("indices must not be empty.")
        if np.any(selected < 0) or np.any(selected >= vectors.shape[1]):
            raise IndexError("indices contains an out-of-range eigenvector index.")
    selected_vectors = vectors[:, selected]
    actions = np.asarray(operator @ selected_vectors, dtype=np.complex128)
    per_mean = np.einsum("ij,ij->j", selected_vectors.conj(), actions).real
    if squared_operator is None:
        per_second = np.einsum("ij,ij->j", actions.conj(), actions).real
    else:
        if squared_operator.shape != operator.shape:
            raise ValueError("squared_operator must have the same shape as operator.")
        if sp.issparse(squared_operator):
            squared_anti = squared_operator - squared_operator.conj().T
            squared_anti_norm = float(sp.linalg.norm(squared_anti))
        else:
            squared_dense = np.asarray(squared_operator, dtype=np.complex128)
            squared_anti_norm = float(np.linalg.norm(squared_dense - squared_dense.conj().T))
        if squared_anti_norm > hermiticity_tolerance:
            raise ValueError("squared_operator must be Hermitian at the stated tolerance.")
        squared_actions = np.asarray(
            squared_operator @ selected_vectors,
            dtype=np.complex128,
        )
        per_second = np.einsum(
            "ij,ij->j",
            selected_vectors.conj(),
            squared_actions,
        ).real

    if weights is None:
        normalized_weights = np.full(selected.size, 1.0 / selected.size, dtype=np.float64)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.size == vectors.shape[1] and selected.size != vectors.shape[1]:
            normalized_weights = normalized_weights[selected]
        if normalized_weights.size != selected.size:
            raise ValueError("weights must match the selected state count or all eigenvectors.")
        if np.any(normalized_weights < 0.0) or not np.all(np.isfinite(normalized_weights)):
            raise ValueError("weights must be finite and non-negative.")
        total = float(np.sum(normalized_weights))
        if total <= 0.0:
            raise ValueError("weights must have a positive sum.")
        normalized_weights = normalized_weights / total

    mean = float(np.dot(normalized_weights, per_mean))
    second = float(np.dot(normalized_weights, per_second))
    variance = float(max(second - mean**2, 0.0))
    effective = float(1.0 / np.sum(normalized_weights**2))
    return SpectralObservableMoments(
        mean=mean,
        second_moment=second,
        variance=variance,
        n_states=int(selected.size),
        effective_state_count=effective,
        minimum_expectation=float(np.min(per_mean)),
        maximum_expectation=float(np.max(per_mean)),
    )


def product_basis_diagonal_phase_factors(
    basis_configs: npt.ArrayLike,
    local_phases: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Return ``exp(i sum_j theta_j n_j)`` for product-basis configurations."""
    configs = np.asarray(basis_configs, dtype=np.float64)
    phases = np.asarray(local_phases, dtype=np.float64).reshape(-1)
    if configs.ndim != 2 or configs.shape[1] != phases.size:
        raise ValueError("basis_configs and local_phases have incompatible shapes.")
    if not np.all(np.isfinite(configs)) or not np.all(np.isfinite(phases)):
        raise ValueError("basis_configs and local_phases must be finite.")
    return np.exp(1.0j * (configs @ phases)).astype(np.complex128, copy=False)


def select_microcanonical_window_by_count(
    eigenvalues: npt.ArrayLike,
    *,
    target_energy: float,
    target_count: int,
    exclude_indices: Sequence[int] = (),
    include_boundary_degeneracy: bool = True,
    degeneracy_tolerance: float = 1.0e-10,
) -> MicrocanonicalWindowSelection:
    """Select the closest finite-size energy levels to a target energy."""
    energies = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative.")
    excluded = {int(index) for index in exclude_indices}
    if any(index < 0 or index >= energies.size for index in excluded):
        raise IndexError("exclude_indices contains an out-of-range level index.")
    candidates = np.asarray(
        [index for index in range(energies.size) if index not in excluded], dtype=np.int64
    )
    if candidates.size < target_count:
        raise ValueError("not enough levels remain to fill the requested window.")
    distances = np.abs(energies[candidates] - float(target_energy))
    order = np.lexsort((energies[candidates], distances))
    initially_selected = candidates[order[:target_count]]
    if include_boundary_degeneracy:
        cutoff = float(np.max(np.abs(energies[initially_selected] - float(target_energy))))
        selected = np.sort(candidates[distances <= cutoff + float(degeneracy_tolerance)])
    else:
        selected = np.sort(initially_selected)
    selected_energies = energies[selected]
    half_width = float(np.max(np.abs(selected_energies - float(target_energy))))
    return MicrocanonicalWindowSelection(
        indices=tuple(int(index) for index in selected),
        target_energy=float(target_energy),
        half_width=half_width,
        energy_min=float(np.min(selected_energies)),
        energy_max=float(np.max(selected_energies)),
        mean_energy=float(np.mean(selected_energies)),
        center_offset=float(np.mean(selected_energies) - float(target_energy)),
    )


def adjacent_gap_ratio_report(
    eigenvalues: npt.ArrayLike,
    *,
    trim_fraction: float = 0.1,
    degeneracy_tolerance: float = 1.0e-10,
) -> AdjacentGapRatioReport:
    """Compute adjacent-gap ratios after trimming spectral edges.

    The input must already belong to one fully desymmetrized sector.  Pairs
    touching a spacing below ``degeneracy_tolerance`` are omitted and their
    number remains visible through ``n_levels_used`` and ``n_ratios``.
    """
    energies = np.sort(np.asarray(eigenvalues, dtype=np.float64).reshape(-1))
    if energies.size < 3:
        raise ValueError("at least three energy levels are required.")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must satisfy 0 <= trim_fraction < 0.5.")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative.")

    trim = int(np.floor(trim_fraction * energies.size))
    trimmed = energies[trim : energies.size - trim] if trim > 0 else energies
    if trimmed.size < 3:
        raise ValueError("too few levels remain after trimming.")
    spacings = np.diff(trimmed)
    ratios: list[float] = []
    for left, right in zip(spacings[:-1], spacings[1:], strict=True):
        if left <= degeneracy_tolerance or right <= degeneracy_tolerance:
            continue
        ratios.append(float(min(left, right) / max(left, right)))
    if not ratios:
        raise ValueError("no nondegenerate adjacent-gap ratios remain.")
    return AdjacentGapRatioReport(
        mean_ratio=float(np.mean(ratios)),
        ratios=tuple(ratios),
        spacings=tuple(float(value) for value in spacings),
        n_levels_input=int(energies.size),
        n_levels_used=int(trimmed.size),
        trim_fraction=float(trim_fraction),
        degeneracy_tolerance=float(degeneracy_tolerance),
    )


def eigenstate_expectations(
    operator: MatrixLike,
    eigenvectors: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return diagonal matrix elements of an operator in supplied eigenstates."""
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[0] != operator.shape[0]:
        raise ValueError("eigenvectors must have shape (dimension, n_states).")
    actions = np.asarray(operator @ vectors, dtype=np.complex128)
    return np.einsum("ij,ij->j", vectors.conj(), actions).real.astype(np.float64)
