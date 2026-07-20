from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
