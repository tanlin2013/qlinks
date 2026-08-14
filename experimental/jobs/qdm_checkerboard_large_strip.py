"""Memory-scalable helpers for the circumference-four QDM evidence workflow.

This module is intentionally experimental.  It keeps the expensive 12x4
scientific machinery out of the reusable :mod:`qlinks` API while giving the
batch notebook deterministic, inspectable primitives for

* zero-momentum ``T_x^2,T_y^2`` sector construction,
* finite-temperature canonical typicality, and
* partial interior spectra from shift-invert.

None of the routines below turns a finite-size result into a thermodynamic
claim.  Callers must export coverage, stochastic error, residual, and method
metadata together with every estimate.
"""

from __future__ import annotations

import itertools
import resource
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qlinks.basis import Basis
from qlinks.caging.analysis.spectral import SymmetrySectorBasis
from qlinks.encoded import BinaryEncodedBasis, encode_binary_config


@dataclass(frozen=True, slots=True)
class CanonicalTypicalityScan:
    beta: npt.NDArray[np.float64]
    partition: npt.NDArray[np.float64]
    energy: npt.NDArray[np.float64]
    energy_stderr: npt.NDArray[np.float64]
    observables: dict[str, npt.NDArray[np.float64]]
    observable_stderr: dict[str, npt.NDArray[np.float64]]
    n_samples: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class EnergyMatchedCanonicalEstimate:
    beta: float
    energy: float
    energy_stderr: float
    observables: dict[str, float]
    observable_stderr: dict[str, float]
    bracket: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PartialSpectrum:
    energies: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.complex128]
    residuals: npt.NDArray[np.float64]
    sigma: float
    target_energy: float
    method: str = "shift_invert_superlu"
    requested_subspace_size: int | None = None
    transformed_residuals: npt.NDArray[np.float64] | None = None
    peak_rss_gib: float | None = None

    @property
    def min_energy(self) -> float:
        return float(np.min(self.energies))

    @property
    def max_energy(self) -> float:
        return float(np.max(self.energies))

    @property
    def maximum_residual(self) -> float:
        return float(np.max(self.residuals, initial=0.0))

    def covers_window(self, half_width: float, *, margin: float = 0.0) -> bool:
        width = float(half_width) + float(margin)
        return bool(
            self.min_energy <= self.target_energy - width
            and self.max_energy >= self.target_energy + width
        )


def process_peak_rss_gib() -> float:
    """Return this process' maximum resident-set size in GiB."""

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_value = value if sys.platform == "darwin" else value * 1024.0
    return bytes_value / float(1024**3)


def project_sparse_operator_to_sector(
    operator: sp.spmatrix | sp.sparray,
    sector: SymmetrySectorBasis | sp.spmatrix | sp.sparray,
) -> sp.csr_array:
    """Project a sparse operator without materializing a dense sector matrix."""

    basis = sector.basis if isinstance(sector, SymmetrySectorBasis) else sector
    if operator.shape[0] != operator.shape[1] or operator.shape[0] != basis.shape[0]:
        raise ValueError("operator and sector basis have incompatible dimensions")
    return sp.csr_array(basis.conj().T @ (operator @ basis))


def materialize_periodic_product_state_from_basis(
    instance: Any,
    basis: Basis | BinaryEncodedBasis,
    *,
    normalize: bool = True,
    tolerance: float = 1.0e-12,
) -> npt.NDArray[np.complex128]:
    """Materialize a periodic product cage using the basis' existing index.

    The package helper accepts only a raw configuration array and therefore
    builds a second all-state Python lookup.  The large-strip evidence job
    already owns a :class:`Basis` with an encoded index, so reusing it avoids a
    multi-gigabyte duplicate lookup on 12x4.
    """

    state = np.zeros(basis.n_states, dtype=np.complex128)
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            instance.padding.exterior_link_ids,
            instance.padding.exterior_config,
            strict=True,
        )
    }
    support_ranges = tuple(range(block.support_size) for block in instance.blocks)
    for support_indices in itertools.product(*support_ranges):
        complete = np.zeros(basis.n_variables, dtype=np.int64)
        for link_id, value in exterior.items():
            complete[link_id] = value
        amplitude = 1.0 + 0.0j
        for block, support_index in zip(instance.blocks, support_indices, strict=True):
            complete[np.asarray(block.link_ids, dtype=np.int64)] = block.support_configs[
                support_index
            ]
            amplitude *= complex(block.amplitudes[support_index])
        try:
            if isinstance(basis, BinaryEncodedBasis):
                basis_index = basis.require_index(encode_binary_config(complete))
            else:
                basis_index = basis.require_index(complete)
        except KeyError as exc:
            raise ValueError(
                "a periodic-product support configuration is absent from the supplied basis"
            ) from exc
        state[basis_index] += amplitude
    norm = float(np.linalg.norm(state))
    if norm <= tolerance:
        raise ValueError("materialized periodic-product state has zero norm")
    if normalize:
        state /= norm
    return state


def binary_basis_configs_uint8(
    basis: Basis | BinaryEncodedBasis,
    *,
    chunk_size: int = 32768,
) -> npt.NDArray[np.uint8]:
    """Materialize binary configurations with one byte per variable.

    The local-witness embedding machinery still consumes an explicit
    configuration table.  For the 12x4 production basis this helper keeps that
    unavoidable table at ~1 byte/configuration-variable instead of the 8-byte
    int64 representation returned by ``to_array_basis()``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(basis, BinaryEncodedBasis):
        result = np.empty((basis.n_states, basis.n_variables), dtype=np.uint8)
        for start in range(0, basis.n_states, int(chunk_size)):
            stop = min(start + int(chunk_size), basis.n_states)
            packed = _encoded_codes_to_packed_bytes(
                basis.codes[start:stop],
                n_variables=basis.n_variables,
            )
            result[start:stop] = np.unpackbits(packed, axis=1, bitorder="little")[
                :, : basis.n_variables
            ]
        return result

    states = np.asarray(basis.states)
    if not np.all((states == 0) | (states == 1)):
        raise ValueError("binary_basis_configs_uint8 requires a binary basis")
    return states.astype(np.uint8, copy=False)


def packed_binary_basis_index(
    basis: Basis | BinaryEncodedBasis,
) -> Mapping[bytes | int, int]:
    """Return the existing compact lookup for a binary dimer basis.

    ``BinaryEncodedBasis`` already owns the most memory-efficient integer-code
    index, so the large-strip path reuses it directly instead of materializing
    millions of explicit 0/1 configurations.  Array bases retain the historical
    packed-byte lookup used by small smoke/known runs.
    """

    if isinstance(basis, BinaryEncodedBasis):
        return basis.index

    states = np.asarray(basis.states)
    if not np.all((states == 0) | (states == 1)):
        raise ValueError("packed_binary_basis_index requires a binary basis")
    packed = np.packbits(states.astype(np.uint8, copy=False), axis=1, bitorder="little")
    return {row.tobytes(): int(index) for index, row in enumerate(packed)}


def _encoded_codes_to_packed_bytes(
    codes: npt.ArrayLike,
    *,
    n_variables: int,
) -> npt.NDArray[np.uint8]:
    """Decode a bounded chunk of Python-int codes into packed little-endian bytes."""

    code_values = np.asarray(codes, dtype=object).reshape(-1)
    byte_width = (int(n_variables) + 7) // 8
    blob = b"".join(int(code).to_bytes(byte_width, "little", signed=False) for code in code_values)
    if not blob:
        return np.zeros((0, byte_width), dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).reshape(code_values.size, byte_width)


def translation_permutation_from_binary_basis(
    model: Any,
    basis: Basis | BinaryEncodedBasis,
    *,
    packed_index: Mapping[bytes | int, int],
    dx: int = 0,
    dy: int = 0,
    chunk_size: int = 32768,
) -> npt.NDArray[np.int64]:
    """Build a dimer-basis translation permutation in bounded temporary memory.

    The production 12x4 bitmask build returns :class:`BinaryEncodedBasis`.
    That path is handled chunk-wise without ever constructing the
    ``n_states x n_variables`` configuration matrix, which would otherwise
    dominate memory before the symmetry projection begins.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    lookup: dict[tuple[int, int, str], int] = {}
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        variable = int(model.layout.link_variable_index(int(link.id)))
        lookup[(int(x), int(y), str(link.kind))] = variable

    target_for_source = np.empty(basis.n_variables, dtype=np.int64)
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        source = int(model.layout.link_variable_index(int(link.id)))
        target = lookup[
            ((int(x) + int(dx)) % int(model.lx), (int(y) + int(dy)) % int(model.ly), str(link.kind))
        ]
        target_for_source[source] = int(target)

    source_for_target = np.empty_like(target_for_source)
    source_for_target[target_for_source] = np.arange(target_for_source.size, dtype=np.int64)
    result = np.empty(basis.n_states, dtype=np.int64)

    if isinstance(basis, BinaryEncodedBasis):
        # Work with <=O(chunk_size * n_variables) bytes at a time.  The
        # encoded basis' int->index dictionary is reused as the lookup.
        byte_width = (basis.n_variables + 7) // 8
        for start in range(0, basis.n_states, int(chunk_size)):
            stop = min(start + int(chunk_size), basis.n_states)
            packed = _encoded_codes_to_packed_bytes(
                basis.codes[start:stop],
                n_variables=basis.n_variables,
            )
            unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, : basis.n_variables]
            transformed = unpacked[:, source_for_target]
            transformed_packed = np.packbits(transformed, axis=1, bitorder="little")
            if transformed_packed.shape[1] != byte_width:
                raise RuntimeError("unexpected packed-width change during translation")
            for offset, row in enumerate(transformed_packed):
                code = int.from_bytes(row.tobytes(), "little", signed=False)
                try:
                    result[start + offset] = int(packed_index[code])
                except KeyError as exc:
                    raise ValueError(
                        "translated configuration is absent from the encoded basis"
                    ) from exc
        return result

    states = np.asarray(basis.states)
    for start in range(0, basis.n_states, int(chunk_size)):
        stop = min(start + int(chunk_size), basis.n_states)
        transformed = states[start:stop, source_for_target]
        packed = np.packbits(transformed.astype(np.uint8, copy=False), axis=1, bitorder="little")
        for offset, row in enumerate(packed):
            try:
                result[start + offset] = int(packed_index[row.tobytes()])
            except KeyError as exc:
                raise ValueError("translated configuration is absent from the basis") from exc
    return result


def zero_momentum_commuting_sector_basis(
    permutations: Sequence[npt.ArrayLike],
    *,
    labels: Mapping[str, object] | None = None,
) -> SymmetrySectorBasis:
    """Build the common zero-momentum sector from commuting permutations.

    The production checkerboard protocol uses only the ``k=0`` character of
    ``T_x^2`` and ``T_y^2``.  For that special case an orbit flood-fill is much
    cheaper than constructing every group power for every seed.
    """

    perms = tuple(np.asarray(value, dtype=np.int64).reshape(-1) for value in permutations)
    if not perms:
        raise ValueError("permutations must not be empty")
    n = int(perms[0].size)
    if any(value.size != n for value in perms):
        raise ValueError("all permutations must have the same size")
    for value in perms:
        if np.unique(value).size != n or np.any(value < 0) or np.any(value >= n):
            raise ValueError("every symmetry action must be a permutation of range(n)")
    for left in range(len(perms)):
        for right in range(left + 1, len(perms)):
            if not np.array_equal(perms[left][perms[right]], perms[right][perms[left]]):
                raise ValueError("the supplied symmetry permutations do not commute")

    visited = np.zeros(n, dtype=bool)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    orbit_sizes: list[int] = []
    column = 0
    for seed in range(n):
        if visited[seed]:
            continue
        orbit: list[int] = []
        stack = [int(seed)]
        visited[seed] = True
        while stack:
            state = stack.pop()
            orbit.append(state)
            for permutation in perms:
                nxt = int(permutation[state])
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        normalization = 1.0 / np.sqrt(float(len(orbit)))
        rows.extend(orbit)
        columns.extend([column] * len(orbit))
        data.extend([normalization] * len(orbit))
        orbit_sizes.append(len(orbit))
        column += 1

    basis = sp.csr_array(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(n, column),
    )
    gram = basis.conj().T @ basis
    residual = float(spla.norm(gram - sp.eye(column, dtype=np.complex128, format="csr")))
    metadata = {} if labels is None else dict(labels)
    metadata.update(
        {
            "symmetry": "commuting_zero_momentum",
            "momentum_indices": tuple(0 for _ in perms),
            "orbit_sizes": tuple(int(value) for value in orbit_sizes),
        }
    )
    return SymmetrySectorBasis(basis=basis, labels=metadata, unitarity_residual=residual)


def canonical_typicality_scan(
    hamiltonian: sp.spmatrix | sp.sparray,
    observables: Mapping[str, sp.spmatrix | sp.sparray],
    *,
    beta_max: float,
    beta_points: int,
    n_samples: int,
    random_seed: int,
) -> CanonicalTypicalityScan:
    """Estimate canonical traces on an evenly spaced positive-beta grid.

    Complex random-phase vectors satisfy ``E[|r><r|]=I``.  The estimator is
    therefore an unbiased stochastic trace estimator before taking ratios.
    Reported uncertainties are leave-one-sample-out jackknife standard errors
    of the corresponding ratio estimates.
    """

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    if h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must be square")
    if beta_max <= 0.0 or beta_points < 2:
        raise ValueError("beta_max must be positive and beta_points must be at least two")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two to estimate stochastic uncertainty")
    obs = {
        name: sp.csr_array(operator, dtype=np.complex128) for name, operator in observables.items()
    }
    if any(operator.shape != h.shape for operator in obs.values()):
        raise ValueError("every observable must have the Hamiltonian shape")

    beta = np.linspace(0.0, float(beta_max), int(beta_points), dtype=np.float64)
    rng = np.random.default_rng(int(random_seed))
    n_beta = beta.size
    z_samples = np.zeros((n_samples, n_beta), dtype=np.float64)
    e_samples = np.zeros_like(z_samples)
    o_samples = {name: np.zeros_like(z_samples) for name in obs}

    generator = -0.5 * h
    for sample in range(int(n_samples)):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=h.shape[0])
        vector = np.exp(1.0j * phases).astype(np.complex128, copy=False)
        evolved = spla.expm_multiply(
            generator,
            vector,
            start=0.0,
            stop=float(beta_max),
            num=int(beta_points),
            endpoint=True,
        )
        for index, state in enumerate(evolved):
            denominator = float(np.vdot(state, state).real)
            z_samples[sample, index] = denominator
            e_samples[sample, index] = float(np.vdot(state, h @ state).real)
            for name, operator in obs.items():
                o_samples[name][sample, index] = float(np.vdot(state, operator @ state).real)

    def ratio_and_jackknife(numerators: npt.NDArray[np.float64]):
        numerator_sum = np.sum(numerators, axis=0)
        denominator_sum = np.sum(z_samples, axis=0)
        ratio = numerator_sum / denominator_sum
        leave_one_out = np.empty_like(numerators)
        for sample in range(n_samples):
            leave_one_out[sample] = (numerator_sum - numerators[sample]) / (
                denominator_sum - z_samples[sample]
            )
        center = np.mean(leave_one_out, axis=0)
        stderr = np.sqrt(
            (n_samples - 1.0) / n_samples * np.sum((leave_one_out - center[None, :]) ** 2, axis=0)
        )
        return ratio, stderr

    energy, energy_stderr = ratio_and_jackknife(e_samples)
    observable_values: dict[str, npt.NDArray[np.float64]] = {}
    observable_stderr: dict[str, npt.NDArray[np.float64]] = {}
    for name, values in o_samples.items():
        estimate, stderr = ratio_and_jackknife(values)
        observable_values[name] = estimate
        observable_stderr[name] = stderr

    return CanonicalTypicalityScan(
        beta=beta,
        partition=np.mean(z_samples, axis=0),
        energy=energy,
        energy_stderr=energy_stderr,
        observables=observable_values,
        observable_stderr=observable_stderr,
        n_samples=int(n_samples),
        random_seed=int(random_seed),
    )


def energy_matched_canonical_estimate(
    scan: CanonicalTypicalityScan,
    *,
    target_energy: float,
) -> EnergyMatchedCanonicalEstimate:
    """Linearly interpolate the canonical scan at the target energy."""

    difference = np.asarray(scan.energy, dtype=float) - float(target_energy)
    exact = np.flatnonzero(np.abs(difference) <= 1.0e-14)
    if exact.size:
        index = int(exact[0])
        return EnergyMatchedCanonicalEstimate(
            beta=float(scan.beta[index]),
            energy=float(scan.energy[index]),
            energy_stderr=float(scan.energy_stderr[index]),
            observables={name: float(values[index]) for name, values in scan.observables.items()},
            observable_stderr={
                name: float(values[index]) for name, values in scan.observable_stderr.items()
            },
            bracket=(float(scan.beta[index]), float(scan.beta[index])),
        )
    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
    if crossings.size == 0:
        raise RuntimeError(
            "canonical beta grid does not bracket the target energy: "
            f"target={target_energy:.12g}, range=[{scan.energy.min():.12g}, "
            f"{scan.energy.max():.12g}]"
        )
    lower = int(crossings[0])
    upper = lower + 1
    e0, e1 = float(scan.energy[lower]), float(scan.energy[upper])
    fraction = (float(target_energy) - e0) / (e1 - e0)

    def interp(values: npt.ArrayLike) -> float:
        arr = np.asarray(values, dtype=float)
        return float(arr[lower] + fraction * (arr[upper] - arr[lower]))

    return EnergyMatchedCanonicalEstimate(
        beta=interp(scan.beta),
        energy=float(target_energy),
        energy_stderr=interp(scan.energy_stderr),
        observables={name: interp(values) for name, values in scan.observables.items()},
        observable_stderr={name: interp(values) for name, values in scan.observable_stderr.items()},
        bracket=(float(scan.beta[lower]), float(scan.beta[upper])),
    )


def folded_spectrum_partial_spectrum(
    hamiltonian: sp.spmatrix | sp.sparray,
    *,
    target_energy: float,
    subspace_size: int,
    tolerance: float = 1.0e-8,
    maxiter: int | None = None,
    ncv_factor: float = 2.05,
    random_seed: int = 20260811,
    cluster_tolerance: float | None = None,
) -> PartialSpectrum:
    """Return interior eigenpairs without factorizing ``H-target_energy``.

    Lanczos is applied to the positive semidefinite folded operator
    ``(H-E_target I)^2``.  Every Krylov action uses two sparse Hamiltonian
    products, so no SuperLU factors are constructed.  Degenerate folded
    eigenspaces are resolved by diagonalizing ``H`` only inside each small
    folded-eigenvalue cluster.

    This is still a partial-spectrum method: callers must require explicit
    window coverage, eigenpair residuals, and budget convergence.
    """

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    n = int(h.shape[0])
    if h.shape[1] != n:
        raise ValueError("hamiltonian must be square")
    if subspace_size <= 0:
        raise ValueError("subspace_size must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if ncv_factor <= 1.0:
        raise ValueError("ncv_factor must exceed one")

    k = min(int(subspace_size), n - 2)
    if k <= 0:
        dense = h.toarray()
        values, vectors = np.linalg.eigh(dense)
        residuals = np.linalg.norm(h @ vectors - vectors * values[None, :], axis=0)
        return PartialSpectrum(
            energies=np.asarray(values, dtype=np.float64),
            eigenvectors=np.asarray(vectors, dtype=np.complex128),
            residuals=np.asarray(residuals, dtype=np.float64),
            sigma=float(target_energy),
            target_energy=float(target_energy),
            method="dense_eigh_small_sector",
            requested_subspace_size=int(subspace_size),
            transformed_residuals=np.zeros_like(values, dtype=np.float64),
            peak_rss_gib=process_peak_rss_gib(),
        )

    shifted = h - float(target_energy) * sp.eye(n, dtype=np.complex128, format="csr")

    def folded_matvec(vector):
        return shifted @ (shifted @ vector)

    def folded_matmat(matrix):
        return shifted @ (shifted @ matrix)

    folded = spla.LinearOperator(
        shape=(n, n), matvec=folded_matvec, matmat=folded_matmat, dtype=np.complex128
    )
    ncv = min(n, max(k + 2, int(np.ceil(float(ncv_factor) * k))))
    rng = np.random.default_rng(int(random_seed))
    v0 = rng.normal(size=n) + 1.0j * rng.normal(size=n)
    v0 = np.asarray(v0 / np.linalg.norm(v0), dtype=np.complex128)
    arpack_status = "converged"
    try:
        folded_values, vectors = spla.eigsh(
            folded,
            k=k,
            which="SA",
            tol=float(tolerance),
            maxiter=maxiter,
            ncv=ncv,
            v0=v0,
            return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        if exc.eigenvalues is None or exc.eigenvectors is None or len(exc.eigenvalues) < 4:
            raise
        folded_values, vectors = exc.eigenvalues, exc.eigenvectors
        arpack_status = "partial_convergence"
        k = int(len(folded_values))
    folded_values = np.asarray(np.real(folded_values), dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.complex128)
    order = np.argsort(folded_values)
    folded_values = folded_values[order]
    vectors = vectors[:, order]

    if cluster_tolerance is None:
        cluster_tolerance = max(1.0e-11, 50.0 * float(tolerance))
    energies_out = []
    vectors_out = []
    transformed_residuals_out = []
    begin = 0
    while begin < k:
        end = begin + 1
        reference = float(folded_values[begin])
        scale = max(1.0, abs(reference))
        while (
            end < k
            and abs(float(folded_values[end]) - reference) <= float(cluster_tolerance) * scale
        ):
            end += 1
        block = vectors[:, begin:end]
        h_block = h @ block
        compressed = block.conj().T @ h_block
        compressed = 0.5 * (compressed + compressed.conj().T)
        local_energies, rotation = np.linalg.eigh(compressed)
        rotated = block @ rotation
        folded_residual = np.linalg.norm(
            folded_matmat(block) - block * folded_values[None, begin:end], axis=0
        )
        energies_out.append(np.asarray(local_energies, dtype=np.float64))
        vectors_out.append(np.asarray(rotated, dtype=np.complex128))
        transformed_residuals_out.append(np.asarray(folded_residual, dtype=np.float64))
        begin = end

    energies = np.concatenate(energies_out)
    eigenvectors = np.column_stack(vectors_out)
    transformed_residuals = np.concatenate(transformed_residuals_out)
    energy_order = np.argsort(energies)
    energies = energies[energy_order]
    eigenvectors = eigenvectors[:, energy_order]
    transformed_residuals = transformed_residuals[energy_order]
    residual_matrix = h @ eigenvectors - eigenvectors * energies[None, :]
    residuals = np.linalg.norm(residual_matrix, axis=0)
    return PartialSpectrum(
        energies=np.asarray(energies, dtype=np.float64),
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        residuals=np.asarray(residuals, dtype=np.float64),
        sigma=float(target_energy),
        target_energy=float(target_energy),
        method=(
            "folded_spectrum_lanczos"
            if arpack_status == "converged"
            else "folded_spectrum_lanczos_partial_arpack"
        ),
        requested_subspace_size=int(subspace_size),
        transformed_residuals=np.asarray(transformed_residuals, dtype=np.float64),
        peak_rss_gib=process_peak_rss_gib(),
    )


def shift_invert_partial_spectrum(
    hamiltonian: sp.spmatrix | sp.sparray,
    *,
    target_energy: float,
    eigenpairs: int,
    sigma_offset: float = 1.0e-6,
    tolerance: float = 1.0e-9,
    maxiter: int | None = None,
    ncv: int | None = None,
) -> PartialSpectrum:
    """Return interior eigenpairs nearest the target using sparse shift-invert.

    ``sigma`` is offset from the exact cage energy because the checkerboard
    family contains an exact eigenstate at the target and factorizing exactly
    at that eigenvalue can make the shifted matrix singular.
    """

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    n = int(h.shape[0])
    if h.shape[1] != n:
        raise ValueError("hamiltonian must be square")
    if eigenpairs <= 0:
        raise ValueError("eigenpairs must be positive")
    k = min(int(eigenpairs), n - 2)
    if k <= 0:
        dense = h.toarray()
        values, vectors = np.linalg.eigh(dense)
    else:
        sigma = float(target_energy) + float(sigma_offset)
        values, vectors = spla.eigsh(
            h,
            k=k,
            sigma=sigma,
            which="LM",
            tol=float(tolerance),
            maxiter=maxiter,
            ncv=ncv,
        )
    values = np.asarray(values, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.complex128)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    residual_matrix = h @ vectors - vectors * values[None, :]
    residuals = np.linalg.norm(residual_matrix, axis=0)
    return PartialSpectrum(
        energies=values,
        eigenvectors=vectors,
        residuals=np.asarray(residuals, dtype=np.float64),
        sigma=float(target_energy) + float(sigma_offset),
        target_energy=float(target_energy),
        method="shift_invert_superlu",
        requested_subspace_size=int(eigenpairs),
        transformed_residuals=None,
        peak_rss_gib=process_peak_rss_gib(),
    )
