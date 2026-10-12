"""Recover Hamiltonian Ritz pairs from a complete saved folded subspace.

Near-degenerate eigenvalues of (H-E*)^2 can mix opposite-energy states even
when the folded residual is small. Resolving H in narrow scalar clusters can
miss that mixing. This module uses the entire returned subspace and its Gram
matrix, then measures actual H residuals and full output orthogonality.
It cannot repair components missing from the input subspace.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from qdm_checkerboard_large_strip import PartialSpectrum, process_peak_rss_gib

RECONSTRUCTION = "full_subspace_generalized_rayleigh_ritz_v1"
REFINED_BACKEND = "primme_full_ritz_v1"
REFINEMENT_PREFIX = "primme_full_ritz_refinement_v1_"


def refinement_backend(tolerance: float) -> str:
    """Keep distinct raw tolerances in distinct refinement checkpoints."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("refinement tolerance must be finite and positive")
    return f"{REFINEMENT_PREFIX}{float(tolerance):.17g}"


def is_primme_checkpoint_backend(name: str) -> bool:
    if name in {"primme", REFINED_BACKEND}:
        return True
    if not name.startswith(REFINEMENT_PREFIX):
        return False
    try:
        tolerance = float(name.removeprefix(REFINEMENT_PREFIX))
        return name == refinement_backend(tolerance)
    except ValueError:
        return False


def reconstruct_hamiltonian_ritz_pairs(
    hamiltonian: sp.spmatrix | sp.sparray,
    vectors: np.ndarray,
    *,
    target_energy: float,
    requested_subspace_size: int,
    block_size: int = 256,
) -> tuple[PartialSpectrum, dict[str, float | str]]:
    """Solve (V†HV)c = E(V†V)c, retaining all input directions.

    Only k-by-k projected matrices are dense; H stays sparse. Row/column
    blocks bound intermediate n-by-k actions and allow memory-mapped inputs.
    The reported orthogonality is the Frobenius norm of the complete X†X-I,
    which also bounds its spectral norm. Neither residual acceptance nor
    window coverage is inferred from the projected problem.
    """

    h = sp.csr_array(hamiltonian, dtype=np.complex128)
    v = np.asarray(vectors)
    if h.shape[0] != h.shape[1] or v.ndim != 2 or v.shape[0] != h.shape[0]:
        raise ValueError("Hamiltonian and reconstruction subspace shapes do not agree")
    n, k = v.shape
    if k == 0 or k > n or block_size <= 0 or not np.isfinite(target_energy):
        raise ValueError("reconstruction needs a nonempty subspace and positive block size")
    if not np.all(np.isfinite(h.data)):
        raise ValueError("reconstruction Hamiltonian contains nonfinite entries")
    print({"reconstruction": "projecting_H_and_Gram", "dimension": n, "vectors": k}, flush=True)
    projected = np.zeros((k, k), dtype=np.complex128)
    gram = np.zeros_like(projected)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        block = v[start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError("reconstruction subspace contains nonfinite entries")
        adjoint = block.conj().T
        projected += adjoint @ (h[start:stop, :] @ v)
        gram += adjoint @ block
    projected = 0.5 * (projected + projected.conj().T)
    gram = 0.5 * (gram + gram.conj().T)
    print({"reconstruction": "diagonalizing_projected_H", "vectors": k}, flush=True)
    try:
        energies, rotation = la.eigh(projected, gram, driver="gvd")
    except la.LinAlgError as exc:
        raise RuntimeError(
            "saved subspace is rank deficient; no directions were discarded"
        ) from exc
    del projected, gram

    print({"reconstruction": "rotating_and_checking_full_Gram"}, flush=True)
    reconstructed = np.empty((n, k), dtype=np.complex128)
    output_gram = np.zeros((k, k), dtype=np.complex128)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        block = v[start:stop] @ rotation
        reconstructed[start:stop] = block
        output_gram += block.conj().T @ block
    del rotation
    output_gram.flat[:: k + 1] -= 1.0
    orthogonality = float(np.linalg.norm(output_gram, ord="fro"))
    del output_gram

    print({"reconstruction": "checking_all_H_and_folded_residuals"}, flush=True)
    residuals = np.empty(k, dtype=np.float64)
    transformed = np.empty(k, dtype=np.float64)
    for start in range(0, k, block_size):
        stop = min(k, start + block_size)
        block = reconstructed[:, start:stop]
        action = h @ block
        residuals[start:stop] = np.linalg.norm(action - block * energies[None, start:stop], axis=0)
        shifted_action = action - float(target_energy) * block
        folded_action = h @ shifted_action - float(target_energy) * shifted_action
        transformed[start:stop] = np.linalg.norm(
            folded_action - block * (energies[None, start:stop] - target_energy) ** 2, axis=0
        )
    if (
        not np.isfinite(orthogonality)
        or not np.all(np.isfinite(energies))
        or not np.all(np.isfinite(residuals))
        or not np.all(np.isfinite(transformed))
    ):
        raise RuntimeError("reconstruction produced nonfinite validation metrics")
    partial = PartialSpectrum(
        energies=np.asarray(energies, dtype=np.float64),
        eigenvectors=reconstructed,
        residuals=residuals,
        sigma=float(target_energy),
        target_energy=float(target_energy),
        method="folded_spectrum_primme_full_ritz",
        requested_subspace_size=int(requested_subspace_size),
        transformed_residuals=transformed,
        peak_rss_gib=process_peak_rss_gib(),
        full_orthogonality_residual=orthogonality,
    )
    return partial, {
        "reconstruction": RECONSTRUCTION,
        "full_orthogonality_frobenius_residual": orthogonality,
        "maximum_reconstructed_H_residual": partial.maximum_residual,
        "maximum_reconstructed_folded_residual": float(np.max(transformed)),
    }
