"""Complete-eigensystem and degeneracy-aware QDM population validation."""

from __future__ import annotations

import numpy as np
import scipy.optimize as optimize

TOL = 1.0e-8


def validate_eigensystem(h, energies, vectors, *, tolerance=TOL):
    """Validate every eigenpair and the complete Gram matrix, not a sample."""
    dimension = h.shape[0]
    if energies.shape != (dimension,) or vectors.shape != (dimension, dimension):
        raise ValueError("a complete eigensystem with matching coordinates is required")
    if not np.isfinite(energies).all() or not np.isfinite(vectors).all():
        raise ValueError("nonfinite eigensystem")
    residual = float(np.max(np.linalg.norm(h @ vectors - vectors * energies, axis=0), initial=0))
    gram = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(dimension)))
    if residual > tolerance or gram > tolerance:
        raise ValueError(f"eigensystem validation failed: residual={residual}, Gram={gram}")
    return {"maximum_eigenpair_residual": residual, "gram_frobenius_residual": gram}


def compare_populations(source, candidate, *, tolerance=TOL):
    """Compare row multisets and rotation-invariant traces of degenerate energy blocks.

    Sorting individual witness values across a degenerate energy block is not a
    valid equivalence test. Its count and trace are invariant under basis rotations.
    Matching never changes the candidate population or its exclusion policy.
    """
    columns = ["energy", "Q_A", "Q_Z"]
    if source is None:
        return {
            "status": "source_missing",
            "population_matches": None,
            "invariant_population_matches": None,
            "interpretation": "Old scatter bytes are required; no cause is inferred.",
        }
    a = source[columns].to_numpy(float)
    b = candidate[columns].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite source or reconstructed scatter")
    result = {"status": "checked", "source_rows": len(a), "candidate_rows": len(b)}
    if len(a) != len(b):
        return {
            **result,
            "population_matches": False,
            "invariant_population_matches": False,
            "interpretation": "population_count_mismatch",
        }
    a = a[np.argsort(a[:, 0], kind="stable")]
    b = b[np.argsort(b[:, 0], kind="stable")]
    energy_error = float(np.max(np.abs(a[:, 0] - b[:, 0]), initial=0))
    result["sorted_energy_max_abs_difference"] = energy_error
    if energy_error > tolerance:
        return {
            **result,
            "population_matches": False,
            "invariant_population_matches": False,
            "interpretation": "spectrum_mismatch",
        }
    boundaries = np.r_[0, np.flatnonzero(np.diff(a[:, 0]) > 1.0e-9) + 1, len(a)]
    largest_row_error = 0.0
    largest_trace_error = 0.0
    degenerate_blocks = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        aa, bb = a[start:end], b[start:end]
        if end - start > 1:
            degenerate_blocks += 1
        cost = np.max(np.abs(aa[:, None, :] - bb[None, :, :]), axis=2)
        rows, cols = optimize.linear_sum_assignment(cost)
        largest_row_error = max(largest_row_error, float(np.max(cost[rows, cols], initial=0)))
        largest_trace_error = max(
            largest_trace_error,
            float(np.max(np.abs(aa[:, 1:].sum(0) - bb[:, 1:].sum(0)), initial=0)),
        )
    matches = largest_row_error <= tolerance
    trace_matches = largest_trace_error <= tolerance
    return {
        **result,
        "population_matches": matches,
        "invariant_population_matches": trace_matches,
        "invariant_scope": "energy multiplicities and Q_A/Q_Z block traces; not vector equivalence",
        "max_row_abs_difference": largest_row_error,
        "max_energy_block_witness_trace_difference": largest_trace_error,
        "energy_block_traces_match": trace_matches,
        "degenerate_energy_blocks": degenerate_blocks,
        "interpretation": "row_multiset_matches"
        if matches
        else (
            "compatible_with_degenerate_basis_rotation"
            if trace_matches
            else "witness_or_exclusion_mismatch"
        ),
    }
