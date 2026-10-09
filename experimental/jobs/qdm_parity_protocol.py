"""Explicit evidence identity and bounded-memory spectral validation."""

from __future__ import annotations

import numpy as np

PROTOCOL = "Tdiag_i_Ty2_plus_measured_target_Sy_v1"


def require_parity_protocol(payload, *, description):
    """Reject legacy or opposite-parity evidence before combining stages."""
    if (
        payload.get("symmetry_protocol") != PROTOCOL
        or payload.get("parity_policy") != "measured_target"
    ):
        raise ValueError(f"{description} is not validated target-parity evidence ({PROTOCOL})")


def validate_partial_eigensystem(h, energies, vectors, *, tolerance=1.0e-6, chunk_size=64):
    """Recompute every residual and every Gram block without copying all vectors."""
    e = np.asarray(energies)
    if e.ndim != 1 or vectors.shape != (h.shape[0], e.size) or not e.size:
        raise ValueError("invalid partial-eigensystem coordinates")
    residual, gram_squared = 0.0, 0.0
    for start in range(0, e.size, chunk_size):
        v = np.asarray(vectors[:, start : start + chunk_size])
        if not np.isfinite(v).all() or not np.isfinite(e[start : start + chunk_size]).all():
            raise ValueError("nonfinite eigenpairs")
        residual = max(
            residual,
            float(np.max(np.linalg.norm(h @ v - v * e[start : start + chunk_size], axis=0))),
        )
        for other in range(0, e.size, chunk_size):
            w = np.asarray(vectors[:, other : other + chunk_size])
            delta = v.conj().T @ w
            if start == other:
                delta -= np.eye(v.shape[1])
            gram_squared += float(np.linalg.norm(delta) ** 2)
    gram = float(np.sqrt(gram_squared))
    if residual > tolerance or gram > tolerance:
        raise ValueError(f"partial eigenpair validation failed: residual={residual}, Gram={gram}")
    return {"maximum_eigenpair_residual": residual, "gram_frobenius_residual": gram}


def window_controls_stable(frame, *, widths=(0.10, 0.20, 0.25, 0.50), tolerance=5e-5):
    """Require two latest budgets for every declared 12x4 target-parity window."""
    required = {
        "requested_subspace_size",
        "window_half_width",
        "Sy",
        "symmetry_protocol",
        "raw_window_state_count",
        "maximum_eigenpair_residual",
        "gram_frobenius_residual",
        "cage_window_projector_weight",
        "tau_A_mc_raw",
        "tau_Z_mc_raw",
        "w_raw",
    }
    if not required.issubset(frame.columns) or frame.empty:
        return False
    budgets = sorted(frame.requested_subspace_size.unique())[-2:]
    if len(budgets) != 2 or not frame.symmetry_protocol.eq(PROTOCOL).all():
        return False
    for width in widths:
        selected = frame[np.isclose(frame.window_half_width, width)]
        selected = selected[selected.requested_subspace_size.isin(budgets)].sort_values(
            "requested_subspace_size"
        )
        if len(selected) != 2:
            return False
        valid = (
            selected.Sy.eq(-1).all()
            and selected.raw_window_state_count.nunique() == 1
            and selected.maximum_eigenpair_residual.le(1e-6).all()
            and selected.gram_frobenius_residual.le(1e-6).all()
            and selected.cage_window_projector_weight.ge(1 - 1e-6).all()
        )
        change = max(
            abs(selected.iloc[-1][key] - selected.iloc[0][key])
            for key in ("tau_A_mc_raw", "tau_Z_mc_raw", "w_raw")
        )
        if not valid or not np.isfinite(change) or change > tolerance:
            return False
    return True
