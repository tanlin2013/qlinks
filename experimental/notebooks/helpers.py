from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from IPython.display import display

from qlinks.caging import (
    CageClassificationConfig,
    classify_cage_state,
)

ZERO_MECHANISM_FIELDS = {
    "all": None,
    "q_empty": "q_empty_zero_indices",
    "closed_by_known_zeros": "closed_by_known_zero_indices",
    "projector_like": "projector_like_zero_indices",
    "unexplained_leakage": "unexplained_leakage_zero_indices",
    "regional": "regional_mechanism_zero_indices",
    "extended": "extended_mechanism_zero_indices",
    "failure": "failure_mechanism_zero_indices",
}


# Generous working canvases for manuscript figures.  The evidence notebooks
# export vector PDF/SVG and the draft can scale them at inclusion time.  These
# sizes match the visual density of the current full-width Fig. 3 better than
# the older 3.35-inch standalone canvases.
PRX_SINGLE_PANEL_FIGSIZE = (6.4, 4.0)
PRX_WIDE_FIGSIZE = (7.2, 4.2)
PRX_TWO_PANEL_FIGSIZE = (7.2, 3.8)
PRX_FOUR_PANEL_FIGSIZE = (7.2, 6.4)


def set_revtex_matplotlib_style(
    *,
    base_font_size: float = 8.0,
    prefer_tex: bool = True,
) -> None:
    """Configure Matplotlib to resemble the current REVTeX manuscript.

    If a working ``latex`` executable is unavailable, the function falls back
    to Matplotlib's built-in mathtext while keeping the Computer-Modern-like
    serif appearance. This makes the notebooks portable on lightweight
    environments such as CI and remote containers.
    """

    use_tex = bool(prefer_tex and shutil.which("latex"))
    if prefer_tex and not use_tex:
        warnings.warn(
            "LaTeX executable not found; falling back to Matplotlib mathtext.",
            RuntimeWarning,
            stacklevel=2,
        )

    mpl.rcParams.update(
        {
            # Render all text and mathematics with LaTeX when available.
            "text.usetex": use_tex,
            # Default REVTeX/LaTeX serif family.
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # Packages and commands used by figure labels.
            # Do not put \documentclass here.
            "text.latex.preamble": (
                r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{bm}
        """
                if use_tex
                else ""
            ),
            # Typography at the final printed figure size.
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "legend.fontsize": base_font_size - 1,
            "figure.titlesize": base_font_size,
            "legend.frameon": False,
            "figure.dpi": 120,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            # Keep mathematical minus signs and ordinary text consistent.
            "axes.unicode_minus": False,
            # Avoid overly thick default figure elements.
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            # Save figures without a rasterized background.
            "savefig.transparent": False,
            "savefig.bbox": "tight",
        }
    )


def classify_cage_search_result(
    search_result,
    *,
    kinetic_matrix,
    basis_configs,
    sector_mask=None,
    config=None,
) -> tuple[pd.DataFrame, list]:
    """Classify all CageRecords in a CageSearchResult.

    Returns
    -------
    df:
        One row per cage record.
    reports:
        The full CageClassificationReport objects, in the same order as df.
    """
    if config is None:
        config = CageClassificationConfig()

    rows = []
    reports = []

    for signature in search_result.signatures:
        records = search_result[signature]

        for record_index, record in enumerate(records):
            report = classify_cage_state(
                record.cage_state,
                kinetic_matrix=kinetic_matrix,
                basis_configs=basis_configs,
                hilbert_size=search_result.hilbert_size,
                sector_mask=sector_mask,
                config=config,
            )

            reports.append(report)

            rows.append(
                {
                    "signature": signature,
                    "kappa": int(signature[0]),
                    "Z": int(signature[1]),
                    "record_index": int(record_index),
                    "global_record_index": len(reports) - 1,
                    "label": report.label,
                    "energy": complex(record.cage_state.energy),
                    "support_size": int(report.support_size),
                    "support_fraction": float(report.support_fraction),
                    "n_nontrivial_zeros": int(report.n_nontrivial_zeros),
                    "n_distinct_local_patterns": int(report.n_distinct_local_patterns),
                    "n_q_empty_source_probes": int(report.n_q_empty_source_probes),
                    "n_closed_by_known_zero_network_source_probes": int(
                        report.n_closed_by_known_zero_network_source_probes
                    ),
                    "n_projector_like_source_probes": int(report.n_projector_like_source_probes),
                    "n_invalid_source_probes": int(report.n_invalid_source_probes),
                    "n_regional_source_probes": int(report.n_regional_source_probes),
                    "n_unexpected_target_probe_failures": int(
                        report.n_unexpected_target_probe_failures
                    ),
                    "n_nonzero_complement_action_probe_failures": int(
                        report.n_nonzero_complement_action_probe_failures
                    ),
                    "n_source_projector_like_probes": int(report.n_source_projector_like_probes),
                    "n_indirect_projector_like_probes": int(
                        report.n_indirect_projector_like_probes
                    ),
                    "mean_q_sector_weight": float(report.mean_q_sector_weight),
                    "max_q_sector_weight": float(report.max_q_sector_weight),
                    "mean_complement_action_norm": float(report.mean_complement_action_norm),
                    "max_complement_action_norm": float(report.max_complement_action_norm),
                    "boundary_residual": float(record.cage_state.boundary_residual),
                    "eigen_residual": float(record.cage_state.eigen_residual),
                    "full_residual": float(record.cage_state.full_residual),
                }
            )

    return pd.DataFrame(rows), reports


def basis_dataframe(
    basis_configs,
    *,
    indices=None,
    amplitudes=None,
    amplitude_digits: int = 6,
    column_prefix: str = "site",
):
    """Render selected basis configurations as a DataFrame.

    Spin-1 local values are shown directly in the site columns. For the
    SpinOneXYChainModel these values should be m_i in {-1, 0, +1}.
    """
    basis_configs = np.asarray(basis_configs)

    if indices is None:
        indices = np.arange(basis_configs.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)

    data = basis_configs[indices]
    columns = [f"{column_prefix}_{site}" for site in range(data.shape[1])]

    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "basis_index", indices)

    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if amplitudes.shape[0] != indices.shape[0]:
            raise ValueError("amplitudes and indices must have the same length.")

        df.insert(1, "amplitude", amplitudes)
        df.insert(2, "abs_amplitude", np.abs(amplitudes))
        df.insert(3, "phase_over_pi", np.angle(amplitudes) / np.pi)
        df["abs_amplitude"] = df["abs_amplitude"].round(amplitude_digits)
        df["phase_over_pi"] = df["phase_over_pi"].round(amplitude_digits)

    return df


def display_basis_dataframe(
    basis_configs,
    *,
    indices=None,
    amplitudes=None,
    max_rows: int = 64,
    title: str | None = None,
):
    df = basis_dataframe(
        basis_configs,
        indices=indices,
        amplitudes=amplitudes,
    )

    if title is not None:
        display(pd.DataFrame({"section": [title], "n_rows": [len(df)]}))

    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        display(df.head(max_rows))

    return df


def zero_indices_from_report(classification_report, *, mechanism: str = "all"):
    """Return nontrivial-zero basis indices, optionally split by mechanism."""
    if mechanism not in ZERO_MECHANISM_FIELDS:
        allowed = ", ".join(ZERO_MECHANISM_FIELDS)
        raise ValueError(f"Unknown mechanism {mechanism!r}. Expected one of: {allowed}.")

    field = ZERO_MECHANISM_FIELDS[mechanism]
    if field is None:
        return np.array(
            [int(zero.zero_index) for zero in classification_report.zero_reports],
            dtype=np.int64,
        )

    return np.asarray(getattr(classification_report, field), dtype=np.int64)


def zero_mechanism_map(classification_report):
    """Map zero basis index to zero-mechanism label."""
    return {
        int(zero.zero_index): str(zero.probe_mechanism_label)
        for zero in classification_report.zero_reports
    }


def interference_zero_dataframe(classification_report, basis_configs, *, mechanism: str = "all"):
    """Return nontrivial interference-zero states as a DataFrame."""
    indices = zero_indices_from_report(classification_report, mechanism=mechanism)
    mechanism_by_index = zero_mechanism_map(classification_report)

    df = basis_dataframe(basis_configs, indices=indices)
    df.insert(1, "mechanism", [mechanism_by_index.get(int(index), "unknown") for index in indices])
    return df


def save_prx_figure(
    fig,
    stem: str,
    *,
    directory: str | Path,
    formats: Sequence[str] = ("pdf", "svg"),
    dpi: int = 300,
    pad_inches: float = 0.02,
    transparent: bool = False,
    close: bool = False,
):
    """Save a Matplotlib figure in manuscript-friendly formats.

    Parameters
    ----------
    fig:
        Matplotlib figure object.
    stem:
        File stem without the extension.
    directory:
        Output directory. It is created automatically.
    formats:
        Iterable of extensions such as ("pdf", "svg") or ("pdf", "svg", "png").
    dpi:
        Raster DPI used when a raster format is requested.
    pad_inches:
        Passed to :meth:`matplotlib.figure.Figure.savefig`.
    transparent:
        Whether to save with a transparent background.
    close:
        Whether to close the figure after saving.

    Returns
    -------
    list[Path]
        The saved file paths.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        output_path = directory / f"{stem}.{fmt}"
        save_kwargs = {
            "pad_inches": pad_inches,
            "transparent": transparent,
            "facecolor": "white",
        }
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff", "webp"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return saved_paths


def orthonormalize_columns(vectors, *, tolerance: float = 1.0e-10):
    """Return an orthonormal basis for the supplied column span."""
    array = np.asarray(vectors, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("vectors must be one- or two-dimensional")
    if array.shape[1] == 0:
        return np.zeros((array.shape[0], 0), dtype=np.complex128)
    q, r = np.linalg.qr(array)
    keep = np.abs(np.diag(r)) > tolerance
    return np.asarray(q[:, keep], dtype=np.complex128)


def projector_deleted_observable_moments(
    window_vectors,
    exceptional_vectors,
    operator,
    *,
    squared_operator=None,
    tolerance: float = 1.0e-10,
):
    """Evaluate a basis-independent microcanonical trace after projector deletion.

    ``window_vectors`` spans the complete energy-window projector.  The
    exceptional vectors are projected into that window and removed as a whole
    subspace, so the result is invariant under rotations inside degenerate
    energy multiplets.
    """
    window = orthonormalize_columns(window_vectors, tolerance=tolerance)
    exceptional = np.asarray(exceptional_vectors, dtype=np.complex128)
    if exceptional.ndim == 1:
        exceptional = exceptional[:, None]
    if exceptional.size == 0:
        exceptional = np.zeros((window.shape[0], 0), dtype=np.complex128)
    projected = window @ (window.conj().T @ exceptional)
    exceptional = orthonormalize_columns(projected, tolerance=tolerance)

    n_window = int(window.shape[1])
    n_exceptional = int(exceptional.shape[1])
    n_retained = n_window - n_exceptional
    if n_retained <= 0:
        raise ValueError("projector deletion removed the complete energy window")

    def projected_trace(matrix, basis):
        if basis.shape[1] == 0:
            return 0.0
        action = matrix @ basis
        return float(np.trace(basis.conj().T @ action).real)

    trace_window = projected_trace(operator, window)
    trace_exceptional = projected_trace(operator, exceptional)
    mean = (trace_window - trace_exceptional) / n_retained

    if squared_operator is None:
        squared_operator = operator.conj().T @ operator
    trace2_window = projected_trace(squared_operator, window)
    trace2_exceptional = projected_trace(squared_operator, exceptional)
    second_moment = (trace2_window - trace2_exceptional) / n_retained

    return {
        "window_rank": n_window,
        "exceptional_rank": n_exceptional,
        "retained_rank": n_retained,
        "removed_fraction": n_exceptional / n_window,
        "mean": float(mean),
        "second_moment": float(second_moment),
        "variance": float(max(0.0, second_moment - mean * mean)),
        "exceptional_projection_residual": (
            float(np.linalg.norm(exceptional - window @ (window.conj().T @ exceptional)))
            if n_exceptional
            else 0.0
        ),
    }


def canonical_beta_match(
    energies,
    target_energy: float,
    *,
    tolerance: float = 1.0e-12,
    maximum_abs_beta: float = 1.0e4,
):
    """Match the canonical mean energy to a finite-sector target energy."""
    from scipy.optimize import brentq

    spectrum = np.asarray(energies, dtype=np.float64).reshape(-1)
    if spectrum.size == 0:
        raise ValueError("energies must not be empty")
    target = float(target_energy)
    if target < spectrum.min() - tolerance or target > spectrum.max() + tolerance:
        raise ValueError("target energy lies outside the spectrum")

    def weights(beta):
        logits = -float(beta) * spectrum
        logits -= np.max(logits)
        raw = np.exp(logits)
        return raw / np.sum(raw)

    def energy_difference(beta):
        return float(np.dot(weights(beta), spectrum) - target)

    at_zero = energy_difference(0.0)
    if abs(at_zero) <= tolerance:
        beta = 0.0
    else:
        direction = 1.0 if at_zero > 0.0 else -1.0
        bound = direction
        while abs(bound) <= maximum_abs_beta and energy_difference(bound) * at_zero > 0.0:
            bound *= 2.0
        if abs(bound) > maximum_abs_beta:
            raise RuntimeError("failed to bracket the energy-matching inverse temperature")
        lo, hi = sorted((0.0, bound))
        beta = float(brentq(energy_difference, lo, hi, xtol=tolerance, rtol=1.0e-12))

    matched_weights = weights(beta)
    return {
        "beta": beta,
        "target_energy": target,
        "matched_energy": float(np.dot(matched_weights, spectrum)),
        "energy_residual": float(np.dot(matched_weights, spectrum) - target),
        "effective_state_count": float(1.0 / np.sum(matched_weights**2)),
        "weights": matched_weights,
    }


def degeneracy_resolved_concentration(
    energies,
    eigenvectors,
    operator,
    indices,
    *,
    energy_tolerance: float = 1.0e-10,
):
    """Basis-independent concentration diagnostic inside an energy window.

    Exact degeneracy blocks are treated as projectors.  Within each block we
    diagonalize the projected observable, so the reported spread does not
    depend on the arbitrary basis returned by the Hamiltonian eigensolver.
    """
    spectrum = np.asarray(energies, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    selected = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
    if selected.size == 0:
        raise ValueError("indices must not be empty")

    groups = []
    current = [int(selected[0])]
    for index in selected[1:]:
        if abs(spectrum[int(index)] - spectrum[current[-1]]) <= energy_tolerance:
            current.append(int(index))
        else:
            groups.append(current)
            current = [int(index)]
    groups.append(current)

    possible_values = []
    block_means = []
    block_sizes = []
    degenerate_states = 0
    for group in groups:
        basis = vectors[:, group]
        block = basis.conj().T @ (operator @ basis)
        block = 0.5 * (block + block.conj().T)
        values = np.linalg.eigvalsh(block).real
        possible_values.extend(values.tolist())
        block_means.append(float(np.mean(values)))
        block_sizes.append(len(group))
        if len(group) > 1:
            degenerate_states += len(group)

    possible = np.asarray(possible_values, dtype=np.float64)
    weights = np.asarray(block_sizes, dtype=np.float64)
    means = np.asarray(block_means, dtype=np.float64)
    global_mean = float(np.mean(possible))
    deviations = np.abs(possible - global_mean)
    block_mean_variance = float(np.average((means - global_mean) ** 2, weights=weights))
    return {
        "window_state_count": int(selected.size),
        "energy_block_count": int(len(groups)),
        "degenerate_state_fraction": float(degenerate_states / selected.size),
        "mean": global_mean,
        "basis_independent_std": float(np.std(possible)),
        "block_mean_std": float(np.sqrt(block_mean_variance)),
        "median_abs_deviation": float(np.median(deviations)),
        "p90_abs_deviation": float(np.quantile(deviations, 0.90)),
        "max_abs_deviation": float(np.max(deviations)),
    }
