from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.sparse as sp
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


# Physical dimensions of the active REVTeX PRX reprint layout.  Figures are
# rendered at their final manuscript size rather than drawn large and scaled
# down by LaTeX, which would also scale the fonts and line widths.
PRX_COLUMN_WIDTH = 246.0 / 72.27  # 3.404 in
PRX_TEXT_WIDTH = 510.0 / 72.27  # 7.057 in

PRX_SINGLE_PANEL_FIGSIZE = (PRX_COLUMN_WIDTH, 2.55)
PRX_WIDE_FIGSIZE = (PRX_TEXT_WIDTH, 3.20)
PRX_TWO_PANEL_FIGSIZE = (PRX_TEXT_WIDTH, 3.15)
PRX_FOUR_PANEL_FIGSIZE = (PRX_TEXT_WIDTH, 5.85)
PRX_PANEL_LABEL_SIZE = 9.0

_FIGURE_AUDIT_ROWS: list[dict] = []


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
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size,
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
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            # Save figures without a rasterized background.
            "savefig.transparent": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )


def use_integer_ticks(ax, *, axis: str = "both") -> None:
    """Restrict visibly discrete axes to integer major ticks."""
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    if axis not in {"x", "y", "both"}:
        raise ValueError("axis must be 'x', 'y', or 'both'")
    if axis in {"x", "both"}:
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True, min_n_ticks=2))
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if axis in {"y", "both"}:
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True, min_n_ticks=2))
        ax.yaxis.set_major_formatter(ScalarFormatter())


def charge_conserving_two_site_hermitian_basis():
    """Return an HS-orthonormal basis of the two-spin charge algebra.

    The fixed-total-``S^z`` blocks have dimensions ``(1, 2, 3, 2, 1)``,
    hence the Hermitian algebra has dimension 19.  The normalized identity is
    returned first.  Remaining candidates are orthonormalized with respect to
    the local Hilbert--Schmidt inner product.
    """

    patterns = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))
    groups: dict[int, list[int]] = {}
    for index, pattern in enumerate(patterns):
        groups.setdefault(sum(pattern), []).append(index)

    candidates: list[tuple[str, np.ndarray]] = [
        ("identity", np.eye(len(patterns), dtype=np.complex128))
    ]
    for index, pattern in enumerate(patterns):
        matrix = np.zeros((len(patterns), len(patterns)), dtype=np.complex128)
        matrix[index, index] = 1.0
        candidates.append((f"diag_{pattern[0]}_{pattern[1]}", matrix))
    for charge, indices in sorted(groups.items()):
        for offset, i in enumerate(indices):
            for j in indices[offset + 1 :]:
                sym = np.zeros((len(patterns), len(patterns)), dtype=np.complex128)
                sym[i, j] = sym[j, i] = 1.0
                asym = np.zeros((len(patterns), len(patterns)), dtype=np.complex128)
                asym[i, j] = -1.0j
                asym[j, i] = 1.0j
                candidates.append((f"q{charge}_sym_{i}_{j}", sym))
                candidates.append((f"q{charge}_asym_{i}_{j}", asym))

    names: list[str] = []
    basis: list[np.ndarray] = []
    for name, candidate in candidates:
        vector = np.asarray(candidate, dtype=np.complex128).copy()
        for prior in basis:
            vector -= np.trace(prior.conj().T @ vector) * prior
        norm = float(np.sqrt(max(np.trace(vector.conj().T @ vector).real, 0.0)))
        if norm <= 1.0e-12:
            continue
        names.append(name)
        basis.append(vector / norm)

    if len(basis) != 19:
        raise RuntimeError(f"expected a 19-dimensional local algebra, got {len(basis)}")
    gram = np.asarray(
        [[np.trace(left.conj().T @ right) for right in basis] for left in basis],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(gram, np.eye(19), atol=1.0e-10)
    return patterns, tuple(names), tuple(basis)


def projector_deleted_block_covariance(
    energies,
    eigenvectors,
    exceptional_vectors,
    operators,
    indices,
    *,
    energy_tolerance: float = 1.0e-10,
    vector_tolerance: float = 1.0e-10,
):
    """Compute a block-invariant covariance over a retained energy window.

    Exact energy degeneracies are treated as projectors.  The exceptional
    subspace is removed independently in every energy block.  For Hermitian
    operators ``O_a`` the returned matrix is

    ``Gamma_ab = D^-1 sum_E Re Tr[B_E(O_a-mu_a)B_E(O_b-mu_b)]``,

    where ``B_E`` denotes compression to the retained part of the energy
    block.  Its largest eigenvalue is therefore the variance of the worst
    Hilbert--Schmidt-normalized local observable in the supplied operator
    span, without choosing an arbitrary eigenbasis inside degenerate blocks.
    """

    spectrum = np.asarray(energies, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    selected = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
    if selected.size == 0:
        raise ValueError("indices must not be empty")
    ops = tuple(
        operator if sp.issparse(operator) else np.asarray(operator, dtype=np.complex128)
        for operator in operators
    )
    if not ops:
        raise ValueError("operators must not be empty")
    if any(operator.shape != (vectors.shape[0], vectors.shape[0]) for operator in ops):
        raise ValueError("every operator must act on the eigenvector Hilbert space")

    exceptional = np.asarray(exceptional_vectors, dtype=np.complex128)
    if exceptional.ndim == 1:
        exceptional = exceptional[:, None]
    if exceptional.size == 0:
        exceptional = np.zeros((vectors.shape[0], 0), dtype=np.complex128)

    groups: list[list[int]] = []
    current = [int(selected[0])]
    for raw_index in selected[1:]:
        index = int(raw_index)
        if abs(spectrum[index] - spectrum[current[-1]]) <= energy_tolerance:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)

    retained_blocks: list[np.ndarray] = []
    removed_rank = 0
    for group in groups:
        block = vectors[:, group]
        block_exceptional = orthonormalize_columns(
            block @ (block.conj().T @ exceptional), tolerance=vector_tolerance
        )
        split = projector_deleted_basis(block, block_exceptional, tolerance=vector_tolerance)
        removed_rank += int(split["exceptional_rank"])
        retained = split["retained_basis"]
        if retained.shape[1]:
            retained_blocks.append(retained)

    retained_rank = int(sum(block.shape[1] for block in retained_blocks))
    if retained_rank <= 0:
        raise ValueError("projector deletion removed the complete energy window")

    compressed: list[list[np.ndarray]] = []
    traces = np.zeros(len(ops), dtype=np.float64)
    for retained in retained_blocks:
        block_ops = []
        for index, operator in enumerate(ops):
            matrix = retained.conj().T @ (operator @ retained)
            matrix = 0.5 * (matrix + matrix.conj().T)
            block_ops.append(matrix)
            traces[index] += float(np.trace(matrix).real)
        compressed.append(block_ops)
    means = traces / retained_rank

    covariance = np.zeros((len(ops), len(ops)), dtype=np.float64)
    for block_ops in compressed:
        identity = np.eye(block_ops[0].shape[0], dtype=np.complex128)
        centered = [matrix - means[index] * identity for index, matrix in enumerate(block_ops)]
        for a, left in enumerate(centered):
            for b in range(a, len(centered)):
                value = float(np.trace(left @ centered[b]).real) / retained_rank
                covariance[a, b] += value
                if b != a:
                    covariance[b, a] += value
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors_cov = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest_index = int(np.argmax(eigenvalues))
    nonidentity = eigenvalues[1:] if eigenvalues.size > 1 else eigenvalues
    return {
        "covariance": covariance,
        "means": means,
        "eigenvalues": eigenvalues,
        "worst_coefficients": eigenvectors_cov[:, largest_index],
        "largest_eigenvalue": float(eigenvalues[largest_index]),
        "largest_width": float(np.sqrt(eigenvalues[largest_index])),
        "median_nonidentity_width": float(np.median(np.sqrt(nonidentity))),
        "window_rank": int(selected.size),
        "exceptional_rank": int(removed_rank),
        "retained_rank": retained_rank,
        "energy_block_count": int(len(groups)),
        "removed_fraction": float(removed_rank / selected.size),
    }


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


def _parse_svg_inches(path: Path) -> tuple[float | None, float | None]:
    try:
        from xml.etree import ElementTree

        root = ElementTree.parse(path).getroot()
    except Exception:
        return None, None

    def to_inches(raw: str | None) -> float | None:
        if raw is None:
            return None
        text = raw.strip().lower()
        factors = {
            "in": 1.0,
            "pt": 1.0 / 72.0,
            "px": 1.0 / 96.0,
            "cm": 1.0 / 2.54,
            "mm": 1.0 / 25.4,
        }
        for unit, factor in factors.items():
            if text.endswith(unit):
                try:
                    return float(text[: -len(unit)]) * factor
                except ValueError:
                    return None
        try:
            return float(text) / 96.0
        except ValueError:
            return None

    return to_inches(root.attrib.get("width")), to_inches(root.attrib.get("height"))


def _parse_pdf_inches(path: Path) -> tuple[float | None, float | None]:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(path))
            box = reader.pages[0].mediabox
            return float(box.width) / 72.0, float(box.height) / 72.0
        except Exception:
            continue
    return None, None


def add_panel_label(ax, label: str, *, x: float = 0.01, y: float = 0.99) -> None:
    """Place a REVTeX-sized panel label inside a reserved plot corner."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PRX_PANEL_LABEL_SIZE,
        fontweight="bold",
        zorder=20,
    )


def save_prx_figure(
    fig,
    stem: str,
    *,
    directory: str | Path,
    formats: Sequence[str] = ("pdf", "svg"),
    dpi: int = 300,
    transparent: bool = False,
    close: bool = False,
    dimension_tolerance_in: float = 0.015,
):
    """Save a figure without changing its declared physical canvas size.

    The usual ``bbox_inches='tight'`` shortcut is deliberately avoided because
    it changes the PDF/SVG dimensions and makes final font sizes depend on the
    surrounding labels.  A small audit record is retained for each output.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    requested_width, requested_height = (float(value) for value in fig.get_size_inches())

    saved_paths = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        output_path = directory / f"{stem}.{fmt}"
        save_kwargs = {
            "transparent": transparent,
            "facecolor": "white",
            "bbox_inches": None,
            "pad_inches": 0.0,
        }
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff", "webp"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)

        if fmt == "svg":
            actual_width, actual_height = _parse_svg_inches(output_path)
        elif fmt == "pdf":
            actual_width, actual_height = _parse_pdf_inches(output_path)
        else:
            actual_width, actual_height = requested_width, requested_height

        dimension_ok = (
            actual_width is None
            or actual_height is None
            or (
                abs(actual_width - requested_width) <= dimension_tolerance_in
                and abs(actual_height - requested_height) <= dimension_tolerance_in
            )
        )
        _FIGURE_AUDIT_ROWS.append(
            {
                "stem": stem,
                "format": fmt,
                "path": str(output_path),
                "requested_width_in": requested_width,
                "requested_height_in": requested_height,
                "actual_width_in": actual_width,
                "actual_height_in": actual_height,
                "dimension_ok": bool(dimension_ok),
                "usetex": bool(mpl.rcParams.get("text.usetex", False)),
                "font_family": repr(mpl.rcParams.get("font.family")),
                "font_size_pt": float(mpl.rcParams.get("font.size", 0.0)),
                "legend_font_size_pt": float(mpl.rcParams.get("legend.fontsize", 0.0)),
            }
        )
        if not dimension_ok:
            raise RuntimeError(
                f"saved figure {output_path} has size {actual_width}x{actual_height} in, "
                f"expected {requested_width}x{requested_height} in"
            )

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return saved_paths


def write_figure_manifest(path: str | Path) -> pd.DataFrame:
    """Write the accumulated figure-size/font audit to JSON and CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(_FIGURE_AUDIT_ROWS)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"figures": _FIGURE_AUDIT_ROWS}, indent=2), encoding="utf-8")
        rows.to_csv(path.with_suffix(".csv"), index=False)
    else:
        rows.to_csv(path, index=False)
        path.with_suffix(".json").write_text(
            json.dumps({"figures": _FIGURE_AUDIT_ROWS}, indent=2), encoding="utf-8"
        )
    return rows


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


def projector_deleted_basis(
    window_vectors,
    exceptional_vectors,
    *,
    tolerance: float = 1.0e-10,
):
    """Return orthonormal window, exceptional, and retained projector bases.

    The exceptional vectors are first projected into the complete window
    projector.  The retained basis is the orthogonal complement inside that
    window, so the result is invariant under rotations of a degenerate
    eigensolver basis.
    """

    window = orthonormalize_columns(window_vectors, tolerance=tolerance)
    exceptional = np.asarray(exceptional_vectors, dtype=np.complex128)
    if exceptional.ndim == 1:
        exceptional = exceptional[:, None]
    if exceptional.size == 0:
        exceptional = np.zeros((window.shape[0], 0), dtype=np.complex128)
    exceptional = orthonormalize_columns(
        window @ (window.conj().T @ exceptional), tolerance=tolerance
    )

    if exceptional.shape[1] == 0:
        retained = window
    else:
        coefficients = window.conj().T @ exceptional
        _u, singular_values, vh = np.linalg.svd(coefficients.conj().T, full_matrices=True)
        rank = int(np.sum(singular_values > tolerance))
        null_coefficients = vh.conj().T[:, rank:]
        retained = orthonormalize_columns(window @ null_coefficients, tolerance=tolerance)

    return {
        "window_basis": window,
        "exceptional_basis": exceptional,
        "retained_basis": retained,
        "window_rank": int(window.shape[1]),
        "exceptional_rank": int(exceptional.shape[1]),
        "retained_rank": int(retained.shape[1]),
        "removed_fraction": (
            float(exceptional.shape[1] / window.shape[1]) if window.shape[1] else 0.0
        ),
    }


def projector_resolved_energy_basis(
    energies,
    eigenvectors,
    exceptional_vectors,
    *,
    energy_tolerance: float = 1.0e-10,
    vector_tolerance: float = 1.0e-10,
):
    """Resolve every degenerate energy block into exceptional and retained parts.

    This produces a basis-independent representation suitable for ETH scatter
    plots: the exceptional projector is aligned inside each exact-degeneracy
    block before the orthogonal retained complement is constructed.
    """

    spectrum = np.asarray(energies, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    exceptional = np.asarray(exceptional_vectors, dtype=np.complex128)
    if exceptional.ndim == 1:
        exceptional = exceptional[:, None]
    if exceptional.size == 0:
        exceptional = np.zeros((vectors.shape[0], 0), dtype=np.complex128)

    groups: list[list[int]] = []
    if spectrum.size:
        current = [0]
        for index in range(1, spectrum.size):
            if abs(spectrum[index] - spectrum[current[-1]]) <= energy_tolerance:
                current.append(index)
            else:
                groups.append(current)
                current = [index]
        groups.append(current)

    basis_columns = []
    resolved_energies = []
    exceptional_flags = []
    block_ids = []
    for block_id, group in enumerate(groups):
        block = vectors[:, group]
        exc = orthonormalize_columns(
            block @ (block.conj().T @ exceptional), tolerance=vector_tolerance
        )
        split = projector_deleted_basis(block, exc, tolerance=vector_tolerance)
        retained = split["retained_basis"]
        for column in range(exc.shape[1]):
            basis_columns.append(exc[:, column])
            resolved_energies.append(float(np.mean(spectrum[group])))
            exceptional_flags.append(True)
            block_ids.append(block_id)
        for column in range(retained.shape[1]):
            basis_columns.append(retained[:, column])
            resolved_energies.append(float(np.mean(spectrum[group])))
            exceptional_flags.append(False)
            block_ids.append(block_id)

    resolved_basis = (
        np.column_stack(basis_columns)
        if basis_columns
        else np.zeros((vectors.shape[0], 0), dtype=np.complex128)
    )
    return {
        "basis": resolved_basis,
        "energies": np.asarray(resolved_energies, dtype=np.float64),
        "is_exceptional": np.asarray(exceptional_flags, dtype=bool),
        "energy_block_id": np.asarray(block_ids, dtype=np.int64),
    }


def projector_deleted_concentration(
    window_vectors,
    exceptional_vectors,
    operator,
    *,
    tolerance: float = 1.0e-10,
):
    """Basis-independent local-observable spread after projector excision.

    The eigenvalues of the observable projected to the retained subspace are
    the possible expectation values under arbitrary rotations of that
    subspace.  Their spread is therefore a conservative, basis-independent
    concentration diagnostic.
    """

    split = projector_deleted_basis(window_vectors, exceptional_vectors, tolerance=tolerance)
    retained = split["retained_basis"]
    if retained.shape[1] == 0:
        raise ValueError("projector deletion left no retained states")
    block = retained.conj().T @ (operator @ retained)
    block = 0.5 * (block + block.conj().T)
    values = np.linalg.eigvalsh(block).real
    mean = float(np.mean(values))
    deviations = np.abs(values - mean)
    return {
        **{
            key: split[key]
            for key in ("window_rank", "exceptional_rank", "retained_rank", "removed_fraction")
        },
        "mean": mean,
        "energy_block_count": np.nan,
        "degenerate_state_fraction": np.nan,
        "basis_independent_std": float(np.std(values)),
        "median_abs_deviation": float(np.median(deviations)),
        "p90_abs_deviation": float(np.quantile(deviations, 0.90)),
        "max_abs_deviation": float(np.max(deviations)),
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
