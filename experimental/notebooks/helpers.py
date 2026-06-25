from __future__ import annotations

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
