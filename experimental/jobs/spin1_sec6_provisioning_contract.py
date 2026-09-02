"""Draft-contract layer for the restartable Spin-1 Sec. VI provisioning workflow.

The core provisioning module owns spectral checkpointing, sector construction,
RDM diagnostics, and orchestration. This module supplies two presentation-level
contracts without duplicating numerical work:

1. use the complete current-convention window grid c L^alpha for
   c in {0.375, 0.5, 0.625}, alpha in {1/2, 1/4, 0}, dropping only sparse
   windows that are not spectrally covered;
2. retain and export the actual 19x19 block-invariant covariance matrices from
   the same calls that produce the concentration summaries.

The prefactors are exactly one half of the historical {0.75, 1, 1.25} grid,
so they select the same physical eigenstate sets under
H_new = H_legacy / 2 at h=D=0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spin1_exchange_convention as convention

import spin1_sec6_provisioning as core

MANUSCRIPT_WINDOW_PREFACTORS = (0.375, 0.5, 0.625)
MANUSCRIPT_WINDOW_EXPONENTS = (0.5, 0.25, 0.0)


def _manuscript_window_specs(
    length: int,
    config: core.Sec6ProvisioningConfig,
    *,
    dense: bool,
) -> list[tuple[str, float, float, float]]:
    """Return the current-convention window grid; coverage is filtered downstream."""

    del dense, config
    rows: list[tuple[str, float, float, float]] = []
    for exponent in MANUSCRIPT_WINDOW_EXPONENTS:
        for prefactor in MANUSCRIPT_WINDOW_PREFACTORS:
            half_width = float(prefactor) * float(length) ** float(exponent)
            role = f"alpha_{exponent:g}_c_{float(prefactor):g}"
            rows.append((role, half_width, float(exponent), float(prefactor)))
    return rows


def _covariance_long_form(
    captures: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for capture in captures:
        names = tuple(str(name) for name in capture["basis_names"])
        matrix = np.asarray(capture["covariance"], dtype=np.float64)
        for row_index, row_name in enumerate(names):
            for column_index, column_name in enumerate(names):
                rows.append(
                    {
                        "L": int(capture["L"]),
                        "M": int(core.TOTAL_SZ),
                        "kappa_over_J": float(capture["kappa_over_J"]),
                        "variant": str(capture["variant"]),
                        "window_half_width": float(capture["window_half_width"]),
                        "row_index": int(row_index),
                        "column_index": int(column_index),
                        "row_operator": row_name,
                        "column_operator": column_name,
                        "covariance": float(matrix[row_index, column_index]),
                        convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                            convention.CURRENT_EXCHANGE_CONVENTION
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_sec6_provisioning(
    config: core.Sec6ProvisioningConfig,
) -> dict[str, pd.DataFrame]:
    """Run the core P0 workflow with the full current window/covariance contract."""

    original_window_specs = core._window_specs
    original_concentration = core._concentration_at_point
    original_covariance = core.projector_deleted_block_covariance

    active_point: dict[str, Any] = {}
    captures: list[dict[str, Any]] = []

    def covariance_with_capture(
        energies,
        eigenvectors,
        exceptional_vectors,
        operators,
        indices,
        *,
        energy_tolerance: float = 1.0e-10,
        vector_tolerance: float = 1.0e-10,
    ):
        result = original_covariance(
            energies,
            eigenvectors,
            exceptional_vectors,
            operators,
            indices,
            energy_tolerance=energy_tolerance,
            vector_tolerance=vector_tolerance,
        )
        if active_point and np.isclose(
            float(energy_tolerance),
            float(config.energy_block_tolerance),
            rtol=0.0,
            atol=1.0e-15,
        ):
            exceptional = np.asarray(exceptional_vectors)
            captures.append(
                {
                    **active_point,
                    "variant": "raw" if exceptional.size == 0 else "clean",
                    "covariance": np.asarray(result["covariance"], dtype=np.float64).copy(),
                }
            )
        return result

    def concentration_with_context(**kwargs):
        context = kwargs["context"]
        active_point.clear()
        active_point.update(
            {
                "L": int(kwargs["length"]),
                "kappa_over_J": float(kwargs["kappa_over_j"]),
                "window_half_width": float(config.concentration_half_width),
                "basis_names": tuple(context["pair"][1]),
            }
        )
        try:
            return original_concentration(**kwargs)
        finally:
            active_point.clear()

    core._window_specs = _manuscript_window_specs
    core._concentration_at_point = concentration_with_context
    core.projector_deleted_block_covariance = covariance_with_capture
    try:
        products = core.run_sec6_provisioning(config)
    finally:
        core.projector_deleted_block_covariance = original_covariance
        core._concentration_at_point = original_concentration
        core._window_specs = original_window_specs

    covariance = _covariance_long_form(captures)
    output = Path(config.output_dir)
    representative = covariance[
        np.isclose(
            covariance.get("kappa_over_J", pd.Series(dtype=float)),
            float(config.representative_kappa_over_j),
        )
    ].copy()
    if not representative.empty:
        representative.to_csv(
            output / "spin1_xy_kappa0p1_concentration_L14_covariance.csv",
            index=False,
        )
    if not covariance.empty:
        covariance.to_csv(
            output / "spin1_xy_large_size_concentration_covariance.csv",
            index=False,
        )

    summary_path = output / "spin1_xy_sec6_provisioning_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update(
        {
            "manuscript_window_prefactors": list(MANUSCRIPT_WINDOW_PREFACTORS),
            "manuscript_window_exponents": list(MANUSCRIPT_WINDOW_EXPONENTS),
            "representative_covariance_matrix_rows": int(len(representative)),
            "all_large_size_covariance_matrix_rows": int(len(covariance)),
            convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                convention.CURRENT_EXCHANGE_CONVENTION
            ),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    products["concentration_L14_covariance"] = representative
    products["large_size_concentration_covariance"] = covariance
    return products
