#!/usr/bin/env python
"""Current-convention three-site Sec. VI concentration from cached spectra only.

Historical three-site checkpoint rows are exactly rescaled in memory: normalized
covariance data are invariant, while the protocol label, energy half-width, and
energy-dimension residual diagnostics change by the uniform factor 1/2. No
eigensolver is introduced by this adapter.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import spin1_exchange_convention as _convention
import spin1_sec6_p1_three_site_concentration_legacy as _legacy

_ORIGINAL_RUN = _legacy.run

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
WINDOW_EXPONENT = _convention.PRIMARY_WINDOW_EXPONENT
WINDOW_PREFACTOR = _convention.PRIMARY_WINDOW_PREFACTOR

_legacy.WINDOW_PROTOCOL = WINDOW_PROTOCOL
_legacy.WINDOW_EXPONENT = WINDOW_EXPONENT


def _expected_half_width(length: int) -> float:
    return WINDOW_PREFACTOR * float(length) ** WINDOW_EXPONENT


def _map_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(row)
    mapped["schema_version"] = 2
    mapped["window_protocol"] = WINDOW_PROTOCOL
    mapped["window_half_width"] = (
        _convention.LEGACY_TO_CURRENT_ENERGY_SCALE * float(row["window_half_width"])
    )
    for key in ("window_max_eigenpair_residual", "window_median_eigenpair_residual", "tower_residual"):
        value = mapped.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            mapped[key] = _convention.LEGACY_TO_CURRENT_ENERGY_SCALE * float(value)
    mapped[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
        _convention.CURRENT_EXCHANGE_CONVENTION
    )
    mapped[_convention.RESCALED_FROM_METADATA_KEY] = (
        _convention.LEGACY_EXCHANGE_CONVENTION
    )
    return mapped


def _validate_row(row: dict[str, Any], *, length: int) -> None:
    if int(row.get("L", -1)) != int(length):
        raise ThreeSiteConcentrationError("checkpoint L mismatch")
    if not math.isclose(
        float(row.get("kappa_over_J", np.nan)),
        REPRESENTATIVE_KAPPA_OVER_J,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ThreeSiteConcentrationError("checkpoint representative kappa mismatch")
    if int(row.get("operator_basis_dimension", -1)) != LOCAL_ALGEBRA_DIMENSION:
        raise ThreeSiteConcentrationError("checkpoint local-algebra dimension mismatch")
    if tuple(row.get("local_sites", ())) != LOCAL_SITES:
        raise ThreeSiteConcentrationError("checkpoint local-region support mismatch")
    if str(row.get("window_protocol", "")) != WINDOW_PROTOCOL:
        raise ThreeSiteConcentrationError("checkpoint window protocol mismatch")
    if not math.isclose(
        float(row.get("window_half_width", np.nan)),
        _expected_half_width(length),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ThreeSiteConcentrationError("checkpoint primary half-width mismatch")
    if row.get(_convention.EXCHANGE_CONVENTION_METADATA_KEY) != (
        _convention.CURRENT_EXCHANGE_CONVENTION
    ):
        raise ThreeSiteConcentrationError("checkpoint exchange-convention mismatch")
    for field in ("w_L_raw", "window_max_eigenpair_residual", "tower_residual"):
        if not math.isfinite(float(row.get(field, np.nan))):
            raise ThreeSiteConcentrationError(f"checkpoint has non-finite {field}")


_legacy._validate_row = _validate_row


def _load_checkpoint(
    cache_root: Path,
    *,
    length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    directory = _legacy._checkpoint_dir(cache_root, length)
    row_path = directory / "row.json"
    worst_path = directory / "worst_eigenoperator.csv"
    metadata_path = directory / "metadata.json"
    present = [path.is_file() for path in (row_path, worst_path, metadata_path)]
    if not any(present):
        return None
    if not all(present):
        raise ThreeSiteConcentrationError(f"partial checkpoint must be inspected: {directory}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = json.loads(row_path.read_text(encoding="utf-8"))
        worst = pd.read_csv(worst_path).to_dict(orient="records")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ThreeSiteConcentrationError(f"invalid checkpoint: {directory}") from exc
    if metadata.get("status") != "complete":
        raise ThreeSiteConcentrationError(f"incomplete checkpoint must be inspected: {directory}")

    convention = metadata.get(_convention.EXCHANGE_CONVENTION_METADATA_KEY)
    if convention is None:
        row = _map_legacy_row(row)
        worst = [
            {
                **item,
                _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                    _convention.CURRENT_EXCHANGE_CONVENTION
                ),
                _convention.RESCALED_FROM_METADATA_KEY: (
                    _convention.LEGACY_EXCHANGE_CONVENTION
                ),
            }
            for item in worst
        ]
    elif convention == _convention.CURRENT_EXCHANGE_CONVENTION:
        row = dict(row)
        row[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = convention
    else:
        raise ThreeSiteConcentrationError(
            f"unsupported exchange convention {convention!r}: {directory}"
        )
    _validate_row(row, length=length)
    return row, worst


_legacy._load_checkpoint = _load_checkpoint


def _write_checkpoint(
    cache_root: Path,
    *,
    row: dict[str, Any],
    worst_rows: list[dict[str, Any]],
) -> None:
    directory = _legacy._checkpoint_dir(cache_root, int(row["L"]))
    directory.mkdir(parents=True, exist_ok=True)
    stamped = dict(row)
    stamped["schema_version"] = 2
    stamped[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
        _convention.CURRENT_EXCHANGE_CONVENTION
    )
    worst = pd.DataFrame(worst_rows)
    worst[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
        _convention.CURRENT_EXCHANGE_CONVENTION
    )
    _legacy._atomic_write_json(directory / "row.json", stamped)
    _legacy._atomic_write_csv(directory / "worst_eigenoperator.csv", worst)
    _legacy._atomic_write_json(
        directory / "metadata.json",
        {
            "schema_version": 2,
            "status": "complete",
            "cache_role": "spin1_sec6_p1_three_site_complete_charge_algebra",
            "L": int(stamped["L"]),
            "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
            "local_sites": list(LOCAL_SITES),
            "operator_basis_dimension": LOCAL_ALGEBRA_DIMENSION,
            "window_protocol": WINDOW_PROTOCOL,
            "window_half_width": float(stamped["window_half_width"]),
            "source_spectrum_checkpoint": str(stamped["source_spectrum_checkpoint"]),
            _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                _convention.CURRENT_EXCHANGE_CONVENTION
            ),
        },
    )


_legacy._write_checkpoint = _write_checkpoint


def _compute_length(*, roots: Iterable[Path], length: int):
    helpers = importlib.import_module("helpers")
    core, context, energies, vectors, metadata = _legacy._validated_spectrum(
        roots=roots,
        length=length,
    )
    half_width = _expected_half_width(length)
    indices = core._window_indices(energies, half_width, ENERGY_BLOCK_TOLERANCE)
    maximum_residual, median_residual = core._window_residuals(
        context["h_sector"], energies, vectors, indices, chunk_size=64
    )
    if maximum_residual > DENSE_RESIDUAL_TOLERANCE:
        raise ThreeSiteConcentrationError(
            f"L={length} dense in-window residual {maximum_residual:.3e} exceeds "
            f"{DENSE_RESIDUAL_TOLERANCE:.1e}"
        )

    patterns, names, local_basis = charge_conserving_three_site_hermitian_basis()
    projected_ops = []
    for local_matrix in local_basis:
        embedded = _legacy._embed_local_matrix(
            local_matrix,
            patterns=patterns,
            configs=context["configs"],
        )
        projected_ops.append(core._project_sparse(embedded, context["sector"]))

    empty = np.zeros((context["sector"].sector_dimension, 0), dtype=np.complex128)
    covariance = helpers.projector_deleted_block_covariance(
        energies,
        vectors,
        empty,
        tuple(projected_ops),
        indices,
        energy_tolerance=ENERGY_BLOCK_TOLERANCE,
        vector_tolerance=1.0e-9,
    )
    tower_residual = float(
        core.diagnose_eigenpair(context["h_sector"], context["tower"]).residual_norm
    )
    coefficients = np.asarray(covariance["worst_coefficients"], dtype=np.complex128)
    dominant = int(np.argmax(np.abs(coefficients)))
    row = {
        "schema_version": 2,
        "L": int(length),
        "M": int(core.TOTAL_SZ),
        "J3_over_J": float(core.J3_OVER_J),
        "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
        "local_sites": list(LOCAL_SITES),
        "local_region_size": len(LOCAL_SITES),
        "operator_basis_dimension": LOCAL_ALGEBRA_DIMENSION,
        "operator_basis_role": "complete magnetization-preserving three-site Hermitian algebra",
        "window_protocol": WINDOW_PROTOCOL,
        "window_half_width": half_width,
        "window_state_count": int(covariance["window_rank"]),
        "energy_block_count": int(covariance["energy_block_count"]),
        "w_L_raw": float(covariance["largest_width"]),
        "median_nonidentity_width_raw": float(covariance["median_nonidentity_width"]),
        "worst_basis_operator": names[dominant],
        "worst_basis_coefficient_abs": float(abs(coefficients[dominant])),
        "window_max_eigenpair_residual": float(maximum_residual),
        "window_median_eigenpair_residual": float(median_residual),
        "tower_residual": tower_residual,
        "source_spectrum_checkpoint": str(metadata["checkpoint_path"]),
        "source_spectrum_returned_eigenpairs": int(metadata["returned_eigenpairs"]),
        "source_spectrum_full": True,
        "source_role": "p1_three_site_from_reused_dense_spectrum",
        _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
            _convention.CURRENT_EXCHANGE_CONVENTION
        ),
        _convention.RESCALED_FROM_METADATA_KEY: metadata.get(
            _convention.RESCALED_FROM_METADATA_KEY,
            "",
        ),
    }
    worst_rows = [
        {
            "L": int(length),
            "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
            "local_sites": "0,1,2",
            "basis_operator": name,
            "coefficient_real": float(complex(coefficient).real),
            "coefficient_imag": float(complex(coefficient).imag),
            "coefficient_abs": float(abs(coefficient)),
            _convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                _convention.CURRENT_EXCHANGE_CONVENTION
            ),
        }
        for name, coefficient in zip(names, coefficients, strict=True)
    ]
    _validate_row(row, length=length)
    return row, worst_rows


_legacy._compute_length = _compute_length


def _stamp_outputs(output_dir: Path) -> None:
    output = Path(output_dir)
    for name in (ROWS_NAME, WORST_NAME):
        path = output / name
        if path.is_file():
            frame = pd.read_csv(path)
            frame[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
                _convention.CURRENT_EXCHANGE_CONVENTION
            )
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
    progress = output / PROGRESS_NAME
    if progress.is_file():
        value = json.loads(progress.read_text(encoding="utf-8"))
        value["schema_version"] = 2
        value["window_protocol"] = WINDOW_PROTOCOL
        value["window_prefactor"] = WINDOW_PREFACTOR
        value[_convention.EXCHANGE_CONVENTION_METADATA_KEY] = (
            _convention.CURRENT_EXCHANGE_CONVENTION
        )
        temporary = progress.with_name(f".{progress.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, progress)


def run(
    *,
    checkpoint_roots: Iterable[Path],
    cache_root: Path,
    output_dir: Path,
    compute_missing: bool,
) -> pd.DataFrame:
    frame = _ORIGINAL_RUN(
        checkpoint_roots=checkpoint_roots,
        cache_root=cache_root,
        output_dir=output_dir,
        compute_missing=compute_missing,
    )
    _stamp_outputs(output_dir)
    return frame


if __name__ == "__main__":
    _legacy.run = run
    _legacy.main()
