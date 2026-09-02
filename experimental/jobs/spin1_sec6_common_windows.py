#!/usr/bin/env python
"""Current-convention cache-only Sec. VI common-window reducer.

The historical reducer is preserved in ``spin1_sec6_common_windows_legacy``.
Legacy spectral payloads are reused only through an explicit in-memory mapping:
eigenvalues and energy-like metadata are multiplied by 1/2 while eigenvectors are
left unchanged and then revalidated against the current Hamiltonian.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import spin1_exchange_convention as _convention
import spin1_sec6_common_windows_legacy as _legacy
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    LEGACY_EXCHANGE_CONVENTION,
    LEGACY_TO_CURRENT_ENERGY_SCALE,
    RESCALED_FROM_METADATA_KEY,
    current_window_half_width,
    exchange_convention_from_metadata,
)

_ORIGINAL_COMPUTE_COMMON_WINDOWS = _legacy.compute_common_windows_from_cache

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

FIXED_CONTROL_HALF_WIDTH = _convention.FIXED_CONTROL_HALF_WIDTH
PRIMARY_WINDOW_EXPONENT = _convention.PRIMARY_WINDOW_EXPONENT
PRIMARY_WINDOW_PREFACTOR = _convention.PRIMARY_WINDOW_PREFACTOR
PRIMARY_WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
FIXED_WINDOW_PROTOCOL = _convention.FIXED_WINDOW_PROTOCOL

# Established normalized concentration anchors are invariant under uniform rescaling.
REFERENCE_L14_FIXED_RAW_WIDTH = _legacy.REFERENCE_L14_FIXED_RAW_WIDTH
REFERENCE_L14_FIXED_CLEAN_WIDTH = _legacy.REFERENCE_L14_FIXED_CLEAN_WIDTH


def _compatible_metadata(metadata: dict[str, Any], *, length: int, kappa_over_j: float) -> bool:
    expected = {
        "L": int(length),
        "M": TOTAL_SZ,
        "J3_over_J": J3_OVER_J,
        "kappa_over_J": float(kappa_over_j),
    }
    if not all(metadata.get(key) == value for key, value in expected.items()):
        return False
    return exchange_convention_from_metadata(metadata) in {
        LEGACY_EXCHANGE_CONVENTION,
        CURRENT_EXCHANGE_CONVENTION,
    }


def _scale_legacy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    scaled = dict(metadata)
    for key in (
        "covered_spectral_half_width",
        "validation_window_half_width",
        "sampled_energy_abs_max",
        "shift",
    ):
        value = scaled.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            scaled[key] = float(value) * LEGACY_TO_CURRENT_ENERGY_SCALE
    scaled[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    scaled[RESCALED_FROM_METADATA_KEY] = LEGACY_EXCHANGE_CONVENTION
    return scaled


def _load_arrays(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = _legacy._metadata(directory / "metadata.json")
    if metadata is None:
        raise CachedSpectrumUnavailableError(f"invalid checkpoint metadata: {directory}")
    vectors_path = directory / "vectors.npy"
    if not vectors_path.is_file():
        vectors_path = directory / "eigenvectors.npy"
    try:
        raw_energies = np.load(directory / "energies.npy", mmap_mode="r", allow_pickle=False)
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CachedSpectrumUnavailableError(f"unreadable checkpoint arrays: {directory}") from exc
    if raw_energies.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != raw_energies.size:
        raise CachedSpectrumUnavailableError(f"checkpoint shape mismatch: {directory}")
    if int(metadata.get("sector_dimension", vectors.shape[0])) != int(vectors.shape[0]):
        raise CachedSpectrumUnavailableError(f"checkpoint sector dimension mismatch: {directory}")
    if not np.all(np.isfinite(raw_energies)):
        raise CachedSpectrumUnavailableError(f"checkpoint has non-finite energies: {directory}")

    convention = exchange_convention_from_metadata(metadata)
    if convention == LEGACY_EXCHANGE_CONVENTION:
        energies = np.asarray(raw_energies, dtype=np.float64) * LEGACY_TO_CURRENT_ENERGY_SCALE
        metadata = _scale_legacy_metadata(metadata)
    elif convention == CURRENT_EXCHANGE_CONVENTION:
        energies = raw_energies
        metadata = dict(metadata)
    else:
        raise CachedSpectrumUnavailableError(
            f"unsupported spin-1 exchange convention {convention!r}: {directory}"
        )
    return energies, vectors, metadata


def _required_validation_half_width(length: int) -> float:
    return max(
        PRIMARY_WINDOW_PREFACTOR * float(length) ** PRIMARY_WINDOW_EXPONENT,
        FIXED_CONTROL_HALF_WIDTH,
    )


def _window_protocols(length: int) -> tuple[tuple[str, float], ...]:
    return (
        (
            PRIMARY_WINDOW_PROTOCOL,
            PRIMARY_WINDOW_PREFACTOR * float(length) ** PRIMARY_WINDOW_EXPONENT,
        ),
        (FIXED_WINDOW_PROTOCOL, FIXED_CONTROL_HALF_WIDTH),
    )


def _expected_keys(lengths: Iterable[int]) -> set[tuple[int, str, str]]:
    return {
        (int(length), protocol, variant)
        for length in lengths
        for protocol in (PRIMARY_WINDOW_PROTOCOL, FIXED_WINDOW_PROTOCOL)
        for variant in ("raw", "clean")
    }


def validate_completed_common_window_export(
    data_dir: Path,
    *,
    lengths: Iterable[int] = TARGET_LENGTHS,
    kappa_over_j: float = REPRESENTATIVE_KAPPA_OVER_J,
) -> pd.DataFrame | None:
    """Validate only explicitly current derived/common-window exports."""

    data = Path(data_dir).resolve(strict=False)
    concentration_path = data / COMMON_NAME
    if not concentration_path.is_file():
        return None
    companions = (CHECKPOINT_AUDIT_NAME, WORST_NAME, TOLERANCE_NAME)
    if any(not (data / name).is_file() for name in companions):
        return None
    try:
        frame = pd.read_csv(concentration_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise CachedSpectrumUnavailableError(
            f"invalid completed common-window export: {concentration_path}"
        ) from exc
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        return None
    conventions = set(frame[EXCHANGE_CONVENTION_METADATA_KEY].astype(str))
    if conventions != {CURRENT_EXCHANGE_CONVENTION}:
        return None
    required = {
        "L",
        "kappa_over_J",
        "variant",
        "window_protocol",
        "window_half_width",
        "w_L",
        "median_nonidentity_width",
        "energy_block_count",
        "removed_projector_rank",
        "removed_fraction",
        "covered_spectral_half_width",
        "window_max_eigenpair_residual",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise CachedSpectrumUnavailableError(
            "completed common-window export is missing required columns: "
            + ", ".join(sorted(missing))
        )
    target_lengths = tuple(int(value) for value in lengths)
    selected = frame[
        frame["L"].astype(int).isin(target_lengths)
        & np.isclose(frame["kappa_over_J"].to_numpy(dtype=float), float(kappa_over_j))
    ].copy()
    actual_keys = {
        (int(row.L), str(row.window_protocol), str(row.variant))
        for row in selected.itertuples(index=False)
    }
    expected_keys = _expected_keys(target_lengths)
    if actual_keys != expected_keys or len(selected) != len(expected_keys):
        return None
    for row in selected.itertuples(index=False):
        expected_half_width = current_window_half_width(int(row.L), str(row.window_protocol))
        if not math.isclose(
            float(row.window_half_width),
            expected_half_width,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise CachedSpectrumUnavailableError(
                "completed common-window export has an invalid half-width at "
                f"L={int(row.L)}, protocol={row.window_protocol}"
            )
        if float(row.covered_spectral_half_width) + 1.0e-10 < expected_half_width:
            raise CachedSpectrumUnavailableError(
                f"completed common-window export exceeds cached spectral coverage at L={int(row.L)}"
            )
        residual = float(row.window_max_eigenpair_residual)
        if math.isfinite(residual) and residual > PHYSICAL_RESIDUAL_TOLERANCE:
            raise CachedSpectrumUnavailableError(
                f"completed common-window export has residual {residual:.3e} at L={int(row.L)}"
            )

    l14 = selected[
        (selected["L"].astype(int) == 14)
        & (selected["window_protocol"].astype(str) == FIXED_WINDOW_PROTOCOL)
    ]
    for variant, expected in {
        "raw": REFERENCE_L14_FIXED_RAW_WIDTH,
        "clean": REFERENCE_L14_FIXED_CLEAN_WIDTH,
    }.items():
        row = l14[l14["variant"].astype(str) == variant]
        if len(row) != 1 or not math.isclose(
            float(row.iloc[0]["w_L"]), expected, rel_tol=0.0, abs_tol=5.0e-8
        ):
            raise CachedSpectrumUnavailableError(
                "completed common-window export does not reproduce the established "
                f"L=14 fixed-width {variant} width"
            )
    return selected.sort_values(["window_protocol", "L", "variant"]).reset_index(drop=True)


def _stamp_current_outputs(output_dir: Path) -> None:
    output = Path(output_dir)
    for path in (
        output / COMMON_NAME,
        output / CHECKPOINT_AUDIT_NAME,
        output / WORST_NAME,
        output / TOLERANCE_NAME,
    ):
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    summary_path = output / SUMMARY_NAME
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            summary[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
            temporary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
            temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, summary_path)


def compute_common_windows_from_cache(*args, **kwargs) -> pd.DataFrame:
    frame = _ORIGINAL_COMPUTE_COMMON_WINDOWS(*args, **kwargs)
    output_dir = kwargs.get("output_dir")
    if output_dir is None and len(args) >= 2:
        output_dir = args[1]
    if output_dir is not None:
        _stamp_current_outputs(Path(output_dir))
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        frame = frame.copy()
        frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    return frame


# Patch global lookups used inside the preserved implementation.
_legacy._compatible_metadata = _compatible_metadata
_legacy._load_arrays = _load_arrays
_legacy._required_validation_half_width = _required_validation_half_width
_legacy._window_protocols = _window_protocols
_legacy._expected_keys = _expected_keys
_legacy.validate_completed_common_window_export = validate_completed_common_window_export
