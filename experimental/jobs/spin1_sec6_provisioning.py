"""Current-convention Sec. VI provisioning adapter.

The pre-2026-09-02 numerical kernel is preserved in
``spin1_sec6_provisioning_legacy`` so historical evidence remains reproducible.
This active module patches only convention-dependent inputs and metadata:
``H=(J/2) sum(S+S-+h.c.)``, current windows, and explicit cache versioning.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import spin1_sec6_provisioning_legacy as _legacy
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    FIXED_CONTROL_HALF_WIDTH,
    LEGACY_EXCHANGE_CONVENTION,
    PRIMARY_WINDOW_PREFACTOR,
    exchange_convention_from_metadata,
)

_LEGACY_RUN_SEC6_PROVISIONING = _legacy.run_sec6_provisioning

# Re-export the established kernel, including private helpers consumed by sibling jobs.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

J1_MATRIX = J_DRAFT
_legacy.J1_MATRIX = J1_MATRIX
_legacy.RAW_WITNESSES = _legacy._make_spin1_witnesses()
RAW_WITNESSES = _legacy.RAW_WITNESSES


@dataclass(frozen=True, slots=True)
class Sec6ProvisioningConfig(_legacy.Sec6ProvisioningConfig):
    """Sec. VI configuration in permanent ``J/2`` ladder units."""

    fixed_half_widths: tuple[float, ...] = (0.375, FIXED_CONTROL_HALF_WIDTH)
    concentration_half_width: float = FIXED_CONTROL_HALF_WIDTH


_legacy.Sec6ProvisioningConfig = Sec6ProvisioningConfig


def _window_specs(
    length: int,
    config: Sec6ProvisioningConfig,
    *,
    dense: bool,
) -> list[tuple[str, float, float, float]]:
    specs: list[tuple[str, float, float, float]] = []
    for half_width in config.fixed_half_widths:
        specs.append((f"fixed_{half_width:g}", float(half_width), 0.0, float(half_width)))
    if config.include_quarter_window:
        specs.append(
            (
                "L_quarter",
                PRIMARY_WINDOW_PREFACTOR * float(length) ** 0.25,
                0.25,
                PRIMARY_WINDOW_PREFACTOR,
            )
        )
    if dense and config.include_sqrt_window_for_dense:
        specs.append(
            (
                "L_sqrt",
                PRIMARY_WINDOW_PREFACTOR * float(length) ** 0.5,
                0.5,
                PRIMARY_WINDOW_PREFACTOR,
            )
        )
    dedup: dict[str, tuple[str, float, float, float]] = {}
    for spec in specs:
        dedup[spec[0]] = spec
    return list(dedup.values())


_legacy._window_specs = _window_specs


def _metadata(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _legacy_checkpoint_matches(
    metadata: dict[str, object],
    *,
    length: int,
    kappa_over_j: float,
    requested: int,
    dimension: int,
) -> bool:
    return (
        int(metadata.get("L", -1)) == int(length)
        and int(metadata.get("M", 10**9)) == int(TOTAL_SZ)
        and float(metadata.get("J3_over_J", np.nan)) == float(J3_OVER_J)
        and float(metadata.get("kappa_over_J", np.nan)) == float(kappa_over_j)
        and int(metadata.get("requested_eigenpairs", -1)) == int(requested)
        and int(metadata.get("sector_dimension", -1)) == int(dimension)
    )


def _partial_spectrum(
    h_sector,
    *,
    length: int,
    kappa_over_j: float,
    eigenpairs: int,
    config: Sec6ProvisioningConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load or solve only checkpoints carrying the explicit current convention."""

    dimension = int(h_sector.shape[0])
    requested = min(int(eigenpairs), dimension - 2)
    if requested <= 0:
        raise ValueError("resolved sector is too small for shift-invert")
    stem = _legacy._checkpoint_name(
        length=length,
        kappa_over_j=kappa_over_j,
        eigenpairs=requested,
    )
    expected = {
        "schema_version": 2,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "L": int(length),
        "M": TOTAL_SZ,
        "J3_over_J": float(J3_OVER_J),
        "kappa_over_J": float(kappa_over_j),
        "requested_eigenpairs": int(requested),
        "sector_dimension": dimension,
        "shift": float(config.shift),
        "arpack_tolerance": float(config.arpack_tolerance),
    }
    if config.reuse_checkpoints:
        for candidate in _legacy._checkpoint_candidates(config, stem):
            metadata = _metadata(candidate / "metadata.json")
            if metadata is not None and _legacy_checkpoint_matches(
                metadata,
                length=length,
                kappa_over_j=kappa_over_j,
                requested=requested,
                dimension=dimension,
            ):
                convention = exchange_convention_from_metadata(metadata)
                if convention == LEGACY_EXCHANGE_CONVENTION:
                    raise RuntimeError(
                        "historical Spin-1 checkpoint matches this point but uses the old "
                        "ladder-prefactor-one convention; refusing silent reuse or fallback. "
                        "Use spin1_exchange_convention_migrate_evidence.py / an explicit "
                        "validated rescaling path."
                    )
            loaded = _legacy._load_checkpoint(candidate, expected=expected)
            if loaded is not None:
                energies, vectors, loaded_metadata = loaded
                loaded_metadata = dict(loaded_metadata)
                loaded_metadata["checkpoint_reused"] = True
                loaded_metadata["checkpoint_path"] = str(candidate)
                return energies, vectors, loaded_metadata

    started = time.perf_counter()
    energies, vectors = _legacy.spla.eigsh(
        h_sector,
        k=requested,
        sigma=float(config.shift),
        which="LM",
        tol=float(config.arpack_tolerance),
    )
    order = np.argsort(energies)
    energies = np.asarray(energies[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    metadata = {
        **expected,
        "returned_eigenpairs": int(energies.size),
        "covered_spectral_half_width": float(min(abs(energies.min()), abs(energies.max()))),
        "solve_seconds": float(time.perf_counter() - started),
        "checkpoint_reused": False,
    }
    if config.write_checkpoints:
        target = config.resolved_checkpoint_dir / stem
        _legacy._save_checkpoint(target, energies=energies, vectors=vectors, metadata=metadata)
        metadata["checkpoint_path"] = str(target)
    return energies, vectors, metadata


_legacy._partial_spectrum = _partial_spectrum


def _require_migrated_source(path: Path | None, *, role: str) -> None:
    if path is None:
        return
    directory = Path(path)
    if not directory.is_dir():
        return
    manifest = directory / "spin1_exchange_convention_migration_manifest.json"
    if manifest.is_file():
        metadata = _metadata(manifest)
        if metadata and metadata.get(EXCHANGE_CONVENTION_METADATA_KEY) == CURRENT_EXCHANGE_CONVENTION:
            return
    raise RuntimeError(
        f"{role} points to an unstamped historical Spin-1 evidence directory: {directory}. "
        "Convert it into a derived convention-migration directory before reuse."
    )


def _stamp_output_convention(output_dir: Path) -> None:
    import pandas as pd

    output = Path(output_dir)
    for path in output.glob("*.csv"):
        frame = pd.read_csv(path)
        frame[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    for path in output.glob("*.json"):
        if path.name == "spin1_exchange_convention_migration_manifest.json":
            continue
        metadata = _metadata(path)
        if metadata is None:
            continue
        metadata[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def run_sec6_provisioning(config: Sec6ProvisioningConfig):
    """Run the established kernel with current semantics and stamped outputs."""

    _require_migrated_source(config.baseline_data_dir, role="baseline_data_dir")
    _require_migrated_source(
        config.sparse_convergence_data_dir,
        role="sparse_convergence_data_dir",
    )
    result = _LEGACY_RUN_SEC6_PROVISIONING(config)
    _stamp_output_convention(config.output_dir)
    return result
