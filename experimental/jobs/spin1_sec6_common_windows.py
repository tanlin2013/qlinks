#!/usr/bin/env python
"""Compute homogeneous Spin-1 Sec. VI concentration windows from cached spectra only.

The validation/reuse path is deliberately lightweight. The heavy Sec. VI numerical
kernel is imported only after no completed derived product can be reused. This module
never calls the eigensolver entry point: missing reusable spectra are provisioning gaps.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir() and (candidate / "experimental").is_dir():
        ROOT = candidate
        break
else:
    ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
NOTEBOOKS = ROOT / "experimental" / "notebooks"
for path in (JOBS, NOTEBOOKS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

REPRESENTATIVE_KAPPA_OVER_J = 0.10
TOTAL_SZ = -2
J3_OVER_J = 0.10
PRIMARY_WINDOW_EXPONENT = 0.25
FIXED_CONTROL_HALF_WIDTH = 1.0
TARGET_LENGTHS = (8, 10, 12, 14)
PHYSICAL_RESIDUAL_TOLERANCE = 1.0e-6
ORTHOGONALITY_TOLERANCE = 1.0e-6
REFERENCE_L14_FIXED_RAW_WIDTH = 0.0237316428
REFERENCE_L14_FIXED_CLEAN_WIDTH = 0.0236713087

COMMON_NAME = "spin1_xy_kappa0p1_concentration_common_windows.csv"
CHECKPOINT_AUDIT_NAME = "spin1_xy_kappa0p1_common_window_checkpoint_audit.csv"
WORST_NAME = "spin1_xy_kappa0p1_common_window_worst_eigenoperator.csv"
TOLERANCE_NAME = "spin1_xy_kappa0p1_common_window_tolerance_audit.csv"
SUMMARY_NAME = "spin1_xy_kappa0p1_common_window_summary.json"


class CachedSpectrumUnavailableError(RuntimeError):
    """Raised when a required reusable eigensystem cannot be validated."""


def _load_core():
    """Load the heavy numerical kernel only after derived-data reuse has failed."""

    core = importlib.import_module("spin1_sec6_provisioning")
    if int(core.TOTAL_SZ) != TOTAL_SZ or not math.isclose(
        float(core.J3_OVER_J), J3_OVER_J, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError("Spin-1 Sec. VI cache contract disagrees with numerical kernel")
    return core


def _metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _compatible_metadata(metadata: dict[str, Any], *, length: int, kappa_over_j: float) -> bool:
    expected = {
        "L": int(length),
        "M": TOTAL_SZ,
        "J3_over_J": J3_OVER_J,
        "kappa_over_J": float(kappa_over_j),
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def discover_checkpoint_directories(
    roots: Iterable[Path], *, length: int, kappa_over_j: float
) -> list[Path]:
    """Return compatible completed checkpoints, largest spectral payload first."""

    candidates: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve(strict=False)
        if not root.is_dir():
            continue
        for metadata_path in root.rglob("metadata.json"):
            directory = metadata_path.parent
            if directory in seen:
                continue
            seen.add(directory)
            metadata = _metadata(metadata_path)
            if metadata is None or not _compatible_metadata(
                metadata, length=length, kappa_over_j=kappa_over_j
            ):
                continue
            energies_path = directory / "energies.npy"
            vectors_path = directory / "vectors.npy"
            if not vectors_path.is_file():
                vectors_path = directory / "eigenvectors.npy"
            if not energies_path.is_file() or not vectors_path.is_file():
                continue
            returned = metadata.get("returned_eigenpairs")
            if returned is None:
                try:
                    returned = int(np.load(energies_path, mmap_mode="r", allow_pickle=False).size)
                except (OSError, ValueError):
                    continue
            candidates.append((int(returned), directory))
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return [directory for _, directory in candidates]


def _load_arrays(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = _metadata(directory / "metadata.json")
    if metadata is None:
        raise CachedSpectrumUnavailableError(f"invalid checkpoint metadata: {directory}")
    vectors_path = directory / "vectors.npy"
    if not vectors_path.is_file():
        vectors_path = directory / "eigenvectors.npy"
    try:
        energies = np.load(directory / "energies.npy", mmap_mode="r", allow_pickle=False)
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CachedSpectrumUnavailableError(f"unreadable checkpoint arrays: {directory}") from exc
    if energies.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != energies.size:
        raise CachedSpectrumUnavailableError(f"checkpoint shape mismatch: {directory}")
    if int(metadata.get("sector_dimension", vectors.shape[0])) != int(vectors.shape[0]):
        raise CachedSpectrumUnavailableError(f"checkpoint sector dimension mismatch: {directory}")
    if not np.all(np.isfinite(energies)):
        raise CachedSpectrumUnavailableError(f"checkpoint has non-finite energies: {directory}")
    return energies, vectors, metadata


def validate_cached_spectrum(
    directory: Path,
    *,
    length: int,
    kappa_over_j: float,
    context: dict[str, Any],
    sample_vectors: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate one reusable spectrum without solving or mutating it."""

    energies, vectors, metadata = _load_arrays(directory)
    if not _compatible_metadata(metadata, length=length, kappa_over_j=kappa_over_j):
        raise CachedSpectrumUnavailableError(f"scientifically incompatible checkpoint: {directory}")
    if vectors.shape[0] != int(context["h_sector"].shape[0]):
        raise CachedSpectrumUnavailableError(
            f"resolved-sector dimension changed for cached checkpoint: {directory}"
        )
    count = min(int(sample_vectors), energies.size)
    sample = (
        np.unique(np.linspace(0, energies.size - 1, count, dtype=np.int64))
        if count
        else np.zeros(0, dtype=np.int64)
    )
    if sample.size:
        block = np.asarray(vectors[:, sample])
        if not np.all(np.isfinite(block)):
            raise CachedSpectrumUnavailableError(
                f"checkpoint has non-finite eigenvectors: {directory}"
            )
        gram = block.conj().T @ block
        orthogonality = float(np.linalg.norm(gram - np.eye(sample.size), ord=2))
        action = context["h_sector"] @ block
        residuals = np.linalg.norm(action - block * np.asarray(energies[sample])[None, :], axis=0)
        maximum_residual = float(np.max(residuals, initial=0.0))
    else:
        orthogonality = 0.0
        maximum_residual = 0.0
    if orthogonality > ORTHOGONALITY_TOLERANCE:
        raise CachedSpectrumUnavailableError(
            f"sample orthogonality residual {orthogonality:.3e} exceeds "
            f"{ORTHOGONALITY_TOLERANCE:.1e}: {directory}"
        )
    if maximum_residual > PHYSICAL_RESIDUAL_TOLERANCE:
        raise CachedSpectrumUnavailableError(
            f"sample physical eigenpair residual {maximum_residual:.3e} exceeds "
            f"{PHYSICAL_RESIDUAL_TOLERANCE:.1e}: {directory}"
        )
    checked = dict(metadata)
    checked.update(
        {
            "checkpoint_path": str(directory),
            "checkpoint_reused": True,
            "sample_orthogonality_residual": orthogonality,
            "sample_maximum_physical_residual": maximum_residual,
            "returned_eigenpairs": int(energies.size),
            "requested_eigenpairs": int(metadata.get("requested_eigenpairs", energies.size)),
        }
    )
    return energies, vectors, checked


def _window_protocols(length: int) -> tuple[tuple[str, float], ...]:
    return (
        ("quarter_power_c1", float(length) ** PRIMARY_WINDOW_EXPONENT),
        ("fixed_width_1", FIXED_CONTROL_HALF_WIDTH),
    )


def _expected_keys(lengths: Iterable[int]) -> set[tuple[int, str, str]]:
    return {
        (int(length), protocol, variant)
        for length in lengths
        for protocol in ("quarter_power_c1", "fixed_width_1")
        for variant in ("raw", "clean")
    }


def validate_completed_common_window_export(
    data_dir: Path,
    *,
    lengths: Iterable[int] = TARGET_LENGTHS,
    kappa_over_j: float = REPRESENTATIVE_KAPPA_OVER_J,
) -> pd.DataFrame | None:
    """Validate/reuse a completed P0-A export before numerical setup."""

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
    if not np.all(np.isfinite(selected["w_L"].to_numpy(dtype=float))):
        raise CachedSpectrumUnavailableError("completed common-window export has non-finite widths")
    if np.any(selected["w_L"].to_numpy(dtype=float) < 0.0):
        raise CachedSpectrumUnavailableError("completed common-window export has negative widths")
    if np.any(selected["removed_fraction"].to_numpy(dtype=float) < 0.0):
        raise CachedSpectrumUnavailableError(
            "completed common-window export has negative removed fraction"
        )
    if np.any(selected["energy_block_count"].to_numpy(dtype=int) <= 0):
        raise CachedSpectrumUnavailableError(
            "completed common-window export has invalid energy blocks"
        )
    for row in selected.itertuples(index=False):
        expected_half_width = (
            float(row.L) ** PRIMARY_WINDOW_EXPONENT
            if str(row.window_protocol) == "quarter_power_c1"
            else FIXED_CONTROL_HALF_WIDTH
        )
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
        & (selected["window_protocol"].astype(str) == "fixed_width_1")
    ]
    anchors = {
        "raw": REFERENCE_L14_FIXED_RAW_WIDTH,
        "clean": REFERENCE_L14_FIXED_CLEAN_WIDTH,
    }
    for variant, expected in anchors.items():
        row = l14[l14["variant"].astype(str) == variant]
        if len(row) != 1 or not math.isclose(
            float(row.iloc[0]["w_L"]), expected, rel_tol=0.0, abs_tol=5.0e-8
        ):
            raise CachedSpectrumUnavailableError(
                "completed common-window export does not reproduce the established "
                f"L=14 fixed-width {variant} width"
            )
    return selected.sort_values(["window_protocol", "L", "variant"]).reset_index(drop=True)


def _copy_completed_products(source: Path, output: Path) -> None:
    names = (COMMON_NAME, CHECKPOINT_AUDIT_NAME, WORST_NAME, TOLERANCE_NAME, SUMMARY_NAME)
    for name in names:
        src = source / name
        dst = output / name
        if src.is_file() and src.resolve() != dst.resolve(strict=False):
            shutil.copy2(src, dst)


def _raw_clean_records(
    *,
    length: int,
    kappa_over_j: float,
    protocol: str,
    half_width: float,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record.update(
            {
                "window_protocol": protocol,
                "window_exponent": (
                    PRIMARY_WINDOW_EXPONENT if protocol == "quarter_power_c1" else 0.0
                ),
                "window_prefactor": 1.0,
                "window_half_width": float(half_width),
                "kappa_over_J": float(kappa_over_j),
                "L": int(length),
                "w_L": float(record.get("w_L", record["largest_covariance_width"])),
                "validated_reusable_spectrum": True,
            }
        )
        output.append(record)
    return output


def compute_common_windows_from_cache(
    *,
    checkpoint_roots: Iterable[Path],
    output_dir: Path,
    lengths: Iterable[int] = TARGET_LENGTHS,
    kappa_over_j: float = REPRESENTATIVE_KAPPA_OVER_J,
    energy_block_tolerance: float = 1.0e-10,
    existing_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute P0-A from validated cached spectra; never invoke an eigensolver."""

    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    target_lengths = tuple(int(value) for value in lengths)
    reuse_source = (
        output if existing_data_dir is None else Path(existing_data_dir).resolve(strict=False)
    )
    completed = validate_completed_common_window_export(
        reuse_source, lengths=target_lengths, kappa_over_j=kappa_over_j
    )
    if completed is not None:
        _copy_completed_products(reuse_source, output)
        return completed

    candidates_by_length = {
        length: discover_checkpoint_directories(
            checkpoint_roots, length=length, kappa_over_j=kappa_over_j
        )
        for length in target_lengths
    }
    missing = [length for length, candidates in candidates_by_length.items() if not candidates]
    if missing:
        pd.DataFrame(
            [
                {
                    "L": length,
                    "kappa_over_J": kappa_over_j,
                    "status": "MISSING_REUSABLE_SPECTRUM",
                    "candidate_count": 0,
                    "validation_errors": "",
                }
                for length in missing
            ]
        ).to_csv(output / CHECKPOINT_AUDIT_NAME, index=False)
        raise CachedSpectrumUnavailableError(
            "missing validated reusable spectra for L="
            + ",".join(str(value) for value in missing)
            + "; no eigensolve was started"
        )

    core = _load_core()
    concentration_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    tolerance_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    invalid: list[int] = []

    for length in target_lengths:
        context = core._point_context(length=length, kappa_over_j=kappa_over_j)
        validated = None
        errors: list[str] = []
        for candidate in candidates_by_length[length]:
            try:
                validated = validate_cached_spectrum(
                    candidate,
                    length=length,
                    kappa_over_j=kappa_over_j,
                    context=context,
                )
                break
            except CachedSpectrumUnavailableError as exc:
                errors.append(str(exc))
        if validated is None:
            invalid.append(length)
            checkpoint_rows.append(
                {
                    "L": length,
                    "kappa_over_J": kappa_over_j,
                    "status": "MISSING_REUSABLE_SPECTRUM",
                    "candidate_count": len(candidates_by_length[length]),
                    "validation_errors": " | ".join(errors),
                }
            )
            continue

        energies, vectors, metadata = validated
        coverage = float(min(abs(float(np.min(energies))), abs(float(np.max(energies)))))
        checkpoint_rows.append(
            {
                "L": length,
                "kappa_over_J": kappa_over_j,
                "status": "VALIDATED_REUSE",
                "checkpoint_path": metadata["checkpoint_path"],
                "returned_eigenpairs": int(energies.size),
                "covered_spectral_half_width": coverage,
                "sample_orthogonality_residual": metadata["sample_orthogonality_residual"],
                "sample_maximum_physical_residual": metadata["sample_maximum_physical_residual"],
            }
        )
        for protocol, half_width in _window_protocols(length):
            if half_width > coverage + energy_block_tolerance:
                raise CachedSpectrumUnavailableError(
                    f"cached L={length} spectrum covers |E|<={coverage:.6g}, below required "
                    f"{protocol} half-width {half_width:.6g}; no solve was started"
                )
            config = core.Sec6ProvisioningConfig(
                output_dir=output,
                dense_sizes=(),
                large_size=length,
                representative_kappa_over_j=kappa_over_j,
                concentration_half_width=half_width,
                energy_block_tolerance=energy_block_tolerance,
                run_large_representative=False,
                run_family_large_size=False,
                reuse_checkpoints=True,
                write_checkpoints=False,
            )
            sparse_metadata = {
                "requested_eigenpairs": int(metadata["requested_eigenpairs"]),
                "checkpoint_reused": True,
                "checkpoint_path": metadata["checkpoint_path"],
            }
            rows, worst, tolerance, _dark, _exceptional = core._concentration_at_point(
                length=length,
                kappa_over_j=kappa_over_j,
                energies=energies,
                vectors=vectors,
                context=context,
                config=config,
                sparse_metadata=sparse_metadata,
                sparse_convergence_passed=True,
                budget_certification_source="validated_reusable_spectrum",
            )
            concentration_rows.extend(
                _raw_clean_records(
                    length=length,
                    kappa_over_j=kappa_over_j,
                    protocol=protocol,
                    half_width=half_width,
                    rows=rows,
                )
            )
            worst_rows.extend(
                {
                    **row,
                    "window_protocol": protocol,
                    "window_half_width": half_width,
                }
                for row in worst
            )
            tolerance_rows.extend(
                {
                    **row,
                    "window_protocol": protocol,
                    "window_half_width": half_width,
                }
                for row in tolerance
            )

    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    checkpoint_frame.to_csv(output / CHECKPOINT_AUDIT_NAME, index=False)
    if invalid:
        raise CachedSpectrumUnavailableError(
            "no validated reusable spectrum remained for L="
            + ",".join(str(value) for value in invalid)
            + "; common-window P0-A was not completed and no eigensolve was started"
        )

    frame = pd.DataFrame(concentration_rows).sort_values(["window_protocol", "L", "variant"])
    frame.to_csv(output / COMMON_NAME, index=False)
    pd.DataFrame(worst_rows).to_csv(output / WORST_NAME, index=False)
    pd.DataFrame(tolerance_rows).to_csv(output / TOLERANCE_NAME, index=False)
    summary: dict[str, Any] = {
        "solve_policy": "cache-only; no eigensolver fallback",
        "lengths": sorted(set(frame["L"].astype(int))),
        "window_protocols": sorted(set(frame["window_protocol"].astype(str))),
        "representative_kappa_over_J": float(kappa_over_j),
        "qualitative_narrowing": {},
        "power_law_fit_computed": False,
    }
    for protocol, group in frame[frame["variant"] == "raw"].groupby("window_protocol"):
        ordered = group.sort_values("L")
        widths = ordered["w_L"].to_numpy(dtype=float)
        summary["qualitative_narrowing"][str(protocol)] = {
            "strictly_decreasing": bool(np.all(np.diff(widths) < 0.0)),
            "widths": [float(value) for value in widths],
            "sizes": [int(value) for value in ordered["L"]],
        }
    (output / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=[],
        help="Reusable checkpoint root. May be supplied multiple times.",
    )
    parser.add_argument("--lengths", default="8,10,12,14")
    parser.add_argument(
        "--existing-data-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior derived-evidence directory. A complete validated common-window "
            "export is copied/reused before covariance reduction is attempted."
        ),
    )
    args = parser.parse_args()
    roots = list(args.checkpoint_root)
    if not roots:
        roots = [ROOT / "experimental" / "data" / "evidence_cache" / "spin1"]
    lengths = tuple(int(token.strip()) for token in args.lengths.split(",") if token.strip())
    if not lengths:
        raise ValueError("--lengths must contain at least one size")
    frame = compute_common_windows_from_cache(
        checkpoint_roots=roots,
        output_dir=args.output_dir,
        lengths=lengths,
        existing_data_dir=args.existing_data_dir,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
