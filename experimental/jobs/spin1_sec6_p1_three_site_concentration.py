#!/usr/bin/env python
"""Three-site complete local-algebra concentration from existing Sec. VI spectra only."""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

TARGET_LENGTHS = (8, 10, 12)
REPRESENTATIVE_KAPPA_OVER_J = 0.10
WINDOW_PROTOCOL = "quarter_power_c1"
WINDOW_EXPONENT = 0.25
LOCAL_SITES = (0, 1, 2)
LOCAL_ALGEBRA_DIMENSION = 141
ENERGY_BLOCK_TOLERANCE = 1.0e-10
DENSE_RESIDUAL_TOLERANCE = 1.0e-8

ROWS_NAME = "spin1_xy_sec6_p1_three_site_concentration.csv"
WORST_NAME = "spin1_xy_sec6_p1_three_site_worst_eigenoperator.csv"
PROGRESS_NAME = "spin1_xy_sec6_p1_three_site_progress.json"


class ThreeSiteConcentrationError(RuntimeError):
    """Raised when cached-spectrum three-site concentration cannot be completed."""


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def charge_conserving_three_site_hermitian_basis() -> tuple[
    np.ndarray, tuple[str, ...], tuple[np.ndarray, ...]
]:
    """Return an HS-orthonormal basis of the three-spin fixed-charge algebra."""

    patterns = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
    dimension = int(patterns.shape[0])
    names: list[str] = ["identity"]
    basis: list[np.ndarray] = [np.eye(dimension, dtype=np.complex128) / math.sqrt(dimension)]

    # A Helmert basis spans the 26 traceless diagonal directions.
    for index in range(1, dimension):
        vector = np.zeros(dimension, dtype=np.float64)
        scale = math.sqrt(index * (index + 1))
        vector[:index] = 1.0 / scale
        vector[index] = -index / scale
        names.append(f"diag_helmert_{index}")
        basis.append(np.diag(vector).astype(np.complex128))

    charges = np.sum(patterns, axis=1)
    for charge in sorted(set(int(value) for value in charges)):
        indices = np.flatnonzero(charges == charge)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                symmetric = np.zeros((dimension, dimension), dtype=np.complex128)
                symmetric[left, right] = symmetric[right, left] = 1.0 / math.sqrt(2.0)
                antisymmetric = np.zeros((dimension, dimension), dtype=np.complex128)
                antisymmetric[left, right] = -1.0j / math.sqrt(2.0)
                antisymmetric[right, left] = 1.0j / math.sqrt(2.0)
                names.append(f"q{charge}_sym_{int(left)}_{int(right)}")
                basis.append(symmetric)
                names.append(f"q{charge}_asym_{int(left)}_{int(right)}")
                basis.append(antisymmetric)

    if len(basis) != LOCAL_ALGEBRA_DIMENSION:
        raise RuntimeError(
            f"expected a {LOCAL_ALGEBRA_DIMENSION}-dimensional algebra, got {len(basis)}"
        )
    gram = np.asarray(
        [[np.trace(left.conj().T @ right) for right in basis] for left in basis],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(gram, np.eye(LOCAL_ALGEBRA_DIMENSION), atol=1.0e-10)
    return patterns, tuple(names), tuple(basis)


def _embed_local_matrix(
    local_matrix: np.ndarray,
    *,
    patterns: np.ndarray,
    configs: np.ndarray,
    sites: tuple[int, ...] = LOCAL_SITES,
) -> sp.csr_array:
    """Embed one charge-preserving local matrix in the fixed-M product basis."""

    configs = np.asarray(configs, dtype=np.int64)
    pattern_lookup = {
        tuple(int(value) for value in pattern): index for index, pattern in enumerate(patterns)
    }
    config_lookup = {
        tuple(int(value) for value in config): index for index, config in enumerate(configs)
    }
    source_pattern_indices = np.asarray(
        [pattern_lookup[tuple(int(config[site]) for site in sites)] for config in configs],
        dtype=np.int64,
    )

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    site_indices = np.asarray(sites, dtype=np.int64)
    for source_basis, source_pattern in enumerate(source_pattern_indices):
        target_patterns = np.flatnonzero(np.abs(local_matrix[:, source_pattern]) > 1.0e-14)
        for target_pattern in target_patterns:
            coefficient = complex(local_matrix[target_pattern, source_pattern])
            target_config = configs[source_basis].copy()
            target_config[site_indices] = patterns[target_pattern]
            target_basis = config_lookup.get(tuple(int(value) for value in target_config))
            if target_basis is None:
                raise ThreeSiteConcentrationError(
                    "charge-preserving local action left the fixed-M basis; cache/model contract changed"
                )
            rows.append(int(target_basis))
            columns.append(int(source_basis))
            values.append(coefficient)
    shape = (configs.shape[0], configs.shape[0])
    return sp.csr_array((values, (rows, columns)), shape=shape, dtype=np.complex128)


def _checkpoint_dir(cache_root: Path, length: int) -> Path:
    return (
        Path(cache_root)
        / "sec6_p1_three_site"
        / f"spin1_L{int(length)}_kappa_p0p100000_three_site"
    )


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
        float(length) ** WINDOW_EXPONENT,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ThreeSiteConcentrationError("checkpoint primary half-width mismatch")
    for field in ("w_L_raw", "window_max_eigenpair_residual", "tower_residual"):
        if not math.isfinite(float(row.get(field, np.nan))):
            raise ThreeSiteConcentrationError(f"checkpoint has non-finite {field}")


def _load_checkpoint(
    cache_root: Path,
    *,
    length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    directory = _checkpoint_dir(cache_root, length)
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
    _validate_row(row, length=length)
    return row, worst


def _write_checkpoint(
    cache_root: Path,
    *,
    row: dict[str, Any],
    worst_rows: list[dict[str, Any]],
) -> None:
    directory = _checkpoint_dir(cache_root, int(row["L"]))
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(directory / "row.json", row)
    _atomic_write_csv(directory / "worst_eigenoperator.csv", pd.DataFrame(worst_rows))
    _atomic_write_json(
        directory / "metadata.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cache_role": "spin1_sec6_p1_three_site_complete_charge_algebra",
            "L": int(row["L"]),
            "kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
            "local_sites": list(LOCAL_SITES),
            "operator_basis_dimension": LOCAL_ALGEBRA_DIMENSION,
            "window_protocol": WINDOW_PROTOCOL,
            "source_spectrum_checkpoint": str(row["source_spectrum_checkpoint"]),
        },
    )


def _validated_spectrum(
    *,
    roots: Iterable[Path],
    length: int,
):
    common = importlib.import_module("spin1_sec6_common_windows")
    core = importlib.import_module("spin1_sec6_provisioning")
    context = core._point_context(length=int(length), kappa_over_j=REPRESENTATIVE_KAPPA_OVER_J)
    errors: list[str] = []
    candidates = common.discover_checkpoint_directories(
        roots,
        length=int(length),
        kappa_over_j=REPRESENTATIVE_KAPPA_OVER_J,
    )
    for candidate in candidates:
        try:
            energies, vectors, metadata = common.validate_cached_spectrum(
                candidate,
                length=int(length),
                kappa_over_j=REPRESENTATIVE_KAPPA_OVER_J,
                context=context,
            )
        except common.CachedSpectrumUnavailableError as exc:
            errors.append(str(exc))
            continue
        if int(metadata.get("returned_eigenpairs", energies.size)) != int(
            context["h_sector"].shape[0]
        ):
            errors.append(f"not a full dense spectrum: {candidate}")
            continue
        if metadata.get("full_spectrum") is False:
            errors.append(f"checkpoint explicitly marks a partial spectrum: {candidate}")
            continue
        return core, context, energies, vectors, metadata
    detail = "; ".join(errors) if errors else "no compatible checkpoint directories found"
    raise ThreeSiteConcentrationError(
        f"no validated full dense representative spectrum remained for L={length}: {detail}; "
        "no eigensolve was started"
    )


def _compute_length(*, roots: Iterable[Path], length: int):
    helpers = importlib.import_module("helpers")
    core, context, energies, vectors, metadata = _validated_spectrum(roots=roots, length=length)
    half_width = float(length) ** WINDOW_EXPONENT
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
        embedded = _embed_local_matrix(
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
        "schema_version": 1,
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
        }
        for name, coefficient in zip(names, coefficients, strict=True)
    ]
    _validate_row(row, length=length)
    return row, worst_rows


def run(
    *,
    checkpoint_roots: Iterable[Path],
    cache_root: Path,
    output_dir: Path,
    compute_missing: bool,
) -> pd.DataFrame:
    cache = Path(cache_root).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    roots = tuple(Path(value).resolve(strict=False) for value in checkpoint_roots)

    records: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for length in TARGET_LENGTHS:
        key = f"L{length}"
        cached = _load_checkpoint(cache, length=length)
        if cached is not None:
            row, point_worst = cached
            records.append(row)
            worst_rows.extend(point_worst)
            statuses[key] = "reused"
            continue
        if not compute_missing:
            statuses[key] = "pending"
            continue
        row, point_worst = _compute_length(roots=roots, length=length)
        _write_checkpoint(cache, row=row, worst_rows=point_worst)
        records.append(row)
        worst_rows.extend(point_worst)
        statuses[key] = "computed"

        # Persist aggregate progress after every completed size.
        current = pd.DataFrame(records).sort_values("L")
        _atomic_write_csv(output / ROWS_NAME, current)
        _atomic_write_csv(
            output / WORST_NAME,
            pd.DataFrame(worst_rows).sort_values(["L", "basis_operator"]),
        )

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values("L").reset_index(drop=True)
        _atomic_write_csv(output / ROWS_NAME, frame)
    if worst_rows:
        _atomic_write_csv(
            output / WORST_NAME,
            pd.DataFrame(worst_rows).sort_values(["L", "basis_operator"]),
        )
    complete = all(status in {"computed", "reused"} for status in statuses.values())
    _atomic_write_json(
        output / PROGRESS_NAME,
        {
            "schema_version": 1,
            "target_lengths": list(TARGET_LENGTHS),
            "representative_kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J,
            "local_sites": list(LOCAL_SITES),
            "operator_basis_dimension": LOCAL_ALGEBRA_DIMENSION,
            "window_protocol": WINDOW_PROTOCOL,
            "point_status": statuses,
            "complete": complete,
            "solve_policy": "cache-only spectral reuse; this stage contains no eigensolver",
            "claim_boundary": (
                "three-site concentration strengthens locality evidence but does not establish "
                "concentration for every bounded region"
            ),
        },
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-missing", action="store_true")
    args = parser.parse_args()
    frame = run(
        checkpoint_roots=args.checkpoint_root,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        compute_missing=args.compute_missing,
    )
    if frame.empty:
        print("no completed three-site rows yet", flush=True)
    else:
        print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
