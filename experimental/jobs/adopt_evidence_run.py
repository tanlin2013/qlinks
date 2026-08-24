#!/usr/bin/env python
"""Validate an existing timestamped evidence run and register reusable products.

The timestamped run remains immutable.  Large Spin-1 spectral checkpoints are
hard-linked (or copied when necessary) into the stable evidence cache, while a
small registry entry records which run currently supplies the logical evidence
family.  Incomplete runs may be adopted explicitly because an independently
validated checkpoint can be useful even when later notebook work failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from evidence_cache import default_cache_root, default_registry_root
from evidence_job_utils import collect_file_manifest, find_repo_root, write_json


def _resolve(raw: Path) -> Path:
    path = raw.expanduser()
    if not path.is_absolute():
        path = find_repo_root() / path
    return path.resolve(strict=False)


def _link_or_copy(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _validate_spin_checkpoint(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    energies_path = directory / "energies.npy"
    vectors_path = directory / "vectors.npy"
    if not all(path.is_file() for path in (metadata_path, energies_path, vectors_path)):
        raise ValueError(f"incomplete Spin-1 checkpoint: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    energies = np.load(energies_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if energies.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != energies.size:
        raise ValueError(f"shape mismatch in Spin-1 checkpoint: {directory}")
    if not np.all(np.isfinite(energies)):
        raise ValueError(f"non-finite energies in Spin-1 checkpoint: {directory}")
    expected_dimension = metadata.get("sector_dimension")
    if expected_dimension is not None and int(expected_dimension) != int(vectors.shape[0]):
        raise ValueError(f"sector-dimension mismatch in Spin-1 checkpoint: {directory}")
    sample_count = min(8, energies.size)
    if sample_count:
        indices = np.unique(np.linspace(0, energies.size - 1, sample_count, dtype=np.int64))
        block = np.asarray(vectors[:, indices])
        if not np.all(np.isfinite(block)):
            raise ValueError(f"non-finite eigenvectors in Spin-1 checkpoint: {directory}")
        gram = block.conj().T @ block
        orthogonality = float(np.linalg.norm(gram - np.eye(indices.size), ord=2))
    else:
        orthogonality = 0.0
    if orthogonality > 1.0e-6:
        raise ValueError(
            f"Spin-1 checkpoint orthogonality residual {orthogonality:.3e} exceeds 1e-6: "
            f"{directory}"
        )
    return {
        "status": "VALID_FINAL",
        "returned_eigenpairs": int(energies.size),
        "sector_dimension": int(vectors.shape[0]),
        "sample_orthogonality_residual": orthogonality,
        "metadata": metadata,
    }


def _spin_metadata_compatible(source: dict[str, Any], destination: dict[str, Any]) -> bool:
    keys = (
        "schema_version",
        "L",
        "M",
        "J3_over_J",
        "kappa_over_J",
        "requested_eigenpairs",
        "sector_dimension",
        "shift",
        "arpack_tolerance",
    )
    return all(source.get(key) == destination.get(key) for key in keys)


def _adopt_spin_checkpoints(
    run_dir: Path,
    *,
    cache_root: Path,
    mode: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for root in (run_dir / "checkpoints", run_dir / "run_artifacts" / "checkpoints"):
        if root.is_dir():
            candidates.extend(path.parent for path in root.glob("*/metadata.json"))
    adopted: list[dict[str, Any]] = []
    destination_root = cache_root / "spin1" / "sec6_sparse"
    for source_dir in sorted(set(candidates)):
        validation = _validate_spin_checkpoint(source_dir)
        destination = destination_root / source_dir.name
        row: dict[str, Any] = {
            "source": str(source_dir),
            "destination": str(destination),
            "validation": validation,
            "files": {},
        }
        if destination.exists():
            destination_validation = _validate_spin_checkpoint(destination)
            if not _spin_metadata_compatible(
                validation["metadata"], destination_validation["metadata"]
            ):
                raise RuntimeError(
                    "refusing to overwrite an incompatible stable Spin-1 checkpoint: "
                    f"{destination}"
                )
            if (
                validation["returned_eigenpairs"]
                != destination_validation["returned_eigenpairs"]
                or validation["sector_dimension"] != destination_validation["sector_dimension"]
            ):
                raise RuntimeError(
                    "stable Spin-1 checkpoint has incompatible array dimensions: "
                    f"{destination}"
                )
            row["mode"] = "existing_validated"
            row["destination_validation"] = destination_validation
            adopted.append(row)
            continue
        if dry_run:
            row["mode"] = "dry-run"
            adopted.append(row)
            continue
        destination.mkdir(parents=True, exist_ok=False)
        # Metadata is placed last so it remains the completion marker.
        for name in ("energies.npy", "vectors.npy"):
            row["files"][name] = _link_or_copy(
                source_dir / name,
                destination / name,
                mode=mode,
            )
        row["files"]["metadata.json"] = _link_or_copy(
            source_dir / "metadata.json",
            destination / "metadata.json",
            mode=mode,
        )
        row["mode"] = "adopted"
        adopted.append(row)
    return adopted


def _default_registry_name(job_name: str, run_dir: Path) -> str:
    lower = f"{job_name} {run_dir.name}".lower()
    if "qdm" in lower:
        return "qdm_checkerboard"
    if "spin1" in lower or "spin_1" in lower or "spin-1" in lower:
        return "spin1_xy"
    return job_name.replace("-", "_") or "evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--register-as", default=None)
    parser.add_argument(
        "--mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="How to place reusable large checkpoint files into the stable cache.",
    )
    parser.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help="Allow adoption from a failed/running parent job after checkpoint-level validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    metadata_path = run_dir / "run_artifacts" / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return_code = run_metadata.get("return_code")
    if return_code not in (0, None) and not args.allow_incomplete_run:
        raise RuntimeError(
            f"run return_code={return_code}; use --allow-incomplete-run only when independently "
            "validated checkpoints from this attempt are worth adopting"
        )
    if (
        return_code is None
        and "finished_at_utc" not in run_metadata
        and not args.allow_incomplete_run
    ):
        raise RuntimeError(
            "run has no completion marker; use --allow-incomplete-run to adopt completed "
            "checkpoint stages from a still-running attempt"
        )

    cache_root = default_cache_root() if args.cache_root is None else _resolve(args.cache_root)
    registry_root = (
        default_registry_root() if args.registry_root is None else _resolve(args.registry_root)
    )
    job_name = str(run_metadata.get("job_name") or run_dir.name)
    registry_name = args.register_as or _default_registry_name(job_name, run_dir)

    adopted_spin = _adopt_spin_checkpoints(
        run_dir,
        cache_root=cache_root,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    manifest = collect_file_manifest(run_dir)
    registry_payload = {
        "schema_version": 1,
        "logical_name": registry_name,
        "run_dir": str(run_dir),
        "job_name": job_name,
        "run_id": run_metadata.get("run_id"),
        "return_code": return_code,
        "finished_at_utc": run_metadata.get("finished_at_utc"),
        "git": run_metadata.get("git"),
        "file_count": len(manifest),
        "adopted_spin_checkpoints": adopted_spin,
        "cache_root": str(cache_root),
        "source_validation": (
            "completed_run" if return_code == 0 else "checkpoint_level_only_incomplete_parent"
        ),
    }
    print(json.dumps(registry_payload, indent=2, sort_keys=True))
    if not args.dry_run:
        registry_root.mkdir(parents=True, exist_ok=True)
        write_json(registry_root / f"{registry_name}.json", registry_payload)


if __name__ == "__main__":
    main()
