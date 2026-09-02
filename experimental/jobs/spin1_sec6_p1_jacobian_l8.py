#!/usr/bin/env python
"""Recompute the Sec. VI cage-conditioning illustration on a generic L=8 ring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlinks.caging.stability import (
    cage_jacobian_conditioning_from_hamiltonian,
    diagnose_cage_stability,
)

from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
)

TOLERANCE = 1.0e-10
LENGTH = 8
TOTAL_SZ = -2
J3_OVER_J = 0.10
KAPPA_OVER_J = 0.10

SUMMARY_NAME = "spin1_xy_sec6_p1_L8_cage_jacobian_conditioning.csv"
GEOMETRY_NAME = "spin1_xy_sec6_p1_L8_geometry_audit.json"


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


def run(*, output_dir: Path) -> dict[str, Any]:
    from qlinks.basis.configs import basis_configs_from_build_result
    from qlinks.caging.analysis.spectral import diagnose_eigenpair
    from qlinks.models import (
        spin_one_xy_hxy_h3_imaginary_j2_model,
        spin_one_xy_periodic_range_couplings,
        spin_one_xy_scar_tower_states,
    )

    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)

    model = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=LENGTH,
        j=1.0,
        j3=J3_OVER_J,
        kappa=KAPPA_OVER_J,
        total_sz=TOTAL_SZ,
        h_z=0.0,
        d_z=0.0,
    )
    result = model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = basis_configs_from_build_result(result)
    tower_states, tower_labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=LENGTH,
        normalize=True,
    )
    if tower_states.shape[1] != 1:
        raise RuntimeError(f"expected one fixed-M tower state, found {tower_labels}")
    tower = tower_states[:, 0]
    support = np.flatnonzero(np.abs(tower) > TOLERANCE)

    conditioning = cage_jacobian_conditioning_from_hamiltonian(
        result.hamiltonian,
        support,
        tower,
        tolerance=TOLERANCE,
    )
    stability = diagnose_cage_stability(
        result.hamiltonian,
        support,
        state=tower,
        tolerance=TOLERANCE,
    )
    tower_residual = float(diagnose_eigenpair(result.hamiltonian, tower).residual_norm)

    row = {
        "L": LENGTH,
        "M": TOTAL_SZ,
        "J_over_J": 1.0,
        "J3_over_J": J3_OVER_J,
        "kappa_over_J": KAPPA_OVER_J,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "tower_support_size": int(support.size),
        "tower_residual": tower_residual,
        "boundary_nullity": int(stability.boundary_nullity),
        "invariant_cage_dimension": int(stability.invariant_cage_dimension),
        "interference_gap": stability.interference_gap,
        **conditioning.to_summary_dict(),
    }
    _atomic_write_csv(output / SUMMARY_NAME, pd.DataFrame([row]))

    range_three_pairs = spin_one_xy_periodic_range_couplings(
        length=LENGTH,
        distance=3,
        coefficient=1.0,
    )
    half_ring_pairs = spin_one_xy_periodic_range_couplings(
        length=6,
        distance=3,
        coefficient=1.0,
    )
    geometry = {
        "L": LENGTH,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "range_three_pair_count": len(range_three_pairs),
        "range_three_pairs": [[int(i), int(j)] for i, j, _ in range_three_pairs],
        "generic_ring_expected_pair_count": LENGTH,
        "generic_ring_geometry": len(range_three_pairs) == LENGTH,
        "L6_range_three_pair_count": len(half_ring_pairs),
        "L6_half_ring_collision": len(half_ring_pairs) != 6,
        "scientific_role": (
            "post-migration generic-ring calibration under the J/2 ladder convention; "
            "the interference gap is remeasured rather than inferred from the old run"
        ),
    }
    _atomic_write_json(output / GEOMETRY_NAME, geometry)
    return {"conditioning": row, "geometry": geometry}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
