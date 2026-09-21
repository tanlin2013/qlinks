#!/usr/bin/env python
"""Report Sec. VII square-QDM fixed-O(1) P0 completion without rebuilding the sector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdm_sec7_fixed_o1 import atomic_write_json
from qdm_sec7_fixed_o1_l12_observables import ACCEPTANCE_NAME as OBSERVABLES_ACCEPTANCE_NAME
from qdm_sec7_fixed_o1_l12_spectrum import (
    ACCEPTANCE_NAME as SPECTRUM_ACCEPTANCE_NAME,
)
from qdm_sec7_fixed_o1_l12_spectrum import (
    RECOMMENDATION_NAME,
)
from qdm_sec7_fixed_o1_sequence import STATUS_NAME as THREE_SIZE_STATUS_NAME

TARGET_ACCEPTANCE_NAME = "qdm_checkerboard_L12_target_block_acceptance.json"
STATUS_NAME = "qdm_checkerboard_fixed_O1_p0_status.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _closed(payload: dict[str, Any] | None) -> bool:
    return bool(payload is not None and payload.get("closed") is True)


def status(*, target_data_dir: Path, output_dir: Path) -> dict[str, Any]:
    target_path = Path(target_data_dir) / TARGET_ACCEPTANCE_NAME
    output = Path(output_dir)
    recommendation_path = output / RECOMMENDATION_NAME
    spectrum_path = output / SPECTRUM_ACCEPTANCE_NAME
    observables_path = output / OBSERVABLES_ACCEPTANCE_NAME
    three_size_path = output / THREE_SIZE_STATUS_NAME

    target = _read_json(target_path)
    recommendation = _read_json(recommendation_path)
    spectrum = _read_json(spectrum_path)
    observables = _read_json(observables_path)
    three_size = _read_json(three_size_path)

    pilot_recommended = bool(
        recommendation is not None
        and recommendation.get("status") == "recommended"
        and recommendation.get("recommended_half_width") is not None
    )
    payload = {
        "schema_version": 1,
        "target_block_closed": _closed(target),
        "fixed_O1_pilot_recommended": pilot_recommended,
        "fixed_O1_primary_half_width": (
            recommendation.get("recommended_half_width") if recommendation is not None else None
        ),
        "L12_spectrum_closed": _closed(spectrum),
        "L12_observables_closed": _closed(observables),
        "three_size_sequence_closed": _closed(three_size),
        "p0_thermal_lane_closed": bool(
            pilot_recommended
            and _closed(spectrum)
            and _closed(observables)
            and _closed(three_size)
        ),
        "files": {
            "target_block_acceptance": str(target_path),
            "pilot_recommendation": str(recommendation_path),
            "L12_spectrum_acceptance": str(spectrum_path),
            "L12_observables_acceptance": str(observables_path),
            "three_size_status": str(three_size_path),
        },
        "claim_boundary": (
            "P0 thermal-lane closure means the homogeneous fixed-O(1) Lx=4,8,12 "
            "numerical sequence is available. Thermodynamic finite-beta targeting, "
            "selected-sector entropy-density proof, phase-uniformity, and broader-region "
            "concentration remain P1/P2 work."
        ),
    }
    atomic_write_json(output / STATUS_NAME, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = status(target_data_dir=args.target_data_dir, output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
