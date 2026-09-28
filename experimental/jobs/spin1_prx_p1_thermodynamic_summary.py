#!/usr/bin/env python
"""Assemble the PRX P1 spin-1 thermodynamic verdict without overclaiming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spin1_prx_p1_resolved_entropy as resolved_entropy
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    PRIMARY_WINDOW_PROTOCOL,
)

BRIDGE_NAMES = ("mc_to_beta0_resolved", "beta0_resolved_to_fixedM")


def _bridge_table_from_args(
    bridge_table: Path | None,
    spin1_data_dir: Path | None,
) -> Path | None:
    if bridge_table is not None:
        return bridge_table.resolve(strict=False)
    if spin1_data_dir is None:
        return None
    candidate = spin1_data_dir.resolve(strict=False) / "spin1_xy_appendix_beta0_bridges_data.csv"
    return candidate if candidate.is_file() else None


def _filter_bridge_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "kappa_over_J" in result:
        result = result[np.isclose(result["kappa_over_J"].astype(float), 0.10)]
    if "window_protocol" in result:
        current = result[result["window_protocol"].astype(str) == PRIMARY_WINDOW_PROTOCOL]
        if not current.empty:
            result = current
    if "variant" in result:
        raw = result[result["variant"].astype(str) == "raw"]
        if not raw.empty:
            result = raw
    return result


def evaluate_a4(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None or not path.is_file():
        return (
            {
                "track": "A4",
                "status": "not attempted",
                "strongest_manuscript_safe_sentence": (
                    "The exact fixed-M beta=0 local values remain auxiliary; no new "
                    "microcanonical-to-beta=0 asymptotic bridge is claimed here."
                ),
                "remaining_gap": (
                    "Supply the validated Appendix-D bridge table, or prove local "
                    "ensemble equivalence for the defining raw microcanonical window."
                ),
            },
            {"bridge_table": None},
        )

    frame = pd.read_csv(path)
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        raise ValueError("bridge table is missing the spin-1 exchange-convention stamp")
    conventions = set(frame[EXCHANGE_CONVENTION_METADATA_KEY].dropna().astype(str))
    if conventions != {CURRENT_EXCHANGE_CONVENTION}:
        raise ValueError(
            "bridge table is not mapped to the permanent J/2 exchange convention: "
            f"{sorted(conventions)!r}"
        )
    frame = _filter_bridge_table(frame)
    missing = {"L", "bridge", "trace_distance"}.difference(frame.columns)
    if missing:
        raise ValueError(f"bridge table is missing required columns: {sorted(missing)}")
    by_bridge = {
        name: frame[frame["bridge"].astype(str) == name].sort_values("L") for name in BRIDGE_NAMES
    }
    if any(group.empty for group in by_bridge.values()):
        raise ValueError("bridge table must contain both microcanonical and fixed-M bridge rows")

    largest_common = min(int(group["L"].max()) for group in by_bridge.values())
    latest = {}
    for name, group in by_bridge.items():
        row = group[group["L"].astype(int) == largest_common]
        if row.empty:
            row = group.iloc[[-1]]
        latest[name] = float(row.iloc[-1]["trace_distance"])

    sentence = (
        "For the tested two-site reduced density matrices through "
        f"L={largest_common}, the defining microcanonical-to-resolved-beta=0 and "
        "resolved-beta=0-to-fixed-M bridges are directly measured; this is controlled "
        "finite-size strengthening, not an asymptotic ensemble-equivalence theorem."
    )
    return (
        {
            "track": "A4",
            "status": "controlled numerical strengthening",
            "strongest_manuscript_safe_sentence": sentence,
            "remaining_gap": (
                "No controlled asymptotic statement has been established that sends "
                "the defining L^(1/4) raw microcanonical ensemble to the resolved/fixed-M "
                "beta=0 local ensemble for arbitrary fixed local support."
            ),
        },
        {
            "bridge_table": str(path),
            "largest_common_L": largest_common,
            "trace_distance_at_largest_common_L": latest,
        },
    )


def run(
    output_dir: Path,
    *,
    bridge_table: Path | None = None,
    spin1_data_dir: Path | None = None,
    minimum_count_length: int = 4,
    maximum_count_length: int = 40,
    magnetization: int = -2,
) -> pd.DataFrame:
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    resolved_entropy.run(
        output,
        minimum_length=minimum_count_length,
        maximum_length=maximum_count_length,
        magnetization=magnetization,
    )

    resolved = json.loads(
        (output / "spin1_resolved_sector_entropy.json").read_text(encoding="utf-8")
    )
    a4_path = _bridge_table_from_args(bridge_table, spin1_data_dir)
    a4_row, a4_metadata = evaluate_a4(a4_path)

    rows = [
        {
            "track": "A1",
            "status": "proved",
            "strongest_manuscript_safe_sentence": (
                "For even L at fixed M=-2, translation resolution changes only "
                "subexponential factors. The staggered tower selects q=pi for L=0 mod 4 "
                "and q=0 for L=2 mod 4, and both selected subsequences have entropy "
                "density log(3). Ordinary inversion is broken at the representative "
                "nonzero-kappa point, so no inversion-parity resolution is imposed."
            ),
            "remaining_gap": (
                "A1 does not control the number of states inside the defining energy "
                "window; that is the separate A2 problem."
            ),
        },
        {
            "track": "A2",
            "status": "inconclusive",
            "strongest_manuscript_safe_sentence": (
                "Positive entropy density of the full resolved M=-2 sector is proved, "
                "but positive entropy density of the defining Delta E=(J/2)L^(1/4) "
                "raw energy window is not established by this package."
            ),
            "remaining_gap": (
                "Spectral reflection fixes symmetry about the tower energy but gives no "
                "lower bound on the number of levels in an L^(1/4) shell. A central-limit "
                "statement on the O(sqrt(L)) energy scale is likewise insufficient for "
                "this shrinking scaled window without a suitable local-limit theorem or "
                "another controlled lower bound after symmetry resolution."
            ),
        },
        {
            "track": "A3",
            "status": "not attempted",
            "strongest_manuscript_safe_sentence": (
                "No thermodynamic exceptional-fraction claim is added because the "
                "denominator N_win(L) has not been controlled in A2."
            ),
            "remaining_gap": (
                "A2 must first establish exponentially many states in the defining raw "
                "window; only then should the full exceptional inventory be compared "
                "against N_win(L)."
            ),
        },
        a4_row,
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "spin1_thermodynamic_upgrade_summary.csv", index=False)

    metadata = {
        "schema_version": 1,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "M": magnetization,
        "resolved_sector_entropy_density": resolved["entropy_density"],
        "defining_window": "Delta E=(J/2)L^(1/4)",
        "A4": a4_metadata,
        "stop_rule_applied": True,
        "large_L_bruteforce_launched": False,
    }
    (output / "spin1_thermodynamic_upgrade_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Spin-1 thermodynamic upgrade summary",
        "",
        f"- Exchange convention: `{CURRENT_EXCHANGE_CONVENTION}`.",
        "- Defining raw window: `Delta E=(J/2)L^(1/4)`.",
        "- Stop rule: applied; no brute-force large-L spectrum campaign was launched.",
        "",
        "| Track | Status | Strongest manuscript-safe sentence | Exact remaining gap |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        safe = str(row["strongest_manuscript_safe_sentence"]).replace("|", "\\|")
        gap = str(row["remaining_gap"]).replace("|", "\\|")
        lines.append(f"| {row['track']} | {row['status']} | {safe} | {gap} |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "A1 can strengthen the manuscript now. A2 remains unresolved under the "
            "ticket's controlled-evidence standard, so A3 is intentionally gated off. "
            "A4 is finite-size strengthening only when the validated bridge table is "
            "present; the exact fixed-M beta=0 values must otherwise remain auxiliary.",
            "",
        ]
    )
    (output / "spin1_thermodynamic_upgrade_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-table", type=Path)
    parser.add_argument("--spin1-data-dir", type=Path)
    parser.add_argument("--minimum-count-length", type=int, default=4)
    parser.add_argument("--maximum-count-length", type=int, default=40)
    parser.add_argument("--magnetization", type=int, default=-2)
    args = parser.parse_args()
    run(
        args.output_dir,
        bridge_table=args.bridge_table,
        spin1_data_dir=args.spin1_data_dir,
        minimum_count_length=args.minimum_count_length,
        maximum_count_length=args.maximum_count_length,
        magnetization=args.magnetization,
    )


if __name__ == "__main__":
    main()
