#!/usr/bin/env python
"""Audit final PRX P1 Fig. 6/Fig. 9 typography, dimensions, and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

FIGURES = {
    "Fig. 6": ("spin1_xy_figure6_prx", "spin1"),
    "Fig. 9": ("qdm_checkerboard_figure9_prx", "qdm"),
}
MANIFEST_CANDIDATES = (
    "evidence_manifest.json",
    "run_manifest.json",
    "manifest.json",
    "claim_manifest.csv",
    "figure_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_hashes(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in MANIFEST_CANDIDATES:
        path = data_dir / name
        if path.is_file():
            result[name] = sha256(path)
    return result


def figure_manifest_record(data_dir: Path, stem: str) -> dict[str, Any]:
    path = data_dir / "figure_manifest.json"
    if not path.is_file():
        return {"status": "missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("figures", [])
        if row.get("stem") == stem and row.get("format") in {"pdf", "svg"}
    ]
    if not rows:
        return {"status": "stem_missing"}
    return {
        "status": "present",
        "usetex": all(bool(row.get("usetex")) for row in rows),
        "font_size_pt": sorted({float(row.get("font_size_pt", 0.0)) for row in rows}),
        "requested_dimensions_in": sorted(
            {
                (
                    float(row.get("requested_width_in", 0.0)),
                    float(row.get("requested_height_in", 0.0)),
                )
                for row in rows
            }
        ),
        "dimension_ok": all(bool(row.get("dimension_ok")) for row in rows),
    }


def pdffonts_audit(path: Path) -> dict[str, Any]:
    executable = shutil.which("pdffonts")
    if executable is None:
        return {
            "status": "unavailable",
            "fonts": [],
            "contains_dejavu": None,
        }
    completed = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "fonts": [],
            "contains_dejavu": None,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    font_names: list[str] = []
    for line in lines[2:]:
        fields = line.split()
        if fields:
            font_names.append(fields[0])
    return {
        "status": "ok",
        "fonts": font_names,
        "contains_dejavu": any("dejavu" in name.lower() for name in font_names),
    }


def qdm_has_l12(data_dir: Path) -> bool:
    thermal = data_dir / "qdm_checkerboard_thermal_overlap.csv"
    if not thermal.is_file():
        thermal = data_dir / "qdm_checkerboard_beta0_overlap.csv"
    if not thermal.is_file():
        return False
    frame = pd.read_csv(thermal)
    if "Lx" not in frame:
        return False
    rows = frame[frame["Lx"].astype(int) == 12]
    if rows.empty:
        return False
    if "window_coverage_complete" in rows:
        rows = rows[rows["window_coverage_complete"].fillna(False).astype(bool)]
    if "converged_vs_previous_budget" in rows:
        rows = rows[rows["converged_vs_previous_budget"].fillna(False).astype(bool)]
    return not rows.empty


def run(
    output_dir: Path,
    *,
    spin1_data_dir: Path,
    qdm_data_dir: Path,
    strict: bool,
) -> dict[str, Any]:
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    spin1 = spin1_data_dir.resolve(strict=False)
    qdm = qdm_data_dir.resolve(strict=False)
    directories = {"spin1": spin1, "qdm": qdm}

    audit: dict[str, Any] = {
        "schema_version": 1,
        "base_font_size_pt": 9.0,
        "render_commands": {
            "Fig. 6": (
                "python experimental/jobs/render_spin1_xy_draft_figures.py "
                "--data-dir <validated-spin1-evidence> --use-tex"
            ),
            "Fig. 9": (
                "python experimental/jobs/render_square_qdm_draft_figures.py "
                "--data-dir <validated-qdm-evidence> --use-tex"
            ),
        },
        "source_evidence": {
            key: {
                "directory": str(value),
                "manifest_sha256": source_manifest_hashes(value),
            }
            for key, value in directories.items()
        },
        "qdm_12x4_verified_row_present": qdm_has_l12(qdm),
        "figures": {},
    }

    errors: list[str] = []
    for label, (stem, key) in FIGURES.items():
        data_dir = directories[key]
        pdf = data_dir / "figures" / f"{stem}.pdf"
        svg = data_dir / "figures" / f"{stem}.svg"
        manifest = figure_manifest_record(data_dir, stem)
        fonts = pdffonts_audit(pdf) if pdf.is_file() else {"status": "pdf_missing"}
        record = {
            "stem": stem,
            "pdf": str(pdf),
            "svg": str(svg),
            "pdf_sha256": sha256(pdf) if pdf.is_file() else None,
            "svg_sha256": sha256(svg) if svg.is_file() else None,
            "figure_manifest": manifest,
            "pdf_fonts": fonts,
        }
        audit["figures"][label] = record
        if not pdf.is_file() or not svg.is_file():
            errors.append(f"{label}: final PDF/SVG pair is incomplete")
        if manifest.get("status") != "present":
            errors.append(f"{label}: figure manifest record is missing")
        elif not manifest.get("usetex", False):
            errors.append(f"{label}: text.usetex was not active")
        elif not manifest.get("dimension_ok", False):
            errors.append(f"{label}: saved dimensions do not match declared dimensions")
        if fonts.get("status") == "ok" and fonts.get("contains_dejavu"):
            errors.append(f"{label}: pdffonts reports DejaVu in the final PDF")
        if strict and fonts.get("status") != "ok":
            errors.append(f"{label}: pdffonts audit did not complete successfully")

    audit["strict"] = bool(strict)
    audit["errors"] = errors
    audit["passed"] = not errors
    (output / "figure_style_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Figure style audit",
        "",
        f"- Overall pass: {audit['passed']}.",
        "- Required final typography: REVTeX/LaTeX, 9 pt at declared physical size.",
        (
            "- QDM verified 12x4 row present in render input: "
            f"{audit['qdm_12x4_verified_row_present']}."
        ),
        "",
    ]
    for label, record in audit["figures"].items():
        manifest = record["figure_manifest"]
        fonts = record["pdf_fonts"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Stem: {record['stem']}",
                f"- text.usetex active: {manifest.get('usetex')}",
                f"- Declared dimensions (in): {manifest.get('requested_dimensions_in')}",
                f"- Base font size (pt): {manifest.get('font_size_pt')}",
                f"- pdffonts status: {fonts.get('status')}",
                f"- DejaVu detected: {fonts.get('contains_dejavu')}",
                f"- PDF fonts: {', '.join(fonts.get('fonts', [])) or 'unavailable'}",
                "",
            ]
        )
    if errors:
        lines.extend(["## Audit errors", "", *[f"- {error}" for error in errors], ""])
    (output / "figure_style_audit.md").write_text("\n".join(lines), encoding="utf-8")

    if strict and errors:
        raise RuntimeError("figure style audit failed: " + "; ".join(errors))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spin1-data-dir", type=Path, required=True)
    parser.add_argument("--qdm-data-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        spin1_data_dir=args.spin1_data_dir,
        qdm_data_dir=args.qdm_data_dir,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
