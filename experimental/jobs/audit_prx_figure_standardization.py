#!/usr/bin/env python
"""Audit standardized PRX figures, provenance, dimensions, and embedded fonts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

PRX_COLUMN_WIDTH = 246.0 / 72.27
BASE_FONT_SIZE = 9.0

ASSETS = (
    {
        "figure": "Fig. 6",
        "stem": "spin1_xy_figure6_prx",
        "dir_key": "spin1_main",
        "expected_width_in": 7.05,
        "sources": (
            "spin1_xy_kappa0p1_sequence.csv",
            "spin1_xy_cage_excised_sequence.csv",
            "spin1_xy_kappa0p1_eth_scatter_Lmax.csv",
            "spin1_xy_kappa0p1_eth_scatter_all_sizes.csv",
            "spin1_xy_kappa0p1_beta0_overlap.csv",
            "spin1_xy_kappa_matching_grid.csv",
            "spin1_xy_kappa_concentration_grid.csv",
        ),
        "command": (
            "python experimental/jobs/render_spin1_xy_draft_figures.py --data-dir {data} --use-tex"
        ),
    },
    {
        "figure": "Fig. 9",
        "stem": "qdm_checkerboard_figure9_prx",
        "dir_key": "qdm_thermal",
        "expected_width_in": 7.05,
        "sources": (
            "qdm_checkerboard_thermal_overlap.csv",
            "qdm_checkerboard_beta0_overlap.csv",
            "qdm_checkerboard_eth_scatter.csv",
            "qdm_checkerboard_concentration_grid.csv",
            "qdm_checkerboard_representative_phase.csv",
        ),
        "command": (
            "python experimental/jobs/render_square_qdm_draft_figures.py "
            "--data-dir {data} --use-tex"
        ),
    },
    {
        "figure": "Fig. 10",
        "stem": "spin1_xy_appendix_beta0_bridges",
        "dir_key": "spin1_appendix",
        "expected_width_in": PRX_COLUMN_WIDTH,
        "sources": ("spin1_xy_appendix_beta0_bridges_data.csv",),
        "command": (
            "python experimental/jobs/render_prx_appendix_figures.py "
            "--spin1-data-dir {data} --use-tex"
        ),
    },
    {
        "figure": "Fig. 11",
        "stem": "spin1_xy_appendix_concentration_windows",
        "dir_key": "spin1_appendix",
        "expected_width_in": PRX_COLUMN_WIDTH,
        "sources": ("spin1_xy_kappa0p1_concentration_common_windows.csv",),
        "command": (
            "python experimental/jobs/render_prx_appendix_figures.py "
            "--spin1-data-dir {data} --use-tex"
        ),
    },
    {
        "figure": "Fig. 14(b)",
        "stem": "qdm_4x4_annihilator_radius",
        "dir_key": "qdm_appendix",
        "expected_width_in": PRX_COLUMN_WIDTH,
        "sources": ("qdm_4x4_minimum_annihilator_radius.csv",),
        "command": (
            "python experimental/jobs/render_prx_appendix_figures.py "
            "--qdm-data-dir {data} --use-tex"
        ),
    },
    {
        "figure": "Fig. 14(c)",
        "stem": "qdm_strip_compatibility_scaling",
        "dir_key": "qdm_appendix",
        "expected_width_in": PRX_COLUMN_WIDTH,
        "sources": ("qdm_4N_by_4_exact_sequence.csv",),
        "command": (
            "python experimental/jobs/render_prx_appendix_figures.py "
            "--qdm-data-dir {data} --use-tex"
        ),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _svg_inches(path: Path) -> tuple[float | None, float | None]:
    root = ElementTree.parse(path).getroot()

    def convert(raw: str | None) -> float | None:
        if raw is None:
            return None
        text = raw.strip().lower()
        factors = {
            "in": 1.0,
            "pt": 1.0 / 72.0,
            "px": 1.0 / 96.0,
            "cm": 1.0 / 2.54,
            "mm": 1.0 / 25.4,
        }
        for unit, factor in factors.items():
            if text.endswith(unit):
                return float(text[: -len(unit)]) * factor
        return float(text) / 96.0

    return convert(root.attrib.get("width")), convert(root.attrib.get("height"))


def _pdfinfo_inches(path: Path) -> tuple[float | None, float | None]:
    executable = shutil.which("pdfinfo")
    if executable is None:
        return None, None
    result = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, None
    for line in result.stdout.splitlines():
        if not line.startswith("Page size:"):
            continue
        fields = line.replace("Page size:", "").strip().split()
        if len(fields) >= 3 and fields[1] == "x":
            return float(fields[0]) / 72.0, float(fields[2]) / 72.0
    return None, None


def _pdffonts(path: Path) -> dict[str, object]:
    executable = shutil.which("pdffonts")
    if executable is None:
        return {"available": False, "returncode": None, "summary": []}
    result = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "returncode": result.returncode,
        "summary": lines[:16],
        "contains_dejavu": any("DejaVu" in line for line in lines),
    }


def _manifest_usetex(data_dir: Path, stem: str) -> bool | None:
    candidates = (
        data_dir / "figure_manifest.json",
        data_dir / "prx_appendix_render_manifest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        figures = payload.get("figures") if isinstance(payload, dict) else None
        if isinstance(figures, list):
            rows = [
                row
                for row in figures
                if isinstance(row, dict) and str(row.get("stem")) == stem
            ]
            if rows:
                values = {bool(row.get("usetex")) for row in rows}
                if len(values) == 1:
                    return values.pop()
        if isinstance(payload, dict) and "text_usetex" in payload:
            return bool(payload["text_usetex"])
    return None


def _verified_qdm_lengths(data_dir: Path) -> list[int]:
    thermal_path = data_dir / "qdm_checkerboard_thermal_overlap.csv"
    if not thermal_path.is_file():
        thermal_path = data_dir / "qdm_checkerboard_beta0_overlap.csv"
    if not thermal_path.is_file():
        return []
    frame = pd.read_csv(thermal_path)
    if frame.empty or "Lx" not in frame.columns:
        return []
    if "window_prefactor" in frame.columns:
        prefactors = frame["window_prefactor"].to_numpy(dtype=float)
        primary = float(prefactors[np.argmin(np.abs(prefactors - 0.75))])
        frame = frame[np.isclose(frame["window_prefactor"], primary)].copy()
    if "window_coverage_complete" in frame.columns:
        large = frame["Lx"].astype(int) >= 12
        frame = frame[~large | frame["window_coverage_complete"].fillna(False).astype(bool)]
    if "converged_vs_previous_budget" in frame.columns:
        large = frame["Lx"].astype(int) >= 12
        converged = frame["converged_vs_previous_budget"].fillna(False).astype(bool)
        frame = frame[~large | converged]
    return sorted(set(frame["Lx"].astype(int)))


def run(
    output_dir: Path,
    *,
    spin1_main_dir: Path,
    qdm_thermal_dir: Path,
    spin1_appendix_dir: Path,
    qdm_appendix_dir: Path,
    strict: bool,
) -> dict[str, object]:
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    dirs = {
        "spin1_main": spin1_main_dir.resolve(strict=False),
        "qdm_thermal": qdm_thermal_dir.resolve(strict=False),
        "spin1_appendix": spin1_appendix_dir.resolve(strict=False),
        "qdm_appendix": qdm_appendix_dir.resolve(strict=False),
    }
    errors: list[str] = []
    records: list[dict[str, object]] = []

    for spec in ASSETS:
        data = dirs[str(spec["dir_key"])]
        figures = data / "figures"
        pdf = figures / (str(spec["stem"]) + ".pdf")
        svg = figures / (str(spec["stem"]) + ".svg")
        missing = [str(path) for path in (pdf, svg) if not path.is_file()]
        if missing:
            errors.append(str(spec["figure"]) + ": missing " + ", ".join(missing))
            continue

        pdf_width, pdf_height = _pdfinfo_inches(pdf)
        svg_width, svg_height = _svg_inches(svg)
        expected_width = float(spec["expected_width_in"])
        widths = [value for value in (pdf_width, svg_width) if value is not None]
        dimension_ok = bool(widths) and all(abs(value - expected_width) <= 0.02 for value in widths)

        font_audit = _pdffonts(pdf)
        if strict and not font_audit.get("available"):
            errors.append(str(spec["figure"]) + ": pdffonts unavailable")
        if strict and font_audit.get("returncode") not in (None, 0):
            errors.append(str(spec["figure"]) + ": pdffonts failed")
        if strict and font_audit.get("contains_dejavu"):
            errors.append(str(spec["figure"]) + ": DejaVu font detected")
        if strict and not dimension_ok:
            errors.append(str(spec["figure"]) + ": physical width mismatch")

        source_hashes = {}
        for name in spec["sources"]:
            source = data / str(name)
            if source.is_file():
                source_hashes[str(name)] = _sha256(source)

        usetex = _manifest_usetex(data, str(spec["stem"]))
        if strict and usetex is not True:
            errors.append(str(spec["figure"]) + ": text.usetex=True not certified")

        records.append(
            {
                "figure": spec["figure"],
                "stem": spec["stem"],
                "renderer_command": str(spec["command"]).format(data=str(data)),
                "evidence_source_directory": str(data),
                "source_sha256": source_hashes,
                "pdf_width_in": pdf_width,
                "pdf_height_in": pdf_height,
                "svg_width_in": svg_width,
                "svg_height_in": svg_height,
                "expected_width_in": expected_width,
                "dimension_ok": dimension_ok,
                "base_font_size_pt": BASE_FONT_SIZE,
                "text_usetex": usetex,
                "pdffonts": font_audit,
                "numerical_recomputation": False,
            }
        )

    verified_qdm_lengths = _verified_qdm_lengths(dirs["qdm_thermal"])
    audit: dict[str, object] = {
        "schema_version": 1,
        "assets": records,
        "verified_qdm_raw_thermal_lengths": verified_qdm_lengths,
        "raw_12x4_thermal_present": 12 in verified_qdm_lengths,
        "numerical_recomputation": False,
        "errors": errors,
    }
    (output / "prx_figure_style_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PRX figure style audit",
        "",
        "- Numerical recomputation: no; all figures are render-only from cached tables.",
        "- Verified QDM raw thermal lengths: "
        + (", ".join(str(value) for value in verified_qdm_lengths) or "none"),
        "- raw_12x4_thermal_present = " + ("true" if 12 in verified_qdm_lengths else "false"),
        "",
    ]
    for row in records:
        lines.extend(
            [
                "## " + str(row["figure"]) + " — " + str(row["stem"]),
                "",
                "- Renderer: " + str(row["renderer_command"]),
                "- Evidence source: " + str(row["evidence_source_directory"]),
                "- Source SHA-256: " + json.dumps(row["source_sha256"], sort_keys=True),
                "- Physical size: PDF "
                + str(row["pdf_width_in"])
                + " x "
                + str(row["pdf_height_in"])
                + " in; SVG "
                + str(row["svg_width_in"])
                + " x "
                + str(row["svg_height_in"])
                + " in.",
                "- Base font: 9.0 pt.",
                "- text.usetex=True: " + ("yes" if row["text_usetex"] is True else "not certified"),
                "- pdffonts: " + " | ".join(row["pdffonts"].get("summary", [])),
                "- Numerical recomputation: no.",
                "",
            ]
        )
    if errors:
        lines.extend(["## Audit errors", "", *["- " + error for error in errors], ""])
    (output / "prx_figure_style_audit.md").write_text("\n".join(lines), encoding="utf-8")

    if strict and errors:
        raise RuntimeError("PRX figure style audit failed: " + "; ".join(errors))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spin1-main-dir", type=Path, required=True)
    parser.add_argument("--qdm-thermal-dir", type=Path, required=True)
    parser.add_argument("--spin1-appendix-dir", type=Path, required=True)
    parser.add_argument("--qdm-appendix-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    audit = run(
        args.output_dir,
        spin1_main_dir=args.spin1_main_dir,
        qdm_thermal_dir=args.qdm_thermal_dir,
        spin1_appendix_dir=args.spin1_appendix_dir,
        qdm_appendix_dir=args.qdm_appendix_dir,
        strict=args.strict,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
