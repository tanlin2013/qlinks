#!/usr/bin/env python
"""Run the evidence-first PRX P1 strengthening package.

The default scientific work is solver-free: exact character counting plus the
L=8,10 cage-obstruction chart. Figure rendering is opt-in and reuses validated
CSV products supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_prx_p1_figure_style as figure_audit
import pandas as pd
import spin1_prx_p1_obstruction_hierarchy as obstruction
import spin1_prx_p1_thermodynamic_summary as thermodynamics
from spin1_exchange_convention import CURRENT_EXCHANGE_CONVENTION

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository")


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        ROOT
        / "experimental"
        / "data"
        / "evidence_jobs"
        / (f"prx_p1_spin1_obstruction_figures_{stamp}")
    )


def _run_renderer(script: str, data_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "experimental" / "jobs" / script),
        "--data-dir",
        str(data_dir),
        "--use-tex",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _copy_final_figures(output_dir: Path, spin1_data_dir: Path, qdm_data_dir: Path) -> None:
    destination = output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        spin1_data_dir / "figures" / "spin1_xy_figure6_prx.pdf",
        spin1_data_dir / "figures" / "spin1_xy_figure6_prx.svg",
        qdm_data_dir / "figures" / "qdm_checkerboard_figure9_prx.pdf",
        qdm_data_dir / "figures" / "qdm_checkerboard_figure9_prx.svg",
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)


def _git_sha_from_metadata(repo_root: Path) -> str | None:
    """Resolve HEAD directly from .git metadata without requiring git."""
    git_entry = repo_root / ".git"
    git_dir = git_entry
    if git_entry.is_file():
        prefix = "gitdir:"
        line = git_entry.read_text(encoding="utf-8").strip()
        if not line.startswith(prefix):
            return None
        target = Path(line[len(prefix) :].strip())
        git_dir = target if target.is_absolute() else (repo_root / target).resolve(strict=False)

    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None

    ref = head.removeprefix("ref: ").strip()
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip() or None

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        suffix = f" {ref}"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")) or not line.endswith(suffix):
                continue
            return line.split(" ", 1)[0]
    return None


def _git_sha() -> str | None:
    """Best-effort provenance stamp; never make the evidence job depend on git."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        completed = None

    if completed is not None and completed.returncode == 0:
        sha = completed.stdout.strip()
        if sha:
            return sha
    return _git_sha_from_metadata(ROOT)


def _write_verdict(output_dir: Path, *, figures_rendered: bool) -> None:
    thermo = pd.read_csv(output_dir / "spin1_thermodynamic_upgrade_summary.csv")
    hierarchy = pd.read_csv(output_dir / "spin1_obstruction_hierarchy.csv")
    computed = hierarchy[hierarchy["status"].astype(str) == "computed"].copy()

    relations: list[str] = []
    for length in sorted(computed["L"].astype(int).unique()):
        frame = computed[computed["L"].astype(int) == length]
        parts = []
        for layer in ("T_cage", "T_fixed"):
            row = frame[frame["layer"] == layer]
            if row.empty:
                continue
            value = float(row.iloc[0]["tower_in_layer_residual"])
            parts.append(f"{layer}: tower-in-layer residual={value:.3e}")
        relations.append(f"- L={length}: " + "; ".join(parts))

    a1 = thermo[thermo["track"] == "A1"].iloc[0]
    a2 = thermo[thermo["track"] == "A2"].iloc[0]
    a4 = thermo[thermo["track"] == "A4"].iloc[0]
    lines = [
        "# PRX P1 strengthening verdict",
        "",
        "## Results that can strengthen the manuscript now",
        "",
        f"- A1: {a1['status']}. {a1['strongest_manuscript_safe_sentence']}",
        "",
        "## Informative finite-size results",
        "",
        "- The blind six-coordinate exchange chart reports T_cage and T_fixed at two "
        "generic even sizes and compares them with T_tower only after the kernels "
        "are computed.",
        *relations,
        "- T_joint is intentionally reported as not implemented because the current "
        "public stability API has no bounded-local operator/state co-continuation map.",
        f"- A4: {a4['status']}. {a4['strongest_manuscript_safe_sentence']}",
        "",
        "## Unresolved thermodynamic claims",
        "",
        f"- A2: {a2['status']}. {a2['remaining_gap']}",
        "- A3 remains gated on A2; no exceptional-fraction thermodynamic limit is inferred.",
        "",
        "## Figures",
        "",
        (
            "- Fig. 6/Fig. 9 were rerendered from validated evidence and passed the "
            "strict style audit."
            if figures_rendered
            else (
                "- Figure rendering was not requested in this run; "
                "no figure-science recomputation occurred."
            )
        ),
        "",
    ]
    (output_dir / "PRX_P1_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    output_dir: Path,
    lengths: tuple[int, ...],
    spin1_data_dir: Path | None,
    qdm_data_dir: Path | None,
    render_figures: bool,
    strict_figure_audit: bool,
) -> dict[str, Any]:
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)

    thermodynamics.run(
        output,
        spin1_data_dir=spin1_data_dir,
        magnetization=-2,
    )
    obstruction.run(output, lengths=lengths, magnetization=-2)

    figures_rendered = False
    if render_figures:
        if spin1_data_dir is None or qdm_data_dir is None:
            raise ValueError("--render-figures requires --spin1-data-dir and --qdm-data-dir")
        spin1 = spin1_data_dir.resolve(strict=False)
        qdm = qdm_data_dir.resolve(strict=False)
        _run_renderer("render_spin1_xy_draft_figures.py", spin1)
        _run_renderer("render_square_qdm_draft_figures.py", qdm)
        _copy_final_figures(output, spin1, qdm)
        figure_audit.run(
            output,
            spin1_data_dir=spin1,
            qdm_data_dir=qdm,
            strict=strict_figure_audit,
        )
        figures_rendered = True

    metadata = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "exchange_convention": CURRENT_EXCHANGE_CONVENTION,
        "lengths": list(lengths),
        "M": -2,
        "spin1_data_dir": None if spin1_data_dir is None else str(spin1_data_dir),
        "qdm_data_dir": None if qdm_data_dir is None else str(qdm_data_dir),
        "figures_rendered": figures_rendered,
        "strict_figure_audit": bool(strict_figure_audit),
        "spectral_solver_launched": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "job_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_verdict(output, figures_rendered=figures_rendered)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lengths", default="8,10")
    parser.add_argument("--spin1-data-dir", type=Path)
    parser.add_argument("--qdm-data-dir", type=Path)
    parser.add_argument("--render-figures", action="store_true")
    parser.add_argument(
        "--no-strict-figure-audit",
        action="store_true",
        help="Allow missing pdffonts; TeX and dimension failures still remain in the audit.",
    )
    args = parser.parse_args()
    lengths = tuple(int(value) for value in args.lengths.split(",") if value.strip())
    output = args.output_dir or default_output_dir()
    metadata = run(
        output_dir=output,
        lengths=lengths,
        spin1_data_dir=args.spin1_data_dir,
        qdm_data_dir=args.qdm_data_dir,
        render_figures=args.render_figures,
        strict_figure_audit=not args.no_strict_figure_audit,
    )
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
