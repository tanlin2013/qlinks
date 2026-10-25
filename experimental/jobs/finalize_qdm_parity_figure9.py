"""Assemble accepted parity data and render Fig. 9 with the 8x4 and 12x4 points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdm_figure9_l12_evidence import load_l12_evidence
from qdm_parity_evidence import submission_gate
from qdm_sec7_fixed_o1_sequence import run as assemble_sequence
from render_qdm_parity_figure9 import load_evidence, render


def finalize(
    output_dir: Path,
    *,
    figure_output_dir: Path | None = None,
    use_tex: bool = True,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
):
    """Run data-only acceptance/assembly steps; never build a model or solve H."""
    output_dir = Path(output_dir)
    print({"figure9": "checking_accepted_inputs"}, flush=True)
    load_evidence(output_dir)
    load_l12_evidence(output_dir)
    print({"figure9": "assembling_parity_sequence"}, flush=True)
    assemble_sequence(output_dir=output_dir, sy_character="target")
    gate = submission_gate(output_dir)
    if gate.get("closed") is not True:
        failed = [name for name, passed in gate["checks"].items() if passed is not True]
        raise ValueError(f"Parity submission gate is open: {failed}")
    print({"figure9": "rendering_original_layout", "sizes": [8, 12]}, flush=True)
    return render(
        output_dir,
        figure_output_dir if figure_output_dir is not None else output_dir / "figures",
        use_tex=use_tex,
        formats=formats,
        include_l12=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-output-dir", type=Path)
    parser.add_argument("--no-tex", action="store_true", help="Use mathtext for a preview")
    parser.add_argument("--figure-formats", default="pdf,svg,png")
    args = parser.parse_args()
    formats = tuple(value.strip() for value in args.figure_formats.split(",") if value.strip())
    if not formats or not set(formats).issubset({"pdf", "svg", "png"}):
        parser.error("--figure-formats must select pdf, svg and/or png")
    print(
        json.dumps(
            finalize(
                args.output_dir,
                figure_output_dir=args.figure_output_dir,
                use_tex=not args.no_tex,
                formats=formats,
            ),
            indent=2,
        ),
        flush=True,
    )
