#!/usr/bin/env python
"""Run or render the square-QDM Sec. VII checkerboard evidence workflow."""

from __future__ import annotations

from evidence_job_utils import (
    build_parser,
    parse_float_tuple,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


def main() -> None:
    parser = build_parser(description=__doc__ or "Run square-QDM checkerboard evidence job.")
    parser.add_argument(
        "--transport-repeats", default=None, help="Local checkerboard-family repeats, e.g. 1,2,3."
    )
    parser.add_argument(
        "--microcanonical-repeats",
        "--ed-repeats",
        dest="ed_repeats",
        default=None,
        help=(
            "Full energy-resolved repeats. Production-safe default: 1,2. "
            "Repeat 3 is disabled unless explicitly acknowledged."
        ),
    )
    parser.add_argument("--transfer-max-length", type=int, default=None)
    parser.add_argument(
        "--phase-values", default=None, help="Checkerboard pilot grid including endpoint control."
    )
    parser.add_argument(
        "--positive-phase-values", default=None, help="Principal positive phase grid."
    )
    parser.add_argument("--representative-phase", type=float, default=None)
    parser.add_argument(
        "--thermal-protocol", choices=("auto", "beta0", "finite-beta"), default="auto"
    )
    parser.add_argument("--energy-match-tolerance", type=float, default=None)
    parser.add_argument("--window-prefactors", default=None)
    parser.add_argument("--primary-window-prefactor", type=float, default=None)
    parser.add_argument("--energy-block-tolerance", type=float, default=None)
    parser.add_argument("--large-strip-repeats", default=None)
    parser.add_argument("--run-large-strip", action="store_true")
    parser.add_argument("--skip-checkerboard-thermal-scan", action="store_true")
    parser.add_argument("--skip-checkerboard-concentration", action="store_true")
    parser.add_argument(
        "--allow-large-dense-ed",
        action="store_true",
        help="Acknowledge the repeat>=3 dense-ED memory risk.",
    )
    args = parser.parse_args()
    if args.stage == "render":
        run_evidence_renderer(
            job_name="square_qdm_draft_evidence",
            renderer_filename="render_square_qdm_draft_figures.py",
            args=args,
        )
        return

    if args.run_large_strip:
        raise NotImplementedError(
            "The third energy-resolved 12x4 strip requires a controlled partial-spectrum or "
            "typicality implementation; "
            "the current dense path is intentionally disabled."
        )
    ed = parse_int_tuple(args.ed_repeats)
    if ed is not None and max(ed) >= 3 and not args.allow_large_dense_ed:
        raise ValueError(
            "QDM full ED repeat >=3 is disabled: the current algorithm can exceed 400 GiB. "
            "Use repeats 1,2, or implement a controlled partial-spectrum/typicality method."
        )
    transport = parse_int_tuple(args.transport_repeats)
    phases = parse_float_tuple(args.phase_values)
    positive = parse_float_tuple(args.positive_phase_values)
    prefactors = parse_float_tuple(args.window_prefactors)
    large = parse_int_tuple(args.large_strip_repeats)
    overrides = {
        "SAVE_FIGURES": bool(args.figure_formats.strip()) and args.stage != "compute",
        "SAVE_PDF": "pdf" in {p.strip().lower() for p in args.figure_formats.split(",")},
        "RUN_CHECKERBOARD_THERMAL_SCAN": not args.skip_checkerboard_thermal_scan,
        "RUN_CHECKERBOARD_CONCENTRATION": not args.skip_checkerboard_concentration,
        "CHECKERBOARD_THERMAL_PROTOCOL": args.thermal_protocol,
        "STRICT_CLAIMS": bool(args.strict_claims),
    }
    if ed is not None:
        overrides["CHECKERBOARD_ED_REPEATS"] = ed
    if transport is not None:
        overrides["CHECKERBOARD_TRANSPORT_REPEATS"] = transport
    if phases is not None:
        overrides["CHECKERBOARD_PHASE_VALUES"] = phases
    if positive is not None:
        overrides["CHECKERBOARD_POSITIVE_PHASE_VALUES"] = positive
    if prefactors is not None:
        overrides["MICROCANONICAL_PREFACTORS"] = prefactors
    if large is not None:
        overrides["LARGE_STRIP_REPEATS"] = large
    if args.transfer_max_length is not None:
        if args.transfer_max_length < 4 or args.transfer_max_length % 4:
            raise ValueError("--transfer-max-length must be a multiple of four >=4")
        overrides["CHECKERBOARD_TRANSFER_MAX_LENGTH"] = int(args.transfer_max_length)
    if args.representative_phase is not None:
        overrides["CHECKERBOARD_REPRESENTATIVE_PHASE"] = float(args.representative_phase)
    if args.energy_match_tolerance is not None:
        overrides["CHECKERBOARD_ENERGY_MATCH_TOL"] = float(args.energy_match_tolerance)
    if args.primary_window_prefactor is not None:
        overrides["PRIMARY_WINDOW_PREFACTOR"] = float(args.primary_window_prefactor)
    if args.energy_block_tolerance is not None:
        overrides["ENERGY_BLOCK_TOL"] = float(args.energy_block_tolerance)
    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
