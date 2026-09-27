#!/usr/bin/env python
"""Exact/analytic validation of the fixed-M spin-1 resolved-sector entropy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def coefficient_for_cycles(length: int, shift: int, magnetization: int) -> int:
    """Trace of T^shift in the fixed-M basis by cycle decomposition."""
    d = math.gcd(length, shift)
    cycle_length = length // d
    counts = {0: 1}
    for _ in range(d):
        nxt: dict[int, int] = {}
        for total, count in counts.items():
            for value in (-cycle_length, 0, cycle_length):
                nxt[total + value] = nxt.get(total + value, 0) + count
        counts = nxt
    return counts.get(magnetization, 0)


def momentum_dimensions(length: int, magnetization: int) -> list[int]:
    """Exact dimensions of all translation-momentum sectors."""
    traces = [coefficient_for_cycles(length, r, magnetization) for r in range(length)]
    dimensions = []
    for q in range(length):
        value = (
            sum(
                traces[r]
                * complex(
                    math.cos(-2 * math.pi * q * r / length),
                    math.sin(-2 * math.pi * q * r / length),
                )
                for r in range(length)
            )
            / length
        )
        if abs(value.imag) > 1e-7 or abs(value.real - round(value.real)) > 1e-7:
            raise RuntimeError(f"character projection failed at L={length}, q={q}: {value}")
        dimensions.append(int(round(value.real)))
    if sum(dimensions) != traces[0]:
        raise RuntimeError("momentum dimensions do not sum to the fixed-M dimension")
    return dimensions


def staggered_tower_momentum_index(length: int, magnetization: int) -> int:
    """Momentum index of the staggered bimagnon tower member at fixed magnetization."""
    if length % 2:
        raise ValueError("the staggered periodic tower sequence requires even length")
    numerator = length + magnetization
    if numerator % 2:
        raise ValueError("magnetization is incompatible with the bimagnon tower")
    excitation_number = numerator // 2
    return 0 if excitation_number % 2 == 0 else length // 2


def run(output_dir: Path, *, minimum_length: int, maximum_length: int, magnetization: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if minimum_length > maximum_length:
        raise ValueError("minimum_length must not exceed maximum_length")
    first_even = minimum_length if minimum_length % 2 == 0 else minimum_length + 1
    rows = []
    for length in range(first_even, maximum_length + 1, 2):
        dims = momentum_dimensions(length, magnetization)
        total = sum(dims)
        asymptotic = (
            3**length
            * math.sqrt(3.0 / (4.0 * math.pi * length))
            * math.exp(-3.0 * magnetization**2 / (4.0 * length))
        )
        selected_q = staggered_tower_momentum_index(length, magnetization)
        for q, dim in enumerate(dims):
            rows.append(
                {
                    "L": length,
                    "M": magnetization,
                    "momentum_index_q": q,
                    "selected_tower_momentum_index_q": selected_q,
                    "is_selected_tower_momentum": q == selected_q,
                    "sector_dimension": dim,
                    "fixed_M_dimension": total,
                    "log_fixed_M_over_L": math.log(total) / length,
                    "log_momentum_dimension_over_L": (
                        math.log(dim) / length if dim else float("-inf")
                    ),
                    "local_clt_asymptotic": asymptotic,
                    "fixed_M_over_asymptotic": total / asymptotic,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "spin1_resolved_sector_counts.csv", index=False)
    summary = f"""# Spin-1 resolved-sector entropy

For fixed magnetization M={magnetization}, the exact block dimension is
T_(L,M) = [z^M](z^-1+1+z)^L.  The lattice local central-limit theorem for a
sum of iid variables uniformly distributed on {{-1,0,1}} (variance 2/3) gives,
for fixed M,

T_(L,M) = 3^L sqrt(3/(4 pi L)) exp(-3 M^2/(4L)) (1+o(1)).

Hence lim_(L->infinity) log T_(L,M)/L = log 3.

For translation resolution, the q-sector character formula is
dim H_(M,q) = L^-1 sum_(r=0)^(L-1) exp(-2 pi i q r/L) Tr_M(T^r).
The identity contribution is T_(L,M)/L.  For r != 0, a configuration fixed by
T^r is constant on gcd(L,r) cycles; because gcd(L,r) is a proper divisor of L,
it is at most L/2.  Therefore every nonidentity character is O(3^(L/2)) and is
exponentially smaller than T_(L,M)/L.  Uniformly in q,

dim H_(M,q) = T_(L,M)/L + O(3^(L/2)),

so every nonempty momentum subsequence has entropy density log 3.  Momentum
resolution changes only a polynomial factor (asymptotically 1/L).

For the staggered bimagnon tower, the member at magnetization M is obtained
after n=(L+M)/2 staggered pair raises.  Its one-site translation eigenvalue is
(-1)^n.  At M=-2 this selects q=pi (momentum index L/2) when L=0 mod 4 and
q=0 when L=2 mod 4.  Both selected subsequences therefore retain entropy
density log 3; the exact selected row is marked in the exported count table.

At the representative nonzero-kappa point used for the thermodynamic evidence,
ordinary inversion is broken, so no inversion-parity label is part of the
physically resolved sector there.  The applicable spatial resolution is the
translation momentum selected above.

Status: proved for the even-L, fixed-M translation-resolved sequence.  The
entropy density is s_resolved = log 3.  This result does not establish positive
entropy density inside the defining L^(1/4) energy window (A2).
"""
    (output_dir / "spin1_resolved_sector_entropy.md").write_text(summary, encoding="utf-8")
    (output_dir / "spin1_resolved_sector_entropy.json").write_text(
        json.dumps(
            {
                "status": "proved",
                "M": magnetization,
                "entropy_density": math.log(3.0),
                "remaining_gap": (
                    "A2 raw microcanonical-window entropy is separate and not implied."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-length", type=int, default=4)
    parser.add_argument("--maximum-length", type=int, default=40)
    parser.add_argument("--magnetization", type=int, default=-2)
    args = parser.parse_args()
    run(
        args.output_dir,
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        magnetization=args.magnetization,
    )


if __name__ == "__main__":
    main()
