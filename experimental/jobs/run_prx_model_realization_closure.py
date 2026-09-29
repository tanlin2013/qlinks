#!/usr/bin/env python
"""Emit the analytic model-realization closure package for the PRX manuscript.

The job is solver-free.  It records exact combinatorial checks and the analytic
bounds proved in experimental/PRX_MODEL_REALIZATION_CLOSURE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from spin1_prx_p1_resolved_entropy import (
    coefficient_for_cycles,
    momentum_dimensions,
    staggered_tower_momentum_index,
)

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository")

TOTAL_SZ = -2
J3_OVER_J = 0.10
KAPPA_MIN_OVER_J = 0.05
KAPPA_MAX_OVER_J = 0.20
KAPPA_STAR_OVER_J = 0.10
ETA = 0.25
WINDOW_POWER = 0.5 + ETA
DEFAULT_WINDOW_PREFACTOR = 1.0
FIXED_M_VARIANCE_COEFFICIENT_MAX = 2.0 * (
    1.0 + J3_OVER_J**2 + KAPPA_MAX_OVER_J**2
)
CHEBYSHEV_DECAY_POWER = 2.0 * ETA
Y_BETA0_LIMIT = 1.0 / 3.0
Y_EVENTUAL_SAFE_LOWER_BOUND = 1.0 / 6.0


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "experimental" / "data" / "evidence_jobs" / (
        "prx_model_realization_closure_" + stamp
    )


def fixed_m_dimension(length: int, magnetization: int = TOTAL_SZ) -> int:
    return coefficient_for_cycles(int(length), 0, int(magnetization))


def fixed_m_local_pattern_probability(
    length: int,
    pattern: tuple[int, ...],
    *,
    magnetization: int = TOTAL_SZ,
) -> float:
    """Exact probability of one fixed local product pattern at beta=0, fixed M."""
    local_sum = sum(int(value) for value in pattern)
    denominator = fixed_m_dimension(length, magnetization)
    if denominator <= 0:
        raise ValueError("empty fixed-magnetization sector")
    numerator = coefficient_for_cycles(
        int(length) - len(pattern),
        0,
        int(magnetization) - local_sum,
    )
    return float(numerator / denominator)


def energy_variance_identity_coefficient(
    kappa_over_j: float,
    *,
    j3_over_j: float = J3_OVER_J,
) -> float:
    """Pointwise fixed-M upper coefficient in Tr(rho H^2)/(J^2 L)."""
    return float(2.0 * (1.0 + j3_over_j**2 + float(kappa_over_j) ** 2))


def chebyshev_asymptotic_coefficient(
    *,
    epsilon: float = 0.1,
    window_prefactor: float = DEFAULT_WINDOW_PREFACTOR,
) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if window_prefactor <= 0.0:
        raise ValueError("window_prefactor must be positive")
    return float(
        (FIXED_M_VARIANCE_COEFFICIENT_MAX + epsilon) / window_prefactor**2
    )


def _git_sha_from_metadata(repo_root: Path) -> str | None:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha.strip() or None

    git_entry = repo_root / ".git"
    git_dir = git_entry
    if git_entry.is_file():
        line = git_entry.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir:"):
            return None
        target = Path(line.removeprefix("gitdir:").strip())
        git_dir = target if target.is_absolute() else (repo_root / target).resolve(strict=False)

    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None

    ref = head.removeprefix("ref: ").strip()
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip() or None

    packed = git_dir / "packed-refs"
    if packed.is_file():
        suffix = " " + ref
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")) or not line.endswith(suffix):
                continue
            return line.split(" ", 1)[0]
    return None


def _closure_rows() -> list[dict[str, str]]:
    return [
        {
            "layer": "exact caged eigenstate",
            "current_status": "closed",
            "after_this_ticket": "closed",
            "proof_evidence": "exact staggered tower; E_psi=0 throughout compatible family",
        },
        {
            "layer": "bounded local caging operator",
            "current_status": "closed",
            "after_this_ticket": "closed",
            "proof_evidence": "existing exact A, Z, and Y constructions",
        },
        {
            "layer": "first-order/predictive deformation selection",
            "current_status": "closed analytically + finite-size blind check",
            "after_this_ticket": "closed",
            "proof_evidence": "existing exchange-phase rule and blind obstruction hierarchy",
        },
        {
            "layer": "positive raw thermodynamic microcanonical witness",
            "current_status": "open",
            "after_this_ticket": "closed on wide window",
            "proof_evidence": "Q_Y beta-zero limit 1/3 plus almost-full-window conditioning",
        },
        {
            "layer": "positive raw-window entropy density",
            "current_status": "open for L^(1/4) protocol",
            "after_this_ticket": "closed for Delta E=c|J|L^(3/4)",
            "proof_evidence": "O(L) resolved variance + Chebyshev + resolved entropy log 3",
        },
        {
            "layer": "vanishing caged fraction",
            "current_status": "gated",
            "after_this_ticket": "closed for declared tower",
            "proof_evidence": (
                "one tower state in fixed M=-2 sector versus "
                "exp[(log3+o(1))L] window"
            ),
        },
        {
            "layer": "all-fixed-bounded-region concentration",
            "current_status": "open",
            "after_this_ticket": "closed",
            "proof_evidence": (
                "momentum compression + fixed-M covariance O(1/L) + "
                "window conditioning"
            ),
        },
        {
            "layer": "literal ICQMBS realization",
            "current_status": "open",
            "after_this_ticket": "closed for spin-1 wide-window sequence",
            "proof_evidence": "Tracks A-D analytic proof",
        },
        {
            "layer": "deformation-stable ICQMBS realization",
            "current_status": "open",
            "after_this_ticket": "closed on 0.05<kappa/J<0.20",
            "proof_evidence": (
                "uniform variance/window bound and Hamiltonian-independent "
                "beta-zero local bounds"
            ),
        },
    ]


def _write_closure_matrix(output: Path) -> dict[str, object]:
    verdicts = {
        "formal_framework_closed": True,
        "model_instantiation_closed": True,
        "literal_icqmbs_realization_closed": True,
        "deformation_stable_icqmbs_closed": True,
    }
    payload = {
        "verdicts": verdicts,
        "realizing_model": "spin1_xy_compatible_exchange_family",
        "raw_window_half_width": "c |J| L^(3/4)",
        "eta": ETA,
        "compatible_open_interval_kappa_over_J": [
            KAPPA_MIN_OVER_J,
            KAPPA_MAX_OVER_J,
        ],
        "rows": _closure_rows(),
        "narrow_L_quarter_window_status": (
            "L^(1/4) finite-size diagnostic only; no thermodynamic theorem claimed"
        ),
    }
    (output / "framework_model_closure_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Framework/model closure matrix",
        "",
        "Top-level verdicts:",
        "",
        "- formal_framework_closed = true",
        "- model_instantiation_closed = true",
        "- literal_icqmbs_realization_closed = true",
        "- deformation_stable_icqmbs_closed = true",
        "",
        "The two thermodynamic upgrades use the analytic raw half-width "
        "Delta E_L=c|J|L^(3/4), not the existing L^(1/4) finite-size protocol.",
        "",
        "| Layer | Current status | After this ticket | Proof/evidence |",
        "|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + row["layer"]
            + " | "
            + row["current_status"]
            + " | "
            + row["after_this_ticket"]
            + " | "
            + row["proof_evidence"]
            + " |"
        )
    (output / "framework_model_closure_matrix.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_bounds(output: Path) -> dict[str, object]:
    payload = {
        "magnetization": TOTAL_SZ,
        "selected_momentum_sequence": {
            "L_mod_4_eq_0": "pi",
            "L_mod_4_eq_2": "0",
        },
        "j3_over_J": J3_OVER_J,
        "kappa_star_over_J": KAPPA_STAR_OVER_J,
        "uniform_closed_interval_kappa_over_J": [
            KAPPA_MIN_OVER_J,
            KAPPA_MAX_OVER_J,
        ],
        "compatible_open_interval_kappa_over_J": [
            KAPPA_MIN_OVER_J,
            KAPPA_MAX_OVER_J,
        ],
        "energy_variance": {
            "fixed_M_pointwise_upper_coefficient_over_J2": (
                FIXED_M_VARIANCE_COEFFICIENT_MAX
            ),
            "resolved_sector_asymptotic": "(2.1+o(1)) J^2 L",
            "resolved_character_correction": "O(J^2 L^(7/2) 3^(-L/2))",
            "mean_energy": 0.0,
            "tower_energy": 0.0,
        },
        "raw_window": {
            "convention": "half_width",
            "half_width": "c |J| L^(3/4)",
            "full_width": "2 c |J| L^(3/4)",
            "default_c": DEFAULT_WINDOW_PREFACTOR,
            "eta": ETA,
            "power": WINDOW_POWER,
            "energy_density_half_width": "c |J| L^(-1/4)",
        },
        "chebyshev": {
            "outside_fraction": "O(L^(-1/2))",
            "decay_power": CHEBYSHEV_DECAY_POWER,
            "asymptotic_coefficient_for_epsilon_0p1_c_1": (
                chebyshev_asymptotic_coefficient()
            ),
        },
        "entropy_density": math.log(3.0),
        "declared_caged_count_in_fixed_M_sector": 1,
        "witness": {
            "name": "Q_Y=Y^dagger Y",
            "beta0_limit": Y_BETA0_LIMIT,
            "eventual_uniform_safe_lower_bound": Y_EVENTUAL_SAFE_LOWER_BOUND,
        },
        "background_concentration": {
            "beta0_translation_average_variance": "O(1/L)",
            "wide_window_variance": "O(1/L)",
            "deviation_fraction": "O(1/(epsilon^2 L))",
        },
    }
    (output / "spin1_wide_window_bounds.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_exact_sequence_checks(output: Path) -> None:
    rows = []
    for length in range(8, 42, 2):
        q = staggered_tower_momentum_index(length, TOTAL_SZ)
        dims = momentum_dimensions(length, TOTAL_SZ)
        fixed = sum(dims)
        y_activity = fixed_m_local_pattern_probability(length, (0,))
        rows.append(
            {
                "L": length,
                "M": TOTAL_SZ,
                "selected_momentum_index_q": q,
                "fixed_M_dimension": fixed,
                "selected_momentum_dimension": dims[q],
                "declared_tower_count": 1,
                "fixed_M_beta0_QY": y_activity,
                "wide_window_power": WINDOW_POWER,
                "chebyshev_decay_power": CHEBYSHEV_DECAY_POWER,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "spin1_wide_window_exact_combinatorial_checks.csv",
        index=False,
    )


def _write_uniformity_audit(output: Path) -> None:
    text = """# Deformation uniformity audit

The thermodynamic proof is uniform on the compact interval
0.05 <= kappa/J <= 0.20 and therefore on the open compatible family
0.05 < kappa/J < 0.20 containing kappa_star/J=0.1.

- Exact tower continuation: already closed by the exchange-phase rule.
- Bounded local caging operators: already closed by the two-site/local Y constructions.
- Energy variance: the fixed-M coefficient is at most 2.1 J^2 L before an exponentially
  small momentum-character correction, uniformly on the interval.
- Raw window: Delta E=c|J|L^(3/4) therefore has a uniform O(L^(-1/2)) outside fraction.
- Local witness: Q_Y is kappa independent and tends to 1/3 in the resolved beta-zero state.
- Background concentration: the beta-zero fixed-(M,q) state is Hamiltonian independent;
  kappa enters only through the uniformly controlled window conditioning.

Verdict: deformation_stable_icqmbs_closed = true on
0.05 < kappa/J < 0.20. The symmetry-enhanced kappa=0 endpoint is not part of this
open-family theorem.
"""
    (output / "deformation_uniformity_audit.md").write_text(text, encoding="utf-8")


def _write_qdm_audit(output: Path) -> None:
    text = """# Square-QDM framework-layer audit

No new expensive QDM run is required to instantiate a missing exact framework layer.

The current QDM construction already supplies the complementary constrained-Hilbert-space
model realization below the thermodynamic gate: an exact repeated cage, bounded local
kinetic reductions on the tested strip construction, the scoped non-gauge checkerboard
direction, and fully resolved finite-size thermal diagnostics.

The pending raw 12x4 result remains useful finite-size strengthening, but it is not a
premise of the spin-1 thermodynamic theorem and it does not by itself establish the QDM
fixed-width thermodynamic classification.

QDM verdict:
- exact/model-construction framework role: closed;
- fixed-width literal thermodynamic ICQMBS classification: open;
- new exact calculation required for model instantiation: no.
"""
    (output / "qdm_framework_layer_audit.md").write_text(text, encoding="utf-8")


def _write_manuscript_handoff(output: Path) -> None:
    text = """# Manuscript claim-upgrade handoff

The claim level changes because the spin-1 compatible family now has an analytic
thermodynamic realization on a new admissible raw window. The existing L^(1/4)
finite-size data remain a stricter numerical diagnostic and must not be described as
the proof window.

## Claim-ledger updates

Proposed ledger changes:
- positive raw thermodynamic microcanonical witness: OPEN -> CLOSED (spin-1, wide window);
- positive raw-window entropy density: OPEN -> CLOSED for Delta E=c|J|L^(3/4);
- vanishing declared caged fraction: GATED -> CLOSED;
- all-fixed-bounded-region concentration: OPEN -> CLOSED;
- literal ICQMBS realization: OPEN -> CLOSED for spin-1;
- deformation-stable ICQMBS realization: OPEN -> CLOSED on 0.05<kappa/J<0.20;
- narrow Delta E=(J/2)L^(1/4) thermodynamic classification: remains unproved.

EDITORIAL_GUIDE.md is not present in the qlinks repository, so these are the exact
ledger edits to apply in the manuscript workspace rather than a silent repository edit.

## Manuscript-safe replacement sentences

Abstract:
The spin-1 compatible exchange family additionally provides a thermodynamic realization:
for an admissible raw window Delta E proportional to L^(3/4), the selected resolved
sector has entropy density log 3, the tower fraction vanishes, a bounded local witness
retains positive activity, and every fixed bounded local observable concentrates in the
background, uniformly on an open compatible deformation interval.

Introduction/model preview:
Beyond the finite-size narrow-window diagnostics, the spin-1 family admits an analytic
thermodynamic closure on the subextensive half-width Delta E=c|J|L^(3/4); this wider
proof window is distinct from the stricter L^(1/4) numerical protocol shown below.

Spin-1 claim boundary:
For even L at M=-2 and the tower momentum, antiunitary spectral reflection fixes the
tower at the beta-zero center and the resolved energy variance is O(L). Consequently
the L^(3/4) raw window contains asymptotically all of the resolved sector. Translation
resolution and fixed-M combinatorics then give a positive Y witness and O(1/L)
background variance for every fixed bounded region. These estimates are uniform for
0.05<kappa/J<0.20.

Conclusions:
The framework is therefore realized thermodynamically by the spin-1 family on an open
compatible deformation interval, while the square QDM remains a complementary
constrained realization whose fixed-width thermodynamic classification is left open.

Popular Summary:
In the spin-1 example, the special states remain locally distinguishable even though
the surrounding energy window contains exponentially many ordinary states; this can be
proved for a shrinking energy-density window and remains true throughout a continuous
family of compatible couplings.

Cover letter:
We now give an analytic model-level closure of the thermodynamic definition in the
spin-1 family. The proof uses a subextensive L^(3/4) raw window, not an extrapolation of
the narrower numerical window, and establishes positive entropy density, a vanishing
tower fraction, a positive bounded local witness, and concentration of arbitrary fixed
bounded local observables uniformly on an open compatible deformation interval.
"""
    (output / "manuscript_claim_upgrade_handoff.md").write_text(text, encoding="utf-8")


def run(output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)

    proof_source = ROOT / "experimental" / "PRX_MODEL_REALIZATION_CLOSURE.md"
    if not proof_source.is_file():
        raise FileNotFoundError(proof_source)
    (output / "spin1_wide_window_thermodynamic_proof.md").write_text(
        proof_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    matrix = _write_closure_matrix(output)
    bounds = _write_bounds(output)
    _write_exact_sequence_checks(output)
    _write_uniformity_audit(output)
    _write_qdm_audit(output)
    _write_manuscript_handoff(output)

    metadata = {
        "schema_version": 1,
        "git_sha": _git_sha_from_metadata(ROOT),
        "solver_free": True,
        "spectral_solver_launched": False,
        "new_large_L_diagonalization": False,
        "window_protocol": "wide_window_c1_eta0p25",
        "window_half_width": "c |J| L^(3/4)",
        "verdicts": matrix["verdicts"],
        "bounds_file": "spin1_wide_window_bounds.json",
        "proof_file": "spin1_wide_window_thermodynamic_proof.md",
        "qdm_12x4_required": False,
        "narrow_L_quarter_window_reclassified": False,
    }
    (output / "job_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(output),
        "metadata": metadata,
        "bounds": bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir is not None else default_output_dir()
    result = run(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
