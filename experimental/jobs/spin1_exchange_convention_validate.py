#!/usr/bin/env python
"""Cheap physical validation of the permanent Spin-1 XY exchange convention.

This job contains no large-size eigensolver. It checks the local/model scaling,
Appendix-D five-site shell, the decorated-PBC counterexample, and (optionally)
a small dense-spectrum legacy-to-current mapping. Historical evidence is read
nowhere and is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    LEGACY_EXCHANGE_CONVENTION,
    LEGACY_TO_CURRENT_ENERGY_SCALE,
)

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.models import SpinOneXYChainModel, spin_one_xy_hxy_h3_imaginary_j2_model

TOLERANCE = 1.0e-10
J = 1.0
J3_OVER_J = 0.10
KAPPA_OVER_J = 0.10
SUMMARY_NAME = "spin1_xy_exchange_convention_validation.json"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _state_from_amplitudes(
    configs: np.ndarray,
    amplitudes: dict[tuple[int, ...], complex],
) -> np.ndarray:
    lookup = {
        tuple(int(value) for value in row): index for index, row in enumerate(np.asarray(configs))
    }
    state = np.zeros(len(configs), dtype=np.complex128)
    for config, amplitude in amplitudes.items():
        try:
            state[lookup[tuple(config)]] = complex(amplitude)
        except KeyError as exc:
            raise RuntimeError(
                f"analytic support configuration is absent from basis: {config}"
            ) from exc
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise RuntimeError("analytic validation state has zero norm")
    return state / norm


def _five_site_state(configs: np.ndarray, *, alternating: bool) -> np.ndarray:
    """Return the two 20-site-cycle modes in the L=5, M=-2 support."""

    amplitudes: dict[tuple[int, ...], complex] = {}
    for plus_site in range(5):
        for zero_site in range(5):
            if zero_site == plus_site:
                continue
            config = [-1] * 5
            config[plus_site] = 1
            config[zero_site] = 0
            distance = (zero_site - plus_site) % 5
            if alternating:
                sign = 1.0 if distance in (1, 3) else -1.0
            else:
                sign = 1.0 if distance in (1, 2) else -1.0
            amplitudes[tuple(config)] = sign
    if len(amplitudes) != 20:
        raise RuntimeError("five-site shell should contain exactly 20 support configurations")
    return _state_from_amplitudes(configs, amplitudes)


def _deleted_chain_l4_n1_state(configs: np.ndarray) -> np.ndarray:
    """Return the L=4,n=1 deleted-chain state used by the PBC counterexample."""

    amplitudes: dict[tuple[int, ...], complex] = {}
    for zero_site in (0, 2):
        r = zero_site + 1
        eta = (-1.0) ** ((r - 1) // 2)
        for plus_site in range(4):
            if plus_site == zero_site:
                continue
            s = plus_site + 1
            chi = (-1.0) ** s if s < r else (-1.0) ** (s - 1)
            config = [-1] * 4
            config[zero_site] = 0
            config[plus_site] = 1
            amplitudes[tuple(config)] = eta * chi
    if len(amplitudes) != 6:
        raise RuntimeError("decorated L=4,n=1 state should contain six configurations")
    return _state_from_amplitudes(configs, amplitudes)


def _residual_report(
    hamiltonian: sp.sparray,
    state: np.ndarray,
    *,
    expected_energy: float,
) -> dict[str, float]:
    action = np.asarray(hamiltonian @ state, dtype=np.complex128)
    support = np.abs(state) > TOLERANCE
    expectation = float(np.vdot(state, action).real)
    return {
        "expectation": expectation,
        "expected_energy": float(expected_energy),
        "eigenpair_residual": float(np.linalg.norm(action - expected_energy * state)),
        "boundary_residual": float(np.linalg.norm(action[~support])),
        "support_size": int(np.count_nonzero(support)),
    }


def _five_site_shell() -> dict[str, Any]:
    model = SpinOneXYChainModel(
        length=5,
        boundary_condition="periodic",
        j_xy=J,
        total_sz=-2,
    )
    build = model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = basis_configs_from_build_result(build)
    zero = _five_site_state(configs, alternating=False)
    minus_two = _five_site_state(configs, alternating=True)
    zero_report = _residual_report(build.hamiltonian, zero, expected_energy=0.0)
    minus_two_report = _residual_report(build.hamiltonian, minus_two, expected_energy=-2.0 * J)
    for label, report in (("S_0", zero_report), ("S_-2", minus_two_report)):
        if report["eigenpair_residual"] > TOLERANCE:
            raise RuntimeError(f"{label} five-site shell residual is too large")
        if report["boundary_residual"] > TOLERANCE:
            raise RuntimeError(f"{label} five-site boundary residual is too large")
    return {"S_0": zero_report, "S_-2": minus_two_report}


def _decorated_pbc_counterexample() -> dict[str, float]:
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=J,
        total_sz=-1,
    )
    build = model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = basis_configs_from_build_result(build)
    state = _deleted_chain_l4_n1_state(configs)
    residual = float(np.linalg.norm(build.hamiltonian @ state))
    expected = math.sqrt(2.0) * abs(J)
    if not math.isclose(residual, expected, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError(
            f"decorated-PBC residual mismatch: expected {expected:.12g}, got {residual:.12g}"
        )
    return {"normalized_residual": residual, "expected_residual": expected}


def _legacy_equivalent_model(*, length: int, d_z: float = 0.0) -> SpinOneXYChainModel:
    from qlinks.models import spin_one_xy_periodic_range_couplings

    extra = list(
        spin_one_xy_periodic_range_couplings(
            length=length,
            distance=3,
            coefficient=2.0 * J3_OVER_J * J,
        )
    ) + list(
        spin_one_xy_periodic_range_couplings(
            length=length,
            distance=2,
            coefficient=2.0j * KAPPA_OVER_J * J,
        )
    )
    return SpinOneXYChainModel(
        length=length,
        boundary_condition="periodic",
        j_xy=2.0 * J,
        total_sz=-2,
        d_z=float(d_z),
        extra_xy_couplings=tuple(extra),
    )


def _matrix_scaling_check(*, length: int, d_new: float = 0.0) -> dict[str, float | int]:
    current = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length,
        j=J,
        j3=J3_OVER_J * J,
        kappa=KAPPA_OVER_J * J,
        total_sz=-2,
        d_z=float(d_new),
    ).build(builder="optimized", basis_solver="dfs", sort_basis=True)
    legacy = _legacy_equivalent_model(length=length, d_z=2.0 * float(d_new)).build(
        builder="optimized", basis_solver="dfs", sort_basis=True
    )
    difference = sp.csr_array(current.hamiltonian) - LEGACY_TO_CURRENT_ENERGY_SCALE * sp.csr_array(
        legacy.hamiltonian
    )
    max_abs = float(np.max(np.abs(difference.data), initial=0.0))
    if max_abs > TOLERANCE:
        raise RuntimeError(f"L={length} Hamiltonian scaling residual is {max_abs:.3e}")
    return {
        "L": int(length),
        "sector_dimension": int(current.hamiltonian.shape[0]),
        "maximum_matrix_scaling_residual": max_abs,
        "D_over_J_current": float(d_new),
        "D_over_J_legacy_display": 2.0 * float(d_new),
    }


def _dense_spectrum_check(length: int) -> dict[str, float | int]:
    current = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length,
        j=J,
        j3=J3_OVER_J * J,
        kappa=KAPPA_OVER_J * J,
        total_sz=-2,
    ).build(builder="optimized", basis_solver="dfs", sort_basis=True)
    legacy = _legacy_equivalent_model(length=length).build(
        builder="optimized", basis_solver="dfs", sort_basis=True
    )
    old_h = np.asarray(sp.csr_array(legacy.hamiltonian).toarray(), dtype=np.complex128)
    old_energies, old_vectors = la.eigh(old_h, check_finite=False)
    new_h = sp.csr_array(current.hamiltonian)
    mapped_energies = LEGACY_TO_CURRENT_ENERGY_SCALE * old_energies
    residuals = np.linalg.norm(
        new_h @ old_vectors - old_vectors * mapped_energies[None, :],
        axis=0,
    )
    max_residual = float(np.max(residuals, initial=0.0))
    current_energies = la.eigvalsh(new_h.toarray(), check_finite=False)
    spectral_error = float(np.max(np.abs(current_energies - mapped_energies), initial=0.0))
    if max(max_residual, spectral_error) > 1.0e-9:
        raise RuntimeError(f"L={length} dense spectrum does not obey exact factor-1/2 mapping")
    return {
        "L": int(length),
        "sector_dimension": int(new_h.shape[0]),
        "maximum_reused_eigenvector_residual": max_residual,
        "maximum_eigenvalue_mapping_error": spectral_error,
        "eigenvector_reuse_overlap": 1.0,
    }


def run(*, output_dir: Path, dense_sizes: tuple[int, ...]) -> dict[str, Any]:
    output = Path(output_dir).resolve(strict=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "historical_exchange_convention": LEGACY_EXCHANGE_CONVENTION,
        "energy_mapping_factor": LEGACY_TO_CURRENT_ENERGY_SCALE,
        "five_site_shell": _five_site_shell(),
        "decorated_pbc_L4_n1": _decorated_pbc_counterexample(),
        "homogeneous_matrix_scaling": [_matrix_scaling_check(length=length) for length in (8, 10)],
        "finite_D_matrix_scaling": _matrix_scaling_check(length=8, d_new=0.315),
        "dense_spectrum_checks": [_dense_spectrum_check(length) for length in dense_sizes],
        "claim": (
            "cheap convention validation only; no large-size Sec. VI evidence was recomputed"
        ),
    }
    _atomic_write_json(output / SUMMARY_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dense-sizes",
        default="8",
        help="Comma-separated dense spot checks. L=8 is CI/desktop-cheap; add L=10 on the server.",
    )
    args = parser.parse_args()
    dense_sizes = tuple(
        int(token.strip()) for token in args.dense_sizes.split(",") if token.strip()
    )
    result = run(output_dir=args.output_dir, dense_sizes=dense_sizes)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
