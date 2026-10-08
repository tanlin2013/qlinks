"""Spin-1 half-spectrum level-statistics part of the PRX referee sanity check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import helpers
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import spin1_exchange_convention as convention
import spin1_sec6_common_windows as cache

import qlinks.caging.analysis.spectral as spectral
import spin1_sec6_provisioning as core

TOTAL_SZ = -2
MANDATORY_KAPPA = 0.10
SYMMETRY_TOL = 1.0e-9
DEGENERACY_TOL = 1.0e-10
BOOTSTRAP_SEED = 20261008


def _metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _vector_path(directory: Path) -> Path | None:
    for name in ("vectors.npy", "eigenvectors.npy"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _looks_complete(directory: Path) -> bool:
    energies_path = directory / "energies.npy"
    vectors_path = _vector_path(directory)
    if not energies_path.is_file() or vectors_path is None:
        return False
    try:
        energies = np.load(energies_path, mmap_mode="r", allow_pickle=False)
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    metadata = _metadata(directory / "metadata.json")
    dimension = int(vectors.shape[0]) if vectors.ndim == 2 else -1
    declared = int(metadata.get("sector_dimension", dimension))
    returned = int(metadata.get("returned_eigenpairs", energies.size))
    return bool(
        vectors.ndim == 2
        and energies.ndim == 1
        and vectors.shape == (dimension, dimension)
        and energies.size == dimension
        and declared == dimension
        and returned == dimension
    )


def _complete_spectrum(
    roots: tuple[Path, ...],
    *,
    length: int,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path] | None:
    candidates = cache.discover_checkpoint_directories(
        roots,
        length=int(length),
        kappa_over_j=float(kappa),
    )
    complete = [directory for directory in candidates if _looks_complete(directory)]
    if not complete:
        return None

    context = core._point_context(length=int(length), kappa_over_j=float(kappa))
    for directory in complete:
        try:
            energies, vectors, metadata = cache.validate_cached_spectrum(
                directory,
                length=int(length),
                kappa_over_j=float(kappa),
                context=context,
                sample_vectors=12,
            )
        except cache.CachedSpectrumUnavailableError:
            continue
        dimension = int(context["h_sector"].shape[0])
        if energies.size != dimension or vectors.shape != (dimension, dimension):
            continue
        return (
            np.asarray(energies, dtype=np.float64),
            np.asarray(vectors, dtype=np.complex128),
            dict(metadata),
            Path(directory),
        )
    return None


def _unitarity(operator: np.ndarray) -> float:
    identity = np.eye(operator.shape[0], dtype=np.complex128)
    return float(
        np.linalg.norm(operator.conj().T @ operator - identity) / math.sqrt(operator.shape[0])
    )


def _symmetry_audit(*, length: int, kappa: float) -> dict[str, Any]:
    context = core._point_context(length=int(length), kappa_over_j=float(kappa))
    hamiltonian = np.asarray(context["h_sector"].toarray(), dtype=np.complex128)
    configs = np.asarray(context["configs"], dtype=np.int64)
    sector = context["sector"]
    dimension = int(hamiltonian.shape[0])
    norm = max(float(np.linalg.norm(hamiltonian)), np.finfo(float).tiny)

    variables = np.mod(-np.arange(int(length), dtype=np.int64), int(length))
    permutation = spectral.basis_permutation_from_variable_permutation(configs, variables)
    inversion = np.asarray(
        spectral.project_operator_to_sector(
            spectral.permutation_matrix(permutation),
            sector,
        ),
        dtype=np.complex128,
    )
    even = np.arange(0, int(length), 2, dtype=np.int64)
    exponent = np.sum(configs[:, even] + 1, axis=1, dtype=np.int64)
    ca_full = sp.diags(np.where(exponent % 2 == 0, 1.0, -1.0), format="csr")
    ca = np.asarray(
        spectral.project_operator_to_sector(ca_full, sector),
        dtype=np.complex128,
    )

    unitary_rows: list[dict[str, Any]] = []
    for name, operator in (("inversion", inversion), ("C_A", ca)):
        unitary_rows.append(
            {
                "name": name,
                "unitarity_residual": _unitarity(operator),
                "commute_residual": float(
                    np.linalg.norm(operator @ hamiltonian @ operator.conj().T - hamiltonian) / norm
                ),
                "anticommute_residual": float(
                    np.linalg.norm(operator @ hamiltonian @ operator.conj().T + hamiltonian) / norm
                ),
            }
        )

    identity = np.eye(dimension, dtype=np.complex128)
    antiunitary_rows: list[dict[str, Any]] = []
    for name, unitary in (
        ("K", identity),
        ("inversion K", inversion),
        ("C_A K", ca),
        ("inversion C_A K", inversion @ ca),
    ):
        transformed = unitary @ hamiltonian.conjugate() @ unitary.conj().T
        square = unitary @ unitary.conjugate()
        antiunitary_rows.append(
            {
                "name": name,
                "commute_residual": float(np.linalg.norm(transformed - hamiltonian) / norm),
                "reflection_residual": float(np.linalg.norm(transformed + hamiltonian) / norm),
                "square_plus_one_residual": float(
                    np.linalg.norm(square - identity) / math.sqrt(dimension)
                ),
                "square_minus_one_residual": float(
                    np.linalg.norm(square + identity) / math.sqrt(dimension)
                ),
            }
        )

    hidden = [
        row["name"]
        for row in unitary_rows
        if row["unitarity_residual"] <= SYMMETRY_TOL and row["commute_residual"] <= SYMMETRY_TOL
    ]
    plus = [
        row["name"]
        for row in antiunitary_rows
        if row["commute_residual"] <= SYMMETRY_TOL
        and row["square_plus_one_residual"] <= SYMMETRY_TOL
    ]
    minus = [
        row["name"]
        for row in antiunitary_rows
        if row["commute_residual"] <= SYMMETRY_TOL
        and row["square_minus_one_residual"] <= SYMMETRY_TOL
    ]
    if hidden:
        rmt_class = "unresolved_unitary_block"
    elif minus:
        rmt_class = "gse_or_symplectic"
    elif plus:
        rmt_class = "goe"
    else:
        rmt_class = "gue"
    return {
        "L": int(length),
        "kappa_over_J": float(kappa),
        "M": TOTAL_SZ,
        "momentum_index": int(context["momentum_index"]),
        "sector_dimension": dimension,
        "unitary_candidates": unitary_rows,
        "antiunitary_candidates": antiunitary_rows,
        "hidden_commuting_unitaries": hidden,
        "rmt_class": rmt_class,
    }


def _bootstrap(ratios: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(ratios, size=(int(samples), ratios.size), replace=True).mean(axis=1)
    return {
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "bootstrap_p16": float(np.quantile(draws, 0.16)),
        "bootstrap_p84": float(np.quantile(draws, 0.84)),
    }


def _expected_mean(name: str) -> float:
    return {"goe": 0.5307, "gue": 0.5996}.get(name, math.nan)


def _level_rows(
    energies: np.ndarray,
    *,
    length: int,
    kappa: float,
    rmt_class: str,
    source: Path,
    metadata: dict[str, Any],
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    values = np.sort(np.asarray(energies, dtype=np.float64))
    scale = max(float(np.max(np.abs(values), initial=1.0)), 1.0)
    zero_tolerance = max(DEGENERACY_TOL, 1.0e-12 * scale)
    halves = {
        "negative": values[values < -zero_tolerance],
        "positive": values[values > zero_tolerance],
    }
    rows: list[dict[str, Any]] = []
    ratios_by_half: dict[str, np.ndarray] = {}
    for half, levels in halves.items():
        if levels.size < 3:
            continue
        report = spectral.adjacent_gap_ratio_report(
            levels,
            trim_fraction=0.0,
            degeneracy_tolerance=DEGENERACY_TOL,
        )
        ratios = np.asarray(report.ratios, dtype=np.float64)
        ratios_by_half[half] = ratios
        expected = _expected_mean(rmt_class)
        rows.append(
            {
                "L": int(length),
                "kappa_over_J": float(kappa),
                "energy_half": half,
                "sector_dimension": int(values.size),
                "n_levels": int(levels.size),
                "n_spacings": int(levels.size - 1),
                "n_ratios": int(ratios.size),
                "energy_min": float(np.min(levels)),
                "energy_max": float(np.max(levels)),
                "mean_r": float(report.mean_ratio),
                **_bootstrap(
                    ratios,
                    samples=bootstrap_samples,
                    seed=(BOOTSTRAP_SEED + 100 * length + int(round(100 * kappa)) + len(half)),
                ),
                "expected_ensemble": rmt_class,
                "expected_mean_r": expected,
                "poisson_mean_r": float(report.expected_poisson),
                "distance_to_expected": (
                    float(abs(report.mean_ratio - expected)) if np.isfinite(expected) else math.nan
                ),
                "distance_to_poisson": float(abs(report.mean_ratio - report.expected_poisson)),
                "checkpoint_path": str(source),
                "exchange_convention": metadata.get(
                    convention.EXCHANGE_CONVENTION_METADATA_KEY,
                    convention.CURRENT_EXCHANGE_CONVENTION,
                ),
            }
        )
    return rows, ratios_by_half


def _theory(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 1001)
    if kind == "poisson":
        pdf = 2.0 / (1.0 + x) ** 2
    else:
        beta = 1.0 if kind == "goe" else 2.0
        pdf = (x + x**2) ** beta / (1.0 + x + x**2) ** (1.0 + 1.5 * beta)
        pdf /= np.trapezoid(pdf, x)
    cdf = np.zeros_like(x)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * np.diff(x))
    cdf /= cdf[-1]
    return x, pdf, cdf


def _plot(
    output: Path,
    frame: pd.DataFrame,
    ratios: dict[tuple[int, float], dict[str, np.ndarray]],
) -> None:
    with backend_pdf.PdfPages(output / "spin1_level_statistics.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(helpers.PRX_TEXT_WIDTH, 3.2))
        if not frame.empty:
            for (kappa, half), group in frame.groupby(["kappa_over_J", "energy_half"]):
                ax.errorbar(
                    group["L"],
                    group["mean_r"],
                    yerr=group["bootstrap_std"],
                    marker="o" if half == "positive" else "s",
                    label=rf"$\kappa/J={kappa:g}$, {half}",
                )
        reference_lines = (
            (0.38629, "Poisson", ":"),
            (0.5307, "GOE", "--"),
            (0.5996, "GUE", "-."),
        )
        for value, label, style in reference_lines:
            ax.axhline(value, ls=style, lw=0.9, label=label)
        ax.set(xlabel="System size $L$", ylabel=r"$\langle r\rangle$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.3, ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for (length, kappa), half_data in sorted(ratios.items()):
            subset = frame[
                (frame["L"] == length)
                & np.isclose(frame["kappa_over_J"].to_numpy(dtype=float), kappa)
            ]
            if subset.empty:
                continue
            wd = str(subset.iloc[0]["expected_ensemble"])
            wd = wd if wd in {"goe", "gue"} else "goe"
            xp, pp, cp = _theory("poisson")
            xw, pw, cw = _theory(wd)
            fig, axes = plt.subplots(1, 2, figsize=(helpers.PRX_TEXT_WIDTH, 2.8))
            for half, data in half_data.items():
                axes[0].hist(
                    data,
                    bins=np.linspace(0, 1, 21),
                    density=True,
                    histtype="step",
                    label=half,
                )
                ordered = np.sort(data)
                axes[1].step(
                    ordered,
                    np.arange(1, ordered.size + 1) / ordered.size,
                    where="post",
                    label=half,
                )
            axes[0].plot(xp, pp, ":", label="Poisson")
            axes[0].plot(xw, pw, "--", label=wd.upper())
            axes[1].plot(xp, cp, ":", label="Poisson")
            axes[1].plot(xw, cw, "--", label=wd.upper())
            axes[0].set(
                xlabel="$r$",
                ylabel="Density",
                title=rf"$L={length}$, $\kappa/J={kappa:g}$",
            )
            axes[1].set(xlabel="$r$", ylabel="CDF")
            for axis in axes:
                axis.set_xlim(0.0, 1.0)
                axis.grid(alpha=0.2)
                axis.legend(fontsize=6.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _write_audit(output: Path, audits: list[dict[str, Any]]) -> None:
    lines = [
        "# Spin-1 symmetry-class audit",
        "",
        "M=-2 and tower momentum are resolved before this audit. C_A K is tested both as",
        "a commuting antiunitary and as a spectral reflection; no RMT class is assumed a priori.",
        "",
    ]
    for audit in audits:
        lines += [
            f"## L={audit['L']}, kappa/J={audit['kappa_over_J']:g}",
            "",
            f"Selected RMT class: `{audit['rmt_class']}`.",
            "",
            "| unitary | unitarity | commute | anticommute |",
            "|---|---:|---:|---:|",
        ]
        for row in audit["unitary_candidates"]:
            lines.append(
                f"| {row['name']} | {row['unitarity_residual']:.3e} | "
                f"{row['commute_residual']:.3e} | "
                f"{row['anticommute_residual']:.3e} |"
            )
        lines += [
            "",
            "| antiunitary | commute | reflection | A^2=+1 | A^2=-1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in audit["antiunitary_candidates"]:
            lines.append(
                f"| {row['name']} | {row['commute_residual']:.3e} | "
                f"{row['reflection_residual']:.3e} | "
                f"{row['square_plus_one_residual']:.3e} | "
                f"{row['square_minus_one_residual']:.3e} |"
            )
        lines.append("")
    (output / "spin1_symmetry_class_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _verdict(frame: pd.DataFrame, audits: list[dict[str, Any]]) -> tuple[str, str]:
    if any(audit["hidden_commuting_unitaries"] for audit in audits):
        return "hidden symmetry/block mixing found", "unresolved_unitary_block"
    if frame.empty:
        return "ambiguous at accessible sizes", "undetermined"
    mandatory = frame[np.isclose(frame["kappa_over_J"].to_numpy(dtype=float), MANDATORY_KAPPA)]
    if mandatory.empty:
        return "ambiguous at accessible sizes", "undetermined"
    classes = set(mandatory["expected_ensemble"].astype(str))
    rmt_class = next(iter(classes)) if len(classes) == 1 else "mixed_or_undetermined"
    if rmt_class not in {"goe", "gue"}:
        return "ambiguous at accessible sizes", rmt_class

    largest = mandatory[mandatory["L"] == mandatory["L"].max()]
    closer = bool((largest["distance_to_expected"] < largest["distance_to_poisson"]).all())
    close = bool((largest["distance_to_expected"] < 0.08).all())
    halves = math.inf
    if len(largest) > 1:
        halves = float(largest["mean_r"].max() - largest["mean_r"].min())
    if closer and close and halves < 0.08:
        verdict = (
            "supports generic Wigner-Dyson/nonintegrable background"
            if mandatory["L"].nunique() >= 3
            else "finite-size but compatible with Wigner-Dyson"
        )
        return verdict, rmt_class
    poisson_closer = largest["distance_to_poisson"] + 0.03 < largest["distance_to_expected"]
    if bool(poisson_closer.all()):
        return "Poisson/integrable-like signature found", rmt_class
    return "ambiguous at accessible sizes", rmt_class


def analyze_spin1(
    output: Path,
    *,
    roots: tuple[Path, ...],
    lengths: tuple[int, ...],
    kappas: tuple[float, ...],
    bootstrap_samples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    ratio_data: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    missing_mandatory: list[int] = []
    for length in lengths:
        for kappa in kappas:
            spectrum = _complete_spectrum(roots, length=length, kappa=kappa)
            if spectrum is None:
                inventory.append(
                    {
                        "L": length,
                        "kappa_over_J": kappa,
                        "status": "CACHE_MISSING_OR_INCOMPLETE",
                    }
                )
                if math.isclose(kappa, MANDATORY_KAPPA, abs_tol=1.0e-12):
                    missing_mandatory.append(length)
                continue

            energies, _vectors, metadata, source = spectrum
            audit = _symmetry_audit(length=length, kappa=kappa)
            audits.append(audit)
            inventory.append(
                {
                    "L": length,
                    "kappa_over_J": kappa,
                    "status": "REUSED_COMPLETE",
                    "source": str(source),
                }
            )
            if audit["rmt_class"] == "unresolved_unitary_block":
                continue
            case_rows, case_ratios = _level_rows(
                energies,
                length=length,
                kappa=kappa,
                rmt_class=audit["rmt_class"],
                source=source,
                metadata=metadata,
                bootstrap_samples=bootstrap_samples,
            )
            rows.extend(case_rows)
            ratio_data[(length, kappa)] = case_ratios

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "spin1_level_statistics.csv", index=False)
    _write_audit(output, audits)
    _plot(output, frame, ratio_data)
    verdict, rmt_class = _verdict(frame, audits)
    return {
        "verdict": verdict,
        "rmt_class": rmt_class,
        "inventory": inventory,
        "missing_mandatory_lengths": sorted(set(missing_mandatory)),
    }
