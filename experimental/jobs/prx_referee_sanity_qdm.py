"""Checkerboard-QDM Fig. 9(a) branch/symmetry part of the PRX referee sanity check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import helpers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qdm_checkerboard_symmetry as qdm_symmetry
import qdm_sec7_fixed_o1 as qdm_fixed
import scipy.linalg as la
import scipy.optimize as optimize
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.stats as stats

import qlinks.caging.analysis.spectral as spectral

QDM_LX = 8
QDM_LY = 4
QDM_REPEATS = 2
SYMMETRY_TOL = 1.0e-9
MATCH_TOL = 1.0e-8


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


def _validate_complete_candidate(
    directory: Path,
    *,
    hamiltonian: sp.csr_array,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path] | None:
    energies_path = directory / "energies.npy"
    vectors_path = _vector_path(directory)
    if not energies_path.is_file() or vectors_path is None:
        return None
    try:
        energies = np.load(energies_path, mmap_mode="r", allow_pickle=False)
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if energies.ndim != 1 or energies.size != dimension:
        return None
    if vectors.shape != (dimension, dimension):
        return None

    sample = np.unique(np.linspace(0, dimension - 1, min(12, dimension), dtype=np.int64))
    block = np.asarray(vectors[:, sample], dtype=np.complex128)
    gram = block.conj().T @ block
    orthogonality = float(np.linalg.norm(gram - np.eye(sample.size), ord=2))
    residuals = np.linalg.norm(
        hamiltonian @ block - block * np.asarray(energies[sample], dtype=float)[None, :],
        axis=0,
    )
    maximum_residual = float(np.max(residuals, initial=0.0))
    if orthogonality > 1.0e-6 or maximum_residual > 1.0e-6:
        return None

    metadata = _metadata(directory / "metadata.json")
    metadata["sample_orthogonality_residual"] = orthogonality
    metadata["sample_maximum_physical_residual"] = maximum_residual
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(vectors, dtype=np.complex128),
        metadata,
        directory,
    )


def _find_complete_spectrum(
    context: Any,
    roots: Iterable[Path],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path] | None:
    dimension = int(context.sector.sector_dimension)
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve(strict=False)
        if not root.is_dir():
            continue
        for energies_path in root.rglob("energies.npy"):
            directory = energies_path.parent
            if directory in seen:
                continue
            seen.add(directory)
            candidate = _validate_complete_candidate(
                directory,
                hamiltonian=context.h_sector,
                dimension=dimension,
            )
            if candidate is not None:
                return candidate
    return None


def _spectrum(
    context: Any,
    *,
    roots: tuple[Path, ...],
    allow_small_dense: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    cached = _find_complete_spectrum(context, roots)
    if cached is not None:
        energies, vectors, metadata, source = cached
        metadata["source_kind"] = "validated_complete_cache"
        return energies, vectors, metadata, str(source)

    dimension = int(context.sector.sector_dimension)
    expected = qdm_fixed.EXPECTED_SECTOR_DIMENSIONS[QDM_LX]
    if dimension != expected:
        raise RuntimeError(
            f"refusing dense fallback: expected QDM Lx=8 dimension {expected}, got {dimension}"
        )
    if not allow_small_dense:
        raise RuntimeError(
            "No complete Lx=8 checkerboard eigensystem was found and "
            "--allow-small-qdm-dense is disabled"
        )

    dense = np.asarray(context.h_sector.toarray(), dtype=np.complex128)
    energies, vectors = la.eigh(dense, check_finite=False, overwrite_a=True)
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(vectors, dtype=np.complex128),
        {
            "source_kind": "small_dense_fallback",
            "solver": "scipy.linalg.eigh",
            "full_spectrum": True,
            "sector_dimension": dimension,
            "Lx": QDM_LX,
            "Ly": QDM_LY,
            "phase": qdm_fixed.REPRESENTATIVE_PHASE,
        },
        "generated_in_memory:Lx8_dense",
    )


def _unitarity(operator: np.ndarray) -> float:
    identity = np.eye(operator.shape[0], dtype=np.complex128)
    return float(
        np.linalg.norm(operator.conj().T @ operator - identity) / math.sqrt(operator.shape[0])
    )


def _compress(operator: sp.spmatrix | sp.sparray, sector: Any) -> np.ndarray:
    basis = sector.basis
    compressed = basis.conj().T @ (operator @ basis)
    if sp.issparse(compressed):
        compressed = compressed.toarray()
    return np.asarray(compressed, dtype=np.complex128)


def _symmetry_audit(context: Any) -> dict[str, Any]:
    permutations = qdm_symmetry.checkerboard_positive_phase_permutations(
        context.model,
        context.basis,
        packed_index=context.packed_index,
        chunk_size=16384,
    )
    h_full = sp.csr_array(context.build.hamiltonian)
    h_sector = np.asarray(context.h_sector.toarray(), dtype=np.complex128)
    norm_full = max(float(sp.linalg.norm(h_full)), np.finfo(float).tiny)
    norm_sector = max(float(np.linalg.norm(h_sector)), np.finfo(float).tiny)
    dimension = int(h_sector.shape[0])
    identity = np.eye(dimension, dtype=np.complex128)

    unitary_rows: list[dict[str, Any]] = []
    hidden: list[str] = []
    compressed: dict[str, np.ndarray] = {}
    for name, permutation in permutations.items():
        operator = sp.csr_array(spectral.permutation_matrix(permutation))
        full_residual = float(sp.linalg.norm(operator @ h_full - h_full @ operator) / norm_full)
        sector_operator = _compress(operator, context.sector)
        compressed[name] = sector_operator
        preservation = _unitarity(sector_operator)
        character = complex(np.trace(sector_operator) / dimension)
        scalar = float(
            np.linalg.norm(sector_operator - character * identity) / math.sqrt(dimension)
        )
        sector_residual = math.nan
        if preservation <= 1.0e-7:
            transformed = sector_operator @ h_sector @ sector_operator.conj().T
            sector_residual = float(np.linalg.norm(transformed - h_sector) / norm_sector)
        exact_inside = bool(
            full_residual <= SYMMETRY_TOL
            and preservation <= SYMMETRY_TOL
            and np.isfinite(sector_residual)
            and sector_residual <= SYMMETRY_TOL
        )
        creates_block = bool(exact_inside and scalar > 1.0e-7)
        if creates_block:
            hidden.append(name)
        unitary_rows.append(
            {
                "name": name,
                "full_unitary_commutator_residual": full_residual,
                "selected_sector_unitarity_residual": preservation,
                "selected_sector_commutator_residual": sector_residual,
                "selected_sector_character_real": float(character.real),
                "selected_sector_character_imag": float(character.imag),
                "selected_sector_scalar_residual": scalar,
                "exact_symmetry_inside_selected_irrep": exact_inside,
                "creates_additional_block": creates_block,
            }
        )

    antiunitary_rows: list[dict[str, Any]] = []
    for name in ("Rx", "Ry", "Sx", "Sy", "C2"):
        operator = compressed[name]
        preservation = _unitarity(operator)
        residual = math.nan
        square = math.nan
        if preservation <= 1.0e-7:
            transformed = operator @ h_sector.conjugate() @ operator.conj().T
            residual = float(np.linalg.norm(transformed - h_sector) / norm_sector)
            square = float(
                np.linalg.norm(operator @ operator.conjugate() - identity) / math.sqrt(dimension)
            )
        antiunitary_rows.append(
            {
                "name": f"{name} K",
                "selected_sector_unitarity_residual": preservation,
                "commuting_residual": residual,
                "square_plus_one_residual": square,
            }
        )

    graph = sp.csr_array(context.h_sector.copy())
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data = (np.abs(graph.data) > 1.0e-12).astype(np.int8)
        graph.eliminate_zeros()
    n_components, labels = csgraph.connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return {
        "Lx": QDM_LX,
        "Ly": QDM_LY,
        "phase": qdm_fixed.REPRESENTATIVE_PHASE,
        "winding": [0, 0],
        "resolved_generators": list(context.resolved_sector.generator_names),
        "resolved_characters": [
            [float(value.real), float(value.imag)]
            for value in context.resolved_sector.generator_characters
        ],
        "point_group_little_group": context.resolved_sector.point_group_little_group,
        "unitary_candidate_rows": unitary_rows,
        "antiunitary_candidate_rows": antiunitary_rows,
        "hidden_commuting_unitaries": hidden,
        "hamiltonian_graph_components": int(n_components),
        "hamiltonian_graph_component_sizes": np.bincount(labels).astype(int).tolist(),
        "disconnected_exact_block_structure": bool(n_components > 1),
    }


def _expectation(operator: sp.spmatrix | sp.sparray, vectors: np.ndarray) -> np.ndarray:
    action = operator @ vectors
    return np.einsum("ij,ij->j", vectors.conj(), action).real.astype(np.float64)


def _source_population(data_dir: Path) -> pd.DataFrame | None:
    path = Path(data_dir) / "qdm_checkerboard_eth_scatter.csv"
    if not path.is_file():
        return None
    source = pd.read_csv(path)
    required = {"energy", "Q_A", "Q_Z"}
    if not required.issubset(source.columns):
        return None
    if "Lx" in source.columns:
        source = source[pd.to_numeric(source["Lx"], errors="coerce").eq(QDM_LX)]
    if "phase" in source.columns:
        phase = pd.to_numeric(source["phase"], errors="coerce").to_numpy(dtype=float)
        source = source[np.isclose(phase, qdm_fixed.REPRESENTATIVE_PHASE)]
    if "background_variant" in source.columns:
        raw = source[source["background_variant"].astype(str).eq("raw_target_excluded")]
        if not raw.empty:
            source = raw
    return source.reset_index(drop=True) if not source.empty else None


def _select_exported_population(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    source = _source_population(data_dir)
    if source is None or len(source) > len(frame):
        return frame.copy()

    columns = ("energy", "Q_A", "Q_Z")
    source_values = source.loc[:, columns].to_numpy(dtype=float)
    candidate_values = frame.loc[:, columns].to_numpy(dtype=float)
    scale = np.std(source_values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    cost = np.sum(
        ((source_values[:, None, :] - candidate_values[None, :, :]) / scale) ** 2,
        axis=2,
    )
    source_rows, candidate_rows = optimize.linear_sum_assignment(cost)
    if source_rows.size != len(source):
        return frame.copy()
    order = np.argsort(source_rows)
    selected = frame.iloc[candidate_rows[order]].copy().reset_index(drop=True)
    selected["source_row_index"] = source_rows[order]
    return selected


def _residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    degree = min(2, max(0, len(x) - 1))
    coefficients = np.polyfit(x, y, degree)
    return y - np.polyval(coefficients, x)


def _classify(frame: pd.DataFrame) -> dict[str, Any]:
    energy = frame["energy"].to_numpy(dtype=float)
    residuals = np.column_stack(
        [
            _residual(energy, frame["Q_A"].to_numpy(dtype=float)),
            _residual(energy, frame["Q_Z"].to_numpy(dtype=float)),
        ]
    )
    scale = np.std(residuals, axis=0)
    scale[scale <= 1.0e-14] = 1.0
    values = residuals / scale
    values -= values.mean(axis=0)
    _, _, vh = np.linalg.svd(values, full_matrices=False)
    score = values @ vh[0]
    order = np.argsort(score)
    sorted_score = score[order]
    n_states = len(score)
    lower = max(1, int(math.ceil(0.10 * n_states)))
    upper = min(n_states - 1, int(math.floor(0.90 * n_states)))
    candidates = np.arange(lower, upper)
    if candidates.size == 0:
        labels = np.zeros(n_states, dtype=int)
        return {
            "score": score,
            "labels": labels,
            "threshold": 0.0,
            "strength": 0.0,
            "pc_direction": vh[0].tolist(),
            "correlations": {},
        }
    split = int(candidates[np.argmax(np.diff(sorted_score)[candidates - 1])])
    threshold = float(0.5 * (sorted_score[split - 1] + sorted_score[split]))
    labels = (score > threshold).astype(int)
    gap = float(sorted_score[split] - sorted_score[split - 1])
    strength = gap / max(float(np.std(score)), np.finfo(float).tiny)
    return {
        "score": score,
        "labels": labels,
        "threshold": threshold,
        "strength": float(strength),
        "pc_direction": vh[0].tolist(),
    }


def _correlations(frame: pd.DataFrame, classifier: dict[str, Any]) -> dict[str, Any]:
    score = np.asarray(classifier["score"], dtype=float)
    labels = np.asarray(classifier["labels"], dtype=int)
    output: dict[str, Any] = {}
    for column in ("potential_energy", "kinetic_energy", "Q_A", "Q_Z", "energy"):
        values = frame[column].to_numpy(dtype=float)
        pearson = stats.pearsonr(score, values)
        spearman = stats.spearmanr(score, values)
        means = [float(np.mean(values[labels == branch])) for branch in (0, 1)]
        denominator = max(float(np.std(values)), np.finfo(float).tiny)
        output[column] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "branch_0_mean": means[0],
            "branch_1_mean": means[1],
            "branch_mean_difference_over_global_std": float((means[1] - means[0]) / denominator),
        }
    return output


def _scatter_table(
    context: Any,
    energies: np.ndarray,
    vectors: np.ndarray,
    *,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = helpers.projector_resolved_energy_basis(
        energies,
        vectors,
        context.tower[:, None],
        energy_tolerance=qdm_fixed.ENERGY_BLOCK_TOL,
        vector_tolerance=1.0e-9,
    )
    tower = np.asarray(resolved["is_exceptional"], dtype=bool)
    basis_vectors = np.asarray(resolved["basis"][:, ~tower], dtype=np.complex128)
    values = np.asarray(resolved["energies"][~tower], dtype=np.float64)
    q_a = _expectation(context.projected_q["A"], basis_vectors)
    q_z = _expectation(context.projected_q["Z"], basis_vectors)

    potential_full = sp.diags(
        np.asarray(context.build.hamiltonian.diagonal(), dtype=np.complex128),
        format="csr",
    )
    sector_basis = context.sector.basis
    potential = sp.csr_array(sector_basis.conj().T @ (potential_full @ sector_basis))
    potential_values = _expectation(potential, basis_vectors)
    frame = pd.DataFrame(
        {
            "resolved_basis_index": np.arange(len(values), dtype=int),
            "energy": values,
            "energy_density": values / float(QDM_LX * QDM_LY),
            "Q_A": q_a,
            "Q_Z": q_z,
            "total_flippability": potential_values,
            "potential_energy": potential_values,
            "kinetic_energy": values - potential_values,
            "is_tower_state": False,
            "background_variant": "raw_target_excluded",
        }
    )
    frame = _select_exported_population(frame, data_dir)
    classifier = _classify(frame)
    frame["eigen_index"] = np.arange(len(frame), dtype=int)
    frame["branch_score"] = classifier["score"]
    frame["branch_label"] = classifier["labels"]
    classifier["correlations"] = _correlations(frame, classifier)
    return frame, classifier


def _provenance(frame: pd.DataFrame, data_dir: Path) -> dict[str, Any]:
    path = Path(data_dir) / "qdm_checkerboard_eth_scatter.csv"
    source = _source_population(data_dir)
    if source is None:
        return {"status": "source_scatter_missing_or_incompatible", "path": str(path)}
    result: dict[str, Any] = {
        "status": "checked",
        "path": str(path),
        "source_rows": int(len(source)),
        "reconstructed_rows": int(len(frame)),
        "all_Lx_8": True,
        "all_phase_0p05": True,
    }
    if len(source) != len(frame):
        result["population_matches"] = False
        return result
    columns = ["energy", "Q_A", "Q_Z"]
    lhs = source[columns].to_numpy(dtype=float)
    rhs = frame[columns].to_numpy(dtype=float)
    difference = np.abs(lhs - rhs)
    result["max_abs_difference"] = float(np.max(difference, initial=0.0))
    result["population_matches"] = bool(np.allclose(lhs, rhs, rtol=MATCH_TOL, atol=MATCH_TOL))
    return result


def _write_audit(
    output: Path,
    audit: dict[str, Any],
    provenance: dict[str, Any],
    classifier: dict[str, Any],
    verdict: str,
) -> None:
    lines = [
        "# Checkerboard-QDM Fig. 9(a) symmetry audit",
        "",
        "The selected raw sector is Lx x Ly = 8 x 4, winding (0,0), phi=0.05,",
        "Tdiag=Tx Ty character i (kdiag=pi/2), and Ty^2 character +1.",
        "",
        f"Track-B verdict: **{verdict}**.",
        f"Implementation little group: `{audit['point_group_little_group']}`.",
        f"Hamiltonian-graph components: {audit['hamiltonian_graph_components']}.",
        f"Hidden commuting unitaries: {audit['hidden_commuting_unitaries'] or 'none'}.",
        "",
        "| operation | full-H commute | sector preservation | sector-H commute | "
        "scalar residual | block |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in audit["unitary_candidate_rows"]:
        lines.append(
            f"| {row['name']} | {row['full_unitary_commutator_residual']:.3e} | "
            f"{row['selected_sector_unitarity_residual']:.3e} | "
            f"{row['selected_sector_commutator_residual']:.3e} | "
            f"{row['selected_sector_scalar_residual']:.3e} | "
            f"{row['creates_additional_block']} |"
        )
    lines += [
        "",
        "## Provenance check",
        "",
        "```json",
        json.dumps(provenance, indent=2, sort_keys=True),
        "```",
        "",
        "## Branch classifier",
        "",
        f"Residual-PCA branch gap / score std: {classifier['strength']:.6g}.",
        "The branch label is a finite-size classifier, not an assumed quantum number.",
        "The eigenstate table contains total flippability, potential/kinetic expectations,",
        "and branch labels; correlations are in `qdm_fig9a_branch_correlations.json`.",
    ]
    (output / "qdm_fig9a_symmetry_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot(output: Path, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(helpers.PRX_TEXT_WIDTH, 4.6), sharex=True)
    color = frame["total_flippability"].to_numpy(dtype=float)
    for axis, witness in zip(axes, ("Q_A", "Q_Z"), strict=True):
        scatter = axis.scatter(
            frame["energy_density"],
            frame[witness],
            c=color,
            s=10,
            alpha=0.7,
        )
        axis.grid(alpha=0.18)
        fig.colorbar(scatter, ax=axis, label="total flippability")
    axes[0].set_ylabel(r"$\langle \widehat Q_R^A\rangle_n$")
    axes[1].set_ylabel(r"$\langle \widehat Q_R^Z\rangle_n$")
    axes[1].set_xlabel(r"Energy density $e=E/(4L_x)$")
    axes[0].set_title("Finite-size branch diagnostic")
    fig.tight_layout()
    fig.savefig(output / "qdm_fig9a_branch_diagnostic.pdf")
    plt.close(fig)


def analyze_qdm(
    output: Path,
    *,
    data_dir: Path,
    cache_root: Path,
    allow_small_dense: bool,
) -> dict[str, Any]:
    reference = qdm_fixed.recover_reference_geometry()
    context = qdm_fixed.build_context(
        reference=reference,
        repeats=QDM_REPEATS,
        phase=qdm_fixed.REPRESENTATIVE_PHASE,
        symmetry_chunk_size=16384,
    )
    energies, vectors, metadata, source = _spectrum(
        context,
        roots=(Path(data_dir), Path(cache_root)),
        allow_small_dense=allow_small_dense,
    )
    audit = _symmetry_audit(context)
    frame, classifier = _scatter_table(context, energies, vectors, data_dir=Path(data_dir))
    provenance = _provenance(frame, Path(data_dir))
    frame.to_csv(output / "qdm_fig9a_branch_audit.csv", index=False)
    serializable = {
        key: value for key, value in classifier.items() if key not in {"score", "labels"}
    }
    (output / "qdm_fig9a_branch_correlations.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "qdm_fig9a_provenance_check.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    disconnected = bool(audit["disconnected_exact_block_structure"])
    hidden = bool(audit["hidden_commuting_unitaries"])
    population_matches = bool(provenance.get("population_matches", False))
    if disconnected:
        verdict = "disconnected exact block structure found — thermal data require re-resolution"
    elif hidden:
        verdict = "hidden exact symmetry found — current fully resolved wording requires correction"
    elif not population_matches:
        verdict = "inconclusive — exact Fig. 9(a) plotted population could not be reproduced"
    else:
        verdict = "no further exact symmetry found — branches are finite-size observable structure"

    _write_audit(output, audit, provenance, classifier, verdict)
    _plot(output, frame)
    return {
        "verdict": verdict,
        "hidden_symmetry_found": hidden,
        "disconnected_block_found": disconnected,
        "recompute_required": bool(hidden or disconnected),
        "source": source,
        "source_kind": metadata.get("source_kind", "unknown"),
        "small_dense_fallback_used": metadata.get("source_kind") == "small_dense_fallback",
        "provenance": provenance,
    }
