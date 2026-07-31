"""Utilities for running draft-evidence notebooks as detached batch jobs.

The evidence notebooks remain the canonical interactive provenance.  The job
scripts in this directory execute those notebooks with explicit parameters and
write all CSV/figure artifacts into a timestamped run directory, so long remote
runs are not tied to a browser or PyCharm notebook frontend.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

RUN_PROFILES = ("smoke", "known", "production")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing the ``qlinks`` package."""

    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "qlinks").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not locate qlinks repository root from {here}")


def utc_run_id(prefix: str) -> str:
    """Return a filesystem-safe timestamped run id."""

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def python_literal(value: Any) -> str:
    """Return a compact Python literal for notebook parameter injection."""

    if isinstance(value, Path):
        return f"Path({str(value)!r})"
    return repr(value)


def replace_assignment(source: str, name: str, value: Any) -> str:
    """Replace a simple top-level assignment in a notebook code cell."""

    literal = python_literal(value)
    pattern = re.compile(
        rf"^(?P<prefix>{re.escape(name)}\s*=\s*)" rf"(?P<rhs>.*?)(?P<comment>\s*#.*)?$",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        comment = match.group("comment") or ""
        return f"{match.group('prefix')}{literal}{comment}"

    replaced, count = pattern.subn(repl, source, count=1)
    if count == 0:
        raise ValueError(f"Could not find top-level assignment for {name!r}")
    return replaced


def patch_notebook_parameters(
    *,
    notebook_path: Path,
    output_path: Path,
    replacements: Mapping[str, Any],
    extra_header: str = "",
) -> None:
    """Copy ``notebook_path`` to ``output_path`` after replacing assignments."""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    normalize_notebook_cells(notebook)
    pending = dict(replacements)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        changed = source
        for name in tuple(pending):
            try:
                changed = replace_assignment(changed, name, pending[name])
            except ValueError:
                continue
            else:
                pending.pop(name)
        if changed != source:
            cell["source"] = changed.splitlines(keepends=True)

    if pending:
        missing = ", ".join(sorted(pending))
        raise ValueError(f"Could not patch notebook parameters: {missing}")

    if extra_header:
        header_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["injected-evidence-job-header"]},
            "outputs": [],
            "source": extra_header.splitlines(keepends=True),
        }
        notebook["cells"].insert(0, header_cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def collect_file_manifest(data_dir: Path) -> list[dict[str, Any]]:
    """Collect file sizes and light CSV schema metadata for a data directory."""

    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir).as_posix()
        entry: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix,
        }
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                    n_rows = sum(1 for _ in reader)
                entry.update({"rows": n_rows, "columns": len(header), "column_names": header})
            except Exception as exc:  # pragma: no cover - manifest should not fail the job
                entry["csv_error"] = repr(exc)
        rows.append(entry)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_metadata(repo_root: Path) -> dict[str, str | None]:
    """Return best-effort git metadata for provenance."""

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": run_git("status", "--short"),
    }


def build_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        choices=RUN_PROFILES,
        default=os.environ.get("QLINKS_EVIDENCE_PROFILE", "known"),
        help="Notebook run profile. Use production for the full remote batch run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output data directory. Defaults to experimental/data/evidence_jobs/<job>_<UTC>.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used when --data-dir is omitted.",
    )
    parser.add_argument(
        "--figure-formats",
        default=os.environ.get("QLINKS_EVIDENCE_FIGURE_FORMATS", "pdf,svg"),
        help="Comma-separated figure formats, or an empty string to skip figure file writes.",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Enable TeX-backed plotting in the notebook. Keep off for unattended numeric runs.",
    )
    parser.add_argument(
        "--stage",
        choices=("compute", "render", "all"),
        default=os.environ.get("QLINKS_EVIDENCE_STAGE", "all"),
        help=(
            "compute writes numerical tables without TeX-backed figure files; "
            "render loads a completed data directory and writes final PDF/SVG; "
            "all executes the notebook and saves figures in one pass."
        ),
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Completed numerical data directory used by --stage render.",
    )
    parser.add_argument(
        "--strict-claims",
        action="store_true",
        help="Fail the job when a mandatory evidence validation reports a "
        "provisional/failed status.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory receiving a copy of final figures and manifests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("QLINKS_EVIDENCE_TIMEOUT", "-1")),
        help="Per-cell timeout passed to nbconvert. -1 disables the timeout.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Only write the patched input notebook and metadata; do not execute it.",
    )
    return parser


def parse_figure_formats(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated positive-integer tuple from CLI/env input."""

    if raw is None:
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        return None
    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError(f"expected positive integers, got {raw!r}")
    return values


def parse_float_tuple(raw: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated finite floating-point tuple."""
    if raw is None:
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        return None
    values = tuple(float(part) for part in parts)
    if any(not __import__("math").isfinite(value) for value in values):
        raise ValueError(f"expected finite numbers, got {raw!r}")
    return values


def normalize_notebook_cells(notebook: dict[str, Any]) -> None:
    """Remove execution-only fields from non-code cells before nbconvert."""

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)
            continue
        cell.pop("outputs", None)
        cell.pop("execution_count", None)


def print_log_tail(log_path: Path, *, n_lines: int = 80) -> None:
    """Print the last lines of an nbconvert log after a failed notebook run."""

    if not log_path.is_file():
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    print("\n--- nbconvert.log tail ---", file=sys.stderr)
    for line in lines[-n_lines:]:
        print(line, file=sys.stderr)
    print("--- end nbconvert.log tail ---\n", file=sys.stderr)


def run_evidence_notebook(
    *,
    job_name: str,
    notebook_filename: str,
    assignment_overrides: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Execute a parameterized evidence notebook and return the data directory."""

    repo_root = find_repo_root()
    notebook_dir = repo_root / "experimental" / "notebooks"
    notebook_path = notebook_dir / notebook_filename
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)

    run_id = args.run_id or utc_run_id(job_name)
    default_data_dir = repo_root / "experimental" / "data" / "evidence_jobs" / run_id
    data_dir = (args.data_dir or default_data_dir).resolve()
    run_artifact_dir = data_dir / "run_artifacts"
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    figure_formats = parse_figure_formats(args.figure_formats)
    if args.stage == "compute":
        figure_formats = ()
    patched_notebook = run_artifact_dir / f"{job_name}_input.ipynb"
    executed_notebook = run_artifact_dir / f"{job_name}_executed.ipynb"
    log_path = run_artifact_dir / "nbconvert.log"

    replacements = {
        "RUN_PROFILE": args.profile,
        "USE_TEX": bool(args.use_tex and args.stage != "compute"),
        "FIGURE_FORMATS": figure_formats,
        "DATA_DIR": data_dir,
        **assignment_overrides,
    }
    header = f"""\
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir({str(notebook_dir)!r})
"""
    patch_notebook_parameters(
        notebook_path=notebook_path,
        output_path=patched_notebook,
        replacements=replacements,
        extra_header=header,
    )

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    metadata = {
        "job_name": job_name,
        "run_id": run_id,
        "profile": args.profile,
        "repo_root": str(repo_root),
        "notebook": str(notebook_path.relative_to(repo_root)),
        "data_dir": str(data_dir),
        "figure_formats": figure_formats,
        "use_tex": bool(args.use_tex and args.stage != "compute"),
        "stage": args.stage,
        "strict_claims": bool(args.strict_claims),
        "python": sys.version,
        "started_at_utc": started_at,
        "git": git_metadata(repo_root),
    }
    write_json(run_artifact_dir / "run_metadata.json", metadata)

    if args.no_execute:
        print(f"Wrote patched input notebook: {patched_notebook}")
        return data_dir

    env = os.environ.copy()
    pythonpath_parts = [str(repo_root), str(notebook_dir)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("MPLBACKEND", "Agg")
    # Keep unattended evidence jobs predictable inside shared Docker hosts.
    # Dense BLAS/LAPACK temporaries are usually the limiting resource; limiting
    # implicit thread pools avoids memory oversubscription and makes OOM
    # behaviour easier to diagnose. Users can override these via docker env.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(name, "1")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        str(executed_notebook.name),
        "--output-dir",
        str(run_artifact_dir),
        str(patched_notebook),
    ]
    print("Executing:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    print("Data directory:", data_dir, flush=True)
    print("Log file:", log_path, flush=True)

    return_code = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
            log.flush()
            result = subprocess.run(
                cmd,
                cwd=notebook_dir,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = result.returncode
        if result.returncode != 0:
            print_log_tail(log_path)
            raise subprocess.CalledProcessError(result.returncode, cmd)
    finally:
        finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        manifest = collect_file_manifest(data_dir)
        write_json(run_artifact_dir / "file_manifest.json", {"files": manifest})
        metadata.update(
            {
                "finished_at_utc": finished_at,
                "return_code": return_code,
                "file_count": len(manifest),
            }
        )
        write_json(run_artifact_dir / "run_metadata.json", metadata)

    print(f"Completed {job_name}. Data written to {data_dir}")
    print(f"Executed notebook: {executed_notebook}")
    print(f"Manifest: {run_artifact_dir / 'file_manifest.json'}")
    return data_dir


def run_evidence_renderer(
    *,
    job_name: str,
    renderer_filename: str,
    args: argparse.Namespace,
) -> Path:
    """Render final manuscript figures from an existing numerical data directory."""
    if args.source_data_dir is None:
        raise ValueError("--stage render requires --source-data-dir")
    repo_root = find_repo_root()
    source_data_dir = args.source_data_dir.resolve()
    if not source_data_dir.is_dir():
        raise FileNotFoundError(source_data_dir)
    renderer = repo_root / "experimental" / "jobs" / renderer_filename
    if not renderer.is_file():
        raise FileNotFoundError(renderer)
    figure_formats = parse_figure_formats(args.figure_formats)
    if not figure_formats:
        raise ValueError("render stage requires at least one --figure-formats entry")

    cmd = [
        sys.executable,
        str(renderer),
        "--data-dir",
        str(source_data_dir),
        "--figure-formats",
        ",".join(figure_formats),
    ]
    if args.use_tex:
        cmd.append("--use-tex")
    print("Rendering:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)

    manifest = collect_file_manifest(source_data_dir)
    run_artifact_dir = source_data_dir / "run_artifacts"
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_artifact_dir / "file_manifest.json", {"files": manifest})
    write_json(
        run_artifact_dir / "render_metadata.json",
        {
            "job_name": job_name,
            "stage": "render",
            "renderer": str(renderer.relative_to(repo_root)),
            "source_data_dir": str(source_data_dir),
            "figure_formats": figure_formats,
            "use_tex": bool(args.use_tex),
            "git": git_metadata(repo_root),
        },
    )

    if args.export_dir is not None:
        export_dir = args.export_dir.resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("figures", "figure_manifest.json", "figure_manifest.csv"):
            source = source_data_dir / relative
            if not source.exists():
                continue
            destination = export_dir / source.name
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    return source_data_dir
