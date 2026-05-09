#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import tomllib

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_tool_versions() -> dict[str, str]:
    path = ROOT / ".tool-versions"
    versions: dict[str, str] = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"Invalid .tool-versions line: {raw_line!r}")

        versions[parts[0]] = parts[1]

    for required in ("python", "poetry"):
        if required not in versions:
            raise SystemExit(f"Missing {required!r} in .tool-versions")

    return versions


def read_requires_python() -> str:
    path = ROOT / "pyproject.toml"
    data = tomllib.loads(path.read_text())

    try:
        return data["project"]["requires-python"]
    except KeyError as exc:
        raise SystemExit("Missing [project].requires-python in pyproject.toml") from exc


def python_minor_candidates() -> list[str]:
    # Keep this finite and explicit because GitHub Actions cannot test
    # arbitrary future Python versions before setup-python supports them.
    return [f"3.{minor}" for minor in range(8, 18)]


def matrix_from_requires_python(spec: str) -> list[str]:
    specifier = SpecifierSet(spec)
    candidates = python_minor_candidates()

    matrix = [
        version
        for version in candidates
        if Version(version) in specifier
    ]

    if not matrix:
        raise SystemExit(f"No Python versions matched requires-python={spec!r}")

    return matrix


def write_github_output(outputs: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")

    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            for key, value in outputs.items():
                fh.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")


def main() -> None:
    tool_versions = read_tool_versions()
    requires_python = read_requires_python()

    default_python = tool_versions["python"]
    poetry_version = tool_versions["poetry"]
    python_matrix = matrix_from_requires_python(requires_python)

    if Version(default_python) not in SpecifierSet(requires_python):
        raise SystemExit(
            f".tool-versions Python {default_python!r} is outside "
            f"pyproject.toml requires-python={requires_python!r}. "
            f"Computed matrix: {python_matrix}"
        )

    write_github_output(
        {
            "default-python": default_python,
            "poetry-version": poetry_version,
            "python-matrix": json.dumps(python_matrix),
        }
    )


if __name__ == "__main__":
    main()
