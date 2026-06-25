#!/usr/bin/env python3
"""Validate release metadata before publishing artifacts."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG_VERSION_RE = re.compile(r"^v(?P<version>[0-9]+(?:\.[0-9]+){1,2}(?:[a-zA-Z0-9_.!+-]+)?)$")


def _project_metadata() -> tuple[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not name:
        raise SystemExit("Missing non-empty [project].name in pyproject.toml")
    if not isinstance(version, str) or not version:
        raise SystemExit("Missing non-empty [project].version in pyproject.toml")
    return name, version


def _validate_tag_version(project_version: str) -> None:
    ref = os.environ.get("GITHUB_REF", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")

    if not ref.startswith("refs/tags/"):
        print("Not a tag build; skipping tag/version consistency check.")
        return

    match = TAG_VERSION_RE.match(ref_name)
    if match is None:
        raise SystemExit(f"Release tags must look like 'v{project_version}', but got {ref_name!r}.")

    tag_version = match.group("version")
    if tag_version != project_version:
        raise SystemExit(
            "Release tag and project version disagree: "
            f"tag {ref_name!r} implies {tag_version!r}, "
            f"but pyproject.toml has {project_version!r}."
        )

    print(f"Tag/version check passed: {ref_name} matches project version {project_version}.")


def main() -> None:
    project_name, project_version = _project_metadata()
    print(f"Project metadata: {project_name} {project_version}")
    _validate_tag_version(project_version)


if __name__ == "__main__":
    main()
