"""Regression tests for direct rendering from a completed mapped P0 run."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

_created_ipython_stub = "IPython.display" not in sys.modules
if _created_ipython_stub:
    ipython = types.ModuleType("IPython")
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *_args, **_kwargs: None
    ipython.display = ipython_display
    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = ipython_display

import spin1_exchange_convention_render_p0 as render_p0  # noqa: E402

if _created_ipython_stub:
    sys.modules.pop("IPython.display", None)
    sys.modules.pop("IPython", None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_migration(data: Path, *, source_run_id: str = "legacy-p0") -> None:
    records = []
    for name in render_p0._REQUIRED_RENDER_INPUTS:
        path = data / name
        path.write_text("placeholder\n", encoding="utf-8")
        records.append({"path": name, "derived_sha256": _sha256(path)})
    manifest = {
        "spin1_xy_exchange_convention": "J_over_2_ladder_v1",
        "source_run_id": source_run_id,
        "converted_files": records,
    }
    (data / "spin1_exchange_convention_migration_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_prepare_and_render_uses_completed_migration_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_migration(tmp_path)
    events: list[str] = []

    def fake_render(
        data: Path,
        *,
        use_tex: bool,
        allow_incomplete: bool,
    ) -> list[str]:
        assert data == tmp_path
        assert use_tex is False
        assert allow_incomplete is False
        events.append("render")
        return ["figures/spin1_xy_figure6.pdf"]

    monkeypatch.setattr(render_p0.renderer, "render", fake_render)

    result = render_p0.prepare_and_render(
        tmp_path,
        use_tex=False,
        allow_incomplete=False,
        source_run_id="legacy-p0",
    )

    assert events == ["render"]
    assert result["verified_render_inputs"] == list(render_p0._REQUIRED_RENDER_INPUTS)
    assert result["rendered"] == ["figures/spin1_xy_figure6.pdf"]


def test_prepare_and_render_refuses_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="migration manifest is missing"):
        render_p0.prepare_and_render(
            tmp_path,
            use_tex=False,
            allow_incomplete=False,
        )


def test_prepare_and_render_reports_all_missing_or_tampered_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_migration(tmp_path)
    missing = render_p0._REQUIRED_RENDER_INPUTS[0]
    tampered = render_p0._REQUIRED_RENDER_INPUTS[1]
    (tmp_path / missing).unlink()
    (tmp_path / tampered).write_text("tampered\n", encoding="utf-8")

    def forbidden_render(*_args, **_kwargs):
        raise AssertionError("renderer must not run after failed migration preflight")

    monkeypatch.setattr(render_p0.renderer, "render", forbidden_render)

    with pytest.raises(RuntimeError) as exc_info:
        render_p0.prepare_and_render(
            tmp_path,
            use_tex=False,
            allow_incomplete=False,
        )

    message = str(exc_info.value)
    assert f"missing mapped renderer input: {missing}" in message
    assert f"renderer input hash mismatch: {tampered}" in message


def test_prepare_and_render_refuses_source_run_mismatch(tmp_path: Path) -> None:
    _write_completed_migration(tmp_path, source_run_id="legacy-p0")

    with pytest.raises(RuntimeError, match="source-run mismatch"):
        render_p0.prepare_and_render(
            tmp_path,
            use_tex=False,
            allow_incomplete=False,
            source_run_id="wrong-source",
        )
