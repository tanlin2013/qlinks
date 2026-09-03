"""Regression tests for mapped-P0 integration before rendering."""

from __future__ import annotations

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


def test_prepare_and_render_builds_required_panel_data_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_integration(source: Path, output: Path) -> object:
        assert source == tmp_path
        assert output == tmp_path
        events.append("integrate")
        for name in render_p0._REQUIRED_RENDER_INPUTS:
            (tmp_path / name).write_text("placeholder\n", encoding="utf-8")
        return object()

    def fake_render(
        data: Path,
        *,
        use_tex: bool,
        allow_incomplete: bool,
    ) -> list[str]:
        assert data == tmp_path
        assert use_tex is False
        assert allow_incomplete is False
        assert all((tmp_path / name).is_file() for name in render_p0._REQUIRED_RENDER_INPUTS)
        events.append("render")
        return ["figures/spin1_xy_figure6.pdf"]

    monkeypatch.setattr(render_p0.integration, "run_integration", fake_integration)
    monkeypatch.setattr(render_p0.renderer, "render", fake_render)

    result = render_p0.prepare_and_render(
        tmp_path,
        use_tex=False,
        allow_incomplete=False,
    )

    assert events == ["integrate", "render"]
    assert result["rendered"] == ["figures/spin1_xy_figure6.pdf"]


def test_prepare_and_render_refuses_missing_standardized_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(render_p0.integration, "run_integration", lambda *_args: object())

    def forbidden_render(*_args, **_kwargs):
        raise AssertionError("renderer must not run without standardized panel inputs")

    monkeypatch.setattr(render_p0.renderer, "render", forbidden_render)

    with pytest.raises(RuntimeError, match="panel_a_scatter"):
        render_p0.prepare_and_render(
            tmp_path,
            use_tex=False,
            allow_incomplete=False,
        )
