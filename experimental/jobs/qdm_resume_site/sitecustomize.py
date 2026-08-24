"""Opt-in interpreter hook for resumable square-QDM production spectra.

This directory is placed on ``PYTHONPATH`` only by the square-QDM evidence job
runner.  Interactive notebooks and ordinary qlinks imports are therefore not
modified.  The hook is deliberately tiny: all cache/solver logic lives in
``qdm_resumable_spectrum.py`` where it can be tested directly.
"""

from __future__ import annotations

import os

if os.environ.get("QLINKS_QDM_RESUMABLE_SPECTRUM", "0") == "1":
    import qdm_checkerboard_large_strip as _large_strip
    from qdm_resumable_spectrum import make_resumable_folded_solver

    import qlinks.caging as _caging
    from qlinks.caging.local_search import (
        LocalQDMCageSearchConfig,
        RobustQDMLocalCageSearchConfig,
        robust_qdm_local_cage_search,
    )

    # The Sec. VII notebook still imports these first-party local-search names
    # from ``qlinks.caging``.  The package refactor intentionally moved the
    # public local-search API to ``qlinks.caging.local_search``; do not restore
    # that broad compatibility surface in the package itself.  Keep this
    # migration bridge scoped to the evidence-job interpreter and remove it
    # once the notebook source is migrated to the new import path.
    for _name, _value in {
        "LocalQDMCageSearchConfig": LocalQDMCageSearchConfig,
        "RobustQDMLocalCageSearchConfig": RobustQDMLocalCageSearchConfig,
        "robust_qdm_local_cage_search": robust_qdm_local_cage_search,
    }.items():
        if not hasattr(_caging, _name):
            setattr(_caging, _name, _value)

    _original = _large_strip.folded_spectrum_partial_spectrum
    if getattr(_original, "__name__", "") != "resumable_folded_spectrum_partial_spectrum":
        _large_strip.folded_spectrum_partial_spectrum = make_resumable_folded_solver(_original)
