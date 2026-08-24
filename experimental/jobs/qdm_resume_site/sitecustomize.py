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

    _original = _large_strip.folded_spectrum_partial_spectrum
    if getattr(_original, "__name__", "") != "resumable_folded_spectrum_partial_spectrum":
        _large_strip.folded_spectrum_partial_spectrum = make_resumable_folded_solver(_original)
