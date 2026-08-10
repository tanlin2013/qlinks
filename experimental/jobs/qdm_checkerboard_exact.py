"""Experimental compatibility import for the exact checkerboard cage theorem.

The algebraic certificate itself lives in :mod:`qlinks.caging.checkerboard_exact`
because it is a compact deterministic caging invariant.  Evidence workflow code
imports it here only to preserve notebook compatibility.
"""

from qlinks.caging.checkerboard_exact import (
    CheckerboardExactPeriodicProductCertificate,
    certify_checkerboard_periodic_product_exact,
)

__all__ = [
    "CheckerboardExactPeriodicProductCertificate",
    "certify_checkerboard_periodic_product_exact",
]
