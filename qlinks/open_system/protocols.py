"""Structural interfaces accepted by open-system construction workflows."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class CageStateRecordLike(Protocol):
    """Minimal cage-state record interface needed by Lindblad construction.

    Concrete cage-search records satisfy this protocol structurally.  Keeping
    the protocol here avoids coupling the open-system layer to the caging
    search implementation and also permits lightweight user-defined records.
    """

    @property
    def support(self) -> NDArray[np.int64]: ...

    @property
    def local_state(self) -> NDArray[np.complex128]: ...

    @property
    def full_state(self) -> NDArray[np.complex128] | None: ...

    @property
    def signature(self) -> tuple[int, int]: ...
