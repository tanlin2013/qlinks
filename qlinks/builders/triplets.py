from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class SparseTripletBuffer:
    """List-backed COO triplet buffer for sparse assembly hot loops.

    Python list appends are very fast, but each builder previously owned its
    own rows/cols/data boilerplate.  This helper centralizes triplet storage and
    materialization while still allowing builders to bind ``rows.append``,
    ``cols.append``, and ``data.append`` locally inside hot loops.
    """

    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    data: list[complex] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of currently stored triplets."""
        return len(self.data)

    def append(self, row: int, col: int, value: complex) -> None:
        """Append one triplet.

        Builders should bind list append methods directly in very hot loops;
        this method is mainly useful for tests and lower-volume call sites.
        """
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(value)

    def validate(self) -> None:
        """Raise if the three triplet lists have inconsistent lengths."""
        if not (len(self.rows) == len(self.cols) == len(self.data)):
            raise ValueError("Sparse triplet lists must have the same length.")

    def index_arrays(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return row/column index arrays."""
        self.validate()
        return (
            np.asarray(self.rows, dtype=np.int64),
            np.asarray(self.cols, dtype=np.int64),
        )

    def data_array(self, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        """Return the data array with the requested dtype."""
        self.validate()
        return np.asarray(self.data, dtype=dtype)
