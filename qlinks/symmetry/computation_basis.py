from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from qlinks.exceptions import InvalidArgumentError


@dataclass(slots=True)
class ComputationBasis:
    """

    Args:
        links: The link data in shape (n_states, n_links).
    """

    links: npt.NDArray[int]
    _index: npt.NDArray[int] = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.links.ndim != 2:
            raise InvalidArgumentError("Computation basis should be a 2D array.")
        self.sort()
        self._index = np.apply_along_axis(
            lambda row: int("".join(map(str, row)), 2), axis=1, arr=self.links
        )
        self._df = pd.DataFrame(self.links, index=self.index, dtype=int)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.links.shape

    @property
    def n_states(self) -> int:
        return self.links.shape[0]

    @property
    def n_links(self) -> int:
        return self.links.shape[1]

    @property
    def index(self) -> npt.NDArray[int]:
        return self._index

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, item: int) -> npt.NDArray[int]:
        return self._df.loc[item].values

    def sort(self) -> None:
        self.links = self.links[self.index.argsort(), :]

