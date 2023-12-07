from dataclasses import dataclass, field
from typing import Tuple, Self

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

    links: npt.NDArray[np.int64]
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._df = pd.DataFrame(self.links, index=self.as_index(self.links), dtype=int)

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
    def index(self) -> npt.NDArray[np.int64 | np.float64]:
        return self._df.index.values

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, index: int | float) -> npt.NDArray[np.int64]:
        return self._df.loc[index].values

    def __hash__(self) -> int:
        return hash(self.links.tobytes())

    @staticmethod
    def as_index(links: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        if links.ndim != 2:
            raise InvalidArgumentError("links should be a 2D array.")
        return np.asarray(
            [int("".join(map(str, links[i, :])), 2) for i in range(links.shape[0])], dtype=object
        )

    def sort(self) -> None:
        self._df.sort_index(inplace=True)
        self.links = self._df.to_numpy()

    def to_parquet(self, *args, **kwargs) -> None:
        self._df.to_parquet(*args, **kwargs)

    @classmethod
    def from_parquet(cls, *args, **kwargs) -> Self:
        return cls(pd.read_parquet(*args, **kwargs).to_numpy())
