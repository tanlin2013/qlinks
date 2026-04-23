# flake8: noqa
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from ortools.sat.python import cp_model


@dataclass(slots=True)
class CpModel:
    shape: Tuple[int, ...]


@dataclass(slots=True)
class PXPModel2D:
    shape: Tuple[int, int]
    periodic: bool = False
