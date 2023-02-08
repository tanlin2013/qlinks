import abc
from dataclasses import dataclass
from types import UnionType
from typing import TypeVar

Real: UnionType = int | float
AnySymmetry = TypeVar("AnySymmetry", bound="AbstractSymmetry")


@dataclass
class AbstractSymmetry(abc.ABC):
    quantum_number: Real
