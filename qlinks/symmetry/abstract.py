import abc
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

Real: TypeAlias = int | float
AnySymmetry = TypeVar("AnySymmetry", bound="AbstractSymmetry")


@dataclass
class AbstractSymmetry(abc.ABC):
    quantum_number: Real
