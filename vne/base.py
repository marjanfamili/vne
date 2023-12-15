from __future__ import annotations

import abc
import enum
from typing import List, Tuple

import numpy as np


class SpatialDims(enum.IntEnum):
    TWO = 2
    THREE = 3


class Datasource(abc.ABC):
    """Abstract datasource."""

    @abc.abstractmethod
    def __call__(self, model_id: str) -> np.ndarray:
        raise NotImplementedError

    def __iter__(self) -> Datasource:
        self._iter = 0
        return self

    def __next__(self) -> Tuple[str, np.ndarray]:
        if self._iter < len(self):
            model_id = self._keys[self._iter]
            self._iter += 1
            return model_id, self(model_id)
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> List[str]:
        return self._keys
