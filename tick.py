from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Deque, Optional

from collections import deque
import numpy as np


T = TypeVar('T')
R = TypeVar('R')

class Tick(ABC, Generic[T, R]):
    @abstractmethod
    def on_tick(self, val: T) -> R:
        """Handle a new tick and optionally return a result."""
        pass

class DequeWindow(Tick[T, Optional[T]], Generic[T]):
    def __init__(self, n: int):
        self._data : Deque[T] = deque(maxlen=n) # the type is Generic Deque and the value is deque a double-ended queue you can push left push right

    def on_tick(self, val: T) -> Optional[T]:
        """Append a value and return the oldes value dropped (if any)."""
        dropped = None
        if self.is_full():
            dropped = self._data[0]
        self._data.append(val)
        return dropped
    
    def is_full(self) -> bool:
        return self.data.maxlen == len(self._data)
    
    def append_left(self, val: T) -> Optional[T]:
        dropped = None
        if self.is_full():
            dropped = self._data[-1]
        self._data.appendleft(val)
        return dropped
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self._data)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(capacity={self._data.maxlen}, values={list(self._data)})"
    

        


