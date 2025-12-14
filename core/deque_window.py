from tick import Tick
from type import T
from typing import Optional, Generic, Deque 
from collections import deque
import numpy as np

class DequeWindow(Tick[T, Optional[T]], Generic[T]):

    def __init__(self, val: int):
        self._data : Deque[T] = deque(maxlen=val)

    def on_tick(self, val: T) -> Optional[T]:
        dropped = None
        if self.is_full():
            dropped = self._data[0]
        self._data.append(val)
        return dropped

    def is_full(self) -> Optional[T]:
        return self._data.maxlen == len(self._data)
    
    def append_left(self, val: T) -> Optional[T]:
        dropped = None
        if self.is_full():
            dropped = self._data[-1]
        self._data.appendleft(val)
        return dropped

    def to_numpy(self) -> np.ndarray :
        return np.array(self._data)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(capacity={self._data.maxlen}, values={list(self._data)})"
    
