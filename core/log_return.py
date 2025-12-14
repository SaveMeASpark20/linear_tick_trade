from tick import Tick
from type import T
from deque_window import DequeWindow
from typing import Optional
import numpy as np

class Log_Return(Tick[T, Optional[T]]):

    def __init__(self):
        self._window = DequeWindow(2)
    
    def on_tick(self, val) -> Optional[T]:
        self._window.on_tick(val)
        if self._window.is_full():
            return np.log(self._window._data[1] / self._window._data[0])
        else:
            return None

    def __repr__(self) -> str :
        cls_name = self.__class__.__name__
        return f"{cls_name}(window={self._window})"
