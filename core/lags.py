from tick import Tick
from type import T
from typing import Optional
from deque_window import DequeWindow
from log_return import Log_Return
import torch


class Log_Return_Lags(Tick[T, torch.Tensor]):
    def __init__(self, lags : int):
        self._lags = DequeWindow(lags)
        self._log_return = Log_Return()

    def on_tick(self, val) -> torch.Tensor | None:
        ret_log = self._log_return.on_tick(val)
        if ret_log is not None:
            self._lags.append_left(ret_log)
            return torch.tensor(self._lags.to_numpy(), dtype=torch.float32) if  self._lags.is_full() else None
        else:   
            return None

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(lags = {self._lags}, log_return = {self._log_return})"
    
