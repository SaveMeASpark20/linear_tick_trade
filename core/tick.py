from abc import ABC, abstractmethod
from type import T, R
from typing import Generic

class Tick(ABC, Generic[T, R]):
    @abstractmethod
    def on_tick(self, value : T) -> R:
        pass