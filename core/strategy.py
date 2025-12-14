from abc import ABC, abstractmethod
from typing import Optional, List
from dataclass import Order
from account import Account

class Strategy(ABC) :
    @abstractmethod
    def on_tick(self, price : float, account: Account) -> Optional[List]:
        pass