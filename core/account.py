from abc import ABC, abstractmethod
from decimal import Decimal
from dataclass import Position, Trade
from typing import Optional, List, Dict

class Account(ABC):
    @abstractmethod
    def balance(self) -> Decimal:
        pass

    @abstractmethod
    def get_position(self, sym : str) -> Optional[Position]:
        pass


class TestAccount(Account):

    def __init__(self, _balance: Decimal) -> None:
        self._balance = _balance
        self._position :  Dict[str, Position] = {}
        self._trade : List[Trade] = []

    def balance(self) -> Decimal:
        return self._balance
    
    def get_position(self, sym: str) -> Optional[Position]:
        return self._position.get(sym)
    
    def __repr__(self) -> str:
        return f"TestAccount(balance={self._balance}, position={self._position}, trade={self._trade})"