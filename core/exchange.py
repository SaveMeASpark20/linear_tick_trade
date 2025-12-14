
from account import Account
from decimal import Decimal
from dataclass import Trade, Position
from abc import abstractmethod
from typing import Optional
from account import TestAccount

class Exchange(Account):
    """Abstract base class representing a tradig exchange/broker"""

    @abstractmethod
    def market_order(self, sym: str, signed_qty: Decimal, price: Decimal) -> Trade:
        pass

    @abstractmethod
    def limit_order(self, sym: str, signed_qty: Decimal, price: Decimal, post_only: bool) -> Optional[Trade]:
        """Execute a limit order and return a Trade if it crosses book."""
        pass


class TestExchange(Exchange):
    _account : TestAccount # type hint only

    def __init__(self, account: TestAccount):
        self._account = account

    def market_order(self, sym : str, signed_qty: Decimal, price: Decimal) -> "Trade" :
        trade = self._update_position(sym, signed_qty, price)
        self._account._balance += trade.pnl
        # account += pnl
        self._account._trade.append(trade)
        #append trade to _account.trade
        return trade
        #return trade
        
    def _update_position(self, sym: str, signed_qty: Decimal, price: Decimal) -> Trade :
        #pop yung old position
        #get the pnl if there's an old position
        #add a position if there's no
        #return Trade

        position = self._account._position.pop(sym, None)
        pnl = Decimal(0.0)
        if position is not None:
            entry_val = position.price * position.signed_qty
            exit_val = price * position.signed_qty
            pnl = exit_val  - entry_val
        else:
            self._account._position[sym] = Position(sym, signed_qty, price)
        
        return Trade(sym, signed_qty, price, pnl)
    

    def balance(self) -> Decimal:
        return self._account.balance()
    
    def get_position(self, sym) -> Optional[Trade]:
        return self._account.get_position()
    
    
    def limit_order(self) -> Decimal:
        raise Exception("not yet implemented")


    def __repr__(self):
        return f"TestExchange(balance={self.balance()}, position={self._account._position}, trades={self._account.balance()})"

    

    


    


