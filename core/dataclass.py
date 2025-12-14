from dataclasses import dataclass
from decimal import Decimal

def decimal_sign(d: Decimal) -> int:
    return 1 if d > Decimal(0) else -1

def is_long(x: Decimal) -> bool:
    return decimal_sign(x) > 0


@dataclass(frozen=True)
class Order:
    sym: str
    signed_qty : Decimal

    def __str__(self):
        sign = "LONG" if self.signed_qty > 0 else "SHORT"
        return f"Order({sign} {self.signed_qty} {self.sym})"

@dataclass
class Position:
    sym : str
    signed_qty : Decimal
    price : Decimal

    def close(self) -> 'Order':
        return Order(self.sym, -self.signed_qty)
    
    def is_long(self) -> bool:
        return is_long(self.signed_qty)
    
    def unrealized_pnl(self, current_price : Decimal) -> Decimal:
        entry_value = self.price * self.signed_qty
        exit_value = current_price * -self.signed_qty
        return entry_value + exit_value
    

@dataclass(frozen=True)
class Trade:
    sym : str
    signed_qty : Decimal
    price : Decimal
    pnl: Decimal

    def __str__(self) -> str:
        sign = 1 if is_long(self.signed_qty) else 0
        return f"Trade({sign}, {self.signed_qty} {self.sym}, {self.price}, {self.pnl})"
    
    def is_long(self) -> bool:
        return is_long(self.signed_qty)
