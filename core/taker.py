from strategy import Strategy
from account import Account
from typing import List, Optional
from dataclass import Order, Position
from lags import Log_Return_Lags
from decimal import Decimal

import torch
import torch.nn as nn
import numpy as np

class BasicTakerStrategy(Strategy):
    def __init__(self,
                 sym : str,
                 model : nn.Module,
                 log_return_lags : Log_Return_Lags,
                 scale_factor : Decimal
                 ):
        
        self.sym = sym
        self.log_return_lags = log_return_lags
        self.model = model
        if scale_factor is None:
            scale_factor = Decimal(1.0)
        self.scale_factor = Decimal(scale_factor)

    def _signed_compound_trade_size(self, y_hat: torch.Tensor, cur_price : Decimal, account: Account, position: Optional[Position])-> Decimal:
        cur_balance = account.balance()
        dir_signal = np.sign(y_hat)
        unrealized_balance =  cur_balance + (position.unrealized_pnl(cur_price) if position else Decimal(0.0))
        qty = unrealized_balance / cur_price
        signed_qty = Decimal(dir_signal) * qty
        return signed_qty * self.scale_factor
        


    def _create_order(self, y_hat: torch.Tensor,price: Decimal, account: Account) -> List[Order]:
        position = account.get_position(self.sym)
        sign_trade_size = self._signed_compound_trade_size(y_hat.item(), price, account, position)
        open_order = Order(self.sym, sign_trade_size)
        if position is not None:
            close_order = Order(position.sym, -position.signed_qty)
            return [close_order, open_order]

        return [open_order]

    def on_tick(self, price: float, account: Account) -> List[Order]:
        X = self.log_return_lags.on_tick(price)
        if X is not None:
            with torch.no_grad():
                y_hat = self.model(X)
                orders = self._create_order(y_hat, Decimal(price), account)
                return orders
        return []
    
    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}=(sym={self.sym}, lags={self.log_return_lags}, models={self.model} )"