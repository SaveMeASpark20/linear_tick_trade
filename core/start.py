from lags import Log_Return_Lags
from account import TestAccount
from taker import BasicTakerStrategy
from decimal import Decimal
from implementation import model
from exchange import TestExchange

linear_model = model

lags = Log_Return_Lags(3)
acc = TestAccount(Decimal(100.0))
strat = BasicTakerStrategy('BTCUSD', linear_model, lags, Decimal(1.0))

#simulate where we have a trained linear model
#that needs a 3 input and will give a 1 output
#100.0 -> lags -> log -> 
# log needs a 2 input to get a log return  ->  none / 100.0, 
# 100 will be just save
strat.on_tick(100.0, acc)
print(strat)

# 150.0 -> lags -> log -> log(150.0 / 100.0)
# lags[0.176] 
strat.on_tick(150.0, acc)
print(strat)

#160.0 -> lags -> log -> log(160.0 /150)
#lags[0.28, 0.176]
strat.on_tick(160.0, acc)
print(strat)

#160.0 -> lags -> log -> log(100/ 160)
#lags[-0.20, 0.28, 0.176] -> lags is full -> 
#return to_numpy(lags, dtype=torch.float32) -> 
#torch.no_grad() don't use the previous prediction 
#y_hat = self.model(X) predicting... ex. the prediction is y = 0.0567
#orders = create_order -> get the position -> (sym, signed_qty, price)
# postion -> access the Account position on particular symbol(self.sym) -> None
# signed_compound_trade_size -> 
# dir_signal  = np.sign(y_hat) 1.0 - ex. 1.0 or -1.0 
# unrealized_balance = cur_bal (100.0) + unrealized_pnl(0.0) 
# qty = unrealized_balance(100.0) / cur_price(100) = 1
# signed_qty = dir_signal(1.0) * qty(1.0) = 1.1
# return -> signed_qty(1.0) * scale_factor(1.0) = 1
# Open Order -> sym = BTCUSD, 1(return)
# return -> Order(1.0, 1.0, BTCUSD)
# position None -> return [open_order]
# return [orders]
orders = strat.on_tick(100.0, acc)

print(strat)
print(orders)

### Execute Order
exchange = TestExchange(acc)

# get the first order ->
# [[Order(sym='BTCUSD', signed_qty=Decimal('1'))]]
order = orders[0]

exchange.market_order(order.sym, order.signed_qty, 100)
print(exchange)

orders = strat.on_tick(115, acc)
print(orders) 

order = orders[1]
exchange.market_order(order.sym, order.signed_qty, 115)

print(exchange)

orders = strat.on_tick(150, acc)
print(orders)

order = orders[0]
exchange.market_order(order.sym, order.signed_qty, 150)
print(exchange)


orders = strat.on_tick(120, acc)
print(orders)

order = orders[1]
exchange.market_order(order.sym, order.signed_qty, 120)
print(exchange)




