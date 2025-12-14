#data and analysis libraries
import polars  as pl            #Fast dataframes for financial  data
import numpy as np              #numerical computing library
from datetime  import datetime, timedelta #Date and time operations
import random

import torch                    #PyTorch framework
import torch.nn as nn           #Neural network modules
import torch.optim as optim     #Optimization algorithms
import researches                 #Model building and training  utilities
import importlib
importlib.reload(researches)

import altair as alt            #Interactive visualization library

import binance                  # Binance market data utilities
importlib.reload(binance)
print(researches.__file__)
import models


#helper
def show(df, n=10):
    print(df.head(n))
    print(df.schema)
### Strategy Development

#Load Model
import models

model = models.LinearModel(3)
#security alert
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)



# Load Time Series
sym = 'BTCUSDT'
time_interval = '12h'

ts = pl.read_csv(f"{sym}_{time_interval}_ohlc.csv", try_parse_dates=True).sort('datetime')


# Add target and features
forecast_horizon = 1
#we will use the 3 lags to predict the close log return
ts = researches.add_log_return_features(ts, 'close', forecast_horizon, max_no_lags=3)


test_size = 0.25
_, trades = researches.timeseries_split(ts, test_size)

#strategy Decision #1 Entry/Exit Signal
#q1. when do we get in? entry signal
#q2. when do we get out? exit signal

# 1.time based
# 2.predicate Base
# predicate example => we only want to trade if our y_hat is above or below threshold
# time based => each row represents a roundtrip trade. trade to close the position at end of interval. each row = 2 trades
### Model's Predictions
target = 'close_log_return'
features = [f'{target}_lag_1', f'{target}_lag_2', f'{target}_lag_3']
trades = researches.add_model_predictions(trades, model, features)
### add directional signal

trades = trades.with_columns(pl.col('y_hat').sign().alias('dir_signal'))

### calculcate trade log return
trades =  trades.with_columns(
    (pl.col('close_log_return') * pl.col('dir_signal')).alias('trade_log_return')
)

### calculate cumulative trade log return

trades = trades.with_columns(
    pl.col('trade_log_return').cum_sum().alias('cum_trade_log_return')
)

### Display equity curve(Log Space)
# plot_cum = researches.plot_column_matplotlib(trades, 'cum_trade_log_return')


### Strategy Decision #2: Trade Sizing

#1. Constant Trade Size
#2. Compounding Trade Size

#1.
capital = 100
ratio = 1.0 #means we will use 100% of our money, if we set 0.5 we use 50% of our money 
trade_value = ratio * capital

# entry trade value and exit trade value
trades = trades.with_columns(
    pl.lit(trade_value).alias('entry_trade_value'),
    (trade_value * pl.col('trade_log_return').exp()).alias('exit_trade_value'),
    (trade_value / pl.col('open')).alias('trade_qty'),
).with_columns(
    (pl.col('trade_qty') * pl.col('dir_signal')).alias('signed_trade_qty')
)


### Add trade Gross PnL
trades = trades.with_columns(
    (pl.col('exit_trade_value') - pl.col('entry_trade_value')).alias('trade_gross_pnl')
)



### Add Transaction Fee
taker_fee = binance.TAKER_FEE #0.000450
maker_fee = binance.MAKER_FEE #0.000450


# trades = trades.with_columns(
#     (pl.col('entry_trade_value') * taker_fee + pl.col('exit_trade_value') * taker_fee).alias('taker_fee'), # 100 * 0.000450 + 100.536332 * 0.0.000450 -> kukuha na sa entry mo tapos kukuha pa sa exit mo parang ganun porsyento
#     (pl.col('entry_trade_value') * maker_fee + pl.col('exit_trade_value') * maker_fee).alias('maker_fee')
# )
trades = trades.with_columns(
    (pl.col('entry_trade_value') * taker_fee + pl.col('exit_trade_value') * taker_fee).alias('taker_fee'),
    (pl.col('entry_trade_value') * maker_fee + pl.col('exit_trade_value') * maker_fee).alias('maker_fee')
)


# print(show(trades))

# print(trades.select('datetime', 'open', 'close', 'trade_log_return', 'y_hat', 'entry_trade_value', 'exit_trade_value', 'signed_trade_qty', 'trade_gross_pnl', 'taker_fee', 'maker_fee'))

### Calculate Trade Net PnL

trades = trades.with_columns(
    (pl.col('trade_gross_pnl') - pl.col('maker_fee')).alias('trade_net_maker_pnl'),
    (pl.col('trade_gross_pnl') - pl.col('taker_fee')).alias('trade_net_taker_pnl')
)


# print(show(trades))

### Display Equity Curves for constant sizing

def equity_curve(capital, col_name, suffix):
    return (capital +(pl.col(col_name).cum_sum())).alias(f'equity_curve_{suffix}')



trades = trades.with_columns(
    equity_curve(capital, 'trade_net_taker_pnl', 'taker'),
    equity_curve(capital, 'trade_net_maker_pnl', 'maker'),
    equity_curve(capital, 'trade_gross_pnl', 'gross'),
)

# print(trades.select("trade_net_maker_pnl", "trade_net_taker_pnl", "trade_gross_pnl", "equity_curve_taker", "equity_curve_maker", "equity_curve_gross"))


 
# ### Display equity curve(Log Space)
# plot_cum = researches.plot_column_matplotlib(trades, 'equity_curve_maker')
# plot_cum = researches.plot_column_matplotlib(trades, 'equity_curve_taker')

# Calculate Total Net Return using Constant Sizing
constant_sizing_net_return = trades['equity_curve_taker'][-1] / capital -1

#experiment with compounding trade sizes
# 2 => 102 (100+ 2)
# 1 => 103 (100 + 2 + 1)
# -1 => 102 (100 + 2 + 1 -1)

# log_return_1  = 0.005439
# pnl1 = capital * np.exp(log_return_1)
# pnl1

# log_return_2 =  0.008597
# pnl2 = pnl1 * np.exp(log_return_2)
# pnl2

# log_return_3 = 0.001385
# pnl3 = pnl2 * np.exp(log_return_3)

### Add Compounding Trade Sizes
trades = trades.with_columns(
    ((pl.col('trade_log_return').exp()) * capital).shift().fill_null(capital).alias('entry_trade_value'),
    ((pl.col('trade_log_return').exp()) * capital).alias('exit_trade_value'),
).with_columns(
    (pl.col('entry_trade_value') / pl.col('open') * pl.col('dir_signal')).alias('signed_trade_qty'),
    (pl.col('exit_trade_value') - pl.col('entry_trade_value')).alias('trade_gross_pnl'),
)



### Add Transactions Fee
trades = researches.add_tx_fees(trades, binance.MAKER_FEE, binance.TAKER_FEE)

### Add Trade Net PnL
trades = trades.with_columns(
    (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_maker_pnl')
)

# print(trades.select('trade_gross_pnl','tx_fee_taker', 'trade_net_taker_pnl'))
# trades = trades.with_columns(
#     (pl.col('trade_net_taker_pnl'))
# )
# print(trades.select('trade_gross_pnl', 'taker_fee', 'trade_net_taker_pnl'))

#Display Equity Curve (Compounding)
trades = researches.add_equity_curve(trades, capital, 'trade_net_taker_pnl', 'taker')


# researches.plot_static_timeseries(trades, sym, 'equity_curve_taker', time_interval)

compound_total_net_return = trades['equity_curve_taker'][-1] / capital -1

# print(constant_sizing_net_return)

# print(np.round(compound_total_net_return - constant_sizing_net_return, 2))


### Key Strategy Decision #3: Leverage

leverage = 4
leverage * capital

print(leverage * capital)

# in theory, we are multiplying our trade size to increase our returns
# we are not multiplying our trade returns
#just remember, it amplifies BOTH your profit and losses so it's important to have a positive expected value
#Leverage only works when you have a high Sharpe model. The more you reduce drawdowns and more leverage you can use
#key decision 3: should we use leverage? if so, how much? what's the sweet spot?
# It's not a golden utopia to multiplying profits, because it can easily wipe you out

trades = researches.add_compounding_trades(trades, capital, leverage, maker_fee, taker_fee)

trades = researches.add_compounding_trades(trades, capital, 8, maker_fee, taker_fee)

# print(trades['equity_curve_taker'][-1] / capital -1)

# reason for this huge return  is the combination of model's edge, compounding and leveraging
# this is possible on small scale because we trading such small size that we are not moving market around


### Show winrate to make sure we don't manipulate the winrate to make the pnl more look good

print(trades.select((pl.col('trade_log_return') > 0).mean()))


#Liquidation is when we go bust. If use too much leverage, then a small price changecan wipe us out.
#Leverage is a double edged sword. you can amplify profits, but too much leverage and you can wipe out all your money

# Equity = Maintenance Margin
#calculation differs from different exchanges

maintenance_margin = 0.005

def long_liquidation_price(p, l, mmr):
    return (p * l) / (l + 1 - mmr * l)

def short_liquidation_price(p, l, mmr):
    return (p * l) / (l - 1 - mmr * l)


### show how leverage affects long positions
# print(long_liquidation_price(200, 2, maintenance_margin))
# print(long_liquidation_price(200, 4, maintenance_margin))
# print(long_liquidation_price(200, 10, maintenance_margin))
# print(long_liquidation_price(200, 50, maintenance_margin))

# print(short_liquidation_price(200, 2, maintenance_margin))
# print(short_liquidation_price(200, 4, maintenance_margin))
# print(short_liquidation_price(200, 10, maintenance_margin))
# print(short_liquidation_price(200, 50, maintenance_margin))


### Add Liquidation Prices
leverage

trades = trades.with_columns(
    pl.when(pl.col("dir_signal") == 1) # long position
        .then(
            (pl.col("open") * leverage) 
            / (leverage + 1 - maintenance_margin * leverage)
        )
        .when(pl.col("dir_signal") == -1) # short position
        .then(
            (pl.col("open") * leverage)
        / (leverage - 1 + maintenance_margin * leverage)
        )
        .otherwise(None)
        .alias("liquidation_price")
)
print(trades.select('datetime','open','high','low','close','liquidation_price','dir_signal'))

### Add Liquidation Flag
trades = trades.with_columns([
    #Worst price based on direction
    pl.when(pl.col("dir_signal") == 1)
        .then(pl.col("low"))
        .otherwise(pl.col("high"))
        .alias("worst_price"),

    #Liquidation flag
    pl.when(
        (pl.col("dir_signal") == 1) & (pl.col("low") <= pl.col("liquidation_price")))
        .then(True)
        .when(
            (pl.col("dir_signal") == - 1) & (pl.col("high") >= pl.col('liquidation_price'))
        )
        .then(True)
        .otherwise(False)
        .alias('liquidated')
])

print(trades.select('datetime','open','low','high','close','dir_signal','worst_price','liquidation_price','liquidated')
)

print(trades.filter(pl.col("liquidated") == True))

# second part finished
# third part is coding this all up and putting this all into action - where i am going to put this live with small amount of money

# topics we haven't covered - this is just a foundation to build upon
# alpha decay (also known as model drift) => where the prediction performance drifts => 
# market impact => we are not trading big sizes => if we were, we could potentially move markets against us
# funding fees/rebates
# slippage => we may not always get the best price, we may get executed at prices below top of the book (best bid/ask)